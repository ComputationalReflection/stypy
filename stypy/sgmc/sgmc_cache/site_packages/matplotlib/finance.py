
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A collection of functions for collecting, analyzing and plotting
3: financial data.
4: 
5: This module is deprecated in 2.0 and has been moved to a module called
6: `mpl_finance`.
7: '''
8: from __future__ import (absolute_import, division, print_function,
9:                         unicode_literals)
10: 
11: import six
12: from six.moves import xrange, zip
13: 
14: import contextlib
15: import os
16: import warnings
17: from six.moves.urllib.request import urlopen
18: 
19: import datetime
20: 
21: import numpy as np
22: 
23: from matplotlib import colors as mcolors, verbose, get_cachedir
24: from matplotlib.dates import date2num
25: from matplotlib.cbook import iterable, mkdirs, warn_deprecated
26: from matplotlib.collections import LineCollection, PolyCollection
27: from matplotlib.lines import Line2D, TICKLEFT, TICKRIGHT
28: from matplotlib.patches import Rectangle
29: from matplotlib.transforms import Affine2D
30: 
31: warn_deprecated(
32:     since=2.0,
33:     message=("The finance module has been deprecated in mpl 2.0 and will "
34:              "be removed in mpl 2.2. Please use the module mpl_finance "
35:              "instead."))
36: 
37: 
38: if six.PY3:
39:     import hashlib
40: 
41:     def md5(x):
42:         return hashlib.md5(x.encode())
43: else:
44:     from hashlib import md5
45: 
46: cachedir = get_cachedir()
47: # cachedir will be None if there is no writable directory.
48: if cachedir is not None:
49:     cachedir = os.path.join(cachedir, 'finance.cache')
50: else:
51:     # Should only happen in a restricted environment (such as Google App
52:     # Engine). Deal with this gracefully by not caching finance data.
53:     cachedir = None
54: 
55: 
56: stock_dt_ohlc = np.dtype([
57:     (str('date'), object),
58:     (str('year'), np.int16),
59:     (str('month'), np.int8),
60:     (str('day'), np.int8),
61:     (str('d'), float),     # mpl datenum
62:     (str('open'), float),
63:     (str('high'), float),
64:     (str('low'), float),
65:     (str('close'), float),
66:     (str('volume'), float),
67:     (str('aclose'), float)])
68: 
69: 
70: stock_dt_ochl = np.dtype(
71:     [(str('date'), object),
72:      (str('year'), np.int16),
73:      (str('month'), np.int8),
74:      (str('day'), np.int8),
75:      (str('d'), float),     # mpl datenum
76:      (str('open'), float),
77:      (str('close'), float),
78:      (str('high'), float),
79:      (str('low'), float),
80:      (str('volume'), float),
81:      (str('aclose'), float)])
82: 
83: 
84: def parse_yahoo_historical_ochl(fh, adjusted=True, asobject=False):
85:     '''Parse the historical data in file handle fh from yahoo finance.
86: 
87:     Parameters
88:     ----------
89: 
90:     adjusted : bool
91:       If True (default) replace open, close, high, low prices with
92:       their adjusted values. The adjustment is by a scale factor, S =
93:       adjusted_close/close. Adjusted prices are actual prices
94:       multiplied by S.
95: 
96:       Volume is not adjusted as it is already backward split adjusted
97:       by Yahoo. If you want to compute dollars traded, multiply volume
98:       by the adjusted close, regardless of whether you choose adjusted
99:       = True|False.
100: 
101: 
102:     asobject : bool or None
103:       If False (default for compatibility with earlier versions)
104:       return a list of tuples containing
105: 
106:         d, open, close, high, low,  volume
107: 
108:       If None (preferred alternative to False), return
109:       a 2-D ndarray corresponding to the list of tuples.
110: 
111:       Otherwise return a numpy recarray with
112: 
113:         date, year, month, day, d, open, close, high, low,
114:         volume, adjusted_close
115: 
116:       where d is a floating poing representation of date,
117:       as returned by date2num, and date is a python standard
118:       library datetime.date instance.
119: 
120:       The name of this kwarg is a historical artifact.  Formerly,
121:       True returned a cbook Bunch
122:       holding 1-D ndarrays.  The behavior of a numpy recarray is
123:       very similar to the Bunch.
124: 
125:     '''
126:     return _parse_yahoo_historical(fh, adjusted=adjusted, asobject=asobject,
127:                            ochl=True)
128: 
129: 
130: def parse_yahoo_historical_ohlc(fh, adjusted=True, asobject=False):
131:     '''Parse the historical data in file handle fh from yahoo finance.
132: 
133:     Parameters
134:     ----------
135: 
136:     adjusted : bool
137:       If True (default) replace open, high, low, close prices with
138:       their adjusted values. The adjustment is by a scale factor, S =
139:       adjusted_close/close. Adjusted prices are actual prices
140:       multiplied by S.
141: 
142:       Volume is not adjusted as it is already backward split adjusted
143:       by Yahoo. If you want to compute dollars traded, multiply volume
144:       by the adjusted close, regardless of whether you choose adjusted
145:       = True|False.
146: 
147: 
148:     asobject : bool or None
149:       If False (default for compatibility with earlier versions)
150:       return a list of tuples containing
151: 
152:         d, open, high, low, close, volume
153: 
154:       If None (preferred alternative to False), return
155:       a 2-D ndarray corresponding to the list of tuples.
156: 
157:       Otherwise return a numpy recarray with
158: 
159:         date, year, month, day, d, open, high, low,  close,
160:         volume, adjusted_close
161: 
162:       where d is a floating poing representation of date,
163:       as returned by date2num, and date is a python standard
164:       library datetime.date instance.
165: 
166:       The name of this kwarg is a historical artifact.  Formerly,
167:       True returned a cbook Bunch
168:       holding 1-D ndarrays.  The behavior of a numpy recarray is
169:       very similar to the Bunch.
170:     '''
171:     return _parse_yahoo_historical(fh, adjusted=adjusted, asobject=asobject,
172:                            ochl=False)
173: 
174: 
175: def _parse_yahoo_historical(fh, adjusted=True, asobject=False,
176:                            ochl=True):
177:     '''Parse the historical data in file handle fh from yahoo finance.
178: 
179: 
180:     Parameters
181:     ----------
182: 
183:     adjusted : bool
184:       If True (default) replace open, high, low, close prices with
185:       their adjusted values. The adjustment is by a scale factor, S =
186:       adjusted_close/close. Adjusted prices are actual prices
187:       multiplied by S.
188: 
189:       Volume is not adjusted as it is already backward split adjusted
190:       by Yahoo. If you want to compute dollars traded, multiply volume
191:       by the adjusted close, regardless of whether you choose adjusted
192:       = True|False.
193: 
194: 
195:     asobject : bool or None
196:       If False (default for compatibility with earlier versions)
197:       return a list of tuples containing
198: 
199:         d, open, high, low, close, volume
200: 
201:        or
202: 
203:         d, open, close, high, low, volume
204: 
205:       depending on `ochl`
206: 
207:       If None (preferred alternative to False), return
208:       a 2-D ndarray corresponding to the list of tuples.
209: 
210:       Otherwise return a numpy recarray with
211: 
212:         date, year, month, day, d, open, high, low, close,
213:         volume, adjusted_close
214: 
215:       where d is a floating poing representation of date,
216:       as returned by date2num, and date is a python standard
217:       library datetime.date instance.
218: 
219:       The name of this kwarg is a historical artifact.  Formerly,
220:       True returned a cbook Bunch
221:       holding 1-D ndarrays.  The behavior of a numpy recarray is
222:       very similar to the Bunch.
223: 
224:     ochl : bool
225:         Selects between ochl and ohlc ordering.
226:         Defaults to True to preserve original functionality.
227: 
228:     '''
229:     if ochl:
230:         stock_dt = stock_dt_ochl
231:     else:
232:         stock_dt = stock_dt_ohlc
233: 
234:     results = []
235: 
236:     #    datefmt = '%Y-%m-%d'
237:     fh.readline()  # discard heading
238:     for line in fh:
239: 
240:         vals = line.split(',')
241:         if len(vals) != 7:
242:             continue      # add warning?
243:         datestr = vals[0]
244:         #dt = datetime.date(*time.strptime(datestr, datefmt)[:3])
245:         # Using strptime doubles the runtime. With the present
246:         # format, we don't need it.
247:         dt = datetime.date(*[int(val) for val in datestr.split('-')])
248:         dnum = date2num(dt)
249:         open, high, low, close = [float(val) for val in vals[1:5]]
250:         volume = float(vals[5])
251:         aclose = float(vals[6])
252:         if ochl:
253:             results.append((dt, dt.year, dt.month, dt.day,
254:                             dnum, open, close, high, low, volume, aclose))
255: 
256:         else:
257:             results.append((dt, dt.year, dt.month, dt.day,
258:                             dnum, open, high, low, close, volume, aclose))
259:     results.reverse()
260:     d = np.array(results, dtype=stock_dt)
261:     if adjusted:
262:         scale = d['aclose'] / d['close']
263:         scale[np.isinf(scale)] = np.nan
264:         d['open'] *= scale
265:         d['high'] *= scale
266:         d['low'] *= scale
267:         d['close'] *= scale
268: 
269:     if not asobject:
270:         # 2-D sequence; formerly list of tuples, now ndarray
271:         ret = np.zeros((len(d), 6), dtype=float)
272:         ret[:, 0] = d['d']
273:         if ochl:
274:             ret[:, 1] = d['open']
275:             ret[:, 2] = d['close']
276:             ret[:, 3] = d['high']
277:             ret[:, 4] = d['low']
278:         else:
279:             ret[:, 1] = d['open']
280:             ret[:, 2] = d['high']
281:             ret[:, 3] = d['low']
282:             ret[:, 4] = d['close']
283:         ret[:, 5] = d['volume']
284:         if asobject is None:
285:             return ret
286:         return [tuple(row) for row in ret]
287: 
288:     return d.view(np.recarray)  # Close enough to former Bunch return
289: 
290: 
291: def fetch_historical_yahoo(ticker, date1, date2, cachename=None,
292:                            dividends=False):
293:     '''
294:     Fetch historical data for ticker between date1 and date2.  date1 and
295:     date2 are date or datetime instances, or (year, month, day) sequences.
296: 
297:     Parameters
298:     ----------
299:     ticker : str
300:         ticker
301: 
302:     date1 : sequence of form (year, month, day), `datetime`, or `date`
303:         start date
304:     date2 : sequence of form (year, month, day), `datetime`, or `date`
305:         end date
306: 
307:     cachename : str
308:         cachename is the name of the local file cache.  If None, will
309:         default to the md5 hash or the url (which incorporates the ticker
310:         and date range)
311: 
312:     dividends : bool
313:         set dividends=True to return dividends instead of price data.  With
314:         this option set, parse functions will not work
315: 
316:     Returns
317:     -------
318:     file_handle : file handle
319:         a file handle is returned
320: 
321: 
322:     Examples
323:     --------
324:     >>> fh = fetch_historical_yahoo('^GSPC', (2000, 1, 1), (2001, 12, 31))
325: 
326:     '''
327: 
328:     ticker = ticker.upper()
329: 
330:     if iterable(date1):
331:         d1 = (date1[1] - 1, date1[2], date1[0])
332:     else:
333:         d1 = (date1.month - 1, date1.day, date1.year)
334:     if iterable(date2):
335:         d2 = (date2[1] - 1, date2[2], date2[0])
336:     else:
337:         d2 = (date2.month - 1, date2.day, date2.year)
338: 
339:     if dividends:
340:         g = 'v'
341:         verbose.report('Retrieving dividends instead of prices')
342:     else:
343:         g = 'd'
344: 
345:     urlFmt = ('http://real-chart.finance.yahoo.com/table.csv?' +
346:               '&s=%s&d=%d&e=%d&f=%d&g=%s&a=%d&b=%d&c=%d&ignore=.csv')
347: 
348:     url = urlFmt % (ticker, d2[0], d2[1], d2[2], g, d1[0], d1[1], d1[2])
349:     # Cache the finance data if cachename is supplied, or there is a writable
350:     # cache directory.
351:     if cachename is None and cachedir is not None:
352:         cachename = os.path.join(cachedir, md5(url).hexdigest())
353:     if cachename is not None:
354:         if os.path.exists(cachename):
355:             fh = open(cachename)
356:             verbose.report('Using cachefile %s for '
357:                            '%s' % (cachename, ticker))
358:         else:
359:             mkdirs(os.path.abspath(os.path.dirname(cachename)))
360:             with contextlib.closing(urlopen(url)) as urlfh:
361:                 with open(cachename, 'wb') as fh:
362:                     fh.write(urlfh.read())
363:             verbose.report('Saved %s data to cache file '
364:                            '%s' % (ticker, cachename))
365:             fh = open(cachename, 'r')
366: 
367:         return fh
368:     else:
369:         return urlopen(url)
370: 
371: 
372: def quotes_historical_yahoo_ochl(ticker, date1, date2, asobject=False,
373:                             adjusted=True, cachename=None):
374:     ''' Get historical data for ticker between date1 and date2.
375: 
376: 
377:     See :func:`parse_yahoo_historical` for explanation of output formats
378:     and the *asobject* and *adjusted* kwargs.
379: 
380:     Parameters
381:     ----------
382:     ticker : str
383:         stock ticker
384: 
385:     date1 : sequence of form (year, month, day), `datetime`, or `date`
386:         start date
387: 
388:     date2 : sequence of form (year, month, day), `datetime`, or `date`
389:         end date
390: 
391:     cachename : str or `None`
392:         is the name of the local file cache.  If None, will
393:         default to the md5 hash or the url (which incorporates the ticker
394:         and date range)
395: 
396:     Examples
397:     --------
398:     >>> sp = f.quotes_historical_yahoo_ochl('^GSPC', d1, d2,
399:                              asobject=True, adjusted=True)
400:     >>> returns = (sp.open[1:] - sp.open[:-1])/sp.open[1:]
401:     >>> [n,bins,patches] = hist(returns, 100)
402:     >>> mu = mean(returns)
403:     >>> sigma = std(returns)
404:     >>> x = normpdf(bins, mu, sigma)
405:     >>> plot(bins, x, color='red', lw=2)
406: 
407:     '''
408: 
409:     return _quotes_historical_yahoo(ticker, date1, date2, asobject=asobject,
410:                              adjusted=adjusted, cachename=cachename,
411:                              ochl=True)
412: 
413: 
414: def quotes_historical_yahoo_ohlc(ticker, date1, date2, asobject=False,
415:                             adjusted=True, cachename=None):
416:     ''' Get historical data for ticker between date1 and date2.
417: 
418: 
419:     See :func:`parse_yahoo_historical` for explanation of output formats
420:     and the *asobject* and *adjusted* kwargs.
421: 
422:     Parameters
423:     ----------
424:     ticker : str
425:         stock ticker
426: 
427:     date1 : sequence of form (year, month, day), `datetime`, or `date`
428:         start date
429: 
430:     date2 : sequence of form (year, month, day), `datetime`, or `date`
431:         end date
432: 
433:     cachename : str or `None`
434:         is the name of the local file cache.  If None, will
435:         default to the md5 hash or the url (which incorporates the ticker
436:         and date range)
437: 
438:     Examples
439:     --------
440:     >>> sp = f.quotes_historical_yahoo_ohlc('^GSPC', d1, d2,
441:                              asobject=True, adjusted=True)
442:     >>> returns = (sp.open[1:] - sp.open[:-1])/sp.open[1:]
443:     >>> [n,bins,patches] = hist(returns, 100)
444:     >>> mu = mean(returns)
445:     >>> sigma = std(returns)
446:     >>> x = normpdf(bins, mu, sigma)
447:     >>> plot(bins, x, color='red', lw=2)
448: 
449:     '''
450: 
451:     return _quotes_historical_yahoo(ticker, date1, date2, asobject=asobject,
452:                              adjusted=adjusted, cachename=cachename,
453:                              ochl=False)
454: 
455: 
456: def _quotes_historical_yahoo(ticker, date1, date2, asobject=False,
457:                             adjusted=True, cachename=None,
458:                             ochl=True):
459:     ''' Get historical data for ticker between date1 and date2.
460: 
461:     See :func:`parse_yahoo_historical` for explanation of output formats
462:     and the *asobject* and *adjusted* kwargs.
463: 
464:     Parameters
465:     ----------
466:     ticker : str
467:         stock ticker
468: 
469:     date1 : sequence of form (year, month, day), `datetime`, or `date`
470:         start date
471: 
472:     date2 : sequence of form (year, month, day), `datetime`, or `date`
473:         end date
474: 
475:     cachename : str or `None`
476:         is the name of the local file cache.  If None, will
477:         default to the md5 hash or the url (which incorporates the ticker
478:         and date range)
479: 
480:     ochl: bool
481:         temporary argument to select between ochl and ohlc ordering
482: 
483: 
484:     Examples
485:     --------
486:     >>> sp = f.quotes_historical_yahoo('^GSPC', d1, d2,
487:                              asobject=True, adjusted=True)
488:     >>> returns = (sp.open[1:] - sp.open[:-1])/sp.open[1:]
489:     >>> [n,bins,patches] = hist(returns, 100)
490:     >>> mu = mean(returns)
491:     >>> sigma = std(returns)
492:     >>> x = normpdf(bins, mu, sigma)
493:     >>> plot(bins, x, color='red', lw=2)
494: 
495:     '''
496:     # Maybe enable a warning later as part of a slow transition
497:     # to using None instead of False.
498:     #if asobject is False:
499:     #    warnings.warn("Recommend changing to asobject=None")
500: 
501:     fh = fetch_historical_yahoo(ticker, date1, date2, cachename)
502: 
503:     try:
504:         ret = _parse_yahoo_historical(fh, asobject=asobject,
505:                                      adjusted=adjusted, ochl=ochl)
506:         if len(ret) == 0:
507:             return None
508:     except IOError as exc:
509:         warnings.warn('fh failure\n%s' % (exc.strerror[1]))
510:         return None
511: 
512:     return ret
513: 
514: 
515: def plot_day_summary_oclh(ax, quotes, ticksize=3,
516:                      colorup='k', colordown='r',
517:                      ):
518:     '''Plots day summary
519: 
520:         Represent the time, open, close, high, low as a vertical line
521:         ranging from low to high.  The left tick is the open and the right
522:         tick is the close.
523: 
524: 
525: 
526:     Parameters
527:     ----------
528:     ax : `Axes`
529:         an `Axes` instance to plot to
530:     quotes : sequence of (time, open, close, high, low, ...) sequences
531:         data to plot.  time must be in float date format - see date2num
532:     ticksize : int
533:         open/close tick marker in points
534:     colorup : color
535:         the color of the lines where close >= open
536:     colordown : color
537:         the color of the lines where close <  open
538: 
539:     Returns
540:     -------
541:     lines : list
542:         list of tuples of the lines added (one tuple per quote)
543:     '''
544:     return _plot_day_summary(ax, quotes, ticksize=ticksize,
545:                      colorup=colorup, colordown=colordown,
546:                      ochl=True)
547: 
548: 
549: def plot_day_summary_ohlc(ax, quotes, ticksize=3,
550:                      colorup='k', colordown='r',
551:                       ):
552:     '''Plots day summary
553: 
554:         Represent the time, open, high, low, close as a vertical line
555:         ranging from low to high.  The left tick is the open and the right
556:         tick is the close.
557: 
558: 
559: 
560:     Parameters
561:     ----------
562:     ax : `Axes`
563:         an `Axes` instance to plot to
564:     quotes : sequence of (time, open, high, low, close, ...) sequences
565:         data to plot.  time must be in float date format - see date2num
566:     ticksize : int
567:         open/close tick marker in points
568:     colorup : color
569:         the color of the lines where close >= open
570:     colordown : color
571:         the color of the lines where close <  open
572: 
573:     Returns
574:     -------
575:     lines : list
576:         list of tuples of the lines added (one tuple per quote)
577:     '''
578:     return _plot_day_summary(ax, quotes, ticksize=ticksize,
579:                      colorup=colorup, colordown=colordown,
580:                      ochl=False)
581: 
582: 
583: def _plot_day_summary(ax, quotes, ticksize=3,
584:                      colorup='k', colordown='r',
585:                      ochl=True
586:                      ):
587:     '''Plots day summary
588: 
589: 
590:         Represent the time, open, high, low, close as a vertical line
591:         ranging from low to high.  The left tick is the open and the right
592:         tick is the close.
593: 
594: 
595: 
596:     Parameters
597:     ----------
598:     ax : `Axes`
599:         an `Axes` instance to plot to
600:     quotes : sequence of quote sequences
601:         data to plot.  time must be in float date format - see date2num
602:         (time, open, high, low, close, ...) vs
603:         (time, open, close, high, low, ...)
604:         set by `ochl`
605:     ticksize : int
606:         open/close tick marker in points
607:     colorup : color
608:         the color of the lines where close >= open
609:     colordown : color
610:         the color of the lines where close <  open
611:     ochl: bool
612:         argument to select between ochl and ohlc ordering of quotes
613: 
614:     Returns
615:     -------
616:     lines : list
617:         list of tuples of the lines added (one tuple per quote)
618:     '''
619:     # unfortunately this has a different return type than plot_day_summary2_*
620:     lines = []
621:     for q in quotes:
622:         if ochl:
623:             t, open, close, high, low = q[:5]
624:         else:
625:             t, open, high, low, close = q[:5]
626: 
627:         if close >= open:
628:             color = colorup
629:         else:
630:             color = colordown
631: 
632:         vline = Line2D(xdata=(t, t), ydata=(low, high),
633:                        color=color,
634:                        antialiased=False,   # no need to antialias vert lines
635:                        )
636: 
637:         oline = Line2D(xdata=(t, t), ydata=(open, open),
638:                        color=color,
639:                        antialiased=False,
640:                        marker=TICKLEFT,
641:                        markersize=ticksize,
642:                        )
643: 
644:         cline = Line2D(xdata=(t, t), ydata=(close, close),
645:                        color=color,
646:                        antialiased=False,
647:                        markersize=ticksize,
648:                        marker=TICKRIGHT)
649: 
650:         lines.extend((vline, oline, cline))
651:         ax.add_line(vline)
652:         ax.add_line(oline)
653:         ax.add_line(cline)
654: 
655:     ax.autoscale_view()
656: 
657:     return lines
658: 
659: 
660: def candlestick_ochl(ax, quotes, width=0.2, colorup='k', colordown='r',
661:                 alpha=1.0):
662: 
663:     '''
664:     Plot the time, open, close, high, low as a vertical line ranging
665:     from low to high.  Use a rectangular bar to represent the
666:     open-close span.  If close >= open, use colorup to color the bar,
667:     otherwise use colordown
668: 
669:     Parameters
670:     ----------
671:     ax : `Axes`
672:         an Axes instance to plot to
673:     quotes : sequence of (time, open, close, high, low, ...) sequences
674:         As long as the first 5 elements are these values,
675:         the record can be as long as you want (e.g., it may store volume).
676: 
677:         time must be in float days format - see date2num
678: 
679:     width : float
680:         fraction of a day for the rectangle width
681:     colorup : color
682:         the color of the rectangle where close >= open
683:     colordown : color
684:          the color of the rectangle where close <  open
685:     alpha : float
686:         the rectangle alpha level
687: 
688:     Returns
689:     -------
690:     ret : tuple
691:         returns (lines, patches) where lines is a list of lines
692:         added and patches is a list of the rectangle patches added
693: 
694:     '''
695:     return _candlestick(ax, quotes, width=width, colorup=colorup,
696:                         colordown=colordown,
697:                         alpha=alpha, ochl=True)
698: 
699: 
700: def candlestick_ohlc(ax, quotes, width=0.2, colorup='k', colordown='r',
701:                 alpha=1.0):
702: 
703:     '''
704:     Plot the time, open, high, low, close as a vertical line ranging
705:     from low to high.  Use a rectangular bar to represent the
706:     open-close span.  If close >= open, use colorup to color the bar,
707:     otherwise use colordown
708: 
709:     Parameters
710:     ----------
711:     ax : `Axes`
712:         an Axes instance to plot to
713:     quotes : sequence of (time, open, high, low, close, ...) sequences
714:         As long as the first 5 elements are these values,
715:         the record can be as long as you want (e.g., it may store volume).
716: 
717:         time must be in float days format - see date2num
718: 
719:     width : float
720:         fraction of a day for the rectangle width
721:     colorup : color
722:         the color of the rectangle where close >= open
723:     colordown : color
724:          the color of the rectangle where close <  open
725:     alpha : float
726:         the rectangle alpha level
727: 
728:     Returns
729:     -------
730:     ret : tuple
731:         returns (lines, patches) where lines is a list of lines
732:         added and patches is a list of the rectangle patches added
733: 
734:     '''
735:     return _candlestick(ax, quotes, width=width, colorup=colorup,
736:                         colordown=colordown,
737:                         alpha=alpha, ochl=False)
738: 
739: 
740: def _candlestick(ax, quotes, width=0.2, colorup='k', colordown='r',
741:                  alpha=1.0, ochl=True):
742: 
743:     '''
744:     Plot the time, open, high, low, close as a vertical line ranging
745:     from low to high.  Use a rectangular bar to represent the
746:     open-close span.  If close >= open, use colorup to color the bar,
747:     otherwise use colordown
748: 
749:     Parameters
750:     ----------
751:     ax : `Axes`
752:         an Axes instance to plot to
753:     quotes : sequence of quote sequences
754:         data to plot.  time must be in float date format - see date2num
755:         (time, open, high, low, close, ...) vs
756:         (time, open, close, high, low, ...)
757:         set by `ochl`
758:     width : float
759:         fraction of a day for the rectangle width
760:     colorup : color
761:         the color of the rectangle where close >= open
762:     colordown : color
763:          the color of the rectangle where close <  open
764:     alpha : float
765:         the rectangle alpha level
766:     ochl: bool
767:         argument to select between ochl and ohlc ordering of quotes
768: 
769:     Returns
770:     -------
771:     ret : tuple
772:         returns (lines, patches) where lines is a list of lines
773:         added and patches is a list of the rectangle patches added
774: 
775:     '''
776: 
777:     OFFSET = width / 2.0
778: 
779:     lines = []
780:     patches = []
781:     for q in quotes:
782:         if ochl:
783:             t, open, close, high, low = q[:5]
784:         else:
785:             t, open, high, low, close = q[:5]
786: 
787:         if close >= open:
788:             color = colorup
789:             lower = open
790:             height = close - open
791:         else:
792:             color = colordown
793:             lower = close
794:             height = open - close
795: 
796:         vline = Line2D(
797:             xdata=(t, t), ydata=(low, high),
798:             color=color,
799:             linewidth=0.5,
800:             antialiased=True,
801:         )
802: 
803:         rect = Rectangle(
804:             xy=(t - OFFSET, lower),
805:             width=width,
806:             height=height,
807:             facecolor=color,
808:             edgecolor=color,
809:         )
810:         rect.set_alpha(alpha)
811: 
812:         lines.append(vline)
813:         patches.append(rect)
814:         ax.add_line(vline)
815:         ax.add_patch(rect)
816:     ax.autoscale_view()
817: 
818:     return lines, patches
819: 
820: 
821: def _check_input(opens, closes, highs, lows, miss=-1):
822:     '''Checks that *opens*, *highs*, *lows* and *closes* have the same length.
823:     NOTE: this code assumes if any value open, high, low, close is
824:     missing (*-1*) they all are missing
825: 
826:     Parameters
827:     ----------
828:     ax : `Axes`
829:         an Axes instance to plot to
830:     opens : sequence
831:         sequence of opening values
832:     highs : sequence
833:         sequence of high values
834:     lows : sequence
835:         sequence of low values
836:     closes : sequence
837:         sequence of closing values
838:     miss : int
839:         identifier of the missing data
840: 
841:     Raises
842:     ------
843:     ValueError
844:         if the input sequences don't have the same length
845:     '''
846: 
847:     def _missing(sequence, miss=-1):
848:         '''Returns the index in *sequence* of the missing data, identified by
849:         *miss*
850: 
851:         Parameters
852:         ----------
853:         sequence :
854:             sequence to evaluate
855:         miss :
856:             identifier of the missing data
857: 
858:         Returns
859:         -------
860:         where_miss: numpy.ndarray
861:             indices of the missing data
862:         '''
863:         return np.where(np.array(sequence) == miss)[0]
864: 
865:     same_length = len(opens) == len(highs) == len(lows) == len(closes)
866:     _missopens = _missing(opens)
867:     same_missing = ((_missopens == _missing(highs)).all() and
868:                     (_missopens == _missing(lows)).all() and
869:                     (_missopens == _missing(closes)).all())
870: 
871:     if not (same_length and same_missing):
872:         msg = ("*opens*, *highs*, *lows* and *closes* must have the same"
873:                " length. NOTE: this code assumes if any value open, high,"
874:                " low, close is missing (*-1*) they all must be missing.")
875:         raise ValueError(msg)
876: 
877: 
878: def plot_day_summary2_ochl(ax, opens, closes, highs, lows, ticksize=4,
879:                           colorup='k', colordown='r',
880:                           ):
881: 
882:     '''Represent the time, open, close, high, low,  as a vertical line
883:     ranging from low to high.  The left tick is the open and the right
884:     tick is the close.
885: 
886:     Parameters
887:     ----------
888:     ax : `Axes`
889:         an Axes instance to plot to
890:     opens : sequence
891:         sequence of opening values
892:     closes : sequence
893:         sequence of closing values
894:     highs : sequence
895:         sequence of high values
896:     lows : sequence
897:         sequence of low values
898:     ticksize : int
899:         size of open and close ticks in points
900:     colorup : color
901:         the color of the lines where close >= open
902:     colordown : color
903:          the color of the lines where close <  open
904: 
905:     Returns
906:     -------
907:     ret : list
908:         a list of lines added to the axes
909:     '''
910: 
911:     return plot_day_summary2_ohlc(ax, opens, highs, lows, closes, ticksize,
912:                                  colorup, colordown)
913: 
914: 
915: def plot_day_summary2_ohlc(ax, opens, highs, lows, closes, ticksize=4,
916:                           colorup='k', colordown='r',
917:                           ):
918: 
919:     '''Represent the time, open, high, low, close as a vertical line
920:     ranging from low to high.  The left tick is the open and the right
921:     tick is the close.
922:     *opens*, *highs*, *lows* and *closes* must have the same length.
923:     NOTE: this code assumes if any value open, high, low, close is
924:     missing (*-1*) they all are missing
925: 
926:     Parameters
927:     ----------
928:     ax : `Axes`
929:         an Axes instance to plot to
930:     opens : sequence
931:         sequence of opening values
932:     highs : sequence
933:         sequence of high values
934:     lows : sequence
935:         sequence of low values
936:     closes : sequence
937:         sequence of closing values
938:     ticksize : int
939:         size of open and close ticks in points
940:     colorup : color
941:         the color of the lines where close >= open
942:     colordown : color
943:          the color of the lines where close <  open
944: 
945:     Returns
946:     -------
947:     ret : list
948:         a list of lines added to the axes
949:     '''
950: 
951:     _check_input(opens, highs, lows, closes)
952: 
953:     rangeSegments = [((i, low), (i, high)) for i, low, high in
954:                      zip(xrange(len(lows)), lows, highs) if low != -1]
955: 
956:     # the ticks will be from ticksize to 0 in points at the origin and
957:     # we'll translate these to the i, close location
958:     openSegments = [((-ticksize, 0), (0, 0))]
959: 
960:     # the ticks will be from 0 to ticksize in points at the origin and
961:     # we'll translate these to the i, close location
962:     closeSegments = [((0, 0), (ticksize, 0))]
963: 
964:     offsetsOpen = [(i, open) for i, open in
965:                    zip(xrange(len(opens)), opens) if open != -1]
966: 
967:     offsetsClose = [(i, close) for i, close in
968:                     zip(xrange(len(closes)), closes) if close != -1]
969: 
970:     scale = ax.figure.dpi * (1.0 / 72.0)
971: 
972:     tickTransform = Affine2D().scale(scale, 0.0)
973: 
974:     colorup = mcolors.to_rgba(colorup)
975:     colordown = mcolors.to_rgba(colordown)
976:     colord = {True: colorup, False: colordown}
977:     colors = [colord[open < close] for open, close in
978:               zip(opens, closes) if open != -1 and close != -1]
979: 
980:     useAA = 0,   # use tuple here
981:     lw = 1,      # and here
982:     rangeCollection = LineCollection(rangeSegments,
983:                                      colors=colors,
984:                                      linewidths=lw,
985:                                      antialiaseds=useAA,
986:                                      )
987: 
988:     openCollection = LineCollection(openSegments,
989:                                     colors=colors,
990:                                     antialiaseds=useAA,
991:                                     linewidths=lw,
992:                                     offsets=offsetsOpen,
993:                                     transOffset=ax.transData,
994:                                     )
995:     openCollection.set_transform(tickTransform)
996: 
997:     closeCollection = LineCollection(closeSegments,
998:                                      colors=colors,
999:                                      antialiaseds=useAA,
1000:                                      linewidths=lw,
1001:                                      offsets=offsetsClose,
1002:                                      transOffset=ax.transData,
1003:                                      )
1004:     closeCollection.set_transform(tickTransform)
1005: 
1006:     minpy, maxx = (0, len(rangeSegments))
1007:     miny = min([low for low in lows if low != -1])
1008:     maxy = max([high for high in highs if high != -1])
1009:     corners = (minpy, miny), (maxx, maxy)
1010:     ax.update_datalim(corners)
1011:     ax.autoscale_view()
1012: 
1013:     # add these last
1014:     ax.add_collection(rangeCollection)
1015:     ax.add_collection(openCollection)
1016:     ax.add_collection(closeCollection)
1017:     return rangeCollection, openCollection, closeCollection
1018: 
1019: 
1020: def candlestick2_ochl(ax, opens, closes, highs, lows,  width=4,
1021:                  colorup='k', colordown='r',
1022:                  alpha=0.75,
1023:                  ):
1024:     '''Represent the open, close as a bar line and high low range as a
1025:     vertical line.
1026: 
1027:     Preserves the original argument order.
1028: 
1029: 
1030:     Parameters
1031:     ----------
1032:     ax : `Axes`
1033:         an Axes instance to plot to
1034:     opens : sequence
1035:         sequence of opening values
1036:     closes : sequence
1037:         sequence of closing values
1038:     highs : sequence
1039:         sequence of high values
1040:     lows : sequence
1041:         sequence of low values
1042:     ticksize : int
1043:         size of open and close ticks in points
1044:     colorup : color
1045:         the color of the lines where close >= open
1046:     colordown : color
1047:         the color of the lines where close <  open
1048:     alpha : float
1049:         bar transparency
1050: 
1051:     Returns
1052:     -------
1053:     ret : tuple
1054:         (lineCollection, barCollection)
1055:     '''
1056: 
1057:     candlestick2_ohlc(ax, opens, highs, lows, closes, width=width,
1058:                      colorup=colorup, colordown=colordown,
1059:                      alpha=alpha)
1060: 
1061: 
1062: def candlestick2_ohlc(ax, opens, highs, lows, closes, width=4,
1063:                  colorup='k', colordown='r',
1064:                  alpha=0.75,
1065:                  ):
1066:     '''Represent the open, close as a bar line and high low range as a
1067:     vertical line.
1068: 
1069:     NOTE: this code assumes if any value open, low, high, close is
1070:     missing they all are missing
1071: 
1072: 
1073:     Parameters
1074:     ----------
1075:     ax : `Axes`
1076:         an Axes instance to plot to
1077:     opens : sequence
1078:         sequence of opening values
1079:     highs : sequence
1080:         sequence of high values
1081:     lows : sequence
1082:         sequence of low values
1083:     closes : sequence
1084:         sequence of closing values
1085:     ticksize : int
1086:         size of open and close ticks in points
1087:     colorup : color
1088:         the color of the lines where close >= open
1089:     colordown : color
1090:         the color of the lines where close <  open
1091:     alpha : float
1092:         bar transparency
1093: 
1094:     Returns
1095:     -------
1096:     ret : tuple
1097:         (lineCollection, barCollection)
1098:     '''
1099: 
1100:     _check_input(opens, highs, lows, closes)
1101: 
1102:     delta = width / 2.
1103:     barVerts = [((i - delta, open),
1104:                  (i - delta, close),
1105:                  (i + delta, close),
1106:                  (i + delta, open))
1107:                 for i, open, close in zip(xrange(len(opens)), opens, closes)
1108:                 if open != -1 and close != -1]
1109: 
1110:     rangeSegments = [((i, low), (i, high))
1111:                      for i, low, high in zip(xrange(len(lows)), lows, highs)
1112:                      if low != -1]
1113: 
1114:     colorup = mcolors.to_rgba(colorup, alpha)
1115:     colordown = mcolors.to_rgba(colordown, alpha)
1116:     colord = {True: colorup, False: colordown}
1117:     colors = [colord[open < close]
1118:               for open, close in zip(opens, closes)
1119:               if open != -1 and close != -1]
1120: 
1121:     useAA = 0,  # use tuple here
1122:     lw = 0.5,   # and here
1123:     rangeCollection = LineCollection(rangeSegments,
1124:                                      colors=((0, 0, 0, 1), ),
1125:                                      linewidths=lw,
1126:                                      antialiaseds=useAA,
1127:                                      )
1128: 
1129:     barCollection = PolyCollection(barVerts,
1130:                                    facecolors=colors,
1131:                                    edgecolors=((0, 0, 0, 1), ),
1132:                                    antialiaseds=useAA,
1133:                                    linewidths=lw,
1134:                                    )
1135: 
1136:     minx, maxx = 0, len(rangeSegments)
1137:     miny = min([low for low in lows if low != -1])
1138:     maxy = max([high for high in highs if high != -1])
1139: 
1140:     corners = (minx, miny), (maxx, maxy)
1141:     ax.update_datalim(corners)
1142:     ax.autoscale_view()
1143: 
1144:     # add these last
1145:     ax.add_collection(rangeCollection)
1146:     ax.add_collection(barCollection)
1147:     return rangeCollection, barCollection
1148: 
1149: 
1150: def volume_overlay(ax, opens, closes, volumes,
1151:                    colorup='k', colordown='r',
1152:                    width=4, alpha=1.0):
1153:     '''Add a volume overlay to the current axes.  The opens and closes
1154:     are used to determine the color of the bar.  -1 is missing.  If a
1155:     value is missing on one it must be missing on all
1156: 
1157:     Parameters
1158:     ----------
1159:     ax : `Axes`
1160:         an Axes instance to plot to
1161:     opens : sequence
1162:         a sequence of opens
1163:     closes : sequence
1164:         a sequence of closes
1165:     volumes : sequence
1166:         a sequence of volumes
1167:     width : int
1168:         the bar width in points
1169:     colorup : color
1170:         the color of the lines where close >= open
1171:     colordown : color
1172:         the color of the lines where close <  open
1173:     alpha : float
1174:         bar transparency
1175: 
1176:     Returns
1177:     -------
1178:     ret : `barCollection`
1179:         The `barrCollection` added to the axes
1180: 
1181:     '''
1182: 
1183:     colorup = mcolors.to_rgba(colorup, alpha)
1184:     colordown = mcolors.to_rgba(colordown, alpha)
1185:     colord = {True: colorup, False: colordown}
1186:     colors = [colord[open < close]
1187:               for open, close in zip(opens, closes)
1188:               if open != -1 and close != -1]
1189: 
1190:     delta = width / 2.
1191:     bars = [((i - delta, 0), (i - delta, v), (i + delta, v), (i + delta, 0))
1192:             for i, v in enumerate(volumes)
1193:             if v != -1]
1194: 
1195:     barCollection = PolyCollection(bars,
1196:                                    facecolors=colors,
1197:                                    edgecolors=((0, 0, 0, 1), ),
1198:                                    antialiaseds=(0,),
1199:                                    linewidths=(0.5,),
1200:                                    )
1201: 
1202:     ax.add_collection(barCollection)
1203:     corners = (0, 0), (len(bars), max(volumes))
1204:     ax.update_datalim(corners)
1205:     ax.autoscale_view()
1206: 
1207:     # add these last
1208:     return barCollection
1209: 
1210: 
1211: def volume_overlay2(ax, closes, volumes,
1212:                     colorup='k', colordown='r',
1213:                     width=4, alpha=1.0):
1214:     '''
1215:     Add a volume overlay to the current axes.  The closes are used to
1216:     determine the color of the bar.  -1 is missing.  If a value is
1217:     missing on one it must be missing on all
1218: 
1219:     nb: first point is not displayed - it is used only for choosing the
1220:     right color
1221: 
1222: 
1223:     Parameters
1224:     ----------
1225:     ax : `Axes`
1226:         an Axes instance to plot to
1227:     closes : sequence
1228:         a sequence of closes
1229:     volumes : sequence
1230:         a sequence of volumes
1231:     width : int
1232:         the bar width in points
1233:     colorup : color
1234:         the color of the lines where close >= open
1235:     colordown : color
1236:         the color of the lines where close <  open
1237:     alpha : float
1238:         bar transparency
1239: 
1240:     Returns
1241:     -------
1242:     ret : `barCollection`
1243:         The `barrCollection` added to the axes
1244: 
1245:     '''
1246: 
1247:     return volume_overlay(ax, closes[:-1], closes[1:], volumes[1:],
1248:                           colorup, colordown, width, alpha)
1249: 
1250: 
1251: def volume_overlay3(ax, quotes,
1252:                     colorup='k', colordown='r',
1253:                     width=4, alpha=1.0):
1254:     '''Add a volume overlay to the current axes.  quotes is a list of (d,
1255:     open, high, low, close, volume) and close-open is used to
1256:     determine the color of the bar
1257: 
1258:     Parameters
1259:     ----------
1260:     ax : `Axes`
1261:         an Axes instance to plot to
1262:     quotes : sequence of (time, open, high, low, close, ...) sequences
1263:         data to plot.  time must be in float date format - see date2num
1264:     width : int
1265:         the bar width in points
1266:     colorup : color
1267:         the color of the lines where close1 >= close0
1268:     colordown : color
1269:         the color of the lines where close1 <  close0
1270:     alpha : float
1271:          bar transparency
1272: 
1273:     Returns
1274:     -------
1275:     ret : `barCollection`
1276:         The `barrCollection` added to the axes
1277: 
1278: 
1279:     '''
1280: 
1281:     colorup = mcolors.to_rgba(colorup, alpha)
1282:     colordown = mcolors.to_rgba(colordown, alpha)
1283:     colord = {True: colorup, False: colordown}
1284: 
1285:     dates, opens, highs, lows, closes, volumes = list(zip(*quotes))
1286:     colors = [colord[close1 >= close0]
1287:               for close0, close1 in zip(closes[:-1], closes[1:])
1288:               if close0 != -1 and close1 != -1]
1289:     colors.insert(0, colord[closes[0] >= opens[0]])
1290: 
1291:     right = width / 2.0
1292:     left = -width / 2.0
1293: 
1294:     bars = [((left, 0), (left, volume), (right, volume), (right, 0))
1295:             for d, open, high, low, close, volume in quotes]
1296: 
1297:     sx = ax.figure.dpi * (1.0 / 72.0)  # scale for points
1298:     sy = ax.bbox.height / ax.viewLim.height
1299: 
1300:     barTransform = Affine2D().scale(sx, sy)
1301: 
1302:     dates = [d for d, open, high, low, close, volume in quotes]
1303:     offsetsBars = [(d, 0) for d in dates]
1304: 
1305:     useAA = 0,  # use tuple here
1306:     lw = 0.5,   # and here
1307:     barCollection = PolyCollection(bars,
1308:                                    facecolors=colors,
1309:                                    edgecolors=((0, 0, 0, 1),),
1310:                                    antialiaseds=useAA,
1311:                                    linewidths=lw,
1312:                                    offsets=offsetsBars,
1313:                                    transOffset=ax.transData,
1314:                                    )
1315:     barCollection.set_transform(barTransform)
1316: 
1317:     minpy, maxx = (min(dates), max(dates))
1318:     miny = 0
1319:     maxy = max([volume for d, open, high, low, close, volume in quotes])
1320:     corners = (minpy, miny), (maxx, maxy)
1321:     ax.update_datalim(corners)
1322:     #print 'datalim', ax.dataLim.bounds
1323:     #print 'viewlim', ax.viewLim.bounds
1324: 
1325:     ax.add_collection(barCollection)
1326:     ax.autoscale_view()
1327: 
1328:     return barCollection
1329: 
1330: 
1331: def index_bar(ax, vals,
1332:               facecolor='b', edgecolor='l',
1333:               width=4, alpha=1.0, ):
1334:     '''Add a bar collection graph with height vals (-1 is missing).
1335: 
1336:     Parameters
1337:     ----------
1338:     ax : `Axes`
1339:         an Axes instance to plot to
1340:     vals : sequence
1341:         a sequence of values
1342:     facecolor : color
1343:         the color of the bar face
1344:     edgecolor : color
1345:         the color of the bar edges
1346:     width : int
1347:         the bar width in points
1348:     alpha : float
1349:        bar transparency
1350: 
1351:     Returns
1352:     -------
1353:     ret : `barCollection`
1354:         The `barrCollection` added to the axes
1355: 
1356:     '''
1357: 
1358:     facecolors = (mcolors.to_rgba(facecolor, alpha),)
1359:     edgecolors = (mcolors.to_rgba(edgecolor, alpha),)
1360: 
1361:     right = width / 2.0
1362:     left = -width / 2.0
1363: 
1364:     bars = [((left, 0), (left, v), (right, v), (right, 0))
1365:             for v in vals if v != -1]
1366: 
1367:     sx = ax.figure.dpi * (1.0 / 72.0)  # scale for points
1368:     sy = ax.bbox.height / ax.viewLim.height
1369: 
1370:     barTransform = Affine2D().scale(sx, sy)
1371: 
1372:     offsetsBars = [(i, 0) for i, v in enumerate(vals) if v != -1]
1373: 
1374:     barCollection = PolyCollection(bars,
1375:                                    facecolors=facecolors,
1376:                                    edgecolors=edgecolors,
1377:                                    antialiaseds=(0,),
1378:                                    linewidths=(0.5,),
1379:                                    offsets=offsetsBars,
1380:                                    transOffset=ax.transData,
1381:                                    )
1382:     barCollection.set_transform(barTransform)
1383: 
1384:     minpy, maxx = (0, len(offsetsBars))
1385:     miny = 0
1386:     maxy = max([v for v in vals if v != -1])
1387:     corners = (minpy, miny), (maxx, maxy)
1388:     ax.update_datalim(corners)
1389:     ax.autoscale_view()
1390: 
1391:     # add these last
1392:     ax.add_collection(barCollection)
1393:     return barCollection
1394: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_54333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'unicode', u'\nA collection of functions for collecting, analyzing and plotting\nfinancial data.\n\nThis module is deprecated in 2.0 and has been moved to a module called\n`mpl_finance`.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import six' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54334 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'six')

if (type(import_54334) is not StypyTypeError):

    if (import_54334 != 'pyd_module'):
        __import__(import_54334)
        sys_modules_54335 = sys.modules[import_54334]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'six', sys_modules_54335.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'six', import_54334)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from six.moves import xrange, zip' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54336 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'six.moves')

if (type(import_54336) is not StypyTypeError):

    if (import_54336 != 'pyd_module'):
        __import__(import_54336)
        sys_modules_54337 = sys.modules[import_54336]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'six.moves', sys_modules_54337.module_type_store, module_type_store, ['xrange', 'zip'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_54337, sys_modules_54337.module_type_store, module_type_store)
    else:
        from six.moves import xrange, zip

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'six.moves', None, module_type_store, ['xrange', 'zip'], [xrange, zip])

else:
    # Assigning a type to the variable 'six.moves' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'six.moves', import_54336)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import contextlib' statement (line 14)
import contextlib

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'contextlib', contextlib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'import os' statement (line 15)
import os

import_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import warnings' statement (line 16)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from six.moves.urllib.request import urlopen' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54338 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'six.moves.urllib.request')

if (type(import_54338) is not StypyTypeError):

    if (import_54338 != 'pyd_module'):
        __import__(import_54338)
        sys_modules_54339 = sys.modules[import_54338]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'six.moves.urllib.request', sys_modules_54339.module_type_store, module_type_store, ['urlopen'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_54339, sys_modules_54339.module_type_store, module_type_store)
    else:
        from six.moves.urllib.request import urlopen

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'six.moves.urllib.request', None, module_type_store, ['urlopen'], [urlopen])

else:
    # Assigning a type to the variable 'six.moves.urllib.request' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'six.moves.urllib.request', import_54338)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import datetime' statement (line 19)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import numpy' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54340 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy')

if (type(import_54340) is not StypyTypeError):

    if (import_54340 != 'pyd_module'):
        __import__(import_54340)
        sys_modules_54341 = sys.modules[import_54340]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'np', sys_modules_54341.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', import_54340)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from matplotlib import mcolors, verbose, get_cachedir' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54342 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib')

if (type(import_54342) is not StypyTypeError):

    if (import_54342 != 'pyd_module'):
        __import__(import_54342)
        sys_modules_54343 = sys.modules[import_54342]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', sys_modules_54343.module_type_store, module_type_store, ['colors', 'verbose', 'get_cachedir'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_54343, sys_modules_54343.module_type_store, module_type_store)
    else:
        from matplotlib import colors as mcolors, verbose, get_cachedir

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', None, module_type_store, ['colors', 'verbose', 'get_cachedir'], [mcolors, verbose, get_cachedir])

else:
    # Assigning a type to the variable 'matplotlib' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', import_54342)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from matplotlib.dates import date2num' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54344 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.dates')

if (type(import_54344) is not StypyTypeError):

    if (import_54344 != 'pyd_module'):
        __import__(import_54344)
        sys_modules_54345 = sys.modules[import_54344]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.dates', sys_modules_54345.module_type_store, module_type_store, ['date2num'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_54345, sys_modules_54345.module_type_store, module_type_store)
    else:
        from matplotlib.dates import date2num

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.dates', None, module_type_store, ['date2num'], [date2num])

else:
    # Assigning a type to the variable 'matplotlib.dates' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib.dates', import_54344)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from matplotlib.cbook import iterable, mkdirs, warn_deprecated' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54346 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib.cbook')

if (type(import_54346) is not StypyTypeError):

    if (import_54346 != 'pyd_module'):
        __import__(import_54346)
        sys_modules_54347 = sys.modules[import_54346]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib.cbook', sys_modules_54347.module_type_store, module_type_store, ['iterable', 'mkdirs', 'warn_deprecated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_54347, sys_modules_54347.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import iterable, mkdirs, warn_deprecated

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib.cbook', None, module_type_store, ['iterable', 'mkdirs', 'warn_deprecated'], [iterable, mkdirs, warn_deprecated])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib.cbook', import_54346)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from matplotlib.collections import LineCollection, PolyCollection' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54348 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.collections')

if (type(import_54348) is not StypyTypeError):

    if (import_54348 != 'pyd_module'):
        __import__(import_54348)
        sys_modules_54349 = sys.modules[import_54348]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.collections', sys_modules_54349.module_type_store, module_type_store, ['LineCollection', 'PolyCollection'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_54349, sys_modules_54349.module_type_store, module_type_store)
    else:
        from matplotlib.collections import LineCollection, PolyCollection

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.collections', None, module_type_store, ['LineCollection', 'PolyCollection'], [LineCollection, PolyCollection])

else:
    # Assigning a type to the variable 'matplotlib.collections' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.collections', import_54348)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from matplotlib.lines import Line2D, TICKLEFT, TICKRIGHT' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54350 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.lines')

if (type(import_54350) is not StypyTypeError):

    if (import_54350 != 'pyd_module'):
        __import__(import_54350)
        sys_modules_54351 = sys.modules[import_54350]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.lines', sys_modules_54351.module_type_store, module_type_store, ['Line2D', 'TICKLEFT', 'TICKRIGHT'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_54351, sys_modules_54351.module_type_store, module_type_store)
    else:
        from matplotlib.lines import Line2D, TICKLEFT, TICKRIGHT

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.lines', None, module_type_store, ['Line2D', 'TICKLEFT', 'TICKRIGHT'], [Line2D, TICKLEFT, TICKRIGHT])

else:
    # Assigning a type to the variable 'matplotlib.lines' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.lines', import_54350)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from matplotlib.patches import Rectangle' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54352 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.patches')

if (type(import_54352) is not StypyTypeError):

    if (import_54352 != 'pyd_module'):
        __import__(import_54352)
        sys_modules_54353 = sys.modules[import_54352]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.patches', sys_modules_54353.module_type_store, module_type_store, ['Rectangle'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_54353, sys_modules_54353.module_type_store, module_type_store)
    else:
        from matplotlib.patches import Rectangle

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.patches', None, module_type_store, ['Rectangle'], [Rectangle])

else:
    # Assigning a type to the variable 'matplotlib.patches' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.patches', import_54352)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from matplotlib.transforms import Affine2D' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_54354 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.transforms')

if (type(import_54354) is not StypyTypeError):

    if (import_54354 != 'pyd_module'):
        __import__(import_54354)
        sys_modules_54355 = sys.modules[import_54354]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.transforms', sys_modules_54355.module_type_store, module_type_store, ['Affine2D'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_54355, sys_modules_54355.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Affine2D

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.transforms', None, module_type_store, ['Affine2D'], [Affine2D])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.transforms', import_54354)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


# Call to warn_deprecated(...): (line 31)
# Processing the call keyword arguments (line 31)
float_54357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 10), 'float')
keyword_54358 = float_54357
unicode_54359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 13), 'unicode', u'The finance module has been deprecated in mpl 2.0 and will be removed in mpl 2.2. Please use the module mpl_finance instead.')
keyword_54360 = unicode_54359
kwargs_54361 = {'message': keyword_54360, 'since': keyword_54358}
# Getting the type of 'warn_deprecated' (line 31)
warn_deprecated_54356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'warn_deprecated', False)
# Calling warn_deprecated(args, kwargs) (line 31)
warn_deprecated_call_result_54362 = invoke(stypy.reporting.localization.Localization(__file__, 31, 0), warn_deprecated_54356, *[], **kwargs_54361)


# Getting the type of 'six' (line 38)
six_54363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 3), 'six')
# Obtaining the member 'PY3' of a type (line 38)
PY3_54364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 3), six_54363, 'PY3')
# Testing the type of an if condition (line 38)
if_condition_54365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 0), PY3_54364)
# Assigning a type to the variable 'if_condition_54365' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'if_condition_54365', if_condition_54365)
# SSA begins for if statement (line 38)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 4))

# 'import hashlib' statement (line 39)
import hashlib

import_module(stypy.reporting.localization.Localization(__file__, 39, 4), 'hashlib', hashlib, module_type_store)


@norecursion
def md5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'md5'
    module_type_store = module_type_store.open_function_context('md5', 41, 4, False)
    
    # Passed parameters checking function
    md5.stypy_localization = localization
    md5.stypy_type_of_self = None
    md5.stypy_type_store = module_type_store
    md5.stypy_function_name = 'md5'
    md5.stypy_param_names_list = ['x']
    md5.stypy_varargs_param_name = None
    md5.stypy_kwargs_param_name = None
    md5.stypy_call_defaults = defaults
    md5.stypy_call_varargs = varargs
    md5.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'md5', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'md5', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'md5(...)' code ##################

    
    # Call to md5(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to encode(...): (line 42)
    # Processing the call keyword arguments (line 42)
    kwargs_54370 = {}
    # Getting the type of 'x' (line 42)
    x_54368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'x', False)
    # Obtaining the member 'encode' of a type (line 42)
    encode_54369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), x_54368, 'encode')
    # Calling encode(args, kwargs) (line 42)
    encode_call_result_54371 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), encode_54369, *[], **kwargs_54370)
    
    # Processing the call keyword arguments (line 42)
    kwargs_54372 = {}
    # Getting the type of 'hashlib' (line 42)
    hashlib_54366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'hashlib', False)
    # Obtaining the member 'md5' of a type (line 42)
    md5_54367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), hashlib_54366, 'md5')
    # Calling md5(args, kwargs) (line 42)
    md5_call_result_54373 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), md5_54367, *[encode_call_result_54371], **kwargs_54372)
    
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', md5_call_result_54373)
    
    # ################# End of 'md5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'md5' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_54374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54374)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'md5'
    return stypy_return_type_54374

# Assigning a type to the variable 'md5' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'md5', md5)
# SSA branch for the else part of an if statement (line 38)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 4))

# 'from hashlib import md5' statement (line 44)
try:
    from hashlib import md5

except:
    md5 = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 44, 4), 'hashlib', None, module_type_store, ['md5'], [md5])

# SSA join for if statement (line 38)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 46):

# Assigning a Call to a Name (line 46):

# Call to get_cachedir(...): (line 46)
# Processing the call keyword arguments (line 46)
kwargs_54376 = {}
# Getting the type of 'get_cachedir' (line 46)
get_cachedir_54375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'get_cachedir', False)
# Calling get_cachedir(args, kwargs) (line 46)
get_cachedir_call_result_54377 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), get_cachedir_54375, *[], **kwargs_54376)

# Assigning a type to the variable 'cachedir' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'cachedir', get_cachedir_call_result_54377)

# Type idiom detected: calculating its left and rigth part (line 48)
# Getting the type of 'cachedir' (line 48)
cachedir_54378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'cachedir')
# Getting the type of 'None' (line 48)
None_54379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'None')

(may_be_54380, more_types_in_union_54381) = may_not_be_none(cachedir_54378, None_54379)

if may_be_54380:

    if more_types_in_union_54381:
        # Runtime conditional SSA (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to join(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'cachedir' (line 49)
    cachedir_54385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'cachedir', False)
    unicode_54386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 38), 'unicode', u'finance.cache')
    # Processing the call keyword arguments (line 49)
    kwargs_54387 = {}
    # Getting the type of 'os' (line 49)
    os_54382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 49)
    path_54383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), os_54382, 'path')
    # Obtaining the member 'join' of a type (line 49)
    join_54384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), path_54383, 'join')
    # Calling join(args, kwargs) (line 49)
    join_call_result_54388 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), join_54384, *[cachedir_54385, unicode_54386], **kwargs_54387)
    
    # Assigning a type to the variable 'cachedir' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'cachedir', join_call_result_54388)

    if more_types_in_union_54381:
        # Runtime conditional SSA for else branch (line 48)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_54380) or more_types_in_union_54381):
    
    # Assigning a Name to a Name (line 53):
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'None' (line 53)
    None_54389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'None')
    # Assigning a type to the variable 'cachedir' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'cachedir', None_54389)

    if (may_be_54380 and more_types_in_union_54381):
        # SSA join for if statement (line 48)
        module_type_store = module_type_store.join_ssa_context()




# Assigning a Call to a Name (line 56):

# Assigning a Call to a Name (line 56):

# Call to dtype(...): (line 56)
# Processing the call arguments (line 56)

# Obtaining an instance of the builtin type 'list' (line 56)
list_54392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 56)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 57)
tuple_54393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 57)
# Adding element type (line 57)

# Call to str(...): (line 57)
# Processing the call arguments (line 57)
unicode_54395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'unicode', u'date')
# Processing the call keyword arguments (line 57)
kwargs_54396 = {}
# Getting the type of 'str' (line 57)
str_54394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 5), 'str', False)
# Calling str(args, kwargs) (line 57)
str_call_result_54397 = invoke(stypy.reporting.localization.Localization(__file__, 57, 5), str_54394, *[unicode_54395], **kwargs_54396)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 5), tuple_54393, str_call_result_54397)
# Adding element type (line 57)
# Getting the type of 'object' (line 57)
object_54398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'object', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 5), tuple_54393, object_54398)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54393)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 58)
tuple_54399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 58)
# Adding element type (line 58)

# Call to str(...): (line 58)
# Processing the call arguments (line 58)
unicode_54401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'unicode', u'year')
# Processing the call keyword arguments (line 58)
kwargs_54402 = {}
# Getting the type of 'str' (line 58)
str_54400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 5), 'str', False)
# Calling str(args, kwargs) (line 58)
str_call_result_54403 = invoke(stypy.reporting.localization.Localization(__file__, 58, 5), str_54400, *[unicode_54401], **kwargs_54402)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 5), tuple_54399, str_call_result_54403)
# Adding element type (line 58)
# Getting the type of 'np' (line 58)
np_54404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'np', False)
# Obtaining the member 'int16' of a type (line 58)
int16_54405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 18), np_54404, 'int16')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 5), tuple_54399, int16_54405)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54399)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 59)
tuple_54406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 59)
# Adding element type (line 59)

# Call to str(...): (line 59)
# Processing the call arguments (line 59)
unicode_54408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 9), 'unicode', u'month')
# Processing the call keyword arguments (line 59)
kwargs_54409 = {}
# Getting the type of 'str' (line 59)
str_54407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 5), 'str', False)
# Calling str(args, kwargs) (line 59)
str_call_result_54410 = invoke(stypy.reporting.localization.Localization(__file__, 59, 5), str_54407, *[unicode_54408], **kwargs_54409)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 5), tuple_54406, str_call_result_54410)
# Adding element type (line 59)
# Getting the type of 'np' (line 59)
np_54411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'np', False)
# Obtaining the member 'int8' of a type (line 59)
int8_54412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 19), np_54411, 'int8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 5), tuple_54406, int8_54412)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54406)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 60)
tuple_54413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 60)
# Adding element type (line 60)

# Call to str(...): (line 60)
# Processing the call arguments (line 60)
unicode_54415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 9), 'unicode', u'day')
# Processing the call keyword arguments (line 60)
kwargs_54416 = {}
# Getting the type of 'str' (line 60)
str_54414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 5), 'str', False)
# Calling str(args, kwargs) (line 60)
str_call_result_54417 = invoke(stypy.reporting.localization.Localization(__file__, 60, 5), str_54414, *[unicode_54415], **kwargs_54416)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 5), tuple_54413, str_call_result_54417)
# Adding element type (line 60)
# Getting the type of 'np' (line 60)
np_54418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'np', False)
# Obtaining the member 'int8' of a type (line 60)
int8_54419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 17), np_54418, 'int8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 5), tuple_54413, int8_54419)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54413)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 61)
tuple_54420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 61)
# Adding element type (line 61)

# Call to str(...): (line 61)
# Processing the call arguments (line 61)
unicode_54422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 9), 'unicode', u'd')
# Processing the call keyword arguments (line 61)
kwargs_54423 = {}
# Getting the type of 'str' (line 61)
str_54421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 5), 'str', False)
# Calling str(args, kwargs) (line 61)
str_call_result_54424 = invoke(stypy.reporting.localization.Localization(__file__, 61, 5), str_54421, *[unicode_54422], **kwargs_54423)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 5), tuple_54420, str_call_result_54424)
# Adding element type (line 61)
# Getting the type of 'float' (line 61)
float_54425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 5), tuple_54420, float_54425)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54420)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 62)
tuple_54426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 62)
# Adding element type (line 62)

# Call to str(...): (line 62)
# Processing the call arguments (line 62)
unicode_54428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 9), 'unicode', u'open')
# Processing the call keyword arguments (line 62)
kwargs_54429 = {}
# Getting the type of 'str' (line 62)
str_54427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 5), 'str', False)
# Calling str(args, kwargs) (line 62)
str_call_result_54430 = invoke(stypy.reporting.localization.Localization(__file__, 62, 5), str_54427, *[unicode_54428], **kwargs_54429)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 5), tuple_54426, str_call_result_54430)
# Adding element type (line 62)
# Getting the type of 'float' (line 62)
float_54431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 5), tuple_54426, float_54431)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54426)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 63)
tuple_54432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 63)
# Adding element type (line 63)

# Call to str(...): (line 63)
# Processing the call arguments (line 63)
unicode_54434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 9), 'unicode', u'high')
# Processing the call keyword arguments (line 63)
kwargs_54435 = {}
# Getting the type of 'str' (line 63)
str_54433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 5), 'str', False)
# Calling str(args, kwargs) (line 63)
str_call_result_54436 = invoke(stypy.reporting.localization.Localization(__file__, 63, 5), str_54433, *[unicode_54434], **kwargs_54435)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 5), tuple_54432, str_call_result_54436)
# Adding element type (line 63)
# Getting the type of 'float' (line 63)
float_54437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 5), tuple_54432, float_54437)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54432)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 64)
tuple_54438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 64)
# Adding element type (line 64)

# Call to str(...): (line 64)
# Processing the call arguments (line 64)
unicode_54440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 9), 'unicode', u'low')
# Processing the call keyword arguments (line 64)
kwargs_54441 = {}
# Getting the type of 'str' (line 64)
str_54439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 5), 'str', False)
# Calling str(args, kwargs) (line 64)
str_call_result_54442 = invoke(stypy.reporting.localization.Localization(__file__, 64, 5), str_54439, *[unicode_54440], **kwargs_54441)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 5), tuple_54438, str_call_result_54442)
# Adding element type (line 64)
# Getting the type of 'float' (line 64)
float_54443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 5), tuple_54438, float_54443)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54438)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 65)
tuple_54444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 65)
# Adding element type (line 65)

# Call to str(...): (line 65)
# Processing the call arguments (line 65)
unicode_54446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'unicode', u'close')
# Processing the call keyword arguments (line 65)
kwargs_54447 = {}
# Getting the type of 'str' (line 65)
str_54445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 5), 'str', False)
# Calling str(args, kwargs) (line 65)
str_call_result_54448 = invoke(stypy.reporting.localization.Localization(__file__, 65, 5), str_54445, *[unicode_54446], **kwargs_54447)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 5), tuple_54444, str_call_result_54448)
# Adding element type (line 65)
# Getting the type of 'float' (line 65)
float_54449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 5), tuple_54444, float_54449)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54444)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 66)
tuple_54450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 66)
# Adding element type (line 66)

# Call to str(...): (line 66)
# Processing the call arguments (line 66)
unicode_54452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 9), 'unicode', u'volume')
# Processing the call keyword arguments (line 66)
kwargs_54453 = {}
# Getting the type of 'str' (line 66)
str_54451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 5), 'str', False)
# Calling str(args, kwargs) (line 66)
str_call_result_54454 = invoke(stypy.reporting.localization.Localization(__file__, 66, 5), str_54451, *[unicode_54452], **kwargs_54453)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 5), tuple_54450, str_call_result_54454)
# Adding element type (line 66)
# Getting the type of 'float' (line 66)
float_54455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 5), tuple_54450, float_54455)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54450)
# Adding element type (line 56)

# Obtaining an instance of the builtin type 'tuple' (line 67)
tuple_54456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 67)
# Adding element type (line 67)

# Call to str(...): (line 67)
# Processing the call arguments (line 67)
unicode_54458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 9), 'unicode', u'aclose')
# Processing the call keyword arguments (line 67)
kwargs_54459 = {}
# Getting the type of 'str' (line 67)
str_54457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 5), 'str', False)
# Calling str(args, kwargs) (line 67)
str_call_result_54460 = invoke(stypy.reporting.localization.Localization(__file__, 67, 5), str_54457, *[unicode_54458], **kwargs_54459)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 5), tuple_54456, str_call_result_54460)
# Adding element type (line 67)
# Getting the type of 'float' (line 67)
float_54461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 5), tuple_54456, float_54461)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), list_54392, tuple_54456)

# Processing the call keyword arguments (line 56)
kwargs_54462 = {}
# Getting the type of 'np' (line 56)
np_54390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'np', False)
# Obtaining the member 'dtype' of a type (line 56)
dtype_54391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), np_54390, 'dtype')
# Calling dtype(args, kwargs) (line 56)
dtype_call_result_54463 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), dtype_54391, *[list_54392], **kwargs_54462)

# Assigning a type to the variable 'stock_dt_ohlc' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stock_dt_ohlc', dtype_call_result_54463)

# Assigning a Call to a Name (line 70):

# Assigning a Call to a Name (line 70):

# Call to dtype(...): (line 70)
# Processing the call arguments (line 70)

# Obtaining an instance of the builtin type 'list' (line 71)
list_54466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 71)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 71)
tuple_54467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 71)
# Adding element type (line 71)

# Call to str(...): (line 71)
# Processing the call arguments (line 71)
unicode_54469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 10), 'unicode', u'date')
# Processing the call keyword arguments (line 71)
kwargs_54470 = {}
# Getting the type of 'str' (line 71)
str_54468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 6), 'str', False)
# Calling str(args, kwargs) (line 71)
str_call_result_54471 = invoke(stypy.reporting.localization.Localization(__file__, 71, 6), str_54468, *[unicode_54469], **kwargs_54470)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 6), tuple_54467, str_call_result_54471)
# Adding element type (line 71)
# Getting the type of 'object' (line 71)
object_54472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'object', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 6), tuple_54467, object_54472)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54467)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 72)
tuple_54473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 72)
# Adding element type (line 72)

# Call to str(...): (line 72)
# Processing the call arguments (line 72)
unicode_54475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 10), 'unicode', u'year')
# Processing the call keyword arguments (line 72)
kwargs_54476 = {}
# Getting the type of 'str' (line 72)
str_54474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 6), 'str', False)
# Calling str(args, kwargs) (line 72)
str_call_result_54477 = invoke(stypy.reporting.localization.Localization(__file__, 72, 6), str_54474, *[unicode_54475], **kwargs_54476)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 6), tuple_54473, str_call_result_54477)
# Adding element type (line 72)
# Getting the type of 'np' (line 72)
np_54478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'np', False)
# Obtaining the member 'int16' of a type (line 72)
int16_54479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 19), np_54478, 'int16')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 6), tuple_54473, int16_54479)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54473)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_54480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)

# Call to str(...): (line 73)
# Processing the call arguments (line 73)
unicode_54482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 10), 'unicode', u'month')
# Processing the call keyword arguments (line 73)
kwargs_54483 = {}
# Getting the type of 'str' (line 73)
str_54481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 6), 'str', False)
# Calling str(args, kwargs) (line 73)
str_call_result_54484 = invoke(stypy.reporting.localization.Localization(__file__, 73, 6), str_54481, *[unicode_54482], **kwargs_54483)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 6), tuple_54480, str_call_result_54484)
# Adding element type (line 73)
# Getting the type of 'np' (line 73)
np_54485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'np', False)
# Obtaining the member 'int8' of a type (line 73)
int8_54486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), np_54485, 'int8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 6), tuple_54480, int8_54486)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54480)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 74)
tuple_54487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 74)
# Adding element type (line 74)

# Call to str(...): (line 74)
# Processing the call arguments (line 74)
unicode_54489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 10), 'unicode', u'day')
# Processing the call keyword arguments (line 74)
kwargs_54490 = {}
# Getting the type of 'str' (line 74)
str_54488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 6), 'str', False)
# Calling str(args, kwargs) (line 74)
str_call_result_54491 = invoke(stypy.reporting.localization.Localization(__file__, 74, 6), str_54488, *[unicode_54489], **kwargs_54490)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 6), tuple_54487, str_call_result_54491)
# Adding element type (line 74)
# Getting the type of 'np' (line 74)
np_54492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'np', False)
# Obtaining the member 'int8' of a type (line 74)
int8_54493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 18), np_54492, 'int8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 6), tuple_54487, int8_54493)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54487)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 75)
tuple_54494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 75)
# Adding element type (line 75)

# Call to str(...): (line 75)
# Processing the call arguments (line 75)
unicode_54496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 10), 'unicode', u'd')
# Processing the call keyword arguments (line 75)
kwargs_54497 = {}
# Getting the type of 'str' (line 75)
str_54495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 6), 'str', False)
# Calling str(args, kwargs) (line 75)
str_call_result_54498 = invoke(stypy.reporting.localization.Localization(__file__, 75, 6), str_54495, *[unicode_54496], **kwargs_54497)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 6), tuple_54494, str_call_result_54498)
# Adding element type (line 75)
# Getting the type of 'float' (line 75)
float_54499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 6), tuple_54494, float_54499)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54494)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_54500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)

# Call to str(...): (line 76)
# Processing the call arguments (line 76)
unicode_54502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 10), 'unicode', u'open')
# Processing the call keyword arguments (line 76)
kwargs_54503 = {}
# Getting the type of 'str' (line 76)
str_54501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 6), 'str', False)
# Calling str(args, kwargs) (line 76)
str_call_result_54504 = invoke(stypy.reporting.localization.Localization(__file__, 76, 6), str_54501, *[unicode_54502], **kwargs_54503)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 6), tuple_54500, str_call_result_54504)
# Adding element type (line 76)
# Getting the type of 'float' (line 76)
float_54505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 6), tuple_54500, float_54505)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54500)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_54506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)

# Call to str(...): (line 77)
# Processing the call arguments (line 77)
unicode_54508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 10), 'unicode', u'close')
# Processing the call keyword arguments (line 77)
kwargs_54509 = {}
# Getting the type of 'str' (line 77)
str_54507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 6), 'str', False)
# Calling str(args, kwargs) (line 77)
str_call_result_54510 = invoke(stypy.reporting.localization.Localization(__file__, 77, 6), str_54507, *[unicode_54508], **kwargs_54509)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 6), tuple_54506, str_call_result_54510)
# Adding element type (line 77)
# Getting the type of 'float' (line 77)
float_54511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 6), tuple_54506, float_54511)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54506)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_54512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)

# Call to str(...): (line 78)
# Processing the call arguments (line 78)
unicode_54514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 10), 'unicode', u'high')
# Processing the call keyword arguments (line 78)
kwargs_54515 = {}
# Getting the type of 'str' (line 78)
str_54513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 6), 'str', False)
# Calling str(args, kwargs) (line 78)
str_call_result_54516 = invoke(stypy.reporting.localization.Localization(__file__, 78, 6), str_54513, *[unicode_54514], **kwargs_54515)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 6), tuple_54512, str_call_result_54516)
# Adding element type (line 78)
# Getting the type of 'float' (line 78)
float_54517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 6), tuple_54512, float_54517)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54512)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 79)
tuple_54518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 79)
# Adding element type (line 79)

# Call to str(...): (line 79)
# Processing the call arguments (line 79)
unicode_54520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 10), 'unicode', u'low')
# Processing the call keyword arguments (line 79)
kwargs_54521 = {}
# Getting the type of 'str' (line 79)
str_54519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 6), 'str', False)
# Calling str(args, kwargs) (line 79)
str_call_result_54522 = invoke(stypy.reporting.localization.Localization(__file__, 79, 6), str_54519, *[unicode_54520], **kwargs_54521)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 6), tuple_54518, str_call_result_54522)
# Adding element type (line 79)
# Getting the type of 'float' (line 79)
float_54523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 6), tuple_54518, float_54523)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54518)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 80)
tuple_54524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 80)
# Adding element type (line 80)

# Call to str(...): (line 80)
# Processing the call arguments (line 80)
unicode_54526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 10), 'unicode', u'volume')
# Processing the call keyword arguments (line 80)
kwargs_54527 = {}
# Getting the type of 'str' (line 80)
str_54525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 6), 'str', False)
# Calling str(args, kwargs) (line 80)
str_call_result_54528 = invoke(stypy.reporting.localization.Localization(__file__, 80, 6), str_54525, *[unicode_54526], **kwargs_54527)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 6), tuple_54524, str_call_result_54528)
# Adding element type (line 80)
# Getting the type of 'float' (line 80)
float_54529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 21), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 6), tuple_54524, float_54529)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54524)
# Adding element type (line 71)

# Obtaining an instance of the builtin type 'tuple' (line 81)
tuple_54530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 81)
# Adding element type (line 81)

# Call to str(...): (line 81)
# Processing the call arguments (line 81)
unicode_54532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 10), 'unicode', u'aclose')
# Processing the call keyword arguments (line 81)
kwargs_54533 = {}
# Getting the type of 'str' (line 81)
str_54531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 6), 'str', False)
# Calling str(args, kwargs) (line 81)
str_call_result_54534 = invoke(stypy.reporting.localization.Localization(__file__, 81, 6), str_54531, *[unicode_54532], **kwargs_54533)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 6), tuple_54530, str_call_result_54534)
# Adding element type (line 81)
# Getting the type of 'float' (line 81)
float_54535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 6), tuple_54530, float_54535)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 4), list_54466, tuple_54530)

# Processing the call keyword arguments (line 70)
kwargs_54536 = {}
# Getting the type of 'np' (line 70)
np_54464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'np', False)
# Obtaining the member 'dtype' of a type (line 70)
dtype_54465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), np_54464, 'dtype')
# Calling dtype(args, kwargs) (line 70)
dtype_call_result_54537 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), dtype_54465, *[list_54466], **kwargs_54536)

# Assigning a type to the variable 'stock_dt_ochl' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stock_dt_ochl', dtype_call_result_54537)

@norecursion
def parse_yahoo_historical_ochl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 84)
    True_54538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 45), 'True')
    # Getting the type of 'False' (line 84)
    False_54539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 60), 'False')
    defaults = [True_54538, False_54539]
    # Create a new context for function 'parse_yahoo_historical_ochl'
    module_type_store = module_type_store.open_function_context('parse_yahoo_historical_ochl', 84, 0, False)
    
    # Passed parameters checking function
    parse_yahoo_historical_ochl.stypy_localization = localization
    parse_yahoo_historical_ochl.stypy_type_of_self = None
    parse_yahoo_historical_ochl.stypy_type_store = module_type_store
    parse_yahoo_historical_ochl.stypy_function_name = 'parse_yahoo_historical_ochl'
    parse_yahoo_historical_ochl.stypy_param_names_list = ['fh', 'adjusted', 'asobject']
    parse_yahoo_historical_ochl.stypy_varargs_param_name = None
    parse_yahoo_historical_ochl.stypy_kwargs_param_name = None
    parse_yahoo_historical_ochl.stypy_call_defaults = defaults
    parse_yahoo_historical_ochl.stypy_call_varargs = varargs
    parse_yahoo_historical_ochl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_yahoo_historical_ochl', ['fh', 'adjusted', 'asobject'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_yahoo_historical_ochl', localization, ['fh', 'adjusted', 'asobject'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_yahoo_historical_ochl(...)' code ##################

    unicode_54540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, (-1)), 'unicode', u'Parse the historical data in file handle fh from yahoo finance.\n\n    Parameters\n    ----------\n\n    adjusted : bool\n      If True (default) replace open, close, high, low prices with\n      their adjusted values. The adjustment is by a scale factor, S =\n      adjusted_close/close. Adjusted prices are actual prices\n      multiplied by S.\n\n      Volume is not adjusted as it is already backward split adjusted\n      by Yahoo. If you want to compute dollars traded, multiply volume\n      by the adjusted close, regardless of whether you choose adjusted\n      = True|False.\n\n\n    asobject : bool or None\n      If False (default for compatibility with earlier versions)\n      return a list of tuples containing\n\n        d, open, close, high, low,  volume\n\n      If None (preferred alternative to False), return\n      a 2-D ndarray corresponding to the list of tuples.\n\n      Otherwise return a numpy recarray with\n\n        date, year, month, day, d, open, close, high, low,\n        volume, adjusted_close\n\n      where d is a floating poing representation of date,\n      as returned by date2num, and date is a python standard\n      library datetime.date instance.\n\n      The name of this kwarg is a historical artifact.  Formerly,\n      True returned a cbook Bunch\n      holding 1-D ndarrays.  The behavior of a numpy recarray is\n      very similar to the Bunch.\n\n    ')
    
    # Call to _parse_yahoo_historical(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'fh' (line 126)
    fh_54542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 35), 'fh', False)
    # Processing the call keyword arguments (line 126)
    # Getting the type of 'adjusted' (line 126)
    adjusted_54543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 48), 'adjusted', False)
    keyword_54544 = adjusted_54543
    # Getting the type of 'asobject' (line 126)
    asobject_54545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 67), 'asobject', False)
    keyword_54546 = asobject_54545
    # Getting the type of 'True' (line 127)
    True_54547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), 'True', False)
    keyword_54548 = True_54547
    kwargs_54549 = {'adjusted': keyword_54544, 'asobject': keyword_54546, 'ochl': keyword_54548}
    # Getting the type of '_parse_yahoo_historical' (line 126)
    _parse_yahoo_historical_54541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), '_parse_yahoo_historical', False)
    # Calling _parse_yahoo_historical(args, kwargs) (line 126)
    _parse_yahoo_historical_call_result_54550 = invoke(stypy.reporting.localization.Localization(__file__, 126, 11), _parse_yahoo_historical_54541, *[fh_54542], **kwargs_54549)
    
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type', _parse_yahoo_historical_call_result_54550)
    
    # ################# End of 'parse_yahoo_historical_ochl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_yahoo_historical_ochl' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_54551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54551)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_yahoo_historical_ochl'
    return stypy_return_type_54551

# Assigning a type to the variable 'parse_yahoo_historical_ochl' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'parse_yahoo_historical_ochl', parse_yahoo_historical_ochl)

@norecursion
def parse_yahoo_historical_ohlc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 130)
    True_54552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 45), 'True')
    # Getting the type of 'False' (line 130)
    False_54553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 60), 'False')
    defaults = [True_54552, False_54553]
    # Create a new context for function 'parse_yahoo_historical_ohlc'
    module_type_store = module_type_store.open_function_context('parse_yahoo_historical_ohlc', 130, 0, False)
    
    # Passed parameters checking function
    parse_yahoo_historical_ohlc.stypy_localization = localization
    parse_yahoo_historical_ohlc.stypy_type_of_self = None
    parse_yahoo_historical_ohlc.stypy_type_store = module_type_store
    parse_yahoo_historical_ohlc.stypy_function_name = 'parse_yahoo_historical_ohlc'
    parse_yahoo_historical_ohlc.stypy_param_names_list = ['fh', 'adjusted', 'asobject']
    parse_yahoo_historical_ohlc.stypy_varargs_param_name = None
    parse_yahoo_historical_ohlc.stypy_kwargs_param_name = None
    parse_yahoo_historical_ohlc.stypy_call_defaults = defaults
    parse_yahoo_historical_ohlc.stypy_call_varargs = varargs
    parse_yahoo_historical_ohlc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_yahoo_historical_ohlc', ['fh', 'adjusted', 'asobject'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_yahoo_historical_ohlc', localization, ['fh', 'adjusted', 'asobject'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_yahoo_historical_ohlc(...)' code ##################

    unicode_54554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, (-1)), 'unicode', u'Parse the historical data in file handle fh from yahoo finance.\n\n    Parameters\n    ----------\n\n    adjusted : bool\n      If True (default) replace open, high, low, close prices with\n      their adjusted values. The adjustment is by a scale factor, S =\n      adjusted_close/close. Adjusted prices are actual prices\n      multiplied by S.\n\n      Volume is not adjusted as it is already backward split adjusted\n      by Yahoo. If you want to compute dollars traded, multiply volume\n      by the adjusted close, regardless of whether you choose adjusted\n      = True|False.\n\n\n    asobject : bool or None\n      If False (default for compatibility with earlier versions)\n      return a list of tuples containing\n\n        d, open, high, low, close, volume\n\n      If None (preferred alternative to False), return\n      a 2-D ndarray corresponding to the list of tuples.\n\n      Otherwise return a numpy recarray with\n\n        date, year, month, day, d, open, high, low,  close,\n        volume, adjusted_close\n\n      where d is a floating poing representation of date,\n      as returned by date2num, and date is a python standard\n      library datetime.date instance.\n\n      The name of this kwarg is a historical artifact.  Formerly,\n      True returned a cbook Bunch\n      holding 1-D ndarrays.  The behavior of a numpy recarray is\n      very similar to the Bunch.\n    ')
    
    # Call to _parse_yahoo_historical(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'fh' (line 171)
    fh_54556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'fh', False)
    # Processing the call keyword arguments (line 171)
    # Getting the type of 'adjusted' (line 171)
    adjusted_54557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 48), 'adjusted', False)
    keyword_54558 = adjusted_54557
    # Getting the type of 'asobject' (line 171)
    asobject_54559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 67), 'asobject', False)
    keyword_54560 = asobject_54559
    # Getting the type of 'False' (line 172)
    False_54561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'False', False)
    keyword_54562 = False_54561
    kwargs_54563 = {'adjusted': keyword_54558, 'asobject': keyword_54560, 'ochl': keyword_54562}
    # Getting the type of '_parse_yahoo_historical' (line 171)
    _parse_yahoo_historical_54555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), '_parse_yahoo_historical', False)
    # Calling _parse_yahoo_historical(args, kwargs) (line 171)
    _parse_yahoo_historical_call_result_54564 = invoke(stypy.reporting.localization.Localization(__file__, 171, 11), _parse_yahoo_historical_54555, *[fh_54556], **kwargs_54563)
    
    # Assigning a type to the variable 'stypy_return_type' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type', _parse_yahoo_historical_call_result_54564)
    
    # ################# End of 'parse_yahoo_historical_ohlc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_yahoo_historical_ohlc' in the type store
    # Getting the type of 'stypy_return_type' (line 130)
    stypy_return_type_54565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54565)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_yahoo_historical_ohlc'
    return stypy_return_type_54565

# Assigning a type to the variable 'parse_yahoo_historical_ohlc' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'parse_yahoo_historical_ohlc', parse_yahoo_historical_ohlc)

@norecursion
def _parse_yahoo_historical(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 175)
    True_54566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 41), 'True')
    # Getting the type of 'False' (line 175)
    False_54567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 56), 'False')
    # Getting the type of 'True' (line 176)
    True_54568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'True')
    defaults = [True_54566, False_54567, True_54568]
    # Create a new context for function '_parse_yahoo_historical'
    module_type_store = module_type_store.open_function_context('_parse_yahoo_historical', 175, 0, False)
    
    # Passed parameters checking function
    _parse_yahoo_historical.stypy_localization = localization
    _parse_yahoo_historical.stypy_type_of_self = None
    _parse_yahoo_historical.stypy_type_store = module_type_store
    _parse_yahoo_historical.stypy_function_name = '_parse_yahoo_historical'
    _parse_yahoo_historical.stypy_param_names_list = ['fh', 'adjusted', 'asobject', 'ochl']
    _parse_yahoo_historical.stypy_varargs_param_name = None
    _parse_yahoo_historical.stypy_kwargs_param_name = None
    _parse_yahoo_historical.stypy_call_defaults = defaults
    _parse_yahoo_historical.stypy_call_varargs = varargs
    _parse_yahoo_historical.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parse_yahoo_historical', ['fh', 'adjusted', 'asobject', 'ochl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parse_yahoo_historical', localization, ['fh', 'adjusted', 'asobject', 'ochl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parse_yahoo_historical(...)' code ##################

    unicode_54569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, (-1)), 'unicode', u'Parse the historical data in file handle fh from yahoo finance.\n\n\n    Parameters\n    ----------\n\n    adjusted : bool\n      If True (default) replace open, high, low, close prices with\n      their adjusted values. The adjustment is by a scale factor, S =\n      adjusted_close/close. Adjusted prices are actual prices\n      multiplied by S.\n\n      Volume is not adjusted as it is already backward split adjusted\n      by Yahoo. If you want to compute dollars traded, multiply volume\n      by the adjusted close, regardless of whether you choose adjusted\n      = True|False.\n\n\n    asobject : bool or None\n      If False (default for compatibility with earlier versions)\n      return a list of tuples containing\n\n        d, open, high, low, close, volume\n\n       or\n\n        d, open, close, high, low, volume\n\n      depending on `ochl`\n\n      If None (preferred alternative to False), return\n      a 2-D ndarray corresponding to the list of tuples.\n\n      Otherwise return a numpy recarray with\n\n        date, year, month, day, d, open, high, low, close,\n        volume, adjusted_close\n\n      where d is a floating poing representation of date,\n      as returned by date2num, and date is a python standard\n      library datetime.date instance.\n\n      The name of this kwarg is a historical artifact.  Formerly,\n      True returned a cbook Bunch\n      holding 1-D ndarrays.  The behavior of a numpy recarray is\n      very similar to the Bunch.\n\n    ochl : bool\n        Selects between ochl and ohlc ordering.\n        Defaults to True to preserve original functionality.\n\n    ')
    
    # Getting the type of 'ochl' (line 229)
    ochl_54570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 7), 'ochl')
    # Testing the type of an if condition (line 229)
    if_condition_54571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 4), ochl_54570)
    # Assigning a type to the variable 'if_condition_54571' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'if_condition_54571', if_condition_54571)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 230):
    
    # Assigning a Name to a Name (line 230):
    # Getting the type of 'stock_dt_ochl' (line 230)
    stock_dt_ochl_54572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'stock_dt_ochl')
    # Assigning a type to the variable 'stock_dt' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'stock_dt', stock_dt_ochl_54572)
    # SSA branch for the else part of an if statement (line 229)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 232):
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'stock_dt_ohlc' (line 232)
    stock_dt_ohlc_54573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'stock_dt_ohlc')
    # Assigning a type to the variable 'stock_dt' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stock_dt', stock_dt_ohlc_54573)
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 234):
    
    # Assigning a List to a Name (line 234):
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_54574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    
    # Assigning a type to the variable 'results' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'results', list_54574)
    
    # Call to readline(...): (line 237)
    # Processing the call keyword arguments (line 237)
    kwargs_54577 = {}
    # Getting the type of 'fh' (line 237)
    fh_54575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'fh', False)
    # Obtaining the member 'readline' of a type (line 237)
    readline_54576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 4), fh_54575, 'readline')
    # Calling readline(args, kwargs) (line 237)
    readline_call_result_54578 = invoke(stypy.reporting.localization.Localization(__file__, 237, 4), readline_54576, *[], **kwargs_54577)
    
    
    # Getting the type of 'fh' (line 238)
    fh_54579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'fh')
    # Testing the type of a for loop iterable (line 238)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 238, 4), fh_54579)
    # Getting the type of the for loop variable (line 238)
    for_loop_var_54580 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 238, 4), fh_54579)
    # Assigning a type to the variable 'line' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'line', for_loop_var_54580)
    # SSA begins for a for statement (line 238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to split(...): (line 240)
    # Processing the call arguments (line 240)
    unicode_54583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 26), 'unicode', u',')
    # Processing the call keyword arguments (line 240)
    kwargs_54584 = {}
    # Getting the type of 'line' (line 240)
    line_54581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'line', False)
    # Obtaining the member 'split' of a type (line 240)
    split_54582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 15), line_54581, 'split')
    # Calling split(args, kwargs) (line 240)
    split_call_result_54585 = invoke(stypy.reporting.localization.Localization(__file__, 240, 15), split_54582, *[unicode_54583], **kwargs_54584)
    
    # Assigning a type to the variable 'vals' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'vals', split_call_result_54585)
    
    
    
    # Call to len(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'vals' (line 241)
    vals_54587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'vals', False)
    # Processing the call keyword arguments (line 241)
    kwargs_54588 = {}
    # Getting the type of 'len' (line 241)
    len_54586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 'len', False)
    # Calling len(args, kwargs) (line 241)
    len_call_result_54589 = invoke(stypy.reporting.localization.Localization(__file__, 241, 11), len_54586, *[vals_54587], **kwargs_54588)
    
    int_54590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 24), 'int')
    # Applying the binary operator '!=' (line 241)
    result_ne_54591 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 11), '!=', len_call_result_54589, int_54590)
    
    # Testing the type of an if condition (line 241)
    if_condition_54592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), result_ne_54591)
    # Assigning a type to the variable 'if_condition_54592' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'if_condition_54592', if_condition_54592)
    # SSA begins for if statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 241)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 243):
    
    # Assigning a Subscript to a Name (line 243):
    
    # Obtaining the type of the subscript
    int_54593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 23), 'int')
    # Getting the type of 'vals' (line 243)
    vals_54594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'vals')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___54595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 18), vals_54594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_54596 = invoke(stypy.reporting.localization.Localization(__file__, 243, 18), getitem___54595, int_54593)
    
    # Assigning a type to the variable 'datestr' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'datestr', subscript_call_result_54596)
    
    # Assigning a Call to a Name (line 247):
    
    # Assigning a Call to a Name (line 247):
    
    # Call to date(...): (line 247)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 247)
    # Processing the call arguments (line 247)
    unicode_54605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 63), 'unicode', u'-')
    # Processing the call keyword arguments (line 247)
    kwargs_54606 = {}
    # Getting the type of 'datestr' (line 247)
    datestr_54603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 49), 'datestr', False)
    # Obtaining the member 'split' of a type (line 247)
    split_54604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 49), datestr_54603, 'split')
    # Calling split(args, kwargs) (line 247)
    split_call_result_54607 = invoke(stypy.reporting.localization.Localization(__file__, 247, 49), split_54604, *[unicode_54605], **kwargs_54606)
    
    comprehension_54608 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 29), split_call_result_54607)
    # Assigning a type to the variable 'val' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'val', comprehension_54608)
    
    # Call to int(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'val' (line 247)
    val_54600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 33), 'val', False)
    # Processing the call keyword arguments (line 247)
    kwargs_54601 = {}
    # Getting the type of 'int' (line 247)
    int_54599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'int', False)
    # Calling int(args, kwargs) (line 247)
    int_call_result_54602 = invoke(stypy.reporting.localization.Localization(__file__, 247, 29), int_54599, *[val_54600], **kwargs_54601)
    
    list_54609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 29), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 29), list_54609, int_call_result_54602)
    # Processing the call keyword arguments (line 247)
    kwargs_54610 = {}
    # Getting the type of 'datetime' (line 247)
    datetime_54597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'datetime', False)
    # Obtaining the member 'date' of a type (line 247)
    date_54598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 13), datetime_54597, 'date')
    # Calling date(args, kwargs) (line 247)
    date_call_result_54611 = invoke(stypy.reporting.localization.Localization(__file__, 247, 13), date_54598, *[list_54609], **kwargs_54610)
    
    # Assigning a type to the variable 'dt' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'dt', date_call_result_54611)
    
    # Assigning a Call to a Name (line 248):
    
    # Assigning a Call to a Name (line 248):
    
    # Call to date2num(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'dt' (line 248)
    dt_54613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'dt', False)
    # Processing the call keyword arguments (line 248)
    kwargs_54614 = {}
    # Getting the type of 'date2num' (line 248)
    date2num_54612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'date2num', False)
    # Calling date2num(args, kwargs) (line 248)
    date2num_call_result_54615 = invoke(stypy.reporting.localization.Localization(__file__, 248, 15), date2num_54612, *[dt_54613], **kwargs_54614)
    
    # Assigning a type to the variable 'dnum' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'dnum', date2num_call_result_54615)
    
    # Assigning a ListComp to a Tuple (line 249):
    
    # Assigning a Subscript to a Name (line 249):
    
    # Obtaining the type of the subscript
    int_54616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 8), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_54621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 61), 'int')
    int_54622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 63), 'int')
    slice_54623 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 249, 56), int_54621, int_54622, None)
    # Getting the type of 'vals' (line 249)
    vals_54624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 56), 'vals')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___54625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 56), vals_54624, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_54626 = invoke(stypy.reporting.localization.Localization(__file__, 249, 56), getitem___54625, slice_54623)
    
    comprehension_54627 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 34), subscript_call_result_54626)
    # Assigning a type to the variable 'val' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 34), 'val', comprehension_54627)
    
    # Call to float(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'val' (line 249)
    val_54618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 40), 'val', False)
    # Processing the call keyword arguments (line 249)
    kwargs_54619 = {}
    # Getting the type of 'float' (line 249)
    float_54617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 34), 'float', False)
    # Calling float(args, kwargs) (line 249)
    float_call_result_54620 = invoke(stypy.reporting.localization.Localization(__file__, 249, 34), float_54617, *[val_54618], **kwargs_54619)
    
    list_54628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 34), list_54628, float_call_result_54620)
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___54629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), list_54628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_54630 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), getitem___54629, int_54616)
    
    # Assigning a type to the variable 'tuple_var_assignment_54294' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'tuple_var_assignment_54294', subscript_call_result_54630)
    
    # Assigning a Subscript to a Name (line 249):
    
    # Obtaining the type of the subscript
    int_54631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 8), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_54636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 61), 'int')
    int_54637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 63), 'int')
    slice_54638 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 249, 56), int_54636, int_54637, None)
    # Getting the type of 'vals' (line 249)
    vals_54639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 56), 'vals')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___54640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 56), vals_54639, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_54641 = invoke(stypy.reporting.localization.Localization(__file__, 249, 56), getitem___54640, slice_54638)
    
    comprehension_54642 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 34), subscript_call_result_54641)
    # Assigning a type to the variable 'val' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 34), 'val', comprehension_54642)
    
    # Call to float(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'val' (line 249)
    val_54633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 40), 'val', False)
    # Processing the call keyword arguments (line 249)
    kwargs_54634 = {}
    # Getting the type of 'float' (line 249)
    float_54632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 34), 'float', False)
    # Calling float(args, kwargs) (line 249)
    float_call_result_54635 = invoke(stypy.reporting.localization.Localization(__file__, 249, 34), float_54632, *[val_54633], **kwargs_54634)
    
    list_54643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 34), list_54643, float_call_result_54635)
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___54644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), list_54643, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_54645 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), getitem___54644, int_54631)
    
    # Assigning a type to the variable 'tuple_var_assignment_54295' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'tuple_var_assignment_54295', subscript_call_result_54645)
    
    # Assigning a Subscript to a Name (line 249):
    
    # Obtaining the type of the subscript
    int_54646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 8), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_54651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 61), 'int')
    int_54652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 63), 'int')
    slice_54653 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 249, 56), int_54651, int_54652, None)
    # Getting the type of 'vals' (line 249)
    vals_54654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 56), 'vals')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___54655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 56), vals_54654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_54656 = invoke(stypy.reporting.localization.Localization(__file__, 249, 56), getitem___54655, slice_54653)
    
    comprehension_54657 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 34), subscript_call_result_54656)
    # Assigning a type to the variable 'val' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 34), 'val', comprehension_54657)
    
    # Call to float(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'val' (line 249)
    val_54648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 40), 'val', False)
    # Processing the call keyword arguments (line 249)
    kwargs_54649 = {}
    # Getting the type of 'float' (line 249)
    float_54647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 34), 'float', False)
    # Calling float(args, kwargs) (line 249)
    float_call_result_54650 = invoke(stypy.reporting.localization.Localization(__file__, 249, 34), float_54647, *[val_54648], **kwargs_54649)
    
    list_54658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 34), list_54658, float_call_result_54650)
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___54659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), list_54658, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_54660 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), getitem___54659, int_54646)
    
    # Assigning a type to the variable 'tuple_var_assignment_54296' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'tuple_var_assignment_54296', subscript_call_result_54660)
    
    # Assigning a Subscript to a Name (line 249):
    
    # Obtaining the type of the subscript
    int_54661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 8), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_54666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 61), 'int')
    int_54667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 63), 'int')
    slice_54668 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 249, 56), int_54666, int_54667, None)
    # Getting the type of 'vals' (line 249)
    vals_54669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 56), 'vals')
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___54670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 56), vals_54669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_54671 = invoke(stypy.reporting.localization.Localization(__file__, 249, 56), getitem___54670, slice_54668)
    
    comprehension_54672 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 34), subscript_call_result_54671)
    # Assigning a type to the variable 'val' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 34), 'val', comprehension_54672)
    
    # Call to float(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'val' (line 249)
    val_54663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 40), 'val', False)
    # Processing the call keyword arguments (line 249)
    kwargs_54664 = {}
    # Getting the type of 'float' (line 249)
    float_54662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 34), 'float', False)
    # Calling float(args, kwargs) (line 249)
    float_call_result_54665 = invoke(stypy.reporting.localization.Localization(__file__, 249, 34), float_54662, *[val_54663], **kwargs_54664)
    
    list_54673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 34), list_54673, float_call_result_54665)
    # Obtaining the member '__getitem__' of a type (line 249)
    getitem___54674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), list_54673, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 249)
    subscript_call_result_54675 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), getitem___54674, int_54661)
    
    # Assigning a type to the variable 'tuple_var_assignment_54297' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'tuple_var_assignment_54297', subscript_call_result_54675)
    
    # Assigning a Name to a Name (line 249):
    # Getting the type of 'tuple_var_assignment_54294' (line 249)
    tuple_var_assignment_54294_54676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'tuple_var_assignment_54294')
    # Assigning a type to the variable 'open' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'open', tuple_var_assignment_54294_54676)
    
    # Assigning a Name to a Name (line 249):
    # Getting the type of 'tuple_var_assignment_54295' (line 249)
    tuple_var_assignment_54295_54677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'tuple_var_assignment_54295')
    # Assigning a type to the variable 'high' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 14), 'high', tuple_var_assignment_54295_54677)
    
    # Assigning a Name to a Name (line 249):
    # Getting the type of 'tuple_var_assignment_54296' (line 249)
    tuple_var_assignment_54296_54678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'tuple_var_assignment_54296')
    # Assigning a type to the variable 'low' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'low', tuple_var_assignment_54296_54678)
    
    # Assigning a Name to a Name (line 249):
    # Getting the type of 'tuple_var_assignment_54297' (line 249)
    tuple_var_assignment_54297_54679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'tuple_var_assignment_54297')
    # Assigning a type to the variable 'close' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'close', tuple_var_assignment_54297_54679)
    
    # Assigning a Call to a Name (line 250):
    
    # Assigning a Call to a Name (line 250):
    
    # Call to float(...): (line 250)
    # Processing the call arguments (line 250)
    
    # Obtaining the type of the subscript
    int_54681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 28), 'int')
    # Getting the type of 'vals' (line 250)
    vals_54682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 23), 'vals', False)
    # Obtaining the member '__getitem__' of a type (line 250)
    getitem___54683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 23), vals_54682, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 250)
    subscript_call_result_54684 = invoke(stypy.reporting.localization.Localization(__file__, 250, 23), getitem___54683, int_54681)
    
    # Processing the call keyword arguments (line 250)
    kwargs_54685 = {}
    # Getting the type of 'float' (line 250)
    float_54680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 17), 'float', False)
    # Calling float(args, kwargs) (line 250)
    float_call_result_54686 = invoke(stypy.reporting.localization.Localization(__file__, 250, 17), float_54680, *[subscript_call_result_54684], **kwargs_54685)
    
    # Assigning a type to the variable 'volume' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'volume', float_call_result_54686)
    
    # Assigning a Call to a Name (line 251):
    
    # Assigning a Call to a Name (line 251):
    
    # Call to float(...): (line 251)
    # Processing the call arguments (line 251)
    
    # Obtaining the type of the subscript
    int_54688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 28), 'int')
    # Getting the type of 'vals' (line 251)
    vals_54689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'vals', False)
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___54690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 23), vals_54689, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_54691 = invoke(stypy.reporting.localization.Localization(__file__, 251, 23), getitem___54690, int_54688)
    
    # Processing the call keyword arguments (line 251)
    kwargs_54692 = {}
    # Getting the type of 'float' (line 251)
    float_54687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 17), 'float', False)
    # Calling float(args, kwargs) (line 251)
    float_call_result_54693 = invoke(stypy.reporting.localization.Localization(__file__, 251, 17), float_54687, *[subscript_call_result_54691], **kwargs_54692)
    
    # Assigning a type to the variable 'aclose' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'aclose', float_call_result_54693)
    
    # Getting the type of 'ochl' (line 252)
    ochl_54694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 11), 'ochl')
    # Testing the type of an if condition (line 252)
    if_condition_54695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 8), ochl_54694)
    # Assigning a type to the variable 'if_condition_54695' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'if_condition_54695', if_condition_54695)
    # SSA begins for if statement (line 252)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 253)
    # Processing the call arguments (line 253)
    
    # Obtaining an instance of the builtin type 'tuple' (line 253)
    tuple_54698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 253)
    # Adding element type (line 253)
    # Getting the type of 'dt' (line 253)
    dt_54699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'dt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, dt_54699)
    # Adding element type (line 253)
    # Getting the type of 'dt' (line 253)
    dt_54700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 32), 'dt', False)
    # Obtaining the member 'year' of a type (line 253)
    year_54701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 32), dt_54700, 'year')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, year_54701)
    # Adding element type (line 253)
    # Getting the type of 'dt' (line 253)
    dt_54702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 41), 'dt', False)
    # Obtaining the member 'month' of a type (line 253)
    month_54703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 41), dt_54702, 'month')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, month_54703)
    # Adding element type (line 253)
    # Getting the type of 'dt' (line 253)
    dt_54704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 51), 'dt', False)
    # Obtaining the member 'day' of a type (line 253)
    day_54705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 51), dt_54704, 'day')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, day_54705)
    # Adding element type (line 253)
    # Getting the type of 'dnum' (line 254)
    dnum_54706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 28), 'dnum', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, dnum_54706)
    # Adding element type (line 253)
    # Getting the type of 'open' (line 254)
    open_54707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'open', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, open_54707)
    # Adding element type (line 253)
    # Getting the type of 'close' (line 254)
    close_54708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 40), 'close', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, close_54708)
    # Adding element type (line 253)
    # Getting the type of 'high' (line 254)
    high_54709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 47), 'high', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, high_54709)
    # Adding element type (line 253)
    # Getting the type of 'low' (line 254)
    low_54710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 53), 'low', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, low_54710)
    # Adding element type (line 253)
    # Getting the type of 'volume' (line 254)
    volume_54711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 58), 'volume', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, volume_54711)
    # Adding element type (line 253)
    # Getting the type of 'aclose' (line 254)
    aclose_54712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 66), 'aclose', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 28), tuple_54698, aclose_54712)
    
    # Processing the call keyword arguments (line 253)
    kwargs_54713 = {}
    # Getting the type of 'results' (line 253)
    results_54696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'results', False)
    # Obtaining the member 'append' of a type (line 253)
    append_54697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), results_54696, 'append')
    # Calling append(args, kwargs) (line 253)
    append_call_result_54714 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), append_54697, *[tuple_54698], **kwargs_54713)
    
    # SSA branch for the else part of an if statement (line 252)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 257)
    # Processing the call arguments (line 257)
    
    # Obtaining an instance of the builtin type 'tuple' (line 257)
    tuple_54717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 257)
    # Adding element type (line 257)
    # Getting the type of 'dt' (line 257)
    dt_54718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'dt', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, dt_54718)
    # Adding element type (line 257)
    # Getting the type of 'dt' (line 257)
    dt_54719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 32), 'dt', False)
    # Obtaining the member 'year' of a type (line 257)
    year_54720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 32), dt_54719, 'year')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, year_54720)
    # Adding element type (line 257)
    # Getting the type of 'dt' (line 257)
    dt_54721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 41), 'dt', False)
    # Obtaining the member 'month' of a type (line 257)
    month_54722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 41), dt_54721, 'month')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, month_54722)
    # Adding element type (line 257)
    # Getting the type of 'dt' (line 257)
    dt_54723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 51), 'dt', False)
    # Obtaining the member 'day' of a type (line 257)
    day_54724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 51), dt_54723, 'day')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, day_54724)
    # Adding element type (line 257)
    # Getting the type of 'dnum' (line 258)
    dnum_54725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 28), 'dnum', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, dnum_54725)
    # Adding element type (line 257)
    # Getting the type of 'open' (line 258)
    open_54726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 34), 'open', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, open_54726)
    # Adding element type (line 257)
    # Getting the type of 'high' (line 258)
    high_54727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 40), 'high', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, high_54727)
    # Adding element type (line 257)
    # Getting the type of 'low' (line 258)
    low_54728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 46), 'low', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, low_54728)
    # Adding element type (line 257)
    # Getting the type of 'close' (line 258)
    close_54729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 51), 'close', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, close_54729)
    # Adding element type (line 257)
    # Getting the type of 'volume' (line 258)
    volume_54730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 58), 'volume', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, volume_54730)
    # Adding element type (line 257)
    # Getting the type of 'aclose' (line 258)
    aclose_54731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 66), 'aclose', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 28), tuple_54717, aclose_54731)
    
    # Processing the call keyword arguments (line 257)
    kwargs_54732 = {}
    # Getting the type of 'results' (line 257)
    results_54715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'results', False)
    # Obtaining the member 'append' of a type (line 257)
    append_54716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), results_54715, 'append')
    # Calling append(args, kwargs) (line 257)
    append_call_result_54733 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), append_54716, *[tuple_54717], **kwargs_54732)
    
    # SSA join for if statement (line 252)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reverse(...): (line 259)
    # Processing the call keyword arguments (line 259)
    kwargs_54736 = {}
    # Getting the type of 'results' (line 259)
    results_54734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'results', False)
    # Obtaining the member 'reverse' of a type (line 259)
    reverse_54735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 4), results_54734, 'reverse')
    # Calling reverse(args, kwargs) (line 259)
    reverse_call_result_54737 = invoke(stypy.reporting.localization.Localization(__file__, 259, 4), reverse_54735, *[], **kwargs_54736)
    
    
    # Assigning a Call to a Name (line 260):
    
    # Assigning a Call to a Name (line 260):
    
    # Call to array(...): (line 260)
    # Processing the call arguments (line 260)
    # Getting the type of 'results' (line 260)
    results_54740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'results', False)
    # Processing the call keyword arguments (line 260)
    # Getting the type of 'stock_dt' (line 260)
    stock_dt_54741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 32), 'stock_dt', False)
    keyword_54742 = stock_dt_54741
    kwargs_54743 = {'dtype': keyword_54742}
    # Getting the type of 'np' (line 260)
    np_54738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 260)
    array_54739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), np_54738, 'array')
    # Calling array(args, kwargs) (line 260)
    array_call_result_54744 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), array_54739, *[results_54740], **kwargs_54743)
    
    # Assigning a type to the variable 'd' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'd', array_call_result_54744)
    
    # Getting the type of 'adjusted' (line 261)
    adjusted_54745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 7), 'adjusted')
    # Testing the type of an if condition (line 261)
    if_condition_54746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 4), adjusted_54745)
    # Assigning a type to the variable 'if_condition_54746' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'if_condition_54746', if_condition_54746)
    # SSA begins for if statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 262):
    
    # Assigning a BinOp to a Name (line 262):
    
    # Obtaining the type of the subscript
    unicode_54747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 18), 'unicode', u'aclose')
    # Getting the type of 'd' (line 262)
    d_54748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'd')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___54749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 16), d_54748, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_54750 = invoke(stypy.reporting.localization.Localization(__file__, 262, 16), getitem___54749, unicode_54747)
    
    
    # Obtaining the type of the subscript
    unicode_54751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 32), 'unicode', u'close')
    # Getting the type of 'd' (line 262)
    d_54752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 30), 'd')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___54753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 30), d_54752, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_54754 = invoke(stypy.reporting.localization.Localization(__file__, 262, 30), getitem___54753, unicode_54751)
    
    # Applying the binary operator 'div' (line 262)
    result_div_54755 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 16), 'div', subscript_call_result_54750, subscript_call_result_54754)
    
    # Assigning a type to the variable 'scale' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'scale', result_div_54755)
    
    # Assigning a Attribute to a Subscript (line 263):
    
    # Assigning a Attribute to a Subscript (line 263):
    # Getting the type of 'np' (line 263)
    np_54756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'np')
    # Obtaining the member 'nan' of a type (line 263)
    nan_54757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 33), np_54756, 'nan')
    # Getting the type of 'scale' (line 263)
    scale_54758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'scale')
    
    # Call to isinf(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'scale' (line 263)
    scale_54761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 23), 'scale', False)
    # Processing the call keyword arguments (line 263)
    kwargs_54762 = {}
    # Getting the type of 'np' (line 263)
    np_54759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 14), 'np', False)
    # Obtaining the member 'isinf' of a type (line 263)
    isinf_54760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 14), np_54759, 'isinf')
    # Calling isinf(args, kwargs) (line 263)
    isinf_call_result_54763 = invoke(stypy.reporting.localization.Localization(__file__, 263, 14), isinf_54760, *[scale_54761], **kwargs_54762)
    
    # Storing an element on a container (line 263)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 8), scale_54758, (isinf_call_result_54763, nan_54757))
    
    # Getting the type of 'd' (line 264)
    d_54764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'd')
    
    # Obtaining the type of the subscript
    unicode_54765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 10), 'unicode', u'open')
    # Getting the type of 'd' (line 264)
    d_54766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'd')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___54767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), d_54766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_54768 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), getitem___54767, unicode_54765)
    
    # Getting the type of 'scale' (line 264)
    scale_54769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 21), 'scale')
    # Applying the binary operator '*=' (line 264)
    result_imul_54770 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 8), '*=', subscript_call_result_54768, scale_54769)
    # Getting the type of 'd' (line 264)
    d_54771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'd')
    unicode_54772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 10), 'unicode', u'open')
    # Storing an element on a container (line 264)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 8), d_54771, (unicode_54772, result_imul_54770))
    
    
    # Getting the type of 'd' (line 265)
    d_54773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'd')
    
    # Obtaining the type of the subscript
    unicode_54774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 10), 'unicode', u'high')
    # Getting the type of 'd' (line 265)
    d_54775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'd')
    # Obtaining the member '__getitem__' of a type (line 265)
    getitem___54776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), d_54775, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 265)
    subscript_call_result_54777 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), getitem___54776, unicode_54774)
    
    # Getting the type of 'scale' (line 265)
    scale_54778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'scale')
    # Applying the binary operator '*=' (line 265)
    result_imul_54779 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 8), '*=', subscript_call_result_54777, scale_54778)
    # Getting the type of 'd' (line 265)
    d_54780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'd')
    unicode_54781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 10), 'unicode', u'high')
    # Storing an element on a container (line 265)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), d_54780, (unicode_54781, result_imul_54779))
    
    
    # Getting the type of 'd' (line 266)
    d_54782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'd')
    
    # Obtaining the type of the subscript
    unicode_54783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 10), 'unicode', u'low')
    # Getting the type of 'd' (line 266)
    d_54784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'd')
    # Obtaining the member '__getitem__' of a type (line 266)
    getitem___54785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), d_54784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 266)
    subscript_call_result_54786 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), getitem___54785, unicode_54783)
    
    # Getting the type of 'scale' (line 266)
    scale_54787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'scale')
    # Applying the binary operator '*=' (line 266)
    result_imul_54788 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 8), '*=', subscript_call_result_54786, scale_54787)
    # Getting the type of 'd' (line 266)
    d_54789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'd')
    unicode_54790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 10), 'unicode', u'low')
    # Storing an element on a container (line 266)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 8), d_54789, (unicode_54790, result_imul_54788))
    
    
    # Getting the type of 'd' (line 267)
    d_54791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'd')
    
    # Obtaining the type of the subscript
    unicode_54792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 10), 'unicode', u'close')
    # Getting the type of 'd' (line 267)
    d_54793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'd')
    # Obtaining the member '__getitem__' of a type (line 267)
    getitem___54794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), d_54793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 267)
    subscript_call_result_54795 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), getitem___54794, unicode_54792)
    
    # Getting the type of 'scale' (line 267)
    scale_54796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 22), 'scale')
    # Applying the binary operator '*=' (line 267)
    result_imul_54797 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 8), '*=', subscript_call_result_54795, scale_54796)
    # Getting the type of 'd' (line 267)
    d_54798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'd')
    unicode_54799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 10), 'unicode', u'close')
    # Storing an element on a container (line 267)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 8), d_54798, (unicode_54799, result_imul_54797))
    
    # SSA join for if statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'asobject' (line 269)
    asobject_54800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'asobject')
    # Applying the 'not' unary operator (line 269)
    result_not__54801 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 7), 'not', asobject_54800)
    
    # Testing the type of an if condition (line 269)
    if_condition_54802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 4), result_not__54801)
    # Assigning a type to the variable 'if_condition_54802' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'if_condition_54802', if_condition_54802)
    # SSA begins for if statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to zeros(...): (line 271)
    # Processing the call arguments (line 271)
    
    # Obtaining an instance of the builtin type 'tuple' (line 271)
    tuple_54805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 271)
    # Adding element type (line 271)
    
    # Call to len(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'd' (line 271)
    d_54807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 28), 'd', False)
    # Processing the call keyword arguments (line 271)
    kwargs_54808 = {}
    # Getting the type of 'len' (line 271)
    len_54806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'len', False)
    # Calling len(args, kwargs) (line 271)
    len_call_result_54809 = invoke(stypy.reporting.localization.Localization(__file__, 271, 24), len_54806, *[d_54807], **kwargs_54808)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 24), tuple_54805, len_call_result_54809)
    # Adding element type (line 271)
    int_54810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 24), tuple_54805, int_54810)
    
    # Processing the call keyword arguments (line 271)
    # Getting the type of 'float' (line 271)
    float_54811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 42), 'float', False)
    keyword_54812 = float_54811
    kwargs_54813 = {'dtype': keyword_54812}
    # Getting the type of 'np' (line 271)
    np_54803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), 'np', False)
    # Obtaining the member 'zeros' of a type (line 271)
    zeros_54804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 14), np_54803, 'zeros')
    # Calling zeros(args, kwargs) (line 271)
    zeros_call_result_54814 = invoke(stypy.reporting.localization.Localization(__file__, 271, 14), zeros_54804, *[tuple_54805], **kwargs_54813)
    
    # Assigning a type to the variable 'ret' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'ret', zeros_call_result_54814)
    
    # Assigning a Subscript to a Subscript (line 272):
    
    # Assigning a Subscript to a Subscript (line 272):
    
    # Obtaining the type of the subscript
    unicode_54815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 22), 'unicode', u'd')
    # Getting the type of 'd' (line 272)
    d_54816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'd')
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___54817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 20), d_54816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_54818 = invoke(stypy.reporting.localization.Localization(__file__, 272, 20), getitem___54817, unicode_54815)
    
    # Getting the type of 'ret' (line 272)
    ret_54819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'ret')
    slice_54820 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 272, 8), None, None, None)
    int_54821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 15), 'int')
    # Storing an element on a container (line 272)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 8), ret_54819, ((slice_54820, int_54821), subscript_call_result_54818))
    
    # Getting the type of 'ochl' (line 273)
    ochl_54822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'ochl')
    # Testing the type of an if condition (line 273)
    if_condition_54823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), ochl_54822)
    # Assigning a type to the variable 'if_condition_54823' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_54823', if_condition_54823)
    # SSA begins for if statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 274):
    
    # Assigning a Subscript to a Subscript (line 274):
    
    # Obtaining the type of the subscript
    unicode_54824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 26), 'unicode', u'open')
    # Getting the type of 'd' (line 274)
    d_54825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'd')
    # Obtaining the member '__getitem__' of a type (line 274)
    getitem___54826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 24), d_54825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 274)
    subscript_call_result_54827 = invoke(stypy.reporting.localization.Localization(__file__, 274, 24), getitem___54826, unicode_54824)
    
    # Getting the type of 'ret' (line 274)
    ret_54828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'ret')
    slice_54829 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 12), None, None, None)
    int_54830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 19), 'int')
    # Storing an element on a container (line 274)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 12), ret_54828, ((slice_54829, int_54830), subscript_call_result_54827))
    
    # Assigning a Subscript to a Subscript (line 275):
    
    # Assigning a Subscript to a Subscript (line 275):
    
    # Obtaining the type of the subscript
    unicode_54831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 26), 'unicode', u'close')
    # Getting the type of 'd' (line 275)
    d_54832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'd')
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___54833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 24), d_54832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_54834 = invoke(stypy.reporting.localization.Localization(__file__, 275, 24), getitem___54833, unicode_54831)
    
    # Getting the type of 'ret' (line 275)
    ret_54835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'ret')
    slice_54836 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 275, 12), None, None, None)
    int_54837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 19), 'int')
    # Storing an element on a container (line 275)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 12), ret_54835, ((slice_54836, int_54837), subscript_call_result_54834))
    
    # Assigning a Subscript to a Subscript (line 276):
    
    # Assigning a Subscript to a Subscript (line 276):
    
    # Obtaining the type of the subscript
    unicode_54838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 26), 'unicode', u'high')
    # Getting the type of 'd' (line 276)
    d_54839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 24), 'd')
    # Obtaining the member '__getitem__' of a type (line 276)
    getitem___54840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 24), d_54839, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 276)
    subscript_call_result_54841 = invoke(stypy.reporting.localization.Localization(__file__, 276, 24), getitem___54840, unicode_54838)
    
    # Getting the type of 'ret' (line 276)
    ret_54842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'ret')
    slice_54843 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 276, 12), None, None, None)
    int_54844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 19), 'int')
    # Storing an element on a container (line 276)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 12), ret_54842, ((slice_54843, int_54844), subscript_call_result_54841))
    
    # Assigning a Subscript to a Subscript (line 277):
    
    # Assigning a Subscript to a Subscript (line 277):
    
    # Obtaining the type of the subscript
    unicode_54845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 26), 'unicode', u'low')
    # Getting the type of 'd' (line 277)
    d_54846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 24), 'd')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___54847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 24), d_54846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_54848 = invoke(stypy.reporting.localization.Localization(__file__, 277, 24), getitem___54847, unicode_54845)
    
    # Getting the type of 'ret' (line 277)
    ret_54849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'ret')
    slice_54850 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 277, 12), None, None, None)
    int_54851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 19), 'int')
    # Storing an element on a container (line 277)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 12), ret_54849, ((slice_54850, int_54851), subscript_call_result_54848))
    # SSA branch for the else part of an if statement (line 273)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 279):
    
    # Assigning a Subscript to a Subscript (line 279):
    
    # Obtaining the type of the subscript
    unicode_54852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 26), 'unicode', u'open')
    # Getting the type of 'd' (line 279)
    d_54853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'd')
    # Obtaining the member '__getitem__' of a type (line 279)
    getitem___54854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 24), d_54853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 279)
    subscript_call_result_54855 = invoke(stypy.reporting.localization.Localization(__file__, 279, 24), getitem___54854, unicode_54852)
    
    # Getting the type of 'ret' (line 279)
    ret_54856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'ret')
    slice_54857 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 279, 12), None, None, None)
    int_54858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 19), 'int')
    # Storing an element on a container (line 279)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 12), ret_54856, ((slice_54857, int_54858), subscript_call_result_54855))
    
    # Assigning a Subscript to a Subscript (line 280):
    
    # Assigning a Subscript to a Subscript (line 280):
    
    # Obtaining the type of the subscript
    unicode_54859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 26), 'unicode', u'high')
    # Getting the type of 'd' (line 280)
    d_54860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'd')
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___54861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 24), d_54860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_54862 = invoke(stypy.reporting.localization.Localization(__file__, 280, 24), getitem___54861, unicode_54859)
    
    # Getting the type of 'ret' (line 280)
    ret_54863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'ret')
    slice_54864 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 280, 12), None, None, None)
    int_54865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 19), 'int')
    # Storing an element on a container (line 280)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 12), ret_54863, ((slice_54864, int_54865), subscript_call_result_54862))
    
    # Assigning a Subscript to a Subscript (line 281):
    
    # Assigning a Subscript to a Subscript (line 281):
    
    # Obtaining the type of the subscript
    unicode_54866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 26), 'unicode', u'low')
    # Getting the type of 'd' (line 281)
    d_54867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'd')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___54868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), d_54867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_54869 = invoke(stypy.reporting.localization.Localization(__file__, 281, 24), getitem___54868, unicode_54866)
    
    # Getting the type of 'ret' (line 281)
    ret_54870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'ret')
    slice_54871 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 281, 12), None, None, None)
    int_54872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 19), 'int')
    # Storing an element on a container (line 281)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 12), ret_54870, ((slice_54871, int_54872), subscript_call_result_54869))
    
    # Assigning a Subscript to a Subscript (line 282):
    
    # Assigning a Subscript to a Subscript (line 282):
    
    # Obtaining the type of the subscript
    unicode_54873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 26), 'unicode', u'close')
    # Getting the type of 'd' (line 282)
    d_54874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'd')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___54875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), d_54874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_54876 = invoke(stypy.reporting.localization.Localization(__file__, 282, 24), getitem___54875, unicode_54873)
    
    # Getting the type of 'ret' (line 282)
    ret_54877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'ret')
    slice_54878 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 282, 12), None, None, None)
    int_54879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 19), 'int')
    # Storing an element on a container (line 282)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 12), ret_54877, ((slice_54878, int_54879), subscript_call_result_54876))
    # SSA join for if statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 283):
    
    # Assigning a Subscript to a Subscript (line 283):
    
    # Obtaining the type of the subscript
    unicode_54880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 22), 'unicode', u'volume')
    # Getting the type of 'd' (line 283)
    d_54881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'd')
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___54882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 20), d_54881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_54883 = invoke(stypy.reporting.localization.Localization(__file__, 283, 20), getitem___54882, unicode_54880)
    
    # Getting the type of 'ret' (line 283)
    ret_54884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'ret')
    slice_54885 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 8), None, None, None)
    int_54886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 15), 'int')
    # Storing an element on a container (line 283)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 8), ret_54884, ((slice_54885, int_54886), subscript_call_result_54883))
    
    # Type idiom detected: calculating its left and rigth part (line 284)
    # Getting the type of 'asobject' (line 284)
    asobject_54887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'asobject')
    # Getting the type of 'None' (line 284)
    None_54888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'None')
    
    (may_be_54889, more_types_in_union_54890) = may_be_none(asobject_54887, None_54888)

    if may_be_54889:

        if more_types_in_union_54890:
            # Runtime conditional SSA (line 284)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'ret' (line 285)
        ret_54891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'stypy_return_type', ret_54891)

        if more_types_in_union_54890:
            # SSA join for if statement (line 284)
            module_type_store = module_type_store.join_ssa_context()


    
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ret' (line 286)
    ret_54896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 38), 'ret')
    comprehension_54897 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 16), ret_54896)
    # Assigning a type to the variable 'row' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'row', comprehension_54897)
    
    # Call to tuple(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'row' (line 286)
    row_54893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'row', False)
    # Processing the call keyword arguments (line 286)
    kwargs_54894 = {}
    # Getting the type of 'tuple' (line 286)
    tuple_54892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 286)
    tuple_call_result_54895 = invoke(stypy.reporting.localization.Localization(__file__, 286, 16), tuple_54892, *[row_54893], **kwargs_54894)
    
    list_54898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 16), list_54898, tuple_call_result_54895)
    # Assigning a type to the variable 'stypy_return_type' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'stypy_return_type', list_54898)
    # SSA join for if statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to view(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'np' (line 288)
    np_54901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'np', False)
    # Obtaining the member 'recarray' of a type (line 288)
    recarray_54902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 18), np_54901, 'recarray')
    # Processing the call keyword arguments (line 288)
    kwargs_54903 = {}
    # Getting the type of 'd' (line 288)
    d_54899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'd', False)
    # Obtaining the member 'view' of a type (line 288)
    view_54900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 11), d_54899, 'view')
    # Calling view(args, kwargs) (line 288)
    view_call_result_54904 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), view_54900, *[recarray_54902], **kwargs_54903)
    
    # Assigning a type to the variable 'stypy_return_type' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type', view_call_result_54904)
    
    # ################# End of '_parse_yahoo_historical(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parse_yahoo_historical' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_54905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_54905)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parse_yahoo_historical'
    return stypy_return_type_54905

# Assigning a type to the variable '_parse_yahoo_historical' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), '_parse_yahoo_historical', _parse_yahoo_historical)

@norecursion
def fetch_historical_yahoo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 291)
    None_54906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 59), 'None')
    # Getting the type of 'False' (line 292)
    False_54907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 37), 'False')
    defaults = [None_54906, False_54907]
    # Create a new context for function 'fetch_historical_yahoo'
    module_type_store = module_type_store.open_function_context('fetch_historical_yahoo', 291, 0, False)
    
    # Passed parameters checking function
    fetch_historical_yahoo.stypy_localization = localization
    fetch_historical_yahoo.stypy_type_of_self = None
    fetch_historical_yahoo.stypy_type_store = module_type_store
    fetch_historical_yahoo.stypy_function_name = 'fetch_historical_yahoo'
    fetch_historical_yahoo.stypy_param_names_list = ['ticker', 'date1', 'date2', 'cachename', 'dividends']
    fetch_historical_yahoo.stypy_varargs_param_name = None
    fetch_historical_yahoo.stypy_kwargs_param_name = None
    fetch_historical_yahoo.stypy_call_defaults = defaults
    fetch_historical_yahoo.stypy_call_varargs = varargs
    fetch_historical_yahoo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fetch_historical_yahoo', ['ticker', 'date1', 'date2', 'cachename', 'dividends'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fetch_historical_yahoo', localization, ['ticker', 'date1', 'date2', 'cachename', 'dividends'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fetch_historical_yahoo(...)' code ##################

    unicode_54908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, (-1)), 'unicode', u"\n    Fetch historical data for ticker between date1 and date2.  date1 and\n    date2 are date or datetime instances, or (year, month, day) sequences.\n\n    Parameters\n    ----------\n    ticker : str\n        ticker\n\n    date1 : sequence of form (year, month, day), `datetime`, or `date`\n        start date\n    date2 : sequence of form (year, month, day), `datetime`, or `date`\n        end date\n\n    cachename : str\n        cachename is the name of the local file cache.  If None, will\n        default to the md5 hash or the url (which incorporates the ticker\n        and date range)\n\n    dividends : bool\n        set dividends=True to return dividends instead of price data.  With\n        this option set, parse functions will not work\n\n    Returns\n    -------\n    file_handle : file handle\n        a file handle is returned\n\n\n    Examples\n    --------\n    >>> fh = fetch_historical_yahoo('^GSPC', (2000, 1, 1), (2001, 12, 31))\n\n    ")
    
    # Assigning a Call to a Name (line 328):
    
    # Assigning a Call to a Name (line 328):
    
    # Call to upper(...): (line 328)
    # Processing the call keyword arguments (line 328)
    kwargs_54911 = {}
    # Getting the type of 'ticker' (line 328)
    ticker_54909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 13), 'ticker', False)
    # Obtaining the member 'upper' of a type (line 328)
    upper_54910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 13), ticker_54909, 'upper')
    # Calling upper(args, kwargs) (line 328)
    upper_call_result_54912 = invoke(stypy.reporting.localization.Localization(__file__, 328, 13), upper_54910, *[], **kwargs_54911)
    
    # Assigning a type to the variable 'ticker' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'ticker', upper_call_result_54912)
    
    
    # Call to iterable(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'date1' (line 330)
    date1_54914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'date1', False)
    # Processing the call keyword arguments (line 330)
    kwargs_54915 = {}
    # Getting the type of 'iterable' (line 330)
    iterable_54913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 7), 'iterable', False)
    # Calling iterable(args, kwargs) (line 330)
    iterable_call_result_54916 = invoke(stypy.reporting.localization.Localization(__file__, 330, 7), iterable_54913, *[date1_54914], **kwargs_54915)
    
    # Testing the type of an if condition (line 330)
    if_condition_54917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 4), iterable_call_result_54916)
    # Assigning a type to the variable 'if_condition_54917' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'if_condition_54917', if_condition_54917)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 331):
    
    # Assigning a Tuple to a Name (line 331):
    
    # Obtaining an instance of the builtin type 'tuple' (line 331)
    tuple_54918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 331)
    # Adding element type (line 331)
    
    # Obtaining the type of the subscript
    int_54919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 20), 'int')
    # Getting the type of 'date1' (line 331)
    date1_54920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 14), 'date1')
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___54921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 14), date1_54920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_54922 = invoke(stypy.reporting.localization.Localization(__file__, 331, 14), getitem___54921, int_54919)
    
    int_54923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 25), 'int')
    # Applying the binary operator '-' (line 331)
    result_sub_54924 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 14), '-', subscript_call_result_54922, int_54923)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 14), tuple_54918, result_sub_54924)
    # Adding element type (line 331)
    
    # Obtaining the type of the subscript
    int_54925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 34), 'int')
    # Getting the type of 'date1' (line 331)
    date1_54926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 'date1')
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___54927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 28), date1_54926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_54928 = invoke(stypy.reporting.localization.Localization(__file__, 331, 28), getitem___54927, int_54925)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 14), tuple_54918, subscript_call_result_54928)
    # Adding element type (line 331)
    
    # Obtaining the type of the subscript
    int_54929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 44), 'int')
    # Getting the type of 'date1' (line 331)
    date1_54930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 38), 'date1')
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___54931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 38), date1_54930, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_54932 = invoke(stypy.reporting.localization.Localization(__file__, 331, 38), getitem___54931, int_54929)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 14), tuple_54918, subscript_call_result_54932)
    
    # Assigning a type to the variable 'd1' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'd1', tuple_54918)
    # SSA branch for the else part of an if statement (line 330)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 333):
    
    # Assigning a Tuple to a Name (line 333):
    
    # Obtaining an instance of the builtin type 'tuple' (line 333)
    tuple_54933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 333)
    # Adding element type (line 333)
    # Getting the type of 'date1' (line 333)
    date1_54934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 14), 'date1')
    # Obtaining the member 'month' of a type (line 333)
    month_54935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 14), date1_54934, 'month')
    int_54936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 28), 'int')
    # Applying the binary operator '-' (line 333)
    result_sub_54937 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 14), '-', month_54935, int_54936)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 14), tuple_54933, result_sub_54937)
    # Adding element type (line 333)
    # Getting the type of 'date1' (line 333)
    date1_54938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 31), 'date1')
    # Obtaining the member 'day' of a type (line 333)
    day_54939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 31), date1_54938, 'day')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 14), tuple_54933, day_54939)
    # Adding element type (line 333)
    # Getting the type of 'date1' (line 333)
    date1_54940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 42), 'date1')
    # Obtaining the member 'year' of a type (line 333)
    year_54941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 42), date1_54940, 'year')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 14), tuple_54933, year_54941)
    
    # Assigning a type to the variable 'd1' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'd1', tuple_54933)
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to iterable(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'date2' (line 334)
    date2_54943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'date2', False)
    # Processing the call keyword arguments (line 334)
    kwargs_54944 = {}
    # Getting the type of 'iterable' (line 334)
    iterable_54942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 7), 'iterable', False)
    # Calling iterable(args, kwargs) (line 334)
    iterable_call_result_54945 = invoke(stypy.reporting.localization.Localization(__file__, 334, 7), iterable_54942, *[date2_54943], **kwargs_54944)
    
    # Testing the type of an if condition (line 334)
    if_condition_54946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 4), iterable_call_result_54945)
    # Assigning a type to the variable 'if_condition_54946' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'if_condition_54946', if_condition_54946)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 335):
    
    # Assigning a Tuple to a Name (line 335):
    
    # Obtaining an instance of the builtin type 'tuple' (line 335)
    tuple_54947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 335)
    # Adding element type (line 335)
    
    # Obtaining the type of the subscript
    int_54948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 20), 'int')
    # Getting the type of 'date2' (line 335)
    date2_54949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'date2')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___54950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 14), date2_54949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_54951 = invoke(stypy.reporting.localization.Localization(__file__, 335, 14), getitem___54950, int_54948)
    
    int_54952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 25), 'int')
    # Applying the binary operator '-' (line 335)
    result_sub_54953 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 14), '-', subscript_call_result_54951, int_54952)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 14), tuple_54947, result_sub_54953)
    # Adding element type (line 335)
    
    # Obtaining the type of the subscript
    int_54954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 34), 'int')
    # Getting the type of 'date2' (line 335)
    date2_54955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 28), 'date2')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___54956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 28), date2_54955, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_54957 = invoke(stypy.reporting.localization.Localization(__file__, 335, 28), getitem___54956, int_54954)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 14), tuple_54947, subscript_call_result_54957)
    # Adding element type (line 335)
    
    # Obtaining the type of the subscript
    int_54958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 44), 'int')
    # Getting the type of 'date2' (line 335)
    date2_54959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 38), 'date2')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___54960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 38), date2_54959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_54961 = invoke(stypy.reporting.localization.Localization(__file__, 335, 38), getitem___54960, int_54958)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 14), tuple_54947, subscript_call_result_54961)
    
    # Assigning a type to the variable 'd2' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'd2', tuple_54947)
    # SSA branch for the else part of an if statement (line 334)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 337):
    
    # Assigning a Tuple to a Name (line 337):
    
    # Obtaining an instance of the builtin type 'tuple' (line 337)
    tuple_54962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 337)
    # Adding element type (line 337)
    # Getting the type of 'date2' (line 337)
    date2_54963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 14), 'date2')
    # Obtaining the member 'month' of a type (line 337)
    month_54964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 14), date2_54963, 'month')
    int_54965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 28), 'int')
    # Applying the binary operator '-' (line 337)
    result_sub_54966 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 14), '-', month_54964, int_54965)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 14), tuple_54962, result_sub_54966)
    # Adding element type (line 337)
    # Getting the type of 'date2' (line 337)
    date2_54967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 31), 'date2')
    # Obtaining the member 'day' of a type (line 337)
    day_54968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 31), date2_54967, 'day')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 14), tuple_54962, day_54968)
    # Adding element type (line 337)
    # Getting the type of 'date2' (line 337)
    date2_54969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 42), 'date2')
    # Obtaining the member 'year' of a type (line 337)
    year_54970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 42), date2_54969, 'year')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 14), tuple_54962, year_54970)
    
    # Assigning a type to the variable 'd2' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'd2', tuple_54962)
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'dividends' (line 339)
    dividends_54971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 7), 'dividends')
    # Testing the type of an if condition (line 339)
    if_condition_54972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 339, 4), dividends_54971)
    # Assigning a type to the variable 'if_condition_54972' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'if_condition_54972', if_condition_54972)
    # SSA begins for if statement (line 339)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 340):
    
    # Assigning a Str to a Name (line 340):
    unicode_54973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 12), 'unicode', u'v')
    # Assigning a type to the variable 'g' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'g', unicode_54973)
    
    # Call to report(...): (line 341)
    # Processing the call arguments (line 341)
    unicode_54976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 23), 'unicode', u'Retrieving dividends instead of prices')
    # Processing the call keyword arguments (line 341)
    kwargs_54977 = {}
    # Getting the type of 'verbose' (line 341)
    verbose_54974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'verbose', False)
    # Obtaining the member 'report' of a type (line 341)
    report_54975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), verbose_54974, 'report')
    # Calling report(args, kwargs) (line 341)
    report_call_result_54978 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), report_54975, *[unicode_54976], **kwargs_54977)
    
    # SSA branch for the else part of an if statement (line 339)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 343):
    
    # Assigning a Str to a Name (line 343):
    unicode_54979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 12), 'unicode', u'd')
    # Assigning a type to the variable 'g' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'g', unicode_54979)
    # SSA join for if statement (line 339)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 345):
    
    # Assigning a BinOp to a Name (line 345):
    unicode_54980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 14), 'unicode', u'http://real-chart.finance.yahoo.com/table.csv?')
    unicode_54981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 14), 'unicode', u'&s=%s&d=%d&e=%d&f=%d&g=%s&a=%d&b=%d&c=%d&ignore=.csv')
    # Applying the binary operator '+' (line 345)
    result_add_54982 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 14), '+', unicode_54980, unicode_54981)
    
    # Assigning a type to the variable 'urlFmt' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'urlFmt', result_add_54982)
    
    # Assigning a BinOp to a Name (line 348):
    
    # Assigning a BinOp to a Name (line 348):
    # Getting the type of 'urlFmt' (line 348)
    urlFmt_54983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 10), 'urlFmt')
    
    # Obtaining an instance of the builtin type 'tuple' (line 348)
    tuple_54984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 348)
    # Adding element type (line 348)
    # Getting the type of 'ticker' (line 348)
    ticker_54985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'ticker')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_54984, ticker_54985)
    # Adding element type (line 348)
    
    # Obtaining the type of the subscript
    int_54986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 31), 'int')
    # Getting the type of 'd2' (line 348)
    d2_54987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 28), 'd2')
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___54988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 28), d2_54987, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_54989 = invoke(stypy.reporting.localization.Localization(__file__, 348, 28), getitem___54988, int_54986)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_54984, subscript_call_result_54989)
    # Adding element type (line 348)
    
    # Obtaining the type of the subscript
    int_54990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 38), 'int')
    # Getting the type of 'd2' (line 348)
    d2_54991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 35), 'd2')
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___54992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 35), d2_54991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_54993 = invoke(stypy.reporting.localization.Localization(__file__, 348, 35), getitem___54992, int_54990)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_54984, subscript_call_result_54993)
    # Adding element type (line 348)
    
    # Obtaining the type of the subscript
    int_54994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 45), 'int')
    # Getting the type of 'd2' (line 348)
    d2_54995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 42), 'd2')
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___54996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 42), d2_54995, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_54997 = invoke(stypy.reporting.localization.Localization(__file__, 348, 42), getitem___54996, int_54994)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_54984, subscript_call_result_54997)
    # Adding element type (line 348)
    # Getting the type of 'g' (line 348)
    g_54998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 49), 'g')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_54984, g_54998)
    # Adding element type (line 348)
    
    # Obtaining the type of the subscript
    int_54999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 55), 'int')
    # Getting the type of 'd1' (line 348)
    d1_55000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 52), 'd1')
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___55001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 52), d1_55000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_55002 = invoke(stypy.reporting.localization.Localization(__file__, 348, 52), getitem___55001, int_54999)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_54984, subscript_call_result_55002)
    # Adding element type (line 348)
    
    # Obtaining the type of the subscript
    int_55003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 62), 'int')
    # Getting the type of 'd1' (line 348)
    d1_55004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 59), 'd1')
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___55005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 59), d1_55004, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_55006 = invoke(stypy.reporting.localization.Localization(__file__, 348, 59), getitem___55005, int_55003)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_54984, subscript_call_result_55006)
    # Adding element type (line 348)
    
    # Obtaining the type of the subscript
    int_55007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 69), 'int')
    # Getting the type of 'd1' (line 348)
    d1_55008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 66), 'd1')
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___55009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 66), d1_55008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_55010 = invoke(stypy.reporting.localization.Localization(__file__, 348, 66), getitem___55009, int_55007)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 20), tuple_54984, subscript_call_result_55010)
    
    # Applying the binary operator '%' (line 348)
    result_mod_55011 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 10), '%', urlFmt_54983, tuple_54984)
    
    # Assigning a type to the variable 'url' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'url', result_mod_55011)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'cachename' (line 351)
    cachename_55012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 7), 'cachename')
    # Getting the type of 'None' (line 351)
    None_55013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 20), 'None')
    # Applying the binary operator 'is' (line 351)
    result_is__55014 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 7), 'is', cachename_55012, None_55013)
    
    
    # Getting the type of 'cachedir' (line 351)
    cachedir_55015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 29), 'cachedir')
    # Getting the type of 'None' (line 351)
    None_55016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 45), 'None')
    # Applying the binary operator 'isnot' (line 351)
    result_is_not_55017 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 29), 'isnot', cachedir_55015, None_55016)
    
    # Applying the binary operator 'and' (line 351)
    result_and_keyword_55018 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 7), 'and', result_is__55014, result_is_not_55017)
    
    # Testing the type of an if condition (line 351)
    if_condition_55019 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 4), result_and_keyword_55018)
    # Assigning a type to the variable 'if_condition_55019' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'if_condition_55019', if_condition_55019)
    # SSA begins for if statement (line 351)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 352):
    
    # Assigning a Call to a Name (line 352):
    
    # Call to join(...): (line 352)
    # Processing the call arguments (line 352)
    # Getting the type of 'cachedir' (line 352)
    cachedir_55023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 33), 'cachedir', False)
    
    # Call to hexdigest(...): (line 352)
    # Processing the call keyword arguments (line 352)
    kwargs_55029 = {}
    
    # Call to md5(...): (line 352)
    # Processing the call arguments (line 352)
    # Getting the type of 'url' (line 352)
    url_55025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 47), 'url', False)
    # Processing the call keyword arguments (line 352)
    kwargs_55026 = {}
    # Getting the type of 'md5' (line 352)
    md5_55024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 43), 'md5', False)
    # Calling md5(args, kwargs) (line 352)
    md5_call_result_55027 = invoke(stypy.reporting.localization.Localization(__file__, 352, 43), md5_55024, *[url_55025], **kwargs_55026)
    
    # Obtaining the member 'hexdigest' of a type (line 352)
    hexdigest_55028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 43), md5_call_result_55027, 'hexdigest')
    # Calling hexdigest(args, kwargs) (line 352)
    hexdigest_call_result_55030 = invoke(stypy.reporting.localization.Localization(__file__, 352, 43), hexdigest_55028, *[], **kwargs_55029)
    
    # Processing the call keyword arguments (line 352)
    kwargs_55031 = {}
    # Getting the type of 'os' (line 352)
    os_55020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 352)
    path_55021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 20), os_55020, 'path')
    # Obtaining the member 'join' of a type (line 352)
    join_55022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 20), path_55021, 'join')
    # Calling join(args, kwargs) (line 352)
    join_call_result_55032 = invoke(stypy.reporting.localization.Localization(__file__, 352, 20), join_55022, *[cachedir_55023, hexdigest_call_result_55030], **kwargs_55031)
    
    # Assigning a type to the variable 'cachename' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'cachename', join_call_result_55032)
    # SSA join for if statement (line 351)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 353)
    # Getting the type of 'cachename' (line 353)
    cachename_55033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'cachename')
    # Getting the type of 'None' (line 353)
    None_55034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 24), 'None')
    
    (may_be_55035, more_types_in_union_55036) = may_not_be_none(cachename_55033, None_55034)

    if may_be_55035:

        if more_types_in_union_55036:
            # Runtime conditional SSA (line 353)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to exists(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'cachename' (line 354)
        cachename_55040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 26), 'cachename', False)
        # Processing the call keyword arguments (line 354)
        kwargs_55041 = {}
        # Getting the type of 'os' (line 354)
        os_55037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 354)
        path_55038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 11), os_55037, 'path')
        # Obtaining the member 'exists' of a type (line 354)
        exists_55039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 11), path_55038, 'exists')
        # Calling exists(args, kwargs) (line 354)
        exists_call_result_55042 = invoke(stypy.reporting.localization.Localization(__file__, 354, 11), exists_55039, *[cachename_55040], **kwargs_55041)
        
        # Testing the type of an if condition (line 354)
        if_condition_55043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 8), exists_call_result_55042)
        # Assigning a type to the variable 'if_condition_55043' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'if_condition_55043', if_condition_55043)
        # SSA begins for if statement (line 354)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 355):
        
        # Assigning a Call to a Name (line 355):
        
        # Call to open(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'cachename' (line 355)
        cachename_55045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'cachename', False)
        # Processing the call keyword arguments (line 355)
        kwargs_55046 = {}
        # Getting the type of 'open' (line 355)
        open_55044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 17), 'open', False)
        # Calling open(args, kwargs) (line 355)
        open_call_result_55047 = invoke(stypy.reporting.localization.Localization(__file__, 355, 17), open_55044, *[cachename_55045], **kwargs_55046)
        
        # Assigning a type to the variable 'fh' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'fh', open_call_result_55047)
        
        # Call to report(...): (line 356)
        # Processing the call arguments (line 356)
        unicode_55050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 27), 'unicode', u'Using cachefile %s for %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 357)
        tuple_55051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 357)
        # Adding element type (line 357)
        # Getting the type of 'cachename' (line 357)
        cachename_55052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 35), 'cachename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 35), tuple_55051, cachename_55052)
        # Adding element type (line 357)
        # Getting the type of 'ticker' (line 357)
        ticker_55053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 46), 'ticker', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 35), tuple_55051, ticker_55053)
        
        # Applying the binary operator '%' (line 356)
        result_mod_55054 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 27), '%', unicode_55050, tuple_55051)
        
        # Processing the call keyword arguments (line 356)
        kwargs_55055 = {}
        # Getting the type of 'verbose' (line 356)
        verbose_55048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'verbose', False)
        # Obtaining the member 'report' of a type (line 356)
        report_55049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 12), verbose_55048, 'report')
        # Calling report(args, kwargs) (line 356)
        report_call_result_55056 = invoke(stypy.reporting.localization.Localization(__file__, 356, 12), report_55049, *[result_mod_55054], **kwargs_55055)
        
        # SSA branch for the else part of an if statement (line 354)
        module_type_store.open_ssa_branch('else')
        
        # Call to mkdirs(...): (line 359)
        # Processing the call arguments (line 359)
        
        # Call to abspath(...): (line 359)
        # Processing the call arguments (line 359)
        
        # Call to dirname(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'cachename' (line 359)
        cachename_55064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 51), 'cachename', False)
        # Processing the call keyword arguments (line 359)
        kwargs_55065 = {}
        # Getting the type of 'os' (line 359)
        os_55061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 35), 'os', False)
        # Obtaining the member 'path' of a type (line 359)
        path_55062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 35), os_55061, 'path')
        # Obtaining the member 'dirname' of a type (line 359)
        dirname_55063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 35), path_55062, 'dirname')
        # Calling dirname(args, kwargs) (line 359)
        dirname_call_result_55066 = invoke(stypy.reporting.localization.Localization(__file__, 359, 35), dirname_55063, *[cachename_55064], **kwargs_55065)
        
        # Processing the call keyword arguments (line 359)
        kwargs_55067 = {}
        # Getting the type of 'os' (line 359)
        os_55058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 359)
        path_55059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 19), os_55058, 'path')
        # Obtaining the member 'abspath' of a type (line 359)
        abspath_55060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 19), path_55059, 'abspath')
        # Calling abspath(args, kwargs) (line 359)
        abspath_call_result_55068 = invoke(stypy.reporting.localization.Localization(__file__, 359, 19), abspath_55060, *[dirname_call_result_55066], **kwargs_55067)
        
        # Processing the call keyword arguments (line 359)
        kwargs_55069 = {}
        # Getting the type of 'mkdirs' (line 359)
        mkdirs_55057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'mkdirs', False)
        # Calling mkdirs(args, kwargs) (line 359)
        mkdirs_call_result_55070 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), mkdirs_55057, *[abspath_call_result_55068], **kwargs_55069)
        
        
        # Call to closing(...): (line 360)
        # Processing the call arguments (line 360)
        
        # Call to urlopen(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'url' (line 360)
        url_55074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 44), 'url', False)
        # Processing the call keyword arguments (line 360)
        kwargs_55075 = {}
        # Getting the type of 'urlopen' (line 360)
        urlopen_55073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 36), 'urlopen', False)
        # Calling urlopen(args, kwargs) (line 360)
        urlopen_call_result_55076 = invoke(stypy.reporting.localization.Localization(__file__, 360, 36), urlopen_55073, *[url_55074], **kwargs_55075)
        
        # Processing the call keyword arguments (line 360)
        kwargs_55077 = {}
        # Getting the type of 'contextlib' (line 360)
        contextlib_55071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 17), 'contextlib', False)
        # Obtaining the member 'closing' of a type (line 360)
        closing_55072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 17), contextlib_55071, 'closing')
        # Calling closing(args, kwargs) (line 360)
        closing_call_result_55078 = invoke(stypy.reporting.localization.Localization(__file__, 360, 17), closing_55072, *[urlopen_call_result_55076], **kwargs_55077)
        
        with_55079 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 360, 17), closing_call_result_55078, 'with parameter', '__enter__', '__exit__')

        if with_55079:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 360)
            enter___55080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 17), closing_call_result_55078, '__enter__')
            with_enter_55081 = invoke(stypy.reporting.localization.Localization(__file__, 360, 17), enter___55080)
            # Assigning a type to the variable 'urlfh' (line 360)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 17), 'urlfh', with_enter_55081)
            
            # Call to open(...): (line 361)
            # Processing the call arguments (line 361)
            # Getting the type of 'cachename' (line 361)
            cachename_55083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 26), 'cachename', False)
            unicode_55084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 37), 'unicode', u'wb')
            # Processing the call keyword arguments (line 361)
            kwargs_55085 = {}
            # Getting the type of 'open' (line 361)
            open_55082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'open', False)
            # Calling open(args, kwargs) (line 361)
            open_call_result_55086 = invoke(stypy.reporting.localization.Localization(__file__, 361, 21), open_55082, *[cachename_55083, unicode_55084], **kwargs_55085)
            
            with_55087 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 361, 21), open_call_result_55086, 'with parameter', '__enter__', '__exit__')

            if with_55087:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 361)
                enter___55088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 21), open_call_result_55086, '__enter__')
                with_enter_55089 = invoke(stypy.reporting.localization.Localization(__file__, 361, 21), enter___55088)
                # Assigning a type to the variable 'fh' (line 361)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'fh', with_enter_55089)
                
                # Call to write(...): (line 362)
                # Processing the call arguments (line 362)
                
                # Call to read(...): (line 362)
                # Processing the call keyword arguments (line 362)
                kwargs_55094 = {}
                # Getting the type of 'urlfh' (line 362)
                urlfh_55092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 29), 'urlfh', False)
                # Obtaining the member 'read' of a type (line 362)
                read_55093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 29), urlfh_55092, 'read')
                # Calling read(args, kwargs) (line 362)
                read_call_result_55095 = invoke(stypy.reporting.localization.Localization(__file__, 362, 29), read_55093, *[], **kwargs_55094)
                
                # Processing the call keyword arguments (line 362)
                kwargs_55096 = {}
                # Getting the type of 'fh' (line 362)
                fh_55090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 20), 'fh', False)
                # Obtaining the member 'write' of a type (line 362)
                write_55091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 20), fh_55090, 'write')
                # Calling write(args, kwargs) (line 362)
                write_call_result_55097 = invoke(stypy.reporting.localization.Localization(__file__, 362, 20), write_55091, *[read_call_result_55095], **kwargs_55096)
                
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 361)
                exit___55098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 21), open_call_result_55086, '__exit__')
                with_exit_55099 = invoke(stypy.reporting.localization.Localization(__file__, 361, 21), exit___55098, None, None, None)

            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 360)
            exit___55100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 17), closing_call_result_55078, '__exit__')
            with_exit_55101 = invoke(stypy.reporting.localization.Localization(__file__, 360, 17), exit___55100, None, None, None)

        
        # Call to report(...): (line 363)
        # Processing the call arguments (line 363)
        unicode_55104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 27), 'unicode', u'Saved %s data to cache file %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 364)
        tuple_55105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 364)
        # Adding element type (line 364)
        # Getting the type of 'ticker' (line 364)
        ticker_55106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 35), 'ticker', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 35), tuple_55105, ticker_55106)
        # Adding element type (line 364)
        # Getting the type of 'cachename' (line 364)
        cachename_55107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 43), 'cachename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 35), tuple_55105, cachename_55107)
        
        # Applying the binary operator '%' (line 363)
        result_mod_55108 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 27), '%', unicode_55104, tuple_55105)
        
        # Processing the call keyword arguments (line 363)
        kwargs_55109 = {}
        # Getting the type of 'verbose' (line 363)
        verbose_55102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'verbose', False)
        # Obtaining the member 'report' of a type (line 363)
        report_55103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), verbose_55102, 'report')
        # Calling report(args, kwargs) (line 363)
        report_call_result_55110 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), report_55103, *[result_mod_55108], **kwargs_55109)
        
        
        # Assigning a Call to a Name (line 365):
        
        # Assigning a Call to a Name (line 365):
        
        # Call to open(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'cachename' (line 365)
        cachename_55112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 22), 'cachename', False)
        unicode_55113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 33), 'unicode', u'r')
        # Processing the call keyword arguments (line 365)
        kwargs_55114 = {}
        # Getting the type of 'open' (line 365)
        open_55111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 17), 'open', False)
        # Calling open(args, kwargs) (line 365)
        open_call_result_55115 = invoke(stypy.reporting.localization.Localization(__file__, 365, 17), open_55111, *[cachename_55112, unicode_55113], **kwargs_55114)
        
        # Assigning a type to the variable 'fh' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'fh', open_call_result_55115)
        # SSA join for if statement (line 354)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'fh' (line 367)
        fh_55116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 15), 'fh')
        # Assigning a type to the variable 'stypy_return_type' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'stypy_return_type', fh_55116)

        if more_types_in_union_55036:
            # Runtime conditional SSA for else branch (line 353)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_55035) or more_types_in_union_55036):
        
        # Call to urlopen(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'url' (line 369)
        url_55118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 23), 'url', False)
        # Processing the call keyword arguments (line 369)
        kwargs_55119 = {}
        # Getting the type of 'urlopen' (line 369)
        urlopen_55117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'urlopen', False)
        # Calling urlopen(args, kwargs) (line 369)
        urlopen_call_result_55120 = invoke(stypy.reporting.localization.Localization(__file__, 369, 15), urlopen_55117, *[url_55118], **kwargs_55119)
        
        # Assigning a type to the variable 'stypy_return_type' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'stypy_return_type', urlopen_call_result_55120)

        if (may_be_55035 and more_types_in_union_55036):
            # SSA join for if statement (line 353)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'fetch_historical_yahoo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fetch_historical_yahoo' in the type store
    # Getting the type of 'stypy_return_type' (line 291)
    stypy_return_type_55121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55121)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fetch_historical_yahoo'
    return stypy_return_type_55121

# Assigning a type to the variable 'fetch_historical_yahoo' (line 291)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), 'fetch_historical_yahoo', fetch_historical_yahoo)

@norecursion
def quotes_historical_yahoo_ochl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 372)
    False_55122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 64), 'False')
    # Getting the type of 'True' (line 373)
    True_55123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 37), 'True')
    # Getting the type of 'None' (line 373)
    None_55124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 53), 'None')
    defaults = [False_55122, True_55123, None_55124]
    # Create a new context for function 'quotes_historical_yahoo_ochl'
    module_type_store = module_type_store.open_function_context('quotes_historical_yahoo_ochl', 372, 0, False)
    
    # Passed parameters checking function
    quotes_historical_yahoo_ochl.stypy_localization = localization
    quotes_historical_yahoo_ochl.stypy_type_of_self = None
    quotes_historical_yahoo_ochl.stypy_type_store = module_type_store
    quotes_historical_yahoo_ochl.stypy_function_name = 'quotes_historical_yahoo_ochl'
    quotes_historical_yahoo_ochl.stypy_param_names_list = ['ticker', 'date1', 'date2', 'asobject', 'adjusted', 'cachename']
    quotes_historical_yahoo_ochl.stypy_varargs_param_name = None
    quotes_historical_yahoo_ochl.stypy_kwargs_param_name = None
    quotes_historical_yahoo_ochl.stypy_call_defaults = defaults
    quotes_historical_yahoo_ochl.stypy_call_varargs = varargs
    quotes_historical_yahoo_ochl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'quotes_historical_yahoo_ochl', ['ticker', 'date1', 'date2', 'asobject', 'adjusted', 'cachename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'quotes_historical_yahoo_ochl', localization, ['ticker', 'date1', 'date2', 'asobject', 'adjusted', 'cachename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'quotes_historical_yahoo_ochl(...)' code ##################

    unicode_55125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, (-1)), 'unicode', u" Get historical data for ticker between date1 and date2.\n\n\n    See :func:`parse_yahoo_historical` for explanation of output formats\n    and the *asobject* and *adjusted* kwargs.\n\n    Parameters\n    ----------\n    ticker : str\n        stock ticker\n\n    date1 : sequence of form (year, month, day), `datetime`, or `date`\n        start date\n\n    date2 : sequence of form (year, month, day), `datetime`, or `date`\n        end date\n\n    cachename : str or `None`\n        is the name of the local file cache.  If None, will\n        default to the md5 hash or the url (which incorporates the ticker\n        and date range)\n\n    Examples\n    --------\n    >>> sp = f.quotes_historical_yahoo_ochl('^GSPC', d1, d2,\n                             asobject=True, adjusted=True)\n    >>> returns = (sp.open[1:] - sp.open[:-1])/sp.open[1:]\n    >>> [n,bins,patches] = hist(returns, 100)\n    >>> mu = mean(returns)\n    >>> sigma = std(returns)\n    >>> x = normpdf(bins, mu, sigma)\n    >>> plot(bins, x, color='red', lw=2)\n\n    ")
    
    # Call to _quotes_historical_yahoo(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'ticker' (line 409)
    ticker_55127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 36), 'ticker', False)
    # Getting the type of 'date1' (line 409)
    date1_55128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 44), 'date1', False)
    # Getting the type of 'date2' (line 409)
    date2_55129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 51), 'date2', False)
    # Processing the call keyword arguments (line 409)
    # Getting the type of 'asobject' (line 409)
    asobject_55130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 67), 'asobject', False)
    keyword_55131 = asobject_55130
    # Getting the type of 'adjusted' (line 410)
    adjusted_55132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 38), 'adjusted', False)
    keyword_55133 = adjusted_55132
    # Getting the type of 'cachename' (line 410)
    cachename_55134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 58), 'cachename', False)
    keyword_55135 = cachename_55134
    # Getting the type of 'True' (line 411)
    True_55136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 34), 'True', False)
    keyword_55137 = True_55136
    kwargs_55138 = {'adjusted': keyword_55133, 'cachename': keyword_55135, 'asobject': keyword_55131, 'ochl': keyword_55137}
    # Getting the type of '_quotes_historical_yahoo' (line 409)
    _quotes_historical_yahoo_55126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), '_quotes_historical_yahoo', False)
    # Calling _quotes_historical_yahoo(args, kwargs) (line 409)
    _quotes_historical_yahoo_call_result_55139 = invoke(stypy.reporting.localization.Localization(__file__, 409, 11), _quotes_historical_yahoo_55126, *[ticker_55127, date1_55128, date2_55129], **kwargs_55138)
    
    # Assigning a type to the variable 'stypy_return_type' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type', _quotes_historical_yahoo_call_result_55139)
    
    # ################# End of 'quotes_historical_yahoo_ochl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'quotes_historical_yahoo_ochl' in the type store
    # Getting the type of 'stypy_return_type' (line 372)
    stypy_return_type_55140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55140)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'quotes_historical_yahoo_ochl'
    return stypy_return_type_55140

# Assigning a type to the variable 'quotes_historical_yahoo_ochl' (line 372)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 0), 'quotes_historical_yahoo_ochl', quotes_historical_yahoo_ochl)

@norecursion
def quotes_historical_yahoo_ohlc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 414)
    False_55141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 64), 'False')
    # Getting the type of 'True' (line 415)
    True_55142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 37), 'True')
    # Getting the type of 'None' (line 415)
    None_55143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 53), 'None')
    defaults = [False_55141, True_55142, None_55143]
    # Create a new context for function 'quotes_historical_yahoo_ohlc'
    module_type_store = module_type_store.open_function_context('quotes_historical_yahoo_ohlc', 414, 0, False)
    
    # Passed parameters checking function
    quotes_historical_yahoo_ohlc.stypy_localization = localization
    quotes_historical_yahoo_ohlc.stypy_type_of_self = None
    quotes_historical_yahoo_ohlc.stypy_type_store = module_type_store
    quotes_historical_yahoo_ohlc.stypy_function_name = 'quotes_historical_yahoo_ohlc'
    quotes_historical_yahoo_ohlc.stypy_param_names_list = ['ticker', 'date1', 'date2', 'asobject', 'adjusted', 'cachename']
    quotes_historical_yahoo_ohlc.stypy_varargs_param_name = None
    quotes_historical_yahoo_ohlc.stypy_kwargs_param_name = None
    quotes_historical_yahoo_ohlc.stypy_call_defaults = defaults
    quotes_historical_yahoo_ohlc.stypy_call_varargs = varargs
    quotes_historical_yahoo_ohlc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'quotes_historical_yahoo_ohlc', ['ticker', 'date1', 'date2', 'asobject', 'adjusted', 'cachename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'quotes_historical_yahoo_ohlc', localization, ['ticker', 'date1', 'date2', 'asobject', 'adjusted', 'cachename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'quotes_historical_yahoo_ohlc(...)' code ##################

    unicode_55144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, (-1)), 'unicode', u" Get historical data for ticker between date1 and date2.\n\n\n    See :func:`parse_yahoo_historical` for explanation of output formats\n    and the *asobject* and *adjusted* kwargs.\n\n    Parameters\n    ----------\n    ticker : str\n        stock ticker\n\n    date1 : sequence of form (year, month, day), `datetime`, or `date`\n        start date\n\n    date2 : sequence of form (year, month, day), `datetime`, or `date`\n        end date\n\n    cachename : str or `None`\n        is the name of the local file cache.  If None, will\n        default to the md5 hash or the url (which incorporates the ticker\n        and date range)\n\n    Examples\n    --------\n    >>> sp = f.quotes_historical_yahoo_ohlc('^GSPC', d1, d2,\n                             asobject=True, adjusted=True)\n    >>> returns = (sp.open[1:] - sp.open[:-1])/sp.open[1:]\n    >>> [n,bins,patches] = hist(returns, 100)\n    >>> mu = mean(returns)\n    >>> sigma = std(returns)\n    >>> x = normpdf(bins, mu, sigma)\n    >>> plot(bins, x, color='red', lw=2)\n\n    ")
    
    # Call to _quotes_historical_yahoo(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'ticker' (line 451)
    ticker_55146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 36), 'ticker', False)
    # Getting the type of 'date1' (line 451)
    date1_55147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 44), 'date1', False)
    # Getting the type of 'date2' (line 451)
    date2_55148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 51), 'date2', False)
    # Processing the call keyword arguments (line 451)
    # Getting the type of 'asobject' (line 451)
    asobject_55149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 67), 'asobject', False)
    keyword_55150 = asobject_55149
    # Getting the type of 'adjusted' (line 452)
    adjusted_55151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 38), 'adjusted', False)
    keyword_55152 = adjusted_55151
    # Getting the type of 'cachename' (line 452)
    cachename_55153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 58), 'cachename', False)
    keyword_55154 = cachename_55153
    # Getting the type of 'False' (line 453)
    False_55155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'False', False)
    keyword_55156 = False_55155
    kwargs_55157 = {'adjusted': keyword_55152, 'cachename': keyword_55154, 'asobject': keyword_55150, 'ochl': keyword_55156}
    # Getting the type of '_quotes_historical_yahoo' (line 451)
    _quotes_historical_yahoo_55145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 11), '_quotes_historical_yahoo', False)
    # Calling _quotes_historical_yahoo(args, kwargs) (line 451)
    _quotes_historical_yahoo_call_result_55158 = invoke(stypy.reporting.localization.Localization(__file__, 451, 11), _quotes_historical_yahoo_55145, *[ticker_55146, date1_55147, date2_55148], **kwargs_55157)
    
    # Assigning a type to the variable 'stypy_return_type' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type', _quotes_historical_yahoo_call_result_55158)
    
    # ################# End of 'quotes_historical_yahoo_ohlc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'quotes_historical_yahoo_ohlc' in the type store
    # Getting the type of 'stypy_return_type' (line 414)
    stypy_return_type_55159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55159)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'quotes_historical_yahoo_ohlc'
    return stypy_return_type_55159

# Assigning a type to the variable 'quotes_historical_yahoo_ohlc' (line 414)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 0), 'quotes_historical_yahoo_ohlc', quotes_historical_yahoo_ohlc)

@norecursion
def _quotes_historical_yahoo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 456)
    False_55160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 60), 'False')
    # Getting the type of 'True' (line 457)
    True_55161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 37), 'True')
    # Getting the type of 'None' (line 457)
    None_55162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 53), 'None')
    # Getting the type of 'True' (line 458)
    True_55163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 33), 'True')
    defaults = [False_55160, True_55161, None_55162, True_55163]
    # Create a new context for function '_quotes_historical_yahoo'
    module_type_store = module_type_store.open_function_context('_quotes_historical_yahoo', 456, 0, False)
    
    # Passed parameters checking function
    _quotes_historical_yahoo.stypy_localization = localization
    _quotes_historical_yahoo.stypy_type_of_self = None
    _quotes_historical_yahoo.stypy_type_store = module_type_store
    _quotes_historical_yahoo.stypy_function_name = '_quotes_historical_yahoo'
    _quotes_historical_yahoo.stypy_param_names_list = ['ticker', 'date1', 'date2', 'asobject', 'adjusted', 'cachename', 'ochl']
    _quotes_historical_yahoo.stypy_varargs_param_name = None
    _quotes_historical_yahoo.stypy_kwargs_param_name = None
    _quotes_historical_yahoo.stypy_call_defaults = defaults
    _quotes_historical_yahoo.stypy_call_varargs = varargs
    _quotes_historical_yahoo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_quotes_historical_yahoo', ['ticker', 'date1', 'date2', 'asobject', 'adjusted', 'cachename', 'ochl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_quotes_historical_yahoo', localization, ['ticker', 'date1', 'date2', 'asobject', 'adjusted', 'cachename', 'ochl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_quotes_historical_yahoo(...)' code ##################

    unicode_55164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, (-1)), 'unicode', u" Get historical data for ticker between date1 and date2.\n\n    See :func:`parse_yahoo_historical` for explanation of output formats\n    and the *asobject* and *adjusted* kwargs.\n\n    Parameters\n    ----------\n    ticker : str\n        stock ticker\n\n    date1 : sequence of form (year, month, day), `datetime`, or `date`\n        start date\n\n    date2 : sequence of form (year, month, day), `datetime`, or `date`\n        end date\n\n    cachename : str or `None`\n        is the name of the local file cache.  If None, will\n        default to the md5 hash or the url (which incorporates the ticker\n        and date range)\n\n    ochl: bool\n        temporary argument to select between ochl and ohlc ordering\n\n\n    Examples\n    --------\n    >>> sp = f.quotes_historical_yahoo('^GSPC', d1, d2,\n                             asobject=True, adjusted=True)\n    >>> returns = (sp.open[1:] - sp.open[:-1])/sp.open[1:]\n    >>> [n,bins,patches] = hist(returns, 100)\n    >>> mu = mean(returns)\n    >>> sigma = std(returns)\n    >>> x = normpdf(bins, mu, sigma)\n    >>> plot(bins, x, color='red', lw=2)\n\n    ")
    
    # Assigning a Call to a Name (line 501):
    
    # Assigning a Call to a Name (line 501):
    
    # Call to fetch_historical_yahoo(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'ticker' (line 501)
    ticker_55166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 32), 'ticker', False)
    # Getting the type of 'date1' (line 501)
    date1_55167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 40), 'date1', False)
    # Getting the type of 'date2' (line 501)
    date2_55168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 47), 'date2', False)
    # Getting the type of 'cachename' (line 501)
    cachename_55169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 54), 'cachename', False)
    # Processing the call keyword arguments (line 501)
    kwargs_55170 = {}
    # Getting the type of 'fetch_historical_yahoo' (line 501)
    fetch_historical_yahoo_55165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 9), 'fetch_historical_yahoo', False)
    # Calling fetch_historical_yahoo(args, kwargs) (line 501)
    fetch_historical_yahoo_call_result_55171 = invoke(stypy.reporting.localization.Localization(__file__, 501, 9), fetch_historical_yahoo_55165, *[ticker_55166, date1_55167, date2_55168, cachename_55169], **kwargs_55170)
    
    # Assigning a type to the variable 'fh' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'fh', fetch_historical_yahoo_call_result_55171)
    
    
    # SSA begins for try-except statement (line 503)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to _parse_yahoo_historical(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'fh' (line 504)
    fh_55173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 38), 'fh', False)
    # Processing the call keyword arguments (line 504)
    # Getting the type of 'asobject' (line 504)
    asobject_55174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 51), 'asobject', False)
    keyword_55175 = asobject_55174
    # Getting the type of 'adjusted' (line 505)
    adjusted_55176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 46), 'adjusted', False)
    keyword_55177 = adjusted_55176
    # Getting the type of 'ochl' (line 505)
    ochl_55178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 61), 'ochl', False)
    keyword_55179 = ochl_55178
    kwargs_55180 = {'adjusted': keyword_55177, 'asobject': keyword_55175, 'ochl': keyword_55179}
    # Getting the type of '_parse_yahoo_historical' (line 504)
    _parse_yahoo_historical_55172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 14), '_parse_yahoo_historical', False)
    # Calling _parse_yahoo_historical(args, kwargs) (line 504)
    _parse_yahoo_historical_call_result_55181 = invoke(stypy.reporting.localization.Localization(__file__, 504, 14), _parse_yahoo_historical_55172, *[fh_55173], **kwargs_55180)
    
    # Assigning a type to the variable 'ret' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'ret', _parse_yahoo_historical_call_result_55181)
    
    
    
    # Call to len(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'ret' (line 506)
    ret_55183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 15), 'ret', False)
    # Processing the call keyword arguments (line 506)
    kwargs_55184 = {}
    # Getting the type of 'len' (line 506)
    len_55182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 11), 'len', False)
    # Calling len(args, kwargs) (line 506)
    len_call_result_55185 = invoke(stypy.reporting.localization.Localization(__file__, 506, 11), len_55182, *[ret_55183], **kwargs_55184)
    
    int_55186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 23), 'int')
    # Applying the binary operator '==' (line 506)
    result_eq_55187 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 11), '==', len_call_result_55185, int_55186)
    
    # Testing the type of an if condition (line 506)
    if_condition_55188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 506, 8), result_eq_55187)
    # Assigning a type to the variable 'if_condition_55188' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'if_condition_55188', if_condition_55188)
    # SSA begins for if statement (line 506)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 507)
    None_55189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 19), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'stypy_return_type', None_55189)
    # SSA join for if statement (line 506)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 503)
    # SSA branch for the except 'IOError' branch of a try statement (line 503)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'IOError' (line 508)
    IOError_55190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 11), 'IOError')
    # Assigning a type to the variable 'exc' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'exc', IOError_55190)
    
    # Call to warn(...): (line 509)
    # Processing the call arguments (line 509)
    unicode_55193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 22), 'unicode', u'fh failure\n%s')
    
    # Obtaining the type of the subscript
    int_55194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 55), 'int')
    # Getting the type of 'exc' (line 509)
    exc_55195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 42), 'exc', False)
    # Obtaining the member 'strerror' of a type (line 509)
    strerror_55196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 42), exc_55195, 'strerror')
    # Obtaining the member '__getitem__' of a type (line 509)
    getitem___55197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 42), strerror_55196, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 509)
    subscript_call_result_55198 = invoke(stypy.reporting.localization.Localization(__file__, 509, 42), getitem___55197, int_55194)
    
    # Applying the binary operator '%' (line 509)
    result_mod_55199 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 22), '%', unicode_55193, subscript_call_result_55198)
    
    # Processing the call keyword arguments (line 509)
    kwargs_55200 = {}
    # Getting the type of 'warnings' (line 509)
    warnings_55191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 509)
    warn_55192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), warnings_55191, 'warn')
    # Calling warn(args, kwargs) (line 509)
    warn_call_result_55201 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), warn_55192, *[result_mod_55199], **kwargs_55200)
    
    # Getting the type of 'None' (line 510)
    None_55202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'stypy_return_type', None_55202)
    # SSA join for try-except statement (line 503)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 512)
    ret_55203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type', ret_55203)
    
    # ################# End of '_quotes_historical_yahoo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_quotes_historical_yahoo' in the type store
    # Getting the type of 'stypy_return_type' (line 456)
    stypy_return_type_55204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55204)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_quotes_historical_yahoo'
    return stypy_return_type_55204

# Assigning a type to the variable '_quotes_historical_yahoo' (line 456)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), '_quotes_historical_yahoo', _quotes_historical_yahoo)

@norecursion
def plot_day_summary_oclh(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_55205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 47), 'int')
    unicode_55206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 29), 'unicode', u'k')
    unicode_55207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 44), 'unicode', u'r')
    defaults = [int_55205, unicode_55206, unicode_55207]
    # Create a new context for function 'plot_day_summary_oclh'
    module_type_store = module_type_store.open_function_context('plot_day_summary_oclh', 515, 0, False)
    
    # Passed parameters checking function
    plot_day_summary_oclh.stypy_localization = localization
    plot_day_summary_oclh.stypy_type_of_self = None
    plot_day_summary_oclh.stypy_type_store = module_type_store
    plot_day_summary_oclh.stypy_function_name = 'plot_day_summary_oclh'
    plot_day_summary_oclh.stypy_param_names_list = ['ax', 'quotes', 'ticksize', 'colorup', 'colordown']
    plot_day_summary_oclh.stypy_varargs_param_name = None
    plot_day_summary_oclh.stypy_kwargs_param_name = None
    plot_day_summary_oclh.stypy_call_defaults = defaults
    plot_day_summary_oclh.stypy_call_varargs = varargs
    plot_day_summary_oclh.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'plot_day_summary_oclh', ['ax', 'quotes', 'ticksize', 'colorup', 'colordown'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'plot_day_summary_oclh', localization, ['ax', 'quotes', 'ticksize', 'colorup', 'colordown'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'plot_day_summary_oclh(...)' code ##################

    unicode_55208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, (-1)), 'unicode', u'Plots day summary\n\n        Represent the time, open, close, high, low as a vertical line\n        ranging from low to high.  The left tick is the open and the right\n        tick is the close.\n\n\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an `Axes` instance to plot to\n    quotes : sequence of (time, open, close, high, low, ...) sequences\n        data to plot.  time must be in float date format - see date2num\n    ticksize : int\n        open/close tick marker in points\n    colorup : color\n        the color of the lines where close >= open\n    colordown : color\n        the color of the lines where close <  open\n\n    Returns\n    -------\n    lines : list\n        list of tuples of the lines added (one tuple per quote)\n    ')
    
    # Call to _plot_day_summary(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'ax' (line 544)
    ax_55210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 29), 'ax', False)
    # Getting the type of 'quotes' (line 544)
    quotes_55211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 33), 'quotes', False)
    # Processing the call keyword arguments (line 544)
    # Getting the type of 'ticksize' (line 544)
    ticksize_55212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 50), 'ticksize', False)
    keyword_55213 = ticksize_55212
    # Getting the type of 'colorup' (line 545)
    colorup_55214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 29), 'colorup', False)
    keyword_55215 = colorup_55214
    # Getting the type of 'colordown' (line 545)
    colordown_55216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 48), 'colordown', False)
    keyword_55217 = colordown_55216
    # Getting the type of 'True' (line 546)
    True_55218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 26), 'True', False)
    keyword_55219 = True_55218
    kwargs_55220 = {'colordown': keyword_55217, 'colorup': keyword_55215, 'ticksize': keyword_55213, 'ochl': keyword_55219}
    # Getting the type of '_plot_day_summary' (line 544)
    _plot_day_summary_55209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 11), '_plot_day_summary', False)
    # Calling _plot_day_summary(args, kwargs) (line 544)
    _plot_day_summary_call_result_55221 = invoke(stypy.reporting.localization.Localization(__file__, 544, 11), _plot_day_summary_55209, *[ax_55210, quotes_55211], **kwargs_55220)
    
    # Assigning a type to the variable 'stypy_return_type' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'stypy_return_type', _plot_day_summary_call_result_55221)
    
    # ################# End of 'plot_day_summary_oclh(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'plot_day_summary_oclh' in the type store
    # Getting the type of 'stypy_return_type' (line 515)
    stypy_return_type_55222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55222)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'plot_day_summary_oclh'
    return stypy_return_type_55222

# Assigning a type to the variable 'plot_day_summary_oclh' (line 515)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'plot_day_summary_oclh', plot_day_summary_oclh)

@norecursion
def plot_day_summary_ohlc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_55223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 47), 'int')
    unicode_55224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 29), 'unicode', u'k')
    unicode_55225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 44), 'unicode', u'r')
    defaults = [int_55223, unicode_55224, unicode_55225]
    # Create a new context for function 'plot_day_summary_ohlc'
    module_type_store = module_type_store.open_function_context('plot_day_summary_ohlc', 549, 0, False)
    
    # Passed parameters checking function
    plot_day_summary_ohlc.stypy_localization = localization
    plot_day_summary_ohlc.stypy_type_of_self = None
    plot_day_summary_ohlc.stypy_type_store = module_type_store
    plot_day_summary_ohlc.stypy_function_name = 'plot_day_summary_ohlc'
    plot_day_summary_ohlc.stypy_param_names_list = ['ax', 'quotes', 'ticksize', 'colorup', 'colordown']
    plot_day_summary_ohlc.stypy_varargs_param_name = None
    plot_day_summary_ohlc.stypy_kwargs_param_name = None
    plot_day_summary_ohlc.stypy_call_defaults = defaults
    plot_day_summary_ohlc.stypy_call_varargs = varargs
    plot_day_summary_ohlc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'plot_day_summary_ohlc', ['ax', 'quotes', 'ticksize', 'colorup', 'colordown'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'plot_day_summary_ohlc', localization, ['ax', 'quotes', 'ticksize', 'colorup', 'colordown'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'plot_day_summary_ohlc(...)' code ##################

    unicode_55226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, (-1)), 'unicode', u'Plots day summary\n\n        Represent the time, open, high, low, close as a vertical line\n        ranging from low to high.  The left tick is the open and the right\n        tick is the close.\n\n\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an `Axes` instance to plot to\n    quotes : sequence of (time, open, high, low, close, ...) sequences\n        data to plot.  time must be in float date format - see date2num\n    ticksize : int\n        open/close tick marker in points\n    colorup : color\n        the color of the lines where close >= open\n    colordown : color\n        the color of the lines where close <  open\n\n    Returns\n    -------\n    lines : list\n        list of tuples of the lines added (one tuple per quote)\n    ')
    
    # Call to _plot_day_summary(...): (line 578)
    # Processing the call arguments (line 578)
    # Getting the type of 'ax' (line 578)
    ax_55228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 29), 'ax', False)
    # Getting the type of 'quotes' (line 578)
    quotes_55229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 33), 'quotes', False)
    # Processing the call keyword arguments (line 578)
    # Getting the type of 'ticksize' (line 578)
    ticksize_55230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 50), 'ticksize', False)
    keyword_55231 = ticksize_55230
    # Getting the type of 'colorup' (line 579)
    colorup_55232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 29), 'colorup', False)
    keyword_55233 = colorup_55232
    # Getting the type of 'colordown' (line 579)
    colordown_55234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 48), 'colordown', False)
    keyword_55235 = colordown_55234
    # Getting the type of 'False' (line 580)
    False_55236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 26), 'False', False)
    keyword_55237 = False_55236
    kwargs_55238 = {'colordown': keyword_55235, 'colorup': keyword_55233, 'ticksize': keyword_55231, 'ochl': keyword_55237}
    # Getting the type of '_plot_day_summary' (line 578)
    _plot_day_summary_55227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 11), '_plot_day_summary', False)
    # Calling _plot_day_summary(args, kwargs) (line 578)
    _plot_day_summary_call_result_55239 = invoke(stypy.reporting.localization.Localization(__file__, 578, 11), _plot_day_summary_55227, *[ax_55228, quotes_55229], **kwargs_55238)
    
    # Assigning a type to the variable 'stypy_return_type' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type', _plot_day_summary_call_result_55239)
    
    # ################# End of 'plot_day_summary_ohlc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'plot_day_summary_ohlc' in the type store
    # Getting the type of 'stypy_return_type' (line 549)
    stypy_return_type_55240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55240)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'plot_day_summary_ohlc'
    return stypy_return_type_55240

# Assigning a type to the variable 'plot_day_summary_ohlc' (line 549)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 0), 'plot_day_summary_ohlc', plot_day_summary_ohlc)

@norecursion
def _plot_day_summary(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_55241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 43), 'int')
    unicode_55242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 29), 'unicode', u'k')
    unicode_55243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 44), 'unicode', u'r')
    # Getting the type of 'True' (line 585)
    True_55244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 26), 'True')
    defaults = [int_55241, unicode_55242, unicode_55243, True_55244]
    # Create a new context for function '_plot_day_summary'
    module_type_store = module_type_store.open_function_context('_plot_day_summary', 583, 0, False)
    
    # Passed parameters checking function
    _plot_day_summary.stypy_localization = localization
    _plot_day_summary.stypy_type_of_self = None
    _plot_day_summary.stypy_type_store = module_type_store
    _plot_day_summary.stypy_function_name = '_plot_day_summary'
    _plot_day_summary.stypy_param_names_list = ['ax', 'quotes', 'ticksize', 'colorup', 'colordown', 'ochl']
    _plot_day_summary.stypy_varargs_param_name = None
    _plot_day_summary.stypy_kwargs_param_name = None
    _plot_day_summary.stypy_call_defaults = defaults
    _plot_day_summary.stypy_call_varargs = varargs
    _plot_day_summary.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_plot_day_summary', ['ax', 'quotes', 'ticksize', 'colorup', 'colordown', 'ochl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_plot_day_summary', localization, ['ax', 'quotes', 'ticksize', 'colorup', 'colordown', 'ochl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_plot_day_summary(...)' code ##################

    unicode_55245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, (-1)), 'unicode', u'Plots day summary\n\n\n        Represent the time, open, high, low, close as a vertical line\n        ranging from low to high.  The left tick is the open and the right\n        tick is the close.\n\n\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an `Axes` instance to plot to\n    quotes : sequence of quote sequences\n        data to plot.  time must be in float date format - see date2num\n        (time, open, high, low, close, ...) vs\n        (time, open, close, high, low, ...)\n        set by `ochl`\n    ticksize : int\n        open/close tick marker in points\n    colorup : color\n        the color of the lines where close >= open\n    colordown : color\n        the color of the lines where close <  open\n    ochl: bool\n        argument to select between ochl and ohlc ordering of quotes\n\n    Returns\n    -------\n    lines : list\n        list of tuples of the lines added (one tuple per quote)\n    ')
    
    # Assigning a List to a Name (line 620):
    
    # Assigning a List to a Name (line 620):
    
    # Obtaining an instance of the builtin type 'list' (line 620)
    list_55246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 620)
    
    # Assigning a type to the variable 'lines' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'lines', list_55246)
    
    # Getting the type of 'quotes' (line 621)
    quotes_55247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 13), 'quotes')
    # Testing the type of a for loop iterable (line 621)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 621, 4), quotes_55247)
    # Getting the type of the for loop variable (line 621)
    for_loop_var_55248 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 621, 4), quotes_55247)
    # Assigning a type to the variable 'q' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'q', for_loop_var_55248)
    # SSA begins for a for statement (line 621)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'ochl' (line 622)
    ochl_55249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 11), 'ochl')
    # Testing the type of an if condition (line 622)
    if_condition_55250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 8), ochl_55249)
    # Assigning a type to the variable 'if_condition_55250' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'if_condition_55250', if_condition_55250)
    # SSA begins for if statement (line 622)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 623):
    
    # Assigning a Subscript to a Name (line 623):
    
    # Obtaining the type of the subscript
    int_55251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 43), 'int')
    slice_55253 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 623, 40), None, int_55252, None)
    # Getting the type of 'q' (line 623)
    q_55254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 40), q_55254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55256 = invoke(stypy.reporting.localization.Localization(__file__, 623, 40), getitem___55255, slice_55253)
    
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 12), subscript_call_result_55256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55258 = invoke(stypy.reporting.localization.Localization(__file__, 623, 12), getitem___55257, int_55251)
    
    # Assigning a type to the variable 'tuple_var_assignment_54298' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54298', subscript_call_result_55258)
    
    # Assigning a Subscript to a Name (line 623):
    
    # Obtaining the type of the subscript
    int_55259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 43), 'int')
    slice_55261 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 623, 40), None, int_55260, None)
    # Getting the type of 'q' (line 623)
    q_55262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 40), q_55262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55264 = invoke(stypy.reporting.localization.Localization(__file__, 623, 40), getitem___55263, slice_55261)
    
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 12), subscript_call_result_55264, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55266 = invoke(stypy.reporting.localization.Localization(__file__, 623, 12), getitem___55265, int_55259)
    
    # Assigning a type to the variable 'tuple_var_assignment_54299' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54299', subscript_call_result_55266)
    
    # Assigning a Subscript to a Name (line 623):
    
    # Obtaining the type of the subscript
    int_55267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 43), 'int')
    slice_55269 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 623, 40), None, int_55268, None)
    # Getting the type of 'q' (line 623)
    q_55270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 40), q_55270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55272 = invoke(stypy.reporting.localization.Localization(__file__, 623, 40), getitem___55271, slice_55269)
    
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 12), subscript_call_result_55272, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55274 = invoke(stypy.reporting.localization.Localization(__file__, 623, 12), getitem___55273, int_55267)
    
    # Assigning a type to the variable 'tuple_var_assignment_54300' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54300', subscript_call_result_55274)
    
    # Assigning a Subscript to a Name (line 623):
    
    # Obtaining the type of the subscript
    int_55275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 43), 'int')
    slice_55277 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 623, 40), None, int_55276, None)
    # Getting the type of 'q' (line 623)
    q_55278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 40), q_55278, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55280 = invoke(stypy.reporting.localization.Localization(__file__, 623, 40), getitem___55279, slice_55277)
    
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 12), subscript_call_result_55280, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55282 = invoke(stypy.reporting.localization.Localization(__file__, 623, 12), getitem___55281, int_55275)
    
    # Assigning a type to the variable 'tuple_var_assignment_54301' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54301', subscript_call_result_55282)
    
    # Assigning a Subscript to a Name (line 623):
    
    # Obtaining the type of the subscript
    int_55283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 43), 'int')
    slice_55285 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 623, 40), None, int_55284, None)
    # Getting the type of 'q' (line 623)
    q_55286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 40), q_55286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55288 = invoke(stypy.reporting.localization.Localization(__file__, 623, 40), getitem___55287, slice_55285)
    
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___55289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 12), subscript_call_result_55288, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_55290 = invoke(stypy.reporting.localization.Localization(__file__, 623, 12), getitem___55289, int_55283)
    
    # Assigning a type to the variable 'tuple_var_assignment_54302' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54302', subscript_call_result_55290)
    
    # Assigning a Name to a Name (line 623):
    # Getting the type of 'tuple_var_assignment_54298' (line 623)
    tuple_var_assignment_54298_55291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54298')
    # Assigning a type to the variable 't' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 't', tuple_var_assignment_54298_55291)
    
    # Assigning a Name to a Name (line 623):
    # Getting the type of 'tuple_var_assignment_54299' (line 623)
    tuple_var_assignment_54299_55292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54299')
    # Assigning a type to the variable 'open' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 15), 'open', tuple_var_assignment_54299_55292)
    
    # Assigning a Name to a Name (line 623):
    # Getting the type of 'tuple_var_assignment_54300' (line 623)
    tuple_var_assignment_54300_55293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54300')
    # Assigning a type to the variable 'close' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 21), 'close', tuple_var_assignment_54300_55293)
    
    # Assigning a Name to a Name (line 623):
    # Getting the type of 'tuple_var_assignment_54301' (line 623)
    tuple_var_assignment_54301_55294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54301')
    # Assigning a type to the variable 'high' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 28), 'high', tuple_var_assignment_54301_55294)
    
    # Assigning a Name to a Name (line 623):
    # Getting the type of 'tuple_var_assignment_54302' (line 623)
    tuple_var_assignment_54302_55295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'tuple_var_assignment_54302')
    # Assigning a type to the variable 'low' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 34), 'low', tuple_var_assignment_54302_55295)
    # SSA branch for the else part of an if statement (line 622)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Tuple (line 625):
    
    # Assigning a Subscript to a Name (line 625):
    
    # Obtaining the type of the subscript
    int_55296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 43), 'int')
    slice_55298 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 625, 40), None, int_55297, None)
    # Getting the type of 'q' (line 625)
    q_55299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 40), q_55299, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55301 = invoke(stypy.reporting.localization.Localization(__file__, 625, 40), getitem___55300, slice_55298)
    
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 12), subscript_call_result_55301, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55303 = invoke(stypy.reporting.localization.Localization(__file__, 625, 12), getitem___55302, int_55296)
    
    # Assigning a type to the variable 'tuple_var_assignment_54303' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54303', subscript_call_result_55303)
    
    # Assigning a Subscript to a Name (line 625):
    
    # Obtaining the type of the subscript
    int_55304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 43), 'int')
    slice_55306 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 625, 40), None, int_55305, None)
    # Getting the type of 'q' (line 625)
    q_55307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 40), q_55307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55309 = invoke(stypy.reporting.localization.Localization(__file__, 625, 40), getitem___55308, slice_55306)
    
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 12), subscript_call_result_55309, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55311 = invoke(stypy.reporting.localization.Localization(__file__, 625, 12), getitem___55310, int_55304)
    
    # Assigning a type to the variable 'tuple_var_assignment_54304' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54304', subscript_call_result_55311)
    
    # Assigning a Subscript to a Name (line 625):
    
    # Obtaining the type of the subscript
    int_55312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 43), 'int')
    slice_55314 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 625, 40), None, int_55313, None)
    # Getting the type of 'q' (line 625)
    q_55315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 40), q_55315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55317 = invoke(stypy.reporting.localization.Localization(__file__, 625, 40), getitem___55316, slice_55314)
    
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 12), subscript_call_result_55317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55319 = invoke(stypy.reporting.localization.Localization(__file__, 625, 12), getitem___55318, int_55312)
    
    # Assigning a type to the variable 'tuple_var_assignment_54305' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54305', subscript_call_result_55319)
    
    # Assigning a Subscript to a Name (line 625):
    
    # Obtaining the type of the subscript
    int_55320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 43), 'int')
    slice_55322 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 625, 40), None, int_55321, None)
    # Getting the type of 'q' (line 625)
    q_55323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 40), q_55323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55325 = invoke(stypy.reporting.localization.Localization(__file__, 625, 40), getitem___55324, slice_55322)
    
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 12), subscript_call_result_55325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55327 = invoke(stypy.reporting.localization.Localization(__file__, 625, 12), getitem___55326, int_55320)
    
    # Assigning a type to the variable 'tuple_var_assignment_54306' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54306', subscript_call_result_55327)
    
    # Assigning a Subscript to a Name (line 625):
    
    # Obtaining the type of the subscript
    int_55328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 43), 'int')
    slice_55330 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 625, 40), None, int_55329, None)
    # Getting the type of 'q' (line 625)
    q_55331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 40), q_55331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55333 = invoke(stypy.reporting.localization.Localization(__file__, 625, 40), getitem___55332, slice_55330)
    
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___55334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 12), subscript_call_result_55333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_55335 = invoke(stypy.reporting.localization.Localization(__file__, 625, 12), getitem___55334, int_55328)
    
    # Assigning a type to the variable 'tuple_var_assignment_54307' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54307', subscript_call_result_55335)
    
    # Assigning a Name to a Name (line 625):
    # Getting the type of 'tuple_var_assignment_54303' (line 625)
    tuple_var_assignment_54303_55336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54303')
    # Assigning a type to the variable 't' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 't', tuple_var_assignment_54303_55336)
    
    # Assigning a Name to a Name (line 625):
    # Getting the type of 'tuple_var_assignment_54304' (line 625)
    tuple_var_assignment_54304_55337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54304')
    # Assigning a type to the variable 'open' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'open', tuple_var_assignment_54304_55337)
    
    # Assigning a Name to a Name (line 625):
    # Getting the type of 'tuple_var_assignment_54305' (line 625)
    tuple_var_assignment_54305_55338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54305')
    # Assigning a type to the variable 'high' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 21), 'high', tuple_var_assignment_54305_55338)
    
    # Assigning a Name to a Name (line 625):
    # Getting the type of 'tuple_var_assignment_54306' (line 625)
    tuple_var_assignment_54306_55339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54306')
    # Assigning a type to the variable 'low' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 27), 'low', tuple_var_assignment_54306_55339)
    
    # Assigning a Name to a Name (line 625):
    # Getting the type of 'tuple_var_assignment_54307' (line 625)
    tuple_var_assignment_54307_55340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'tuple_var_assignment_54307')
    # Assigning a type to the variable 'close' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 32), 'close', tuple_var_assignment_54307_55340)
    # SSA join for if statement (line 622)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'close' (line 627)
    close_55341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 11), 'close')
    # Getting the type of 'open' (line 627)
    open_55342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 20), 'open')
    # Applying the binary operator '>=' (line 627)
    result_ge_55343 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 11), '>=', close_55341, open_55342)
    
    # Testing the type of an if condition (line 627)
    if_condition_55344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 627, 8), result_ge_55343)
    # Assigning a type to the variable 'if_condition_55344' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'if_condition_55344', if_condition_55344)
    # SSA begins for if statement (line 627)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 628):
    
    # Assigning a Name to a Name (line 628):
    # Getting the type of 'colorup' (line 628)
    colorup_55345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 20), 'colorup')
    # Assigning a type to the variable 'color' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 12), 'color', colorup_55345)
    # SSA branch for the else part of an if statement (line 627)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 630):
    
    # Assigning a Name to a Name (line 630):
    # Getting the type of 'colordown' (line 630)
    colordown_55346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 20), 'colordown')
    # Assigning a type to the variable 'color' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'color', colordown_55346)
    # SSA join for if statement (line 627)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 632):
    
    # Assigning a Call to a Name (line 632):
    
    # Call to Line2D(...): (line 632)
    # Processing the call keyword arguments (line 632)
    
    # Obtaining an instance of the builtin type 'tuple' (line 632)
    tuple_55348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 632)
    # Adding element type (line 632)
    # Getting the type of 't' (line 632)
    t_55349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 30), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 30), tuple_55348, t_55349)
    # Adding element type (line 632)
    # Getting the type of 't' (line 632)
    t_55350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 33), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 30), tuple_55348, t_55350)
    
    keyword_55351 = tuple_55348
    
    # Obtaining an instance of the builtin type 'tuple' (line 632)
    tuple_55352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 632)
    # Adding element type (line 632)
    # Getting the type of 'low' (line 632)
    low_55353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 44), 'low', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 44), tuple_55352, low_55353)
    # Adding element type (line 632)
    # Getting the type of 'high' (line 632)
    high_55354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 49), 'high', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 44), tuple_55352, high_55354)
    
    keyword_55355 = tuple_55352
    # Getting the type of 'color' (line 633)
    color_55356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 29), 'color', False)
    keyword_55357 = color_55356
    # Getting the type of 'False' (line 634)
    False_55358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 35), 'False', False)
    keyword_55359 = False_55358
    kwargs_55360 = {'color': keyword_55357, 'antialiased': keyword_55359, 'ydata': keyword_55355, 'xdata': keyword_55351}
    # Getting the type of 'Line2D' (line 632)
    Line2D_55347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 16), 'Line2D', False)
    # Calling Line2D(args, kwargs) (line 632)
    Line2D_call_result_55361 = invoke(stypy.reporting.localization.Localization(__file__, 632, 16), Line2D_55347, *[], **kwargs_55360)
    
    # Assigning a type to the variable 'vline' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'vline', Line2D_call_result_55361)
    
    # Assigning a Call to a Name (line 637):
    
    # Assigning a Call to a Name (line 637):
    
    # Call to Line2D(...): (line 637)
    # Processing the call keyword arguments (line 637)
    
    # Obtaining an instance of the builtin type 'tuple' (line 637)
    tuple_55363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 637)
    # Adding element type (line 637)
    # Getting the type of 't' (line 637)
    t_55364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 30), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 30), tuple_55363, t_55364)
    # Adding element type (line 637)
    # Getting the type of 't' (line 637)
    t_55365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 33), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 30), tuple_55363, t_55365)
    
    keyword_55366 = tuple_55363
    
    # Obtaining an instance of the builtin type 'tuple' (line 637)
    tuple_55367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 637)
    # Adding element type (line 637)
    # Getting the type of 'open' (line 637)
    open_55368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 44), 'open', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 44), tuple_55367, open_55368)
    # Adding element type (line 637)
    # Getting the type of 'open' (line 637)
    open_55369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 50), 'open', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 637, 44), tuple_55367, open_55369)
    
    keyword_55370 = tuple_55367
    # Getting the type of 'color' (line 638)
    color_55371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 29), 'color', False)
    keyword_55372 = color_55371
    # Getting the type of 'False' (line 639)
    False_55373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 35), 'False', False)
    keyword_55374 = False_55373
    # Getting the type of 'TICKLEFT' (line 640)
    TICKLEFT_55375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 30), 'TICKLEFT', False)
    keyword_55376 = TICKLEFT_55375
    # Getting the type of 'ticksize' (line 641)
    ticksize_55377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 34), 'ticksize', False)
    keyword_55378 = ticksize_55377
    kwargs_55379 = {'color': keyword_55372, 'markersize': keyword_55378, 'antialiased': keyword_55374, 'xdata': keyword_55366, 'marker': keyword_55376, 'ydata': keyword_55370}
    # Getting the type of 'Line2D' (line 637)
    Line2D_55362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 16), 'Line2D', False)
    # Calling Line2D(args, kwargs) (line 637)
    Line2D_call_result_55380 = invoke(stypy.reporting.localization.Localization(__file__, 637, 16), Line2D_55362, *[], **kwargs_55379)
    
    # Assigning a type to the variable 'oline' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'oline', Line2D_call_result_55380)
    
    # Assigning a Call to a Name (line 644):
    
    # Assigning a Call to a Name (line 644):
    
    # Call to Line2D(...): (line 644)
    # Processing the call keyword arguments (line 644)
    
    # Obtaining an instance of the builtin type 'tuple' (line 644)
    tuple_55382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 644)
    # Adding element type (line 644)
    # Getting the type of 't' (line 644)
    t_55383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 30), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 30), tuple_55382, t_55383)
    # Adding element type (line 644)
    # Getting the type of 't' (line 644)
    t_55384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 33), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 30), tuple_55382, t_55384)
    
    keyword_55385 = tuple_55382
    
    # Obtaining an instance of the builtin type 'tuple' (line 644)
    tuple_55386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 644)
    # Adding element type (line 644)
    # Getting the type of 'close' (line 644)
    close_55387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 44), 'close', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 44), tuple_55386, close_55387)
    # Adding element type (line 644)
    # Getting the type of 'close' (line 644)
    close_55388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 51), 'close', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 44), tuple_55386, close_55388)
    
    keyword_55389 = tuple_55386
    # Getting the type of 'color' (line 645)
    color_55390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 29), 'color', False)
    keyword_55391 = color_55390
    # Getting the type of 'False' (line 646)
    False_55392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 35), 'False', False)
    keyword_55393 = False_55392
    # Getting the type of 'ticksize' (line 647)
    ticksize_55394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 34), 'ticksize', False)
    keyword_55395 = ticksize_55394
    # Getting the type of 'TICKRIGHT' (line 648)
    TICKRIGHT_55396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 30), 'TICKRIGHT', False)
    keyword_55397 = TICKRIGHT_55396
    kwargs_55398 = {'color': keyword_55391, 'markersize': keyword_55395, 'antialiased': keyword_55393, 'xdata': keyword_55385, 'marker': keyword_55397, 'ydata': keyword_55389}
    # Getting the type of 'Line2D' (line 644)
    Line2D_55381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 16), 'Line2D', False)
    # Calling Line2D(args, kwargs) (line 644)
    Line2D_call_result_55399 = invoke(stypy.reporting.localization.Localization(__file__, 644, 16), Line2D_55381, *[], **kwargs_55398)
    
    # Assigning a type to the variable 'cline' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'cline', Line2D_call_result_55399)
    
    # Call to extend(...): (line 650)
    # Processing the call arguments (line 650)
    
    # Obtaining an instance of the builtin type 'tuple' (line 650)
    tuple_55402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 650)
    # Adding element type (line 650)
    # Getting the type of 'vline' (line 650)
    vline_55403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 22), 'vline', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 22), tuple_55402, vline_55403)
    # Adding element type (line 650)
    # Getting the type of 'oline' (line 650)
    oline_55404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 29), 'oline', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 22), tuple_55402, oline_55404)
    # Adding element type (line 650)
    # Getting the type of 'cline' (line 650)
    cline_55405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 36), 'cline', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 22), tuple_55402, cline_55405)
    
    # Processing the call keyword arguments (line 650)
    kwargs_55406 = {}
    # Getting the type of 'lines' (line 650)
    lines_55400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'lines', False)
    # Obtaining the member 'extend' of a type (line 650)
    extend_55401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 8), lines_55400, 'extend')
    # Calling extend(args, kwargs) (line 650)
    extend_call_result_55407 = invoke(stypy.reporting.localization.Localization(__file__, 650, 8), extend_55401, *[tuple_55402], **kwargs_55406)
    
    
    # Call to add_line(...): (line 651)
    # Processing the call arguments (line 651)
    # Getting the type of 'vline' (line 651)
    vline_55410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 20), 'vline', False)
    # Processing the call keyword arguments (line 651)
    kwargs_55411 = {}
    # Getting the type of 'ax' (line 651)
    ax_55408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'ax', False)
    # Obtaining the member 'add_line' of a type (line 651)
    add_line_55409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 8), ax_55408, 'add_line')
    # Calling add_line(args, kwargs) (line 651)
    add_line_call_result_55412 = invoke(stypy.reporting.localization.Localization(__file__, 651, 8), add_line_55409, *[vline_55410], **kwargs_55411)
    
    
    # Call to add_line(...): (line 652)
    # Processing the call arguments (line 652)
    # Getting the type of 'oline' (line 652)
    oline_55415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 20), 'oline', False)
    # Processing the call keyword arguments (line 652)
    kwargs_55416 = {}
    # Getting the type of 'ax' (line 652)
    ax_55413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'ax', False)
    # Obtaining the member 'add_line' of a type (line 652)
    add_line_55414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 8), ax_55413, 'add_line')
    # Calling add_line(args, kwargs) (line 652)
    add_line_call_result_55417 = invoke(stypy.reporting.localization.Localization(__file__, 652, 8), add_line_55414, *[oline_55415], **kwargs_55416)
    
    
    # Call to add_line(...): (line 653)
    # Processing the call arguments (line 653)
    # Getting the type of 'cline' (line 653)
    cline_55420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'cline', False)
    # Processing the call keyword arguments (line 653)
    kwargs_55421 = {}
    # Getting the type of 'ax' (line 653)
    ax_55418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'ax', False)
    # Obtaining the member 'add_line' of a type (line 653)
    add_line_55419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 8), ax_55418, 'add_line')
    # Calling add_line(args, kwargs) (line 653)
    add_line_call_result_55422 = invoke(stypy.reporting.localization.Localization(__file__, 653, 8), add_line_55419, *[cline_55420], **kwargs_55421)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to autoscale_view(...): (line 655)
    # Processing the call keyword arguments (line 655)
    kwargs_55425 = {}
    # Getting the type of 'ax' (line 655)
    ax_55423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'ax', False)
    # Obtaining the member 'autoscale_view' of a type (line 655)
    autoscale_view_55424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 4), ax_55423, 'autoscale_view')
    # Calling autoscale_view(args, kwargs) (line 655)
    autoscale_view_call_result_55426 = invoke(stypy.reporting.localization.Localization(__file__, 655, 4), autoscale_view_55424, *[], **kwargs_55425)
    
    # Getting the type of 'lines' (line 657)
    lines_55427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 11), 'lines')
    # Assigning a type to the variable 'stypy_return_type' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'stypy_return_type', lines_55427)
    
    # ################# End of '_plot_day_summary(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_plot_day_summary' in the type store
    # Getting the type of 'stypy_return_type' (line 583)
    stypy_return_type_55428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55428)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_plot_day_summary'
    return stypy_return_type_55428

# Assigning a type to the variable '_plot_day_summary' (line 583)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 0), '_plot_day_summary', _plot_day_summary)

@norecursion
def candlestick_ochl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_55429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 39), 'float')
    unicode_55430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 52), 'unicode', u'k')
    unicode_55431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 67), 'unicode', u'r')
    float_55432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 22), 'float')
    defaults = [float_55429, unicode_55430, unicode_55431, float_55432]
    # Create a new context for function 'candlestick_ochl'
    module_type_store = module_type_store.open_function_context('candlestick_ochl', 660, 0, False)
    
    # Passed parameters checking function
    candlestick_ochl.stypy_localization = localization
    candlestick_ochl.stypy_type_of_self = None
    candlestick_ochl.stypy_type_store = module_type_store
    candlestick_ochl.stypy_function_name = 'candlestick_ochl'
    candlestick_ochl.stypy_param_names_list = ['ax', 'quotes', 'width', 'colorup', 'colordown', 'alpha']
    candlestick_ochl.stypy_varargs_param_name = None
    candlestick_ochl.stypy_kwargs_param_name = None
    candlestick_ochl.stypy_call_defaults = defaults
    candlestick_ochl.stypy_call_varargs = varargs
    candlestick_ochl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'candlestick_ochl', ['ax', 'quotes', 'width', 'colorup', 'colordown', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'candlestick_ochl', localization, ['ax', 'quotes', 'width', 'colorup', 'colordown', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'candlestick_ochl(...)' code ##################

    unicode_55433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, (-1)), 'unicode', u'\n    Plot the time, open, close, high, low as a vertical line ranging\n    from low to high.  Use a rectangular bar to represent the\n    open-close span.  If close >= open, use colorup to color the bar,\n    otherwise use colordown\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    quotes : sequence of (time, open, close, high, low, ...) sequences\n        As long as the first 5 elements are these values,\n        the record can be as long as you want (e.g., it may store volume).\n\n        time must be in float days format - see date2num\n\n    width : float\n        fraction of a day for the rectangle width\n    colorup : color\n        the color of the rectangle where close >= open\n    colordown : color\n         the color of the rectangle where close <  open\n    alpha : float\n        the rectangle alpha level\n\n    Returns\n    -------\n    ret : tuple\n        returns (lines, patches) where lines is a list of lines\n        added and patches is a list of the rectangle patches added\n\n    ')
    
    # Call to _candlestick(...): (line 695)
    # Processing the call arguments (line 695)
    # Getting the type of 'ax' (line 695)
    ax_55435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 24), 'ax', False)
    # Getting the type of 'quotes' (line 695)
    quotes_55436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 28), 'quotes', False)
    # Processing the call keyword arguments (line 695)
    # Getting the type of 'width' (line 695)
    width_55437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 42), 'width', False)
    keyword_55438 = width_55437
    # Getting the type of 'colorup' (line 695)
    colorup_55439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 57), 'colorup', False)
    keyword_55440 = colorup_55439
    # Getting the type of 'colordown' (line 696)
    colordown_55441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 34), 'colordown', False)
    keyword_55442 = colordown_55441
    # Getting the type of 'alpha' (line 697)
    alpha_55443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 30), 'alpha', False)
    keyword_55444 = alpha_55443
    # Getting the type of 'True' (line 697)
    True_55445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 42), 'True', False)
    keyword_55446 = True_55445
    kwargs_55447 = {'colordown': keyword_55442, 'width': keyword_55438, 'alpha': keyword_55444, 'ochl': keyword_55446, 'colorup': keyword_55440}
    # Getting the type of '_candlestick' (line 695)
    _candlestick_55434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 11), '_candlestick', False)
    # Calling _candlestick(args, kwargs) (line 695)
    _candlestick_call_result_55448 = invoke(stypy.reporting.localization.Localization(__file__, 695, 11), _candlestick_55434, *[ax_55435, quotes_55436], **kwargs_55447)
    
    # Assigning a type to the variable 'stypy_return_type' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'stypy_return_type', _candlestick_call_result_55448)
    
    # ################# End of 'candlestick_ochl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'candlestick_ochl' in the type store
    # Getting the type of 'stypy_return_type' (line 660)
    stypy_return_type_55449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55449)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'candlestick_ochl'
    return stypy_return_type_55449

# Assigning a type to the variable 'candlestick_ochl' (line 660)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 0), 'candlestick_ochl', candlestick_ochl)

@norecursion
def candlestick_ohlc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_55450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 39), 'float')
    unicode_55451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 52), 'unicode', u'k')
    unicode_55452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 67), 'unicode', u'r')
    float_55453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 22), 'float')
    defaults = [float_55450, unicode_55451, unicode_55452, float_55453]
    # Create a new context for function 'candlestick_ohlc'
    module_type_store = module_type_store.open_function_context('candlestick_ohlc', 700, 0, False)
    
    # Passed parameters checking function
    candlestick_ohlc.stypy_localization = localization
    candlestick_ohlc.stypy_type_of_self = None
    candlestick_ohlc.stypy_type_store = module_type_store
    candlestick_ohlc.stypy_function_name = 'candlestick_ohlc'
    candlestick_ohlc.stypy_param_names_list = ['ax', 'quotes', 'width', 'colorup', 'colordown', 'alpha']
    candlestick_ohlc.stypy_varargs_param_name = None
    candlestick_ohlc.stypy_kwargs_param_name = None
    candlestick_ohlc.stypy_call_defaults = defaults
    candlestick_ohlc.stypy_call_varargs = varargs
    candlestick_ohlc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'candlestick_ohlc', ['ax', 'quotes', 'width', 'colorup', 'colordown', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'candlestick_ohlc', localization, ['ax', 'quotes', 'width', 'colorup', 'colordown', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'candlestick_ohlc(...)' code ##################

    unicode_55454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, (-1)), 'unicode', u'\n    Plot the time, open, high, low, close as a vertical line ranging\n    from low to high.  Use a rectangular bar to represent the\n    open-close span.  If close >= open, use colorup to color the bar,\n    otherwise use colordown\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    quotes : sequence of (time, open, high, low, close, ...) sequences\n        As long as the first 5 elements are these values,\n        the record can be as long as you want (e.g., it may store volume).\n\n        time must be in float days format - see date2num\n\n    width : float\n        fraction of a day for the rectangle width\n    colorup : color\n        the color of the rectangle where close >= open\n    colordown : color\n         the color of the rectangle where close <  open\n    alpha : float\n        the rectangle alpha level\n\n    Returns\n    -------\n    ret : tuple\n        returns (lines, patches) where lines is a list of lines\n        added and patches is a list of the rectangle patches added\n\n    ')
    
    # Call to _candlestick(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'ax' (line 735)
    ax_55456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 24), 'ax', False)
    # Getting the type of 'quotes' (line 735)
    quotes_55457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 28), 'quotes', False)
    # Processing the call keyword arguments (line 735)
    # Getting the type of 'width' (line 735)
    width_55458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 42), 'width', False)
    keyword_55459 = width_55458
    # Getting the type of 'colorup' (line 735)
    colorup_55460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 57), 'colorup', False)
    keyword_55461 = colorup_55460
    # Getting the type of 'colordown' (line 736)
    colordown_55462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 34), 'colordown', False)
    keyword_55463 = colordown_55462
    # Getting the type of 'alpha' (line 737)
    alpha_55464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 30), 'alpha', False)
    keyword_55465 = alpha_55464
    # Getting the type of 'False' (line 737)
    False_55466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 42), 'False', False)
    keyword_55467 = False_55466
    kwargs_55468 = {'colordown': keyword_55463, 'width': keyword_55459, 'alpha': keyword_55465, 'ochl': keyword_55467, 'colorup': keyword_55461}
    # Getting the type of '_candlestick' (line 735)
    _candlestick_55455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 11), '_candlestick', False)
    # Calling _candlestick(args, kwargs) (line 735)
    _candlestick_call_result_55469 = invoke(stypy.reporting.localization.Localization(__file__, 735, 11), _candlestick_55455, *[ax_55456, quotes_55457], **kwargs_55468)
    
    # Assigning a type to the variable 'stypy_return_type' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'stypy_return_type', _candlestick_call_result_55469)
    
    # ################# End of 'candlestick_ohlc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'candlestick_ohlc' in the type store
    # Getting the type of 'stypy_return_type' (line 700)
    stypy_return_type_55470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55470)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'candlestick_ohlc'
    return stypy_return_type_55470

# Assigning a type to the variable 'candlestick_ohlc' (line 700)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 0), 'candlestick_ohlc', candlestick_ohlc)

@norecursion
def _candlestick(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_55471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 35), 'float')
    unicode_55472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 48), 'unicode', u'k')
    unicode_55473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 63), 'unicode', u'r')
    float_55474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 23), 'float')
    # Getting the type of 'True' (line 741)
    True_55475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 33), 'True')
    defaults = [float_55471, unicode_55472, unicode_55473, float_55474, True_55475]
    # Create a new context for function '_candlestick'
    module_type_store = module_type_store.open_function_context('_candlestick', 740, 0, False)
    
    # Passed parameters checking function
    _candlestick.stypy_localization = localization
    _candlestick.stypy_type_of_self = None
    _candlestick.stypy_type_store = module_type_store
    _candlestick.stypy_function_name = '_candlestick'
    _candlestick.stypy_param_names_list = ['ax', 'quotes', 'width', 'colorup', 'colordown', 'alpha', 'ochl']
    _candlestick.stypy_varargs_param_name = None
    _candlestick.stypy_kwargs_param_name = None
    _candlestick.stypy_call_defaults = defaults
    _candlestick.stypy_call_varargs = varargs
    _candlestick.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_candlestick', ['ax', 'quotes', 'width', 'colorup', 'colordown', 'alpha', 'ochl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_candlestick', localization, ['ax', 'quotes', 'width', 'colorup', 'colordown', 'alpha', 'ochl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_candlestick(...)' code ##################

    unicode_55476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, (-1)), 'unicode', u'\n    Plot the time, open, high, low, close as a vertical line ranging\n    from low to high.  Use a rectangular bar to represent the\n    open-close span.  If close >= open, use colorup to color the bar,\n    otherwise use colordown\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    quotes : sequence of quote sequences\n        data to plot.  time must be in float date format - see date2num\n        (time, open, high, low, close, ...) vs\n        (time, open, close, high, low, ...)\n        set by `ochl`\n    width : float\n        fraction of a day for the rectangle width\n    colorup : color\n        the color of the rectangle where close >= open\n    colordown : color\n         the color of the rectangle where close <  open\n    alpha : float\n        the rectangle alpha level\n    ochl: bool\n        argument to select between ochl and ohlc ordering of quotes\n\n    Returns\n    -------\n    ret : tuple\n        returns (lines, patches) where lines is a list of lines\n        added and patches is a list of the rectangle patches added\n\n    ')
    
    # Assigning a BinOp to a Name (line 777):
    
    # Assigning a BinOp to a Name (line 777):
    # Getting the type of 'width' (line 777)
    width_55477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 13), 'width')
    float_55478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 21), 'float')
    # Applying the binary operator 'div' (line 777)
    result_div_55479 = python_operator(stypy.reporting.localization.Localization(__file__, 777, 13), 'div', width_55477, float_55478)
    
    # Assigning a type to the variable 'OFFSET' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 4), 'OFFSET', result_div_55479)
    
    # Assigning a List to a Name (line 779):
    
    # Assigning a List to a Name (line 779):
    
    # Obtaining an instance of the builtin type 'list' (line 779)
    list_55480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 779)
    
    # Assigning a type to the variable 'lines' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'lines', list_55480)
    
    # Assigning a List to a Name (line 780):
    
    # Assigning a List to a Name (line 780):
    
    # Obtaining an instance of the builtin type 'list' (line 780)
    list_55481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 780)
    
    # Assigning a type to the variable 'patches' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 4), 'patches', list_55481)
    
    # Getting the type of 'quotes' (line 781)
    quotes_55482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 13), 'quotes')
    # Testing the type of a for loop iterable (line 781)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 781, 4), quotes_55482)
    # Getting the type of the for loop variable (line 781)
    for_loop_var_55483 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 781, 4), quotes_55482)
    # Assigning a type to the variable 'q' (line 781)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 4), 'q', for_loop_var_55483)
    # SSA begins for a for statement (line 781)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'ochl' (line 782)
    ochl_55484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 11), 'ochl')
    # Testing the type of an if condition (line 782)
    if_condition_55485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 782, 8), ochl_55484)
    # Assigning a type to the variable 'if_condition_55485' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'if_condition_55485', if_condition_55485)
    # SSA begins for if statement (line 782)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 783):
    
    # Assigning a Subscript to a Name (line 783):
    
    # Obtaining the type of the subscript
    int_55486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 43), 'int')
    slice_55488 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 783, 40), None, int_55487, None)
    # Getting the type of 'q' (line 783)
    q_55489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 40), q_55489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55491 = invoke(stypy.reporting.localization.Localization(__file__, 783, 40), getitem___55490, slice_55488)
    
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 12), subscript_call_result_55491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55493 = invoke(stypy.reporting.localization.Localization(__file__, 783, 12), getitem___55492, int_55486)
    
    # Assigning a type to the variable 'tuple_var_assignment_54308' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54308', subscript_call_result_55493)
    
    # Assigning a Subscript to a Name (line 783):
    
    # Obtaining the type of the subscript
    int_55494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 43), 'int')
    slice_55496 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 783, 40), None, int_55495, None)
    # Getting the type of 'q' (line 783)
    q_55497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 40), q_55497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55499 = invoke(stypy.reporting.localization.Localization(__file__, 783, 40), getitem___55498, slice_55496)
    
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 12), subscript_call_result_55499, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55501 = invoke(stypy.reporting.localization.Localization(__file__, 783, 12), getitem___55500, int_55494)
    
    # Assigning a type to the variable 'tuple_var_assignment_54309' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54309', subscript_call_result_55501)
    
    # Assigning a Subscript to a Name (line 783):
    
    # Obtaining the type of the subscript
    int_55502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 43), 'int')
    slice_55504 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 783, 40), None, int_55503, None)
    # Getting the type of 'q' (line 783)
    q_55505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 40), q_55505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55507 = invoke(stypy.reporting.localization.Localization(__file__, 783, 40), getitem___55506, slice_55504)
    
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 12), subscript_call_result_55507, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55509 = invoke(stypy.reporting.localization.Localization(__file__, 783, 12), getitem___55508, int_55502)
    
    # Assigning a type to the variable 'tuple_var_assignment_54310' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54310', subscript_call_result_55509)
    
    # Assigning a Subscript to a Name (line 783):
    
    # Obtaining the type of the subscript
    int_55510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 43), 'int')
    slice_55512 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 783, 40), None, int_55511, None)
    # Getting the type of 'q' (line 783)
    q_55513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 40), q_55513, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55515 = invoke(stypy.reporting.localization.Localization(__file__, 783, 40), getitem___55514, slice_55512)
    
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 12), subscript_call_result_55515, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55517 = invoke(stypy.reporting.localization.Localization(__file__, 783, 12), getitem___55516, int_55510)
    
    # Assigning a type to the variable 'tuple_var_assignment_54311' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54311', subscript_call_result_55517)
    
    # Assigning a Subscript to a Name (line 783):
    
    # Obtaining the type of the subscript
    int_55518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 43), 'int')
    slice_55520 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 783, 40), None, int_55519, None)
    # Getting the type of 'q' (line 783)
    q_55521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 40), q_55521, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55523 = invoke(stypy.reporting.localization.Localization(__file__, 783, 40), getitem___55522, slice_55520)
    
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___55524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 12), subscript_call_result_55523, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_55525 = invoke(stypy.reporting.localization.Localization(__file__, 783, 12), getitem___55524, int_55518)
    
    # Assigning a type to the variable 'tuple_var_assignment_54312' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54312', subscript_call_result_55525)
    
    # Assigning a Name to a Name (line 783):
    # Getting the type of 'tuple_var_assignment_54308' (line 783)
    tuple_var_assignment_54308_55526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54308')
    # Assigning a type to the variable 't' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 't', tuple_var_assignment_54308_55526)
    
    # Assigning a Name to a Name (line 783):
    # Getting the type of 'tuple_var_assignment_54309' (line 783)
    tuple_var_assignment_54309_55527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54309')
    # Assigning a type to the variable 'open' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 15), 'open', tuple_var_assignment_54309_55527)
    
    # Assigning a Name to a Name (line 783):
    # Getting the type of 'tuple_var_assignment_54310' (line 783)
    tuple_var_assignment_54310_55528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54310')
    # Assigning a type to the variable 'close' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 21), 'close', tuple_var_assignment_54310_55528)
    
    # Assigning a Name to a Name (line 783):
    # Getting the type of 'tuple_var_assignment_54311' (line 783)
    tuple_var_assignment_54311_55529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54311')
    # Assigning a type to the variable 'high' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 28), 'high', tuple_var_assignment_54311_55529)
    
    # Assigning a Name to a Name (line 783):
    # Getting the type of 'tuple_var_assignment_54312' (line 783)
    tuple_var_assignment_54312_55530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'tuple_var_assignment_54312')
    # Assigning a type to the variable 'low' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 34), 'low', tuple_var_assignment_54312_55530)
    # SSA branch for the else part of an if statement (line 782)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Tuple (line 785):
    
    # Assigning a Subscript to a Name (line 785):
    
    # Obtaining the type of the subscript
    int_55531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 43), 'int')
    slice_55533 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 785, 40), None, int_55532, None)
    # Getting the type of 'q' (line 785)
    q_55534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 40), q_55534, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55536 = invoke(stypy.reporting.localization.Localization(__file__, 785, 40), getitem___55535, slice_55533)
    
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 12), subscript_call_result_55536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55538 = invoke(stypy.reporting.localization.Localization(__file__, 785, 12), getitem___55537, int_55531)
    
    # Assigning a type to the variable 'tuple_var_assignment_54313' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54313', subscript_call_result_55538)
    
    # Assigning a Subscript to a Name (line 785):
    
    # Obtaining the type of the subscript
    int_55539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 43), 'int')
    slice_55541 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 785, 40), None, int_55540, None)
    # Getting the type of 'q' (line 785)
    q_55542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 40), q_55542, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55544 = invoke(stypy.reporting.localization.Localization(__file__, 785, 40), getitem___55543, slice_55541)
    
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 12), subscript_call_result_55544, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55546 = invoke(stypy.reporting.localization.Localization(__file__, 785, 12), getitem___55545, int_55539)
    
    # Assigning a type to the variable 'tuple_var_assignment_54314' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54314', subscript_call_result_55546)
    
    # Assigning a Subscript to a Name (line 785):
    
    # Obtaining the type of the subscript
    int_55547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 43), 'int')
    slice_55549 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 785, 40), None, int_55548, None)
    # Getting the type of 'q' (line 785)
    q_55550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 40), q_55550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55552 = invoke(stypy.reporting.localization.Localization(__file__, 785, 40), getitem___55551, slice_55549)
    
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 12), subscript_call_result_55552, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55554 = invoke(stypy.reporting.localization.Localization(__file__, 785, 12), getitem___55553, int_55547)
    
    # Assigning a type to the variable 'tuple_var_assignment_54315' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54315', subscript_call_result_55554)
    
    # Assigning a Subscript to a Name (line 785):
    
    # Obtaining the type of the subscript
    int_55555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 43), 'int')
    slice_55557 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 785, 40), None, int_55556, None)
    # Getting the type of 'q' (line 785)
    q_55558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 40), q_55558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55560 = invoke(stypy.reporting.localization.Localization(__file__, 785, 40), getitem___55559, slice_55557)
    
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 12), subscript_call_result_55560, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55562 = invoke(stypy.reporting.localization.Localization(__file__, 785, 12), getitem___55561, int_55555)
    
    # Assigning a type to the variable 'tuple_var_assignment_54316' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54316', subscript_call_result_55562)
    
    # Assigning a Subscript to a Name (line 785):
    
    # Obtaining the type of the subscript
    int_55563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 12), 'int')
    
    # Obtaining the type of the subscript
    int_55564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 43), 'int')
    slice_55565 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 785, 40), None, int_55564, None)
    # Getting the type of 'q' (line 785)
    q_55566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 40), 'q')
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 40), q_55566, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55568 = invoke(stypy.reporting.localization.Localization(__file__, 785, 40), getitem___55567, slice_55565)
    
    # Obtaining the member '__getitem__' of a type (line 785)
    getitem___55569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 12), subscript_call_result_55568, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 785)
    subscript_call_result_55570 = invoke(stypy.reporting.localization.Localization(__file__, 785, 12), getitem___55569, int_55563)
    
    # Assigning a type to the variable 'tuple_var_assignment_54317' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54317', subscript_call_result_55570)
    
    # Assigning a Name to a Name (line 785):
    # Getting the type of 'tuple_var_assignment_54313' (line 785)
    tuple_var_assignment_54313_55571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54313')
    # Assigning a type to the variable 't' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 't', tuple_var_assignment_54313_55571)
    
    # Assigning a Name to a Name (line 785):
    # Getting the type of 'tuple_var_assignment_54314' (line 785)
    tuple_var_assignment_54314_55572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54314')
    # Assigning a type to the variable 'open' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 15), 'open', tuple_var_assignment_54314_55572)
    
    # Assigning a Name to a Name (line 785):
    # Getting the type of 'tuple_var_assignment_54315' (line 785)
    tuple_var_assignment_54315_55573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54315')
    # Assigning a type to the variable 'high' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 21), 'high', tuple_var_assignment_54315_55573)
    
    # Assigning a Name to a Name (line 785):
    # Getting the type of 'tuple_var_assignment_54316' (line 785)
    tuple_var_assignment_54316_55574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54316')
    # Assigning a type to the variable 'low' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 27), 'low', tuple_var_assignment_54316_55574)
    
    # Assigning a Name to a Name (line 785):
    # Getting the type of 'tuple_var_assignment_54317' (line 785)
    tuple_var_assignment_54317_55575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 12), 'tuple_var_assignment_54317')
    # Assigning a type to the variable 'close' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 32), 'close', tuple_var_assignment_54317_55575)
    # SSA join for if statement (line 782)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'close' (line 787)
    close_55576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 11), 'close')
    # Getting the type of 'open' (line 787)
    open_55577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 20), 'open')
    # Applying the binary operator '>=' (line 787)
    result_ge_55578 = python_operator(stypy.reporting.localization.Localization(__file__, 787, 11), '>=', close_55576, open_55577)
    
    # Testing the type of an if condition (line 787)
    if_condition_55579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 787, 8), result_ge_55578)
    # Assigning a type to the variable 'if_condition_55579' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'if_condition_55579', if_condition_55579)
    # SSA begins for if statement (line 787)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 788):
    
    # Assigning a Name to a Name (line 788):
    # Getting the type of 'colorup' (line 788)
    colorup_55580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 20), 'colorup')
    # Assigning a type to the variable 'color' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 12), 'color', colorup_55580)
    
    # Assigning a Name to a Name (line 789):
    
    # Assigning a Name to a Name (line 789):
    # Getting the type of 'open' (line 789)
    open_55581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 20), 'open')
    # Assigning a type to the variable 'lower' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 12), 'lower', open_55581)
    
    # Assigning a BinOp to a Name (line 790):
    
    # Assigning a BinOp to a Name (line 790):
    # Getting the type of 'close' (line 790)
    close_55582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 21), 'close')
    # Getting the type of 'open' (line 790)
    open_55583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 29), 'open')
    # Applying the binary operator '-' (line 790)
    result_sub_55584 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 21), '-', close_55582, open_55583)
    
    # Assigning a type to the variable 'height' (line 790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 12), 'height', result_sub_55584)
    # SSA branch for the else part of an if statement (line 787)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 792):
    
    # Assigning a Name to a Name (line 792):
    # Getting the type of 'colordown' (line 792)
    colordown_55585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 20), 'colordown')
    # Assigning a type to the variable 'color' (line 792)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 12), 'color', colordown_55585)
    
    # Assigning a Name to a Name (line 793):
    
    # Assigning a Name to a Name (line 793):
    # Getting the type of 'close' (line 793)
    close_55586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 20), 'close')
    # Assigning a type to the variable 'lower' (line 793)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 12), 'lower', close_55586)
    
    # Assigning a BinOp to a Name (line 794):
    
    # Assigning a BinOp to a Name (line 794):
    # Getting the type of 'open' (line 794)
    open_55587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 21), 'open')
    # Getting the type of 'close' (line 794)
    close_55588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 28), 'close')
    # Applying the binary operator '-' (line 794)
    result_sub_55589 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 21), '-', open_55587, close_55588)
    
    # Assigning a type to the variable 'height' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), 'height', result_sub_55589)
    # SSA join for if statement (line 787)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 796):
    
    # Assigning a Call to a Name (line 796):
    
    # Call to Line2D(...): (line 796)
    # Processing the call keyword arguments (line 796)
    
    # Obtaining an instance of the builtin type 'tuple' (line 797)
    tuple_55591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 797)
    # Adding element type (line 797)
    # Getting the type of 't' (line 797)
    t_55592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 19), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 19), tuple_55591, t_55592)
    # Adding element type (line 797)
    # Getting the type of 't' (line 797)
    t_55593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 22), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 19), tuple_55591, t_55593)
    
    keyword_55594 = tuple_55591
    
    # Obtaining an instance of the builtin type 'tuple' (line 797)
    tuple_55595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 797)
    # Adding element type (line 797)
    # Getting the type of 'low' (line 797)
    low_55596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 33), 'low', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 33), tuple_55595, low_55596)
    # Adding element type (line 797)
    # Getting the type of 'high' (line 797)
    high_55597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 38), 'high', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 33), tuple_55595, high_55597)
    
    keyword_55598 = tuple_55595
    # Getting the type of 'color' (line 798)
    color_55599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 18), 'color', False)
    keyword_55600 = color_55599
    float_55601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 22), 'float')
    keyword_55602 = float_55601
    # Getting the type of 'True' (line 800)
    True_55603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 24), 'True', False)
    keyword_55604 = True_55603
    kwargs_55605 = {'color': keyword_55600, 'antialiased': keyword_55604, 'linewidth': keyword_55602, 'ydata': keyword_55598, 'xdata': keyword_55594}
    # Getting the type of 'Line2D' (line 796)
    Line2D_55590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 16), 'Line2D', False)
    # Calling Line2D(args, kwargs) (line 796)
    Line2D_call_result_55606 = invoke(stypy.reporting.localization.Localization(__file__, 796, 16), Line2D_55590, *[], **kwargs_55605)
    
    # Assigning a type to the variable 'vline' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 8), 'vline', Line2D_call_result_55606)
    
    # Assigning a Call to a Name (line 803):
    
    # Assigning a Call to a Name (line 803):
    
    # Call to Rectangle(...): (line 803)
    # Processing the call keyword arguments (line 803)
    
    # Obtaining an instance of the builtin type 'tuple' (line 804)
    tuple_55608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 804)
    # Adding element type (line 804)
    # Getting the type of 't' (line 804)
    t_55609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 16), 't', False)
    # Getting the type of 'OFFSET' (line 804)
    OFFSET_55610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 20), 'OFFSET', False)
    # Applying the binary operator '-' (line 804)
    result_sub_55611 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 16), '-', t_55609, OFFSET_55610)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 16), tuple_55608, result_sub_55611)
    # Adding element type (line 804)
    # Getting the type of 'lower' (line 804)
    lower_55612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 28), 'lower', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 16), tuple_55608, lower_55612)
    
    keyword_55613 = tuple_55608
    # Getting the type of 'width' (line 805)
    width_55614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 18), 'width', False)
    keyword_55615 = width_55614
    # Getting the type of 'height' (line 806)
    height_55616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 19), 'height', False)
    keyword_55617 = height_55616
    # Getting the type of 'color' (line 807)
    color_55618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 22), 'color', False)
    keyword_55619 = color_55618
    # Getting the type of 'color' (line 808)
    color_55620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 22), 'color', False)
    keyword_55621 = color_55620
    kwargs_55622 = {'edgecolor': keyword_55621, 'width': keyword_55615, 'xy': keyword_55613, 'facecolor': keyword_55619, 'height': keyword_55617}
    # Getting the type of 'Rectangle' (line 803)
    Rectangle_55607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 15), 'Rectangle', False)
    # Calling Rectangle(args, kwargs) (line 803)
    Rectangle_call_result_55623 = invoke(stypy.reporting.localization.Localization(__file__, 803, 15), Rectangle_55607, *[], **kwargs_55622)
    
    # Assigning a type to the variable 'rect' (line 803)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'rect', Rectangle_call_result_55623)
    
    # Call to set_alpha(...): (line 810)
    # Processing the call arguments (line 810)
    # Getting the type of 'alpha' (line 810)
    alpha_55626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 23), 'alpha', False)
    # Processing the call keyword arguments (line 810)
    kwargs_55627 = {}
    # Getting the type of 'rect' (line 810)
    rect_55624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'rect', False)
    # Obtaining the member 'set_alpha' of a type (line 810)
    set_alpha_55625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 8), rect_55624, 'set_alpha')
    # Calling set_alpha(args, kwargs) (line 810)
    set_alpha_call_result_55628 = invoke(stypy.reporting.localization.Localization(__file__, 810, 8), set_alpha_55625, *[alpha_55626], **kwargs_55627)
    
    
    # Call to append(...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'vline' (line 812)
    vline_55631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 21), 'vline', False)
    # Processing the call keyword arguments (line 812)
    kwargs_55632 = {}
    # Getting the type of 'lines' (line 812)
    lines_55629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'lines', False)
    # Obtaining the member 'append' of a type (line 812)
    append_55630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 8), lines_55629, 'append')
    # Calling append(args, kwargs) (line 812)
    append_call_result_55633 = invoke(stypy.reporting.localization.Localization(__file__, 812, 8), append_55630, *[vline_55631], **kwargs_55632)
    
    
    # Call to append(...): (line 813)
    # Processing the call arguments (line 813)
    # Getting the type of 'rect' (line 813)
    rect_55636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 23), 'rect', False)
    # Processing the call keyword arguments (line 813)
    kwargs_55637 = {}
    # Getting the type of 'patches' (line 813)
    patches_55634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 8), 'patches', False)
    # Obtaining the member 'append' of a type (line 813)
    append_55635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 8), patches_55634, 'append')
    # Calling append(args, kwargs) (line 813)
    append_call_result_55638 = invoke(stypy.reporting.localization.Localization(__file__, 813, 8), append_55635, *[rect_55636], **kwargs_55637)
    
    
    # Call to add_line(...): (line 814)
    # Processing the call arguments (line 814)
    # Getting the type of 'vline' (line 814)
    vline_55641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 20), 'vline', False)
    # Processing the call keyword arguments (line 814)
    kwargs_55642 = {}
    # Getting the type of 'ax' (line 814)
    ax_55639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 8), 'ax', False)
    # Obtaining the member 'add_line' of a type (line 814)
    add_line_55640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 8), ax_55639, 'add_line')
    # Calling add_line(args, kwargs) (line 814)
    add_line_call_result_55643 = invoke(stypy.reporting.localization.Localization(__file__, 814, 8), add_line_55640, *[vline_55641], **kwargs_55642)
    
    
    # Call to add_patch(...): (line 815)
    # Processing the call arguments (line 815)
    # Getting the type of 'rect' (line 815)
    rect_55646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 21), 'rect', False)
    # Processing the call keyword arguments (line 815)
    kwargs_55647 = {}
    # Getting the type of 'ax' (line 815)
    ax_55644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'ax', False)
    # Obtaining the member 'add_patch' of a type (line 815)
    add_patch_55645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 8), ax_55644, 'add_patch')
    # Calling add_patch(args, kwargs) (line 815)
    add_patch_call_result_55648 = invoke(stypy.reporting.localization.Localization(__file__, 815, 8), add_patch_55645, *[rect_55646], **kwargs_55647)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to autoscale_view(...): (line 816)
    # Processing the call keyword arguments (line 816)
    kwargs_55651 = {}
    # Getting the type of 'ax' (line 816)
    ax_55649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 4), 'ax', False)
    # Obtaining the member 'autoscale_view' of a type (line 816)
    autoscale_view_55650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 4), ax_55649, 'autoscale_view')
    # Calling autoscale_view(args, kwargs) (line 816)
    autoscale_view_call_result_55652 = invoke(stypy.reporting.localization.Localization(__file__, 816, 4), autoscale_view_55650, *[], **kwargs_55651)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 818)
    tuple_55653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 818)
    # Adding element type (line 818)
    # Getting the type of 'lines' (line 818)
    lines_55654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 11), 'lines')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 11), tuple_55653, lines_55654)
    # Adding element type (line 818)
    # Getting the type of 'patches' (line 818)
    patches_55655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 18), 'patches')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 11), tuple_55653, patches_55655)
    
    # Assigning a type to the variable 'stypy_return_type' (line 818)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 4), 'stypy_return_type', tuple_55653)
    
    # ################# End of '_candlestick(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_candlestick' in the type store
    # Getting the type of 'stypy_return_type' (line 740)
    stypy_return_type_55656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55656)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_candlestick'
    return stypy_return_type_55656

# Assigning a type to the variable '_candlestick' (line 740)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 0), '_candlestick', _candlestick)

@norecursion
def _check_input(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_55657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 50), 'int')
    defaults = [int_55657]
    # Create a new context for function '_check_input'
    module_type_store = module_type_store.open_function_context('_check_input', 821, 0, False)
    
    # Passed parameters checking function
    _check_input.stypy_localization = localization
    _check_input.stypy_type_of_self = None
    _check_input.stypy_type_store = module_type_store
    _check_input.stypy_function_name = '_check_input'
    _check_input.stypy_param_names_list = ['opens', 'closes', 'highs', 'lows', 'miss']
    _check_input.stypy_varargs_param_name = None
    _check_input.stypy_kwargs_param_name = None
    _check_input.stypy_call_defaults = defaults
    _check_input.stypy_call_varargs = varargs
    _check_input.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_input', ['opens', 'closes', 'highs', 'lows', 'miss'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_input', localization, ['opens', 'closes', 'highs', 'lows', 'miss'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_input(...)' code ##################

    unicode_55658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, (-1)), 'unicode', u"Checks that *opens*, *highs*, *lows* and *closes* have the same length.\n    NOTE: this code assumes if any value open, high, low, close is\n    missing (*-1*) they all are missing\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    opens : sequence\n        sequence of opening values\n    highs : sequence\n        sequence of high values\n    lows : sequence\n        sequence of low values\n    closes : sequence\n        sequence of closing values\n    miss : int\n        identifier of the missing data\n\n    Raises\n    ------\n    ValueError\n        if the input sequences don't have the same length\n    ")

    @norecursion
    def _missing(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_55659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 32), 'int')
        defaults = [int_55659]
        # Create a new context for function '_missing'
        module_type_store = module_type_store.open_function_context('_missing', 847, 4, False)
        
        # Passed parameters checking function
        _missing.stypy_localization = localization
        _missing.stypy_type_of_self = None
        _missing.stypy_type_store = module_type_store
        _missing.stypy_function_name = '_missing'
        _missing.stypy_param_names_list = ['sequence', 'miss']
        _missing.stypy_varargs_param_name = None
        _missing.stypy_kwargs_param_name = None
        _missing.stypy_call_defaults = defaults
        _missing.stypy_call_varargs = varargs
        _missing.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_missing', ['sequence', 'miss'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_missing', localization, ['sequence', 'miss'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_missing(...)' code ##################

        unicode_55660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, (-1)), 'unicode', u'Returns the index in *sequence* of the missing data, identified by\n        *miss*\n\n        Parameters\n        ----------\n        sequence :\n            sequence to evaluate\n        miss :\n            identifier of the missing data\n\n        Returns\n        -------\n        where_miss: numpy.ndarray\n            indices of the missing data\n        ')
        
        # Obtaining the type of the subscript
        int_55661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 52), 'int')
        
        # Call to where(...): (line 863)
        # Processing the call arguments (line 863)
        
        
        # Call to array(...): (line 863)
        # Processing the call arguments (line 863)
        # Getting the type of 'sequence' (line 863)
        sequence_55666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 33), 'sequence', False)
        # Processing the call keyword arguments (line 863)
        kwargs_55667 = {}
        # Getting the type of 'np' (line 863)
        np_55664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 24), 'np', False)
        # Obtaining the member 'array' of a type (line 863)
        array_55665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 24), np_55664, 'array')
        # Calling array(args, kwargs) (line 863)
        array_call_result_55668 = invoke(stypy.reporting.localization.Localization(__file__, 863, 24), array_55665, *[sequence_55666], **kwargs_55667)
        
        # Getting the type of 'miss' (line 863)
        miss_55669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 46), 'miss', False)
        # Applying the binary operator '==' (line 863)
        result_eq_55670 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 24), '==', array_call_result_55668, miss_55669)
        
        # Processing the call keyword arguments (line 863)
        kwargs_55671 = {}
        # Getting the type of 'np' (line 863)
        np_55662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 863)
        where_55663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 15), np_55662, 'where')
        # Calling where(args, kwargs) (line 863)
        where_call_result_55672 = invoke(stypy.reporting.localization.Localization(__file__, 863, 15), where_55663, *[result_eq_55670], **kwargs_55671)
        
        # Obtaining the member '__getitem__' of a type (line 863)
        getitem___55673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 15), where_call_result_55672, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 863)
        subscript_call_result_55674 = invoke(stypy.reporting.localization.Localization(__file__, 863, 15), getitem___55673, int_55661)
        
        # Assigning a type to the variable 'stypy_return_type' (line 863)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 8), 'stypy_return_type', subscript_call_result_55674)
        
        # ################# End of '_missing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_missing' in the type store
        # Getting the type of 'stypy_return_type' (line 847)
        stypy_return_type_55675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55675)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_missing'
        return stypy_return_type_55675

    # Assigning a type to the variable '_missing' (line 847)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 4), '_missing', _missing)
    
    # Assigning a Compare to a Name (line 865):
    
    # Assigning a Compare to a Name (line 865):
    
    
    # Call to len(...): (line 865)
    # Processing the call arguments (line 865)
    # Getting the type of 'opens' (line 865)
    opens_55677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 22), 'opens', False)
    # Processing the call keyword arguments (line 865)
    kwargs_55678 = {}
    # Getting the type of 'len' (line 865)
    len_55676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 18), 'len', False)
    # Calling len(args, kwargs) (line 865)
    len_call_result_55679 = invoke(stypy.reporting.localization.Localization(__file__, 865, 18), len_55676, *[opens_55677], **kwargs_55678)
    
    
    # Call to len(...): (line 865)
    # Processing the call arguments (line 865)
    # Getting the type of 'highs' (line 865)
    highs_55681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 36), 'highs', False)
    # Processing the call keyword arguments (line 865)
    kwargs_55682 = {}
    # Getting the type of 'len' (line 865)
    len_55680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 32), 'len', False)
    # Calling len(args, kwargs) (line 865)
    len_call_result_55683 = invoke(stypy.reporting.localization.Localization(__file__, 865, 32), len_55680, *[highs_55681], **kwargs_55682)
    
    # Applying the binary operator '==' (line 865)
    result_eq_55684 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 18), '==', len_call_result_55679, len_call_result_55683)
    
    # Call to len(...): (line 865)
    # Processing the call arguments (line 865)
    # Getting the type of 'lows' (line 865)
    lows_55686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 50), 'lows', False)
    # Processing the call keyword arguments (line 865)
    kwargs_55687 = {}
    # Getting the type of 'len' (line 865)
    len_55685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 46), 'len', False)
    # Calling len(args, kwargs) (line 865)
    len_call_result_55688 = invoke(stypy.reporting.localization.Localization(__file__, 865, 46), len_55685, *[lows_55686], **kwargs_55687)
    
    # Applying the binary operator '==' (line 865)
    result_eq_55689 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 18), '==', len_call_result_55683, len_call_result_55688)
    # Applying the binary operator '&' (line 865)
    result_and__55690 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 18), '&', result_eq_55684, result_eq_55689)
    
    # Call to len(...): (line 865)
    # Processing the call arguments (line 865)
    # Getting the type of 'closes' (line 865)
    closes_55692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 63), 'closes', False)
    # Processing the call keyword arguments (line 865)
    kwargs_55693 = {}
    # Getting the type of 'len' (line 865)
    len_55691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 59), 'len', False)
    # Calling len(args, kwargs) (line 865)
    len_call_result_55694 = invoke(stypy.reporting.localization.Localization(__file__, 865, 59), len_55691, *[closes_55692], **kwargs_55693)
    
    # Applying the binary operator '==' (line 865)
    result_eq_55695 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 18), '==', len_call_result_55688, len_call_result_55694)
    # Applying the binary operator '&' (line 865)
    result_and__55696 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 18), '&', result_and__55690, result_eq_55695)
    
    # Assigning a type to the variable 'same_length' (line 865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 4), 'same_length', result_and__55696)
    
    # Assigning a Call to a Name (line 866):
    
    # Assigning a Call to a Name (line 866):
    
    # Call to _missing(...): (line 866)
    # Processing the call arguments (line 866)
    # Getting the type of 'opens' (line 866)
    opens_55698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 26), 'opens', False)
    # Processing the call keyword arguments (line 866)
    kwargs_55699 = {}
    # Getting the type of '_missing' (line 866)
    _missing_55697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 17), '_missing', False)
    # Calling _missing(args, kwargs) (line 866)
    _missing_call_result_55700 = invoke(stypy.reporting.localization.Localization(__file__, 866, 17), _missing_55697, *[opens_55698], **kwargs_55699)
    
    # Assigning a type to the variable '_missopens' (line 866)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 4), '_missopens', _missing_call_result_55700)
    
    # Assigning a BoolOp to a Name (line 867):
    
    # Assigning a BoolOp to a Name (line 867):
    
    # Evaluating a boolean operation
    
    # Call to all(...): (line 867)
    # Processing the call keyword arguments (line 867)
    kwargs_55708 = {}
    
    # Getting the type of '_missopens' (line 867)
    _missopens_55701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 21), '_missopens', False)
    
    # Call to _missing(...): (line 867)
    # Processing the call arguments (line 867)
    # Getting the type of 'highs' (line 867)
    highs_55703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 44), 'highs', False)
    # Processing the call keyword arguments (line 867)
    kwargs_55704 = {}
    # Getting the type of '_missing' (line 867)
    _missing_55702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 35), '_missing', False)
    # Calling _missing(args, kwargs) (line 867)
    _missing_call_result_55705 = invoke(stypy.reporting.localization.Localization(__file__, 867, 35), _missing_55702, *[highs_55703], **kwargs_55704)
    
    # Applying the binary operator '==' (line 867)
    result_eq_55706 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 21), '==', _missopens_55701, _missing_call_result_55705)
    
    # Obtaining the member 'all' of a type (line 867)
    all_55707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 21), result_eq_55706, 'all')
    # Calling all(args, kwargs) (line 867)
    all_call_result_55709 = invoke(stypy.reporting.localization.Localization(__file__, 867, 21), all_55707, *[], **kwargs_55708)
    
    
    # Call to all(...): (line 868)
    # Processing the call keyword arguments (line 868)
    kwargs_55717 = {}
    
    # Getting the type of '_missopens' (line 868)
    _missopens_55710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 21), '_missopens', False)
    
    # Call to _missing(...): (line 868)
    # Processing the call arguments (line 868)
    # Getting the type of 'lows' (line 868)
    lows_55712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 44), 'lows', False)
    # Processing the call keyword arguments (line 868)
    kwargs_55713 = {}
    # Getting the type of '_missing' (line 868)
    _missing_55711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 35), '_missing', False)
    # Calling _missing(args, kwargs) (line 868)
    _missing_call_result_55714 = invoke(stypy.reporting.localization.Localization(__file__, 868, 35), _missing_55711, *[lows_55712], **kwargs_55713)
    
    # Applying the binary operator '==' (line 868)
    result_eq_55715 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 21), '==', _missopens_55710, _missing_call_result_55714)
    
    # Obtaining the member 'all' of a type (line 868)
    all_55716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 21), result_eq_55715, 'all')
    # Calling all(args, kwargs) (line 868)
    all_call_result_55718 = invoke(stypy.reporting.localization.Localization(__file__, 868, 21), all_55716, *[], **kwargs_55717)
    
    # Applying the binary operator 'and' (line 867)
    result_and_keyword_55719 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 20), 'and', all_call_result_55709, all_call_result_55718)
    
    # Call to all(...): (line 869)
    # Processing the call keyword arguments (line 869)
    kwargs_55727 = {}
    
    # Getting the type of '_missopens' (line 869)
    _missopens_55720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 21), '_missopens', False)
    
    # Call to _missing(...): (line 869)
    # Processing the call arguments (line 869)
    # Getting the type of 'closes' (line 869)
    closes_55722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 44), 'closes', False)
    # Processing the call keyword arguments (line 869)
    kwargs_55723 = {}
    # Getting the type of '_missing' (line 869)
    _missing_55721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 35), '_missing', False)
    # Calling _missing(args, kwargs) (line 869)
    _missing_call_result_55724 = invoke(stypy.reporting.localization.Localization(__file__, 869, 35), _missing_55721, *[closes_55722], **kwargs_55723)
    
    # Applying the binary operator '==' (line 869)
    result_eq_55725 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 21), '==', _missopens_55720, _missing_call_result_55724)
    
    # Obtaining the member 'all' of a type (line 869)
    all_55726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 21), result_eq_55725, 'all')
    # Calling all(args, kwargs) (line 869)
    all_call_result_55728 = invoke(stypy.reporting.localization.Localization(__file__, 869, 21), all_55726, *[], **kwargs_55727)
    
    # Applying the binary operator 'and' (line 867)
    result_and_keyword_55729 = python_operator(stypy.reporting.localization.Localization(__file__, 867, 20), 'and', result_and_keyword_55719, all_call_result_55728)
    
    # Assigning a type to the variable 'same_missing' (line 867)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 4), 'same_missing', result_and_keyword_55729)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'same_length' (line 871)
    same_length_55730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 12), 'same_length')
    # Getting the type of 'same_missing' (line 871)
    same_missing_55731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 28), 'same_missing')
    # Applying the binary operator 'and' (line 871)
    result_and_keyword_55732 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 12), 'and', same_length_55730, same_missing_55731)
    
    # Applying the 'not' unary operator (line 871)
    result_not__55733 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 7), 'not', result_and_keyword_55732)
    
    # Testing the type of an if condition (line 871)
    if_condition_55734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 871, 4), result_not__55733)
    # Assigning a type to the variable 'if_condition_55734' (line 871)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 4), 'if_condition_55734', if_condition_55734)
    # SSA begins for if statement (line 871)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 872):
    
    # Assigning a Str to a Name (line 872):
    unicode_55735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 15), 'unicode', u'*opens*, *highs*, *lows* and *closes* must have the same length. NOTE: this code assumes if any value open, high, low, close is missing (*-1*) they all must be missing.')
    # Assigning a type to the variable 'msg' (line 872)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 8), 'msg', unicode_55735)
    
    # Call to ValueError(...): (line 875)
    # Processing the call arguments (line 875)
    # Getting the type of 'msg' (line 875)
    msg_55737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 25), 'msg', False)
    # Processing the call keyword arguments (line 875)
    kwargs_55738 = {}
    # Getting the type of 'ValueError' (line 875)
    ValueError_55736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 875)
    ValueError_call_result_55739 = invoke(stypy.reporting.localization.Localization(__file__, 875, 14), ValueError_55736, *[msg_55737], **kwargs_55738)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 875, 8), ValueError_call_result_55739, 'raise parameter', BaseException)
    # SSA join for if statement (line 871)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_input(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_input' in the type store
    # Getting the type of 'stypy_return_type' (line 821)
    stypy_return_type_55740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55740)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_input'
    return stypy_return_type_55740

# Assigning a type to the variable '_check_input' (line 821)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 0), '_check_input', _check_input)

@norecursion
def plot_day_summary2_ochl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_55741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, 68), 'int')
    unicode_55742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 34), 'unicode', u'k')
    unicode_55743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 49), 'unicode', u'r')
    defaults = [int_55741, unicode_55742, unicode_55743]
    # Create a new context for function 'plot_day_summary2_ochl'
    module_type_store = module_type_store.open_function_context('plot_day_summary2_ochl', 878, 0, False)
    
    # Passed parameters checking function
    plot_day_summary2_ochl.stypy_localization = localization
    plot_day_summary2_ochl.stypy_type_of_self = None
    plot_day_summary2_ochl.stypy_type_store = module_type_store
    plot_day_summary2_ochl.stypy_function_name = 'plot_day_summary2_ochl'
    plot_day_summary2_ochl.stypy_param_names_list = ['ax', 'opens', 'closes', 'highs', 'lows', 'ticksize', 'colorup', 'colordown']
    plot_day_summary2_ochl.stypy_varargs_param_name = None
    plot_day_summary2_ochl.stypy_kwargs_param_name = None
    plot_day_summary2_ochl.stypy_call_defaults = defaults
    plot_day_summary2_ochl.stypy_call_varargs = varargs
    plot_day_summary2_ochl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'plot_day_summary2_ochl', ['ax', 'opens', 'closes', 'highs', 'lows', 'ticksize', 'colorup', 'colordown'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'plot_day_summary2_ochl', localization, ['ax', 'opens', 'closes', 'highs', 'lows', 'ticksize', 'colorup', 'colordown'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'plot_day_summary2_ochl(...)' code ##################

    unicode_55744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, (-1)), 'unicode', u'Represent the time, open, close, high, low,  as a vertical line\n    ranging from low to high.  The left tick is the open and the right\n    tick is the close.\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    opens : sequence\n        sequence of opening values\n    closes : sequence\n        sequence of closing values\n    highs : sequence\n        sequence of high values\n    lows : sequence\n        sequence of low values\n    ticksize : int\n        size of open and close ticks in points\n    colorup : color\n        the color of the lines where close >= open\n    colordown : color\n         the color of the lines where close <  open\n\n    Returns\n    -------\n    ret : list\n        a list of lines added to the axes\n    ')
    
    # Call to plot_day_summary2_ohlc(...): (line 911)
    # Processing the call arguments (line 911)
    # Getting the type of 'ax' (line 911)
    ax_55746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 34), 'ax', False)
    # Getting the type of 'opens' (line 911)
    opens_55747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 38), 'opens', False)
    # Getting the type of 'highs' (line 911)
    highs_55748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 45), 'highs', False)
    # Getting the type of 'lows' (line 911)
    lows_55749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 52), 'lows', False)
    # Getting the type of 'closes' (line 911)
    closes_55750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 58), 'closes', False)
    # Getting the type of 'ticksize' (line 911)
    ticksize_55751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 66), 'ticksize', False)
    # Getting the type of 'colorup' (line 912)
    colorup_55752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 33), 'colorup', False)
    # Getting the type of 'colordown' (line 912)
    colordown_55753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 42), 'colordown', False)
    # Processing the call keyword arguments (line 911)
    kwargs_55754 = {}
    # Getting the type of 'plot_day_summary2_ohlc' (line 911)
    plot_day_summary2_ohlc_55745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 11), 'plot_day_summary2_ohlc', False)
    # Calling plot_day_summary2_ohlc(args, kwargs) (line 911)
    plot_day_summary2_ohlc_call_result_55755 = invoke(stypy.reporting.localization.Localization(__file__, 911, 11), plot_day_summary2_ohlc_55745, *[ax_55746, opens_55747, highs_55748, lows_55749, closes_55750, ticksize_55751, colorup_55752, colordown_55753], **kwargs_55754)
    
    # Assigning a type to the variable 'stypy_return_type' (line 911)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 4), 'stypy_return_type', plot_day_summary2_ohlc_call_result_55755)
    
    # ################# End of 'plot_day_summary2_ochl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'plot_day_summary2_ochl' in the type store
    # Getting the type of 'stypy_return_type' (line 878)
    stypy_return_type_55756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55756)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'plot_day_summary2_ochl'
    return stypy_return_type_55756

# Assigning a type to the variable 'plot_day_summary2_ochl' (line 878)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 0), 'plot_day_summary2_ochl', plot_day_summary2_ochl)

@norecursion
def plot_day_summary2_ohlc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_55757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 68), 'int')
    unicode_55758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 34), 'unicode', u'k')
    unicode_55759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 49), 'unicode', u'r')
    defaults = [int_55757, unicode_55758, unicode_55759]
    # Create a new context for function 'plot_day_summary2_ohlc'
    module_type_store = module_type_store.open_function_context('plot_day_summary2_ohlc', 915, 0, False)
    
    # Passed parameters checking function
    plot_day_summary2_ohlc.stypy_localization = localization
    plot_day_summary2_ohlc.stypy_type_of_self = None
    plot_day_summary2_ohlc.stypy_type_store = module_type_store
    plot_day_summary2_ohlc.stypy_function_name = 'plot_day_summary2_ohlc'
    plot_day_summary2_ohlc.stypy_param_names_list = ['ax', 'opens', 'highs', 'lows', 'closes', 'ticksize', 'colorup', 'colordown']
    plot_day_summary2_ohlc.stypy_varargs_param_name = None
    plot_day_summary2_ohlc.stypy_kwargs_param_name = None
    plot_day_summary2_ohlc.stypy_call_defaults = defaults
    plot_day_summary2_ohlc.stypy_call_varargs = varargs
    plot_day_summary2_ohlc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'plot_day_summary2_ohlc', ['ax', 'opens', 'highs', 'lows', 'closes', 'ticksize', 'colorup', 'colordown'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'plot_day_summary2_ohlc', localization, ['ax', 'opens', 'highs', 'lows', 'closes', 'ticksize', 'colorup', 'colordown'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'plot_day_summary2_ohlc(...)' code ##################

    unicode_55760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, (-1)), 'unicode', u'Represent the time, open, high, low, close as a vertical line\n    ranging from low to high.  The left tick is the open and the right\n    tick is the close.\n    *opens*, *highs*, *lows* and *closes* must have the same length.\n    NOTE: this code assumes if any value open, high, low, close is\n    missing (*-1*) they all are missing\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    opens : sequence\n        sequence of opening values\n    highs : sequence\n        sequence of high values\n    lows : sequence\n        sequence of low values\n    closes : sequence\n        sequence of closing values\n    ticksize : int\n        size of open and close ticks in points\n    colorup : color\n        the color of the lines where close >= open\n    colordown : color\n         the color of the lines where close <  open\n\n    Returns\n    -------\n    ret : list\n        a list of lines added to the axes\n    ')
    
    # Call to _check_input(...): (line 951)
    # Processing the call arguments (line 951)
    # Getting the type of 'opens' (line 951)
    opens_55762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 17), 'opens', False)
    # Getting the type of 'highs' (line 951)
    highs_55763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 24), 'highs', False)
    # Getting the type of 'lows' (line 951)
    lows_55764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 31), 'lows', False)
    # Getting the type of 'closes' (line 951)
    closes_55765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 37), 'closes', False)
    # Processing the call keyword arguments (line 951)
    kwargs_55766 = {}
    # Getting the type of '_check_input' (line 951)
    _check_input_55761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 4), '_check_input', False)
    # Calling _check_input(args, kwargs) (line 951)
    _check_input_call_result_55767 = invoke(stypy.reporting.localization.Localization(__file__, 951, 4), _check_input_55761, *[opens_55762, highs_55763, lows_55764, closes_55765], **kwargs_55766)
    
    
    # Assigning a ListComp to a Name (line 953):
    
    # Assigning a ListComp to a Name (line 953):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 954)
    # Processing the call arguments (line 954)
    
    # Call to xrange(...): (line 954)
    # Processing the call arguments (line 954)
    
    # Call to len(...): (line 954)
    # Processing the call arguments (line 954)
    # Getting the type of 'lows' (line 954)
    lows_55781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 36), 'lows', False)
    # Processing the call keyword arguments (line 954)
    kwargs_55782 = {}
    # Getting the type of 'len' (line 954)
    len_55780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 32), 'len', False)
    # Calling len(args, kwargs) (line 954)
    len_call_result_55783 = invoke(stypy.reporting.localization.Localization(__file__, 954, 32), len_55780, *[lows_55781], **kwargs_55782)
    
    # Processing the call keyword arguments (line 954)
    kwargs_55784 = {}
    # Getting the type of 'xrange' (line 954)
    xrange_55779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 25), 'xrange', False)
    # Calling xrange(args, kwargs) (line 954)
    xrange_call_result_55785 = invoke(stypy.reporting.localization.Localization(__file__, 954, 25), xrange_55779, *[len_call_result_55783], **kwargs_55784)
    
    # Getting the type of 'lows' (line 954)
    lows_55786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 44), 'lows', False)
    # Getting the type of 'highs' (line 954)
    highs_55787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 50), 'highs', False)
    # Processing the call keyword arguments (line 954)
    kwargs_55788 = {}
    # Getting the type of 'zip' (line 954)
    zip_55778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 21), 'zip', False)
    # Calling zip(args, kwargs) (line 954)
    zip_call_result_55789 = invoke(stypy.reporting.localization.Localization(__file__, 954, 21), zip_55778, *[xrange_call_result_55785, lows_55786, highs_55787], **kwargs_55788)
    
    comprehension_55790 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 21), zip_call_result_55789)
    # Assigning a type to the variable 'i' (line 953)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 21), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 21), comprehension_55790))
    # Assigning a type to the variable 'low' (line 953)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 21), 'low', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 21), comprehension_55790))
    # Assigning a type to the variable 'high' (line 953)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 21), 'high', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 21), comprehension_55790))
    
    # Getting the type of 'low' (line 954)
    low_55775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 60), 'low')
    int_55776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 67), 'int')
    # Applying the binary operator '!=' (line 954)
    result_ne_55777 = python_operator(stypy.reporting.localization.Localization(__file__, 954, 60), '!=', low_55775, int_55776)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 953)
    tuple_55768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 953)
    # Adding element type (line 953)
    
    # Obtaining an instance of the builtin type 'tuple' (line 953)
    tuple_55769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 953)
    # Adding element type (line 953)
    # Getting the type of 'i' (line 953)
    i_55770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 23), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 23), tuple_55769, i_55770)
    # Adding element type (line 953)
    # Getting the type of 'low' (line 953)
    low_55771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 26), 'low')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 23), tuple_55769, low_55771)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 22), tuple_55768, tuple_55769)
    # Adding element type (line 953)
    
    # Obtaining an instance of the builtin type 'tuple' (line 953)
    tuple_55772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 953)
    # Adding element type (line 953)
    # Getting the type of 'i' (line 953)
    i_55773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 33), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 33), tuple_55772, i_55773)
    # Adding element type (line 953)
    # Getting the type of 'high' (line 953)
    high_55774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 36), 'high')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 33), tuple_55772, high_55774)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 22), tuple_55768, tuple_55772)
    
    list_55791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 21), list_55791, tuple_55768)
    # Assigning a type to the variable 'rangeSegments' (line 953)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 4), 'rangeSegments', list_55791)
    
    # Assigning a List to a Name (line 958):
    
    # Assigning a List to a Name (line 958):
    
    # Obtaining an instance of the builtin type 'list' (line 958)
    list_55792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 958)
    # Adding element type (line 958)
    
    # Obtaining an instance of the builtin type 'tuple' (line 958)
    tuple_55793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 958)
    # Adding element type (line 958)
    
    # Obtaining an instance of the builtin type 'tuple' (line 958)
    tuple_55794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 958)
    # Adding element type (line 958)
    
    # Getting the type of 'ticksize' (line 958)
    ticksize_55795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 23), 'ticksize')
    # Applying the 'usub' unary operator (line 958)
    result___neg___55796 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 22), 'usub', ticksize_55795)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 958, 22), tuple_55794, result___neg___55796)
    # Adding element type (line 958)
    int_55797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 958, 22), tuple_55794, int_55797)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 958, 21), tuple_55793, tuple_55794)
    # Adding element type (line 958)
    
    # Obtaining an instance of the builtin type 'tuple' (line 958)
    tuple_55798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 958)
    # Adding element type (line 958)
    int_55799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 958, 38), tuple_55798, int_55799)
    # Adding element type (line 958)
    int_55800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 958, 38), tuple_55798, int_55800)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 958, 21), tuple_55793, tuple_55798)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 958, 19), list_55792, tuple_55793)
    
    # Assigning a type to the variable 'openSegments' (line 958)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 958, 4), 'openSegments', list_55792)
    
    # Assigning a List to a Name (line 962):
    
    # Assigning a List to a Name (line 962):
    
    # Obtaining an instance of the builtin type 'list' (line 962)
    list_55801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 962)
    # Adding element type (line 962)
    
    # Obtaining an instance of the builtin type 'tuple' (line 962)
    tuple_55802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 962)
    # Adding element type (line 962)
    
    # Obtaining an instance of the builtin type 'tuple' (line 962)
    tuple_55803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 962)
    # Adding element type (line 962)
    int_55804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 23), tuple_55803, int_55804)
    # Adding element type (line 962)
    int_55805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 23), tuple_55803, int_55805)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 22), tuple_55802, tuple_55803)
    # Adding element type (line 962)
    
    # Obtaining an instance of the builtin type 'tuple' (line 962)
    tuple_55806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 962)
    # Adding element type (line 962)
    # Getting the type of 'ticksize' (line 962)
    ticksize_55807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 31), 'ticksize')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 31), tuple_55806, ticksize_55807)
    # Adding element type (line 962)
    int_55808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 31), tuple_55806, int_55808)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 22), tuple_55802, tuple_55806)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 20), list_55801, tuple_55802)
    
    # Assigning a type to the variable 'closeSegments' (line 962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 4), 'closeSegments', list_55801)
    
    # Assigning a ListComp to a Name (line 964):
    
    # Assigning a ListComp to a Name (line 964):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 965)
    # Processing the call arguments (line 965)
    
    # Call to xrange(...): (line 965)
    # Processing the call arguments (line 965)
    
    # Call to len(...): (line 965)
    # Processing the call arguments (line 965)
    # Getting the type of 'opens' (line 965)
    opens_55818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 34), 'opens', False)
    # Processing the call keyword arguments (line 965)
    kwargs_55819 = {}
    # Getting the type of 'len' (line 965)
    len_55817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 30), 'len', False)
    # Calling len(args, kwargs) (line 965)
    len_call_result_55820 = invoke(stypy.reporting.localization.Localization(__file__, 965, 30), len_55817, *[opens_55818], **kwargs_55819)
    
    # Processing the call keyword arguments (line 965)
    kwargs_55821 = {}
    # Getting the type of 'xrange' (line 965)
    xrange_55816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 23), 'xrange', False)
    # Calling xrange(args, kwargs) (line 965)
    xrange_call_result_55822 = invoke(stypy.reporting.localization.Localization(__file__, 965, 23), xrange_55816, *[len_call_result_55820], **kwargs_55821)
    
    # Getting the type of 'opens' (line 965)
    opens_55823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 43), 'opens', False)
    # Processing the call keyword arguments (line 965)
    kwargs_55824 = {}
    # Getting the type of 'zip' (line 965)
    zip_55815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 19), 'zip', False)
    # Calling zip(args, kwargs) (line 965)
    zip_call_result_55825 = invoke(stypy.reporting.localization.Localization(__file__, 965, 19), zip_55815, *[xrange_call_result_55822, opens_55823], **kwargs_55824)
    
    comprehension_55826 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 19), zip_call_result_55825)
    # Assigning a type to the variable 'i' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 19), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 19), comprehension_55826))
    # Assigning a type to the variable 'open' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 19), 'open', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 19), comprehension_55826))
    
    # Getting the type of 'open' (line 965)
    open_55812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 53), 'open')
    int_55813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 61), 'int')
    # Applying the binary operator '!=' (line 965)
    result_ne_55814 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 53), '!=', open_55812, int_55813)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 964)
    tuple_55809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 964)
    # Adding element type (line 964)
    # Getting the type of 'i' (line 964)
    i_55810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 20), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 20), tuple_55809, i_55810)
    # Adding element type (line 964)
    # Getting the type of 'open' (line 964)
    open_55811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 23), 'open')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 20), tuple_55809, open_55811)
    
    list_55827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 19), list_55827, tuple_55809)
    # Assigning a type to the variable 'offsetsOpen' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 4), 'offsetsOpen', list_55827)
    
    # Assigning a ListComp to a Name (line 967):
    
    # Assigning a ListComp to a Name (line 967):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 968)
    # Processing the call arguments (line 968)
    
    # Call to xrange(...): (line 968)
    # Processing the call arguments (line 968)
    
    # Call to len(...): (line 968)
    # Processing the call arguments (line 968)
    # Getting the type of 'closes' (line 968)
    closes_55837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 35), 'closes', False)
    # Processing the call keyword arguments (line 968)
    kwargs_55838 = {}
    # Getting the type of 'len' (line 968)
    len_55836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 31), 'len', False)
    # Calling len(args, kwargs) (line 968)
    len_call_result_55839 = invoke(stypy.reporting.localization.Localization(__file__, 968, 31), len_55836, *[closes_55837], **kwargs_55838)
    
    # Processing the call keyword arguments (line 968)
    kwargs_55840 = {}
    # Getting the type of 'xrange' (line 968)
    xrange_55835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 24), 'xrange', False)
    # Calling xrange(args, kwargs) (line 968)
    xrange_call_result_55841 = invoke(stypy.reporting.localization.Localization(__file__, 968, 24), xrange_55835, *[len_call_result_55839], **kwargs_55840)
    
    # Getting the type of 'closes' (line 968)
    closes_55842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 45), 'closes', False)
    # Processing the call keyword arguments (line 968)
    kwargs_55843 = {}
    # Getting the type of 'zip' (line 968)
    zip_55834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 20), 'zip', False)
    # Calling zip(args, kwargs) (line 968)
    zip_call_result_55844 = invoke(stypy.reporting.localization.Localization(__file__, 968, 20), zip_55834, *[xrange_call_result_55841, closes_55842], **kwargs_55843)
    
    comprehension_55845 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 967, 20), zip_call_result_55844)
    # Assigning a type to the variable 'i' (line 967)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 20), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 967, 20), comprehension_55845))
    # Assigning a type to the variable 'close' (line 967)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 20), 'close', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 967, 20), comprehension_55845))
    
    # Getting the type of 'close' (line 968)
    close_55831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 56), 'close')
    int_55832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 65), 'int')
    # Applying the binary operator '!=' (line 968)
    result_ne_55833 = python_operator(stypy.reporting.localization.Localization(__file__, 968, 56), '!=', close_55831, int_55832)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 967)
    tuple_55828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 967)
    # Adding element type (line 967)
    # Getting the type of 'i' (line 967)
    i_55829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 21), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 967, 21), tuple_55828, i_55829)
    # Adding element type (line 967)
    # Getting the type of 'close' (line 967)
    close_55830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 24), 'close')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 967, 21), tuple_55828, close_55830)
    
    list_55846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 967, 20), list_55846, tuple_55828)
    # Assigning a type to the variable 'offsetsClose' (line 967)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 4), 'offsetsClose', list_55846)
    
    # Assigning a BinOp to a Name (line 970):
    
    # Assigning a BinOp to a Name (line 970):
    # Getting the type of 'ax' (line 970)
    ax_55847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 12), 'ax')
    # Obtaining the member 'figure' of a type (line 970)
    figure_55848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 12), ax_55847, 'figure')
    # Obtaining the member 'dpi' of a type (line 970)
    dpi_55849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 12), figure_55848, 'dpi')
    float_55850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 29), 'float')
    float_55851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 35), 'float')
    # Applying the binary operator 'div' (line 970)
    result_div_55852 = python_operator(stypy.reporting.localization.Localization(__file__, 970, 29), 'div', float_55850, float_55851)
    
    # Applying the binary operator '*' (line 970)
    result_mul_55853 = python_operator(stypy.reporting.localization.Localization(__file__, 970, 12), '*', dpi_55849, result_div_55852)
    
    # Assigning a type to the variable 'scale' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'scale', result_mul_55853)
    
    # Assigning a Call to a Name (line 972):
    
    # Assigning a Call to a Name (line 972):
    
    # Call to scale(...): (line 972)
    # Processing the call arguments (line 972)
    # Getting the type of 'scale' (line 972)
    scale_55858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 37), 'scale', False)
    float_55859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 44), 'float')
    # Processing the call keyword arguments (line 972)
    kwargs_55860 = {}
    
    # Call to Affine2D(...): (line 972)
    # Processing the call keyword arguments (line 972)
    kwargs_55855 = {}
    # Getting the type of 'Affine2D' (line 972)
    Affine2D_55854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 20), 'Affine2D', False)
    # Calling Affine2D(args, kwargs) (line 972)
    Affine2D_call_result_55856 = invoke(stypy.reporting.localization.Localization(__file__, 972, 20), Affine2D_55854, *[], **kwargs_55855)
    
    # Obtaining the member 'scale' of a type (line 972)
    scale_55857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 20), Affine2D_call_result_55856, 'scale')
    # Calling scale(args, kwargs) (line 972)
    scale_call_result_55861 = invoke(stypy.reporting.localization.Localization(__file__, 972, 20), scale_55857, *[scale_55858, float_55859], **kwargs_55860)
    
    # Assigning a type to the variable 'tickTransform' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 4), 'tickTransform', scale_call_result_55861)
    
    # Assigning a Call to a Name (line 974):
    
    # Assigning a Call to a Name (line 974):
    
    # Call to to_rgba(...): (line 974)
    # Processing the call arguments (line 974)
    # Getting the type of 'colorup' (line 974)
    colorup_55864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 30), 'colorup', False)
    # Processing the call keyword arguments (line 974)
    kwargs_55865 = {}
    # Getting the type of 'mcolors' (line 974)
    mcolors_55862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 14), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 974)
    to_rgba_55863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 14), mcolors_55862, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 974)
    to_rgba_call_result_55866 = invoke(stypy.reporting.localization.Localization(__file__, 974, 14), to_rgba_55863, *[colorup_55864], **kwargs_55865)
    
    # Assigning a type to the variable 'colorup' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 4), 'colorup', to_rgba_call_result_55866)
    
    # Assigning a Call to a Name (line 975):
    
    # Assigning a Call to a Name (line 975):
    
    # Call to to_rgba(...): (line 975)
    # Processing the call arguments (line 975)
    # Getting the type of 'colordown' (line 975)
    colordown_55869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 32), 'colordown', False)
    # Processing the call keyword arguments (line 975)
    kwargs_55870 = {}
    # Getting the type of 'mcolors' (line 975)
    mcolors_55867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 16), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 975)
    to_rgba_55868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 16), mcolors_55867, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 975)
    to_rgba_call_result_55871 = invoke(stypy.reporting.localization.Localization(__file__, 975, 16), to_rgba_55868, *[colordown_55869], **kwargs_55870)
    
    # Assigning a type to the variable 'colordown' (line 975)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 4), 'colordown', to_rgba_call_result_55871)
    
    # Assigning a Dict to a Name (line 976):
    
    # Assigning a Dict to a Name (line 976):
    
    # Obtaining an instance of the builtin type 'dict' (line 976)
    dict_55872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 976)
    # Adding element type (key, value) (line 976)
    # Getting the type of 'True' (line 976)
    True_55873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 14), 'True')
    # Getting the type of 'colorup' (line 976)
    colorup_55874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 20), 'colorup')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 976, 13), dict_55872, (True_55873, colorup_55874))
    # Adding element type (key, value) (line 976)
    # Getting the type of 'False' (line 976)
    False_55875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 29), 'False')
    # Getting the type of 'colordown' (line 976)
    colordown_55876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 36), 'colordown')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 976, 13), dict_55872, (False_55875, colordown_55876))
    
    # Assigning a type to the variable 'colord' (line 976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 4), 'colord', dict_55872)
    
    # Assigning a ListComp to a Name (line 977):
    
    # Assigning a ListComp to a Name (line 977):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 978)
    # Processing the call arguments (line 978)
    # Getting the type of 'opens' (line 978)
    opens_55891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 18), 'opens', False)
    # Getting the type of 'closes' (line 978)
    closes_55892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 25), 'closes', False)
    # Processing the call keyword arguments (line 978)
    kwargs_55893 = {}
    # Getting the type of 'zip' (line 978)
    zip_55890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 14), 'zip', False)
    # Calling zip(args, kwargs) (line 978)
    zip_call_result_55894 = invoke(stypy.reporting.localization.Localization(__file__, 978, 14), zip_55890, *[opens_55891, closes_55892], **kwargs_55893)
    
    comprehension_55895 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 977, 14), zip_call_result_55894)
    # Assigning a type to the variable 'open' (line 977)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 14), 'open', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 977, 14), comprehension_55895))
    # Assigning a type to the variable 'close' (line 977)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 14), 'close', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 977, 14), comprehension_55895))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'open' (line 978)
    open_55883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 36), 'open')
    int_55884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 44), 'int')
    # Applying the binary operator '!=' (line 978)
    result_ne_55885 = python_operator(stypy.reporting.localization.Localization(__file__, 978, 36), '!=', open_55883, int_55884)
    
    
    # Getting the type of 'close' (line 978)
    close_55886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 51), 'close')
    int_55887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 60), 'int')
    # Applying the binary operator '!=' (line 978)
    result_ne_55888 = python_operator(stypy.reporting.localization.Localization(__file__, 978, 51), '!=', close_55886, int_55887)
    
    # Applying the binary operator 'and' (line 978)
    result_and_keyword_55889 = python_operator(stypy.reporting.localization.Localization(__file__, 978, 36), 'and', result_ne_55885, result_ne_55888)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'open' (line 977)
    open_55877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 21), 'open')
    # Getting the type of 'close' (line 977)
    close_55878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 28), 'close')
    # Applying the binary operator '<' (line 977)
    result_lt_55879 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 21), '<', open_55877, close_55878)
    
    # Getting the type of 'colord' (line 977)
    colord_55880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 14), 'colord')
    # Obtaining the member '__getitem__' of a type (line 977)
    getitem___55881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 977, 14), colord_55880, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 977)
    subscript_call_result_55882 = invoke(stypy.reporting.localization.Localization(__file__, 977, 14), getitem___55881, result_lt_55879)
    
    list_55896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 977, 14), list_55896, subscript_call_result_55882)
    # Assigning a type to the variable 'colors' (line 977)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 4), 'colors', list_55896)
    
    # Assigning a Tuple to a Name (line 980):
    
    # Assigning a Tuple to a Name (line 980):
    
    # Obtaining an instance of the builtin type 'tuple' (line 980)
    tuple_55897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 980)
    # Adding element type (line 980)
    int_55898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 980, 12), tuple_55897, int_55898)
    
    # Assigning a type to the variable 'useAA' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 4), 'useAA', tuple_55897)
    
    # Assigning a Tuple to a Name (line 981):
    
    # Assigning a Tuple to a Name (line 981):
    
    # Obtaining an instance of the builtin type 'tuple' (line 981)
    tuple_55899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 981)
    # Adding element type (line 981)
    int_55900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 981, 9), tuple_55899, int_55900)
    
    # Assigning a type to the variable 'lw' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 4), 'lw', tuple_55899)
    
    # Assigning a Call to a Name (line 982):
    
    # Assigning a Call to a Name (line 982):
    
    # Call to LineCollection(...): (line 982)
    # Processing the call arguments (line 982)
    # Getting the type of 'rangeSegments' (line 982)
    rangeSegments_55902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 37), 'rangeSegments', False)
    # Processing the call keyword arguments (line 982)
    # Getting the type of 'colors' (line 983)
    colors_55903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 44), 'colors', False)
    keyword_55904 = colors_55903
    # Getting the type of 'lw' (line 984)
    lw_55905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 48), 'lw', False)
    keyword_55906 = lw_55905
    # Getting the type of 'useAA' (line 985)
    useAA_55907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 50), 'useAA', False)
    keyword_55908 = useAA_55907
    kwargs_55909 = {'colors': keyword_55904, 'antialiaseds': keyword_55908, 'linewidths': keyword_55906}
    # Getting the type of 'LineCollection' (line 982)
    LineCollection_55901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 22), 'LineCollection', False)
    # Calling LineCollection(args, kwargs) (line 982)
    LineCollection_call_result_55910 = invoke(stypy.reporting.localization.Localization(__file__, 982, 22), LineCollection_55901, *[rangeSegments_55902], **kwargs_55909)
    
    # Assigning a type to the variable 'rangeCollection' (line 982)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 4), 'rangeCollection', LineCollection_call_result_55910)
    
    # Assigning a Call to a Name (line 988):
    
    # Assigning a Call to a Name (line 988):
    
    # Call to LineCollection(...): (line 988)
    # Processing the call arguments (line 988)
    # Getting the type of 'openSegments' (line 988)
    openSegments_55912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 36), 'openSegments', False)
    # Processing the call keyword arguments (line 988)
    # Getting the type of 'colors' (line 989)
    colors_55913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 43), 'colors', False)
    keyword_55914 = colors_55913
    # Getting the type of 'useAA' (line 990)
    useAA_55915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 49), 'useAA', False)
    keyword_55916 = useAA_55915
    # Getting the type of 'lw' (line 991)
    lw_55917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 47), 'lw', False)
    keyword_55918 = lw_55917
    # Getting the type of 'offsetsOpen' (line 992)
    offsetsOpen_55919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 44), 'offsetsOpen', False)
    keyword_55920 = offsetsOpen_55919
    # Getting the type of 'ax' (line 993)
    ax_55921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 48), 'ax', False)
    # Obtaining the member 'transData' of a type (line 993)
    transData_55922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 48), ax_55921, 'transData')
    keyword_55923 = transData_55922
    kwargs_55924 = {'colors': keyword_55914, 'offsets': keyword_55920, 'transOffset': keyword_55923, 'antialiaseds': keyword_55916, 'linewidths': keyword_55918}
    # Getting the type of 'LineCollection' (line 988)
    LineCollection_55911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 21), 'LineCollection', False)
    # Calling LineCollection(args, kwargs) (line 988)
    LineCollection_call_result_55925 = invoke(stypy.reporting.localization.Localization(__file__, 988, 21), LineCollection_55911, *[openSegments_55912], **kwargs_55924)
    
    # Assigning a type to the variable 'openCollection' (line 988)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 4), 'openCollection', LineCollection_call_result_55925)
    
    # Call to set_transform(...): (line 995)
    # Processing the call arguments (line 995)
    # Getting the type of 'tickTransform' (line 995)
    tickTransform_55928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 33), 'tickTransform', False)
    # Processing the call keyword arguments (line 995)
    kwargs_55929 = {}
    # Getting the type of 'openCollection' (line 995)
    openCollection_55926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 4), 'openCollection', False)
    # Obtaining the member 'set_transform' of a type (line 995)
    set_transform_55927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 4), openCollection_55926, 'set_transform')
    # Calling set_transform(args, kwargs) (line 995)
    set_transform_call_result_55930 = invoke(stypy.reporting.localization.Localization(__file__, 995, 4), set_transform_55927, *[tickTransform_55928], **kwargs_55929)
    
    
    # Assigning a Call to a Name (line 997):
    
    # Assigning a Call to a Name (line 997):
    
    # Call to LineCollection(...): (line 997)
    # Processing the call arguments (line 997)
    # Getting the type of 'closeSegments' (line 997)
    closeSegments_55932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 37), 'closeSegments', False)
    # Processing the call keyword arguments (line 997)
    # Getting the type of 'colors' (line 998)
    colors_55933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 44), 'colors', False)
    keyword_55934 = colors_55933
    # Getting the type of 'useAA' (line 999)
    useAA_55935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 50), 'useAA', False)
    keyword_55936 = useAA_55935
    # Getting the type of 'lw' (line 1000)
    lw_55937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 48), 'lw', False)
    keyword_55938 = lw_55937
    # Getting the type of 'offsetsClose' (line 1001)
    offsetsClose_55939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 45), 'offsetsClose', False)
    keyword_55940 = offsetsClose_55939
    # Getting the type of 'ax' (line 1002)
    ax_55941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 49), 'ax', False)
    # Obtaining the member 'transData' of a type (line 1002)
    transData_55942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 49), ax_55941, 'transData')
    keyword_55943 = transData_55942
    kwargs_55944 = {'colors': keyword_55934, 'offsets': keyword_55940, 'transOffset': keyword_55943, 'antialiaseds': keyword_55936, 'linewidths': keyword_55938}
    # Getting the type of 'LineCollection' (line 997)
    LineCollection_55931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 22), 'LineCollection', False)
    # Calling LineCollection(args, kwargs) (line 997)
    LineCollection_call_result_55945 = invoke(stypy.reporting.localization.Localization(__file__, 997, 22), LineCollection_55931, *[closeSegments_55932], **kwargs_55944)
    
    # Assigning a type to the variable 'closeCollection' (line 997)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 4), 'closeCollection', LineCollection_call_result_55945)
    
    # Call to set_transform(...): (line 1004)
    # Processing the call arguments (line 1004)
    # Getting the type of 'tickTransform' (line 1004)
    tickTransform_55948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 34), 'tickTransform', False)
    # Processing the call keyword arguments (line 1004)
    kwargs_55949 = {}
    # Getting the type of 'closeCollection' (line 1004)
    closeCollection_55946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 4), 'closeCollection', False)
    # Obtaining the member 'set_transform' of a type (line 1004)
    set_transform_55947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1004, 4), closeCollection_55946, 'set_transform')
    # Calling set_transform(args, kwargs) (line 1004)
    set_transform_call_result_55950 = invoke(stypy.reporting.localization.Localization(__file__, 1004, 4), set_transform_55947, *[tickTransform_55948], **kwargs_55949)
    
    
    # Assigning a Tuple to a Tuple (line 1006):
    
    # Assigning a Num to a Name (line 1006):
    int_55951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 19), 'int')
    # Assigning a type to the variable 'tuple_assignment_54318' (line 1006)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 4), 'tuple_assignment_54318', int_55951)
    
    # Assigning a Call to a Name (line 1006):
    
    # Call to len(...): (line 1006)
    # Processing the call arguments (line 1006)
    # Getting the type of 'rangeSegments' (line 1006)
    rangeSegments_55953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 26), 'rangeSegments', False)
    # Processing the call keyword arguments (line 1006)
    kwargs_55954 = {}
    # Getting the type of 'len' (line 1006)
    len_55952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 22), 'len', False)
    # Calling len(args, kwargs) (line 1006)
    len_call_result_55955 = invoke(stypy.reporting.localization.Localization(__file__, 1006, 22), len_55952, *[rangeSegments_55953], **kwargs_55954)
    
    # Assigning a type to the variable 'tuple_assignment_54319' (line 1006)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 4), 'tuple_assignment_54319', len_call_result_55955)
    
    # Assigning a Name to a Name (line 1006):
    # Getting the type of 'tuple_assignment_54318' (line 1006)
    tuple_assignment_54318_55956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 4), 'tuple_assignment_54318')
    # Assigning a type to the variable 'minpy' (line 1006)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 4), 'minpy', tuple_assignment_54318_55956)
    
    # Assigning a Name to a Name (line 1006):
    # Getting the type of 'tuple_assignment_54319' (line 1006)
    tuple_assignment_54319_55957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 4), 'tuple_assignment_54319')
    # Assigning a type to the variable 'maxx' (line 1006)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 11), 'maxx', tuple_assignment_54319_55957)
    
    # Assigning a Call to a Name (line 1007):
    
    # Assigning a Call to a Name (line 1007):
    
    # Call to min(...): (line 1007)
    # Processing the call arguments (line 1007)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'lows' (line 1007)
    lows_55963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 31), 'lows', False)
    comprehension_55964 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1007, 16), lows_55963)
    # Assigning a type to the variable 'low' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 16), 'low', comprehension_55964)
    
    # Getting the type of 'low' (line 1007)
    low_55960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 39), 'low', False)
    int_55961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 46), 'int')
    # Applying the binary operator '!=' (line 1007)
    result_ne_55962 = python_operator(stypy.reporting.localization.Localization(__file__, 1007, 39), '!=', low_55960, int_55961)
    
    # Getting the type of 'low' (line 1007)
    low_55959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 16), 'low', False)
    list_55965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1007, 16), list_55965, low_55959)
    # Processing the call keyword arguments (line 1007)
    kwargs_55966 = {}
    # Getting the type of 'min' (line 1007)
    min_55958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 11), 'min', False)
    # Calling min(args, kwargs) (line 1007)
    min_call_result_55967 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 11), min_55958, *[list_55965], **kwargs_55966)
    
    # Assigning a type to the variable 'miny' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'miny', min_call_result_55967)
    
    # Assigning a Call to a Name (line 1008):
    
    # Assigning a Call to a Name (line 1008):
    
    # Call to max(...): (line 1008)
    # Processing the call arguments (line 1008)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'highs' (line 1008)
    highs_55973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 33), 'highs', False)
    comprehension_55974 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1008, 16), highs_55973)
    # Assigning a type to the variable 'high' (line 1008)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'high', comprehension_55974)
    
    # Getting the type of 'high' (line 1008)
    high_55970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 42), 'high', False)
    int_55971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 50), 'int')
    # Applying the binary operator '!=' (line 1008)
    result_ne_55972 = python_operator(stypy.reporting.localization.Localization(__file__, 1008, 42), '!=', high_55970, int_55971)
    
    # Getting the type of 'high' (line 1008)
    high_55969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'high', False)
    list_55975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1008, 16), list_55975, high_55969)
    # Processing the call keyword arguments (line 1008)
    kwargs_55976 = {}
    # Getting the type of 'max' (line 1008)
    max_55968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 11), 'max', False)
    # Calling max(args, kwargs) (line 1008)
    max_call_result_55977 = invoke(stypy.reporting.localization.Localization(__file__, 1008, 11), max_55968, *[list_55975], **kwargs_55976)
    
    # Assigning a type to the variable 'maxy' (line 1008)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 4), 'maxy', max_call_result_55977)
    
    # Assigning a Tuple to a Name (line 1009):
    
    # Assigning a Tuple to a Name (line 1009):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1009)
    tuple_55978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1009)
    # Adding element type (line 1009)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1009)
    tuple_55979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1009)
    # Adding element type (line 1009)
    # Getting the type of 'minpy' (line 1009)
    minpy_55980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 15), 'minpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 15), tuple_55979, minpy_55980)
    # Adding element type (line 1009)
    # Getting the type of 'miny' (line 1009)
    miny_55981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 22), 'miny')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 15), tuple_55979, miny_55981)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 14), tuple_55978, tuple_55979)
    # Adding element type (line 1009)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1009)
    tuple_55982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1009)
    # Adding element type (line 1009)
    # Getting the type of 'maxx' (line 1009)
    maxx_55983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 30), 'maxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 30), tuple_55982, maxx_55983)
    # Adding element type (line 1009)
    # Getting the type of 'maxy' (line 1009)
    maxy_55984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 36), 'maxy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 30), tuple_55982, maxy_55984)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1009, 14), tuple_55978, tuple_55982)
    
    # Assigning a type to the variable 'corners' (line 1009)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 4), 'corners', tuple_55978)
    
    # Call to update_datalim(...): (line 1010)
    # Processing the call arguments (line 1010)
    # Getting the type of 'corners' (line 1010)
    corners_55987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 22), 'corners', False)
    # Processing the call keyword arguments (line 1010)
    kwargs_55988 = {}
    # Getting the type of 'ax' (line 1010)
    ax_55985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 4), 'ax', False)
    # Obtaining the member 'update_datalim' of a type (line 1010)
    update_datalim_55986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1010, 4), ax_55985, 'update_datalim')
    # Calling update_datalim(args, kwargs) (line 1010)
    update_datalim_call_result_55989 = invoke(stypy.reporting.localization.Localization(__file__, 1010, 4), update_datalim_55986, *[corners_55987], **kwargs_55988)
    
    
    # Call to autoscale_view(...): (line 1011)
    # Processing the call keyword arguments (line 1011)
    kwargs_55992 = {}
    # Getting the type of 'ax' (line 1011)
    ax_55990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 4), 'ax', False)
    # Obtaining the member 'autoscale_view' of a type (line 1011)
    autoscale_view_55991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1011, 4), ax_55990, 'autoscale_view')
    # Calling autoscale_view(args, kwargs) (line 1011)
    autoscale_view_call_result_55993 = invoke(stypy.reporting.localization.Localization(__file__, 1011, 4), autoscale_view_55991, *[], **kwargs_55992)
    
    
    # Call to add_collection(...): (line 1014)
    # Processing the call arguments (line 1014)
    # Getting the type of 'rangeCollection' (line 1014)
    rangeCollection_55996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 22), 'rangeCollection', False)
    # Processing the call keyword arguments (line 1014)
    kwargs_55997 = {}
    # Getting the type of 'ax' (line 1014)
    ax_55994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 1014)
    add_collection_55995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 4), ax_55994, 'add_collection')
    # Calling add_collection(args, kwargs) (line 1014)
    add_collection_call_result_55998 = invoke(stypy.reporting.localization.Localization(__file__, 1014, 4), add_collection_55995, *[rangeCollection_55996], **kwargs_55997)
    
    
    # Call to add_collection(...): (line 1015)
    # Processing the call arguments (line 1015)
    # Getting the type of 'openCollection' (line 1015)
    openCollection_56001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 22), 'openCollection', False)
    # Processing the call keyword arguments (line 1015)
    kwargs_56002 = {}
    # Getting the type of 'ax' (line 1015)
    ax_55999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 1015)
    add_collection_56000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 4), ax_55999, 'add_collection')
    # Calling add_collection(args, kwargs) (line 1015)
    add_collection_call_result_56003 = invoke(stypy.reporting.localization.Localization(__file__, 1015, 4), add_collection_56000, *[openCollection_56001], **kwargs_56002)
    
    
    # Call to add_collection(...): (line 1016)
    # Processing the call arguments (line 1016)
    # Getting the type of 'closeCollection' (line 1016)
    closeCollection_56006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 22), 'closeCollection', False)
    # Processing the call keyword arguments (line 1016)
    kwargs_56007 = {}
    # Getting the type of 'ax' (line 1016)
    ax_56004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 1016)
    add_collection_56005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1016, 4), ax_56004, 'add_collection')
    # Calling add_collection(args, kwargs) (line 1016)
    add_collection_call_result_56008 = invoke(stypy.reporting.localization.Localization(__file__, 1016, 4), add_collection_56005, *[closeCollection_56006], **kwargs_56007)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1017)
    tuple_56009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1017)
    # Adding element type (line 1017)
    # Getting the type of 'rangeCollection' (line 1017)
    rangeCollection_56010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 11), 'rangeCollection')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1017, 11), tuple_56009, rangeCollection_56010)
    # Adding element type (line 1017)
    # Getting the type of 'openCollection' (line 1017)
    openCollection_56011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 28), 'openCollection')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1017, 11), tuple_56009, openCollection_56011)
    # Adding element type (line 1017)
    # Getting the type of 'closeCollection' (line 1017)
    closeCollection_56012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 44), 'closeCollection')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1017, 11), tuple_56009, closeCollection_56012)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1017)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 4), 'stypy_return_type', tuple_56009)
    
    # ################# End of 'plot_day_summary2_ohlc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'plot_day_summary2_ohlc' in the type store
    # Getting the type of 'stypy_return_type' (line 915)
    stypy_return_type_56013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_56013)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'plot_day_summary2_ohlc'
    return stypy_return_type_56013

# Assigning a type to the variable 'plot_day_summary2_ohlc' (line 915)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 0), 'plot_day_summary2_ohlc', plot_day_summary2_ohlc)

@norecursion
def candlestick2_ochl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_56014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 61), 'int')
    unicode_56015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 25), 'unicode', u'k')
    unicode_56016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 40), 'unicode', u'r')
    float_56017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, 23), 'float')
    defaults = [int_56014, unicode_56015, unicode_56016, float_56017]
    # Create a new context for function 'candlestick2_ochl'
    module_type_store = module_type_store.open_function_context('candlestick2_ochl', 1020, 0, False)
    
    # Passed parameters checking function
    candlestick2_ochl.stypy_localization = localization
    candlestick2_ochl.stypy_type_of_self = None
    candlestick2_ochl.stypy_type_store = module_type_store
    candlestick2_ochl.stypy_function_name = 'candlestick2_ochl'
    candlestick2_ochl.stypy_param_names_list = ['ax', 'opens', 'closes', 'highs', 'lows', 'width', 'colorup', 'colordown', 'alpha']
    candlestick2_ochl.stypy_varargs_param_name = None
    candlestick2_ochl.stypy_kwargs_param_name = None
    candlestick2_ochl.stypy_call_defaults = defaults
    candlestick2_ochl.stypy_call_varargs = varargs
    candlestick2_ochl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'candlestick2_ochl', ['ax', 'opens', 'closes', 'highs', 'lows', 'width', 'colorup', 'colordown', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'candlestick2_ochl', localization, ['ax', 'opens', 'closes', 'highs', 'lows', 'width', 'colorup', 'colordown', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'candlestick2_ochl(...)' code ##################

    unicode_56018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, (-1)), 'unicode', u'Represent the open, close as a bar line and high low range as a\n    vertical line.\n\n    Preserves the original argument order.\n\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    opens : sequence\n        sequence of opening values\n    closes : sequence\n        sequence of closing values\n    highs : sequence\n        sequence of high values\n    lows : sequence\n        sequence of low values\n    ticksize : int\n        size of open and close ticks in points\n    colorup : color\n        the color of the lines where close >= open\n    colordown : color\n        the color of the lines where close <  open\n    alpha : float\n        bar transparency\n\n    Returns\n    -------\n    ret : tuple\n        (lineCollection, barCollection)\n    ')
    
    # Call to candlestick2_ohlc(...): (line 1057)
    # Processing the call arguments (line 1057)
    # Getting the type of 'ax' (line 1057)
    ax_56020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 22), 'ax', False)
    # Getting the type of 'opens' (line 1057)
    opens_56021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 26), 'opens', False)
    # Getting the type of 'highs' (line 1057)
    highs_56022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 33), 'highs', False)
    # Getting the type of 'lows' (line 1057)
    lows_56023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 40), 'lows', False)
    # Getting the type of 'closes' (line 1057)
    closes_56024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 46), 'closes', False)
    # Processing the call keyword arguments (line 1057)
    # Getting the type of 'width' (line 1057)
    width_56025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 60), 'width', False)
    keyword_56026 = width_56025
    # Getting the type of 'colorup' (line 1058)
    colorup_56027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 29), 'colorup', False)
    keyword_56028 = colorup_56027
    # Getting the type of 'colordown' (line 1058)
    colordown_56029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 48), 'colordown', False)
    keyword_56030 = colordown_56029
    # Getting the type of 'alpha' (line 1059)
    alpha_56031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 27), 'alpha', False)
    keyword_56032 = alpha_56031
    kwargs_56033 = {'colordown': keyword_56030, 'width': keyword_56026, 'alpha': keyword_56032, 'colorup': keyword_56028}
    # Getting the type of 'candlestick2_ohlc' (line 1057)
    candlestick2_ohlc_56019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 4), 'candlestick2_ohlc', False)
    # Calling candlestick2_ohlc(args, kwargs) (line 1057)
    candlestick2_ohlc_call_result_56034 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 4), candlestick2_ohlc_56019, *[ax_56020, opens_56021, highs_56022, lows_56023, closes_56024], **kwargs_56033)
    
    
    # ################# End of 'candlestick2_ochl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'candlestick2_ochl' in the type store
    # Getting the type of 'stypy_return_type' (line 1020)
    stypy_return_type_56035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1020, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_56035)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'candlestick2_ochl'
    return stypy_return_type_56035

# Assigning a type to the variable 'candlestick2_ochl' (line 1020)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1020, 0), 'candlestick2_ochl', candlestick2_ochl)

@norecursion
def candlestick2_ohlc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_56036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 60), 'int')
    unicode_56037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1063, 25), 'unicode', u'k')
    unicode_56038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1063, 40), 'unicode', u'r')
    float_56039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, 23), 'float')
    defaults = [int_56036, unicode_56037, unicode_56038, float_56039]
    # Create a new context for function 'candlestick2_ohlc'
    module_type_store = module_type_store.open_function_context('candlestick2_ohlc', 1062, 0, False)
    
    # Passed parameters checking function
    candlestick2_ohlc.stypy_localization = localization
    candlestick2_ohlc.stypy_type_of_self = None
    candlestick2_ohlc.stypy_type_store = module_type_store
    candlestick2_ohlc.stypy_function_name = 'candlestick2_ohlc'
    candlestick2_ohlc.stypy_param_names_list = ['ax', 'opens', 'highs', 'lows', 'closes', 'width', 'colorup', 'colordown', 'alpha']
    candlestick2_ohlc.stypy_varargs_param_name = None
    candlestick2_ohlc.stypy_kwargs_param_name = None
    candlestick2_ohlc.stypy_call_defaults = defaults
    candlestick2_ohlc.stypy_call_varargs = varargs
    candlestick2_ohlc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'candlestick2_ohlc', ['ax', 'opens', 'highs', 'lows', 'closes', 'width', 'colorup', 'colordown', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'candlestick2_ohlc', localization, ['ax', 'opens', 'highs', 'lows', 'closes', 'width', 'colorup', 'colordown', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'candlestick2_ohlc(...)' code ##################

    unicode_56040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, (-1)), 'unicode', u'Represent the open, close as a bar line and high low range as a\n    vertical line.\n\n    NOTE: this code assumes if any value open, low, high, close is\n    missing they all are missing\n\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    opens : sequence\n        sequence of opening values\n    highs : sequence\n        sequence of high values\n    lows : sequence\n        sequence of low values\n    closes : sequence\n        sequence of closing values\n    ticksize : int\n        size of open and close ticks in points\n    colorup : color\n        the color of the lines where close >= open\n    colordown : color\n        the color of the lines where close <  open\n    alpha : float\n        bar transparency\n\n    Returns\n    -------\n    ret : tuple\n        (lineCollection, barCollection)\n    ')
    
    # Call to _check_input(...): (line 1100)
    # Processing the call arguments (line 1100)
    # Getting the type of 'opens' (line 1100)
    opens_56042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 17), 'opens', False)
    # Getting the type of 'highs' (line 1100)
    highs_56043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 24), 'highs', False)
    # Getting the type of 'lows' (line 1100)
    lows_56044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 31), 'lows', False)
    # Getting the type of 'closes' (line 1100)
    closes_56045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 37), 'closes', False)
    # Processing the call keyword arguments (line 1100)
    kwargs_56046 = {}
    # Getting the type of '_check_input' (line 1100)
    _check_input_56041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 4), '_check_input', False)
    # Calling _check_input(args, kwargs) (line 1100)
    _check_input_call_result_56047 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 4), _check_input_56041, *[opens_56042, highs_56043, lows_56044, closes_56045], **kwargs_56046)
    
    
    # Assigning a BinOp to a Name (line 1102):
    
    # Assigning a BinOp to a Name (line 1102):
    # Getting the type of 'width' (line 1102)
    width_56048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 12), 'width')
    float_56049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 20), 'float')
    # Applying the binary operator 'div' (line 1102)
    result_div_56050 = python_operator(stypy.reporting.localization.Localization(__file__, 1102, 12), 'div', width_56048, float_56049)
    
    # Assigning a type to the variable 'delta' (line 1102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1102, 4), 'delta', result_div_56050)
    
    # Assigning a ListComp to a Name (line 1103):
    
    # Assigning a ListComp to a Name (line 1103):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1107)
    # Processing the call arguments (line 1107)
    
    # Call to xrange(...): (line 1107)
    # Processing the call arguments (line 1107)
    
    # Call to len(...): (line 1107)
    # Processing the call arguments (line 1107)
    # Getting the type of 'opens' (line 1107)
    opens_56082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 53), 'opens', False)
    # Processing the call keyword arguments (line 1107)
    kwargs_56083 = {}
    # Getting the type of 'len' (line 1107)
    len_56081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 49), 'len', False)
    # Calling len(args, kwargs) (line 1107)
    len_call_result_56084 = invoke(stypy.reporting.localization.Localization(__file__, 1107, 49), len_56081, *[opens_56082], **kwargs_56083)
    
    # Processing the call keyword arguments (line 1107)
    kwargs_56085 = {}
    # Getting the type of 'xrange' (line 1107)
    xrange_56080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 42), 'xrange', False)
    # Calling xrange(args, kwargs) (line 1107)
    xrange_call_result_56086 = invoke(stypy.reporting.localization.Localization(__file__, 1107, 42), xrange_56080, *[len_call_result_56084], **kwargs_56085)
    
    # Getting the type of 'opens' (line 1107)
    opens_56087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 62), 'opens', False)
    # Getting the type of 'closes' (line 1107)
    closes_56088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 69), 'closes', False)
    # Processing the call keyword arguments (line 1107)
    kwargs_56089 = {}
    # Getting the type of 'zip' (line 1107)
    zip_56079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 38), 'zip', False)
    # Calling zip(args, kwargs) (line 1107)
    zip_call_result_56090 = invoke(stypy.reporting.localization.Localization(__file__, 1107, 38), zip_56079, *[xrange_call_result_56086, opens_56087, closes_56088], **kwargs_56089)
    
    comprehension_56091 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 16), zip_call_result_56090)
    # Assigning a type to the variable 'i' (line 1103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 16), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 16), comprehension_56091))
    # Assigning a type to the variable 'open' (line 1103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 16), 'open', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 16), comprehension_56091))
    # Assigning a type to the variable 'close' (line 1103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 16), 'close', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 16), comprehension_56091))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'open' (line 1108)
    open_56072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 19), 'open')
    int_56073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 27), 'int')
    # Applying the binary operator '!=' (line 1108)
    result_ne_56074 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 19), '!=', open_56072, int_56073)
    
    
    # Getting the type of 'close' (line 1108)
    close_56075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 34), 'close')
    int_56076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 43), 'int')
    # Applying the binary operator '!=' (line 1108)
    result_ne_56077 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 34), '!=', close_56075, int_56076)
    
    # Applying the binary operator 'and' (line 1108)
    result_and_keyword_56078 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 19), 'and', result_ne_56074, result_ne_56077)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1103)
    tuple_56051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1103, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1103)
    # Adding element type (line 1103)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1103)
    tuple_56052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1103, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1103)
    # Adding element type (line 1103)
    # Getting the type of 'i' (line 1103)
    i_56053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 18), 'i')
    # Getting the type of 'delta' (line 1103)
    delta_56054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 22), 'delta')
    # Applying the binary operator '-' (line 1103)
    result_sub_56055 = python_operator(stypy.reporting.localization.Localization(__file__, 1103, 18), '-', i_56053, delta_56054)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 18), tuple_56052, result_sub_56055)
    # Adding element type (line 1103)
    # Getting the type of 'open' (line 1103)
    open_56056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 29), 'open')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 18), tuple_56052, open_56056)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 17), tuple_56051, tuple_56052)
    # Adding element type (line 1103)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1104)
    tuple_56057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1104)
    # Adding element type (line 1104)
    # Getting the type of 'i' (line 1104)
    i_56058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 18), 'i')
    # Getting the type of 'delta' (line 1104)
    delta_56059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 22), 'delta')
    # Applying the binary operator '-' (line 1104)
    result_sub_56060 = python_operator(stypy.reporting.localization.Localization(__file__, 1104, 18), '-', i_56058, delta_56059)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 18), tuple_56057, result_sub_56060)
    # Adding element type (line 1104)
    # Getting the type of 'close' (line 1104)
    close_56061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 29), 'close')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 18), tuple_56057, close_56061)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 17), tuple_56051, tuple_56057)
    # Adding element type (line 1103)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1105)
    tuple_56062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1105)
    # Adding element type (line 1105)
    # Getting the type of 'i' (line 1105)
    i_56063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 18), 'i')
    # Getting the type of 'delta' (line 1105)
    delta_56064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 22), 'delta')
    # Applying the binary operator '+' (line 1105)
    result_add_56065 = python_operator(stypy.reporting.localization.Localization(__file__, 1105, 18), '+', i_56063, delta_56064)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1105, 18), tuple_56062, result_add_56065)
    # Adding element type (line 1105)
    # Getting the type of 'close' (line 1105)
    close_56066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 29), 'close')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1105, 18), tuple_56062, close_56066)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 17), tuple_56051, tuple_56062)
    # Adding element type (line 1103)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1106)
    tuple_56067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1106)
    # Adding element type (line 1106)
    # Getting the type of 'i' (line 1106)
    i_56068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 18), 'i')
    # Getting the type of 'delta' (line 1106)
    delta_56069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 22), 'delta')
    # Applying the binary operator '+' (line 1106)
    result_add_56070 = python_operator(stypy.reporting.localization.Localization(__file__, 1106, 18), '+', i_56068, delta_56069)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 18), tuple_56067, result_add_56070)
    # Adding element type (line 1106)
    # Getting the type of 'open' (line 1106)
    open_56071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 29), 'open')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 18), tuple_56067, open_56071)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 17), tuple_56051, tuple_56067)
    
    list_56092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1103, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1103, 16), list_56092, tuple_56051)
    # Assigning a type to the variable 'barVerts' (line 1103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 4), 'barVerts', list_56092)
    
    # Assigning a ListComp to a Name (line 1110):
    
    # Assigning a ListComp to a Name (line 1110):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1111)
    # Processing the call arguments (line 1111)
    
    # Call to xrange(...): (line 1111)
    # Processing the call arguments (line 1111)
    
    # Call to len(...): (line 1111)
    # Processing the call arguments (line 1111)
    # Getting the type of 'lows' (line 1111)
    lows_56106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 56), 'lows', False)
    # Processing the call keyword arguments (line 1111)
    kwargs_56107 = {}
    # Getting the type of 'len' (line 1111)
    len_56105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 52), 'len', False)
    # Calling len(args, kwargs) (line 1111)
    len_call_result_56108 = invoke(stypy.reporting.localization.Localization(__file__, 1111, 52), len_56105, *[lows_56106], **kwargs_56107)
    
    # Processing the call keyword arguments (line 1111)
    kwargs_56109 = {}
    # Getting the type of 'xrange' (line 1111)
    xrange_56104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 45), 'xrange', False)
    # Calling xrange(args, kwargs) (line 1111)
    xrange_call_result_56110 = invoke(stypy.reporting.localization.Localization(__file__, 1111, 45), xrange_56104, *[len_call_result_56108], **kwargs_56109)
    
    # Getting the type of 'lows' (line 1111)
    lows_56111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 64), 'lows', False)
    # Getting the type of 'highs' (line 1111)
    highs_56112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 70), 'highs', False)
    # Processing the call keyword arguments (line 1111)
    kwargs_56113 = {}
    # Getting the type of 'zip' (line 1111)
    zip_56103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 41), 'zip', False)
    # Calling zip(args, kwargs) (line 1111)
    zip_call_result_56114 = invoke(stypy.reporting.localization.Localization(__file__, 1111, 41), zip_56103, *[xrange_call_result_56110, lows_56111, highs_56112], **kwargs_56113)
    
    comprehension_56115 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 21), zip_call_result_56114)
    # Assigning a type to the variable 'i' (line 1110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 21), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 21), comprehension_56115))
    # Assigning a type to the variable 'low' (line 1110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 21), 'low', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 21), comprehension_56115))
    # Assigning a type to the variable 'high' (line 1110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 21), 'high', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 21), comprehension_56115))
    
    # Getting the type of 'low' (line 1112)
    low_56100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 24), 'low')
    int_56101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1112, 31), 'int')
    # Applying the binary operator '!=' (line 1112)
    result_ne_56102 = python_operator(stypy.reporting.localization.Localization(__file__, 1112, 24), '!=', low_56100, int_56101)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1110)
    tuple_56093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1110, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1110)
    # Adding element type (line 1110)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1110)
    tuple_56094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1110, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1110)
    # Adding element type (line 1110)
    # Getting the type of 'i' (line 1110)
    i_56095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 23), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 23), tuple_56094, i_56095)
    # Adding element type (line 1110)
    # Getting the type of 'low' (line 1110)
    low_56096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 26), 'low')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 23), tuple_56094, low_56096)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 22), tuple_56093, tuple_56094)
    # Adding element type (line 1110)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1110)
    tuple_56097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1110, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1110)
    # Adding element type (line 1110)
    # Getting the type of 'i' (line 1110)
    i_56098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 33), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 33), tuple_56097, i_56098)
    # Adding element type (line 1110)
    # Getting the type of 'high' (line 1110)
    high_56099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 36), 'high')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 33), tuple_56097, high_56099)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 22), tuple_56093, tuple_56097)
    
    list_56116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1110, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1110, 21), list_56116, tuple_56093)
    # Assigning a type to the variable 'rangeSegments' (line 1110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 4), 'rangeSegments', list_56116)
    
    # Assigning a Call to a Name (line 1114):
    
    # Assigning a Call to a Name (line 1114):
    
    # Call to to_rgba(...): (line 1114)
    # Processing the call arguments (line 1114)
    # Getting the type of 'colorup' (line 1114)
    colorup_56119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 30), 'colorup', False)
    # Getting the type of 'alpha' (line 1114)
    alpha_56120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 39), 'alpha', False)
    # Processing the call keyword arguments (line 1114)
    kwargs_56121 = {}
    # Getting the type of 'mcolors' (line 1114)
    mcolors_56117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 14), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 1114)
    to_rgba_56118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 14), mcolors_56117, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 1114)
    to_rgba_call_result_56122 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 14), to_rgba_56118, *[colorup_56119, alpha_56120], **kwargs_56121)
    
    # Assigning a type to the variable 'colorup' (line 1114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 4), 'colorup', to_rgba_call_result_56122)
    
    # Assigning a Call to a Name (line 1115):
    
    # Assigning a Call to a Name (line 1115):
    
    # Call to to_rgba(...): (line 1115)
    # Processing the call arguments (line 1115)
    # Getting the type of 'colordown' (line 1115)
    colordown_56125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 32), 'colordown', False)
    # Getting the type of 'alpha' (line 1115)
    alpha_56126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 43), 'alpha', False)
    # Processing the call keyword arguments (line 1115)
    kwargs_56127 = {}
    # Getting the type of 'mcolors' (line 1115)
    mcolors_56123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 16), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 1115)
    to_rgba_56124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1115, 16), mcolors_56123, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 1115)
    to_rgba_call_result_56128 = invoke(stypy.reporting.localization.Localization(__file__, 1115, 16), to_rgba_56124, *[colordown_56125, alpha_56126], **kwargs_56127)
    
    # Assigning a type to the variable 'colordown' (line 1115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 4), 'colordown', to_rgba_call_result_56128)
    
    # Assigning a Dict to a Name (line 1116):
    
    # Assigning a Dict to a Name (line 1116):
    
    # Obtaining an instance of the builtin type 'dict' (line 1116)
    dict_56129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1116)
    # Adding element type (key, value) (line 1116)
    # Getting the type of 'True' (line 1116)
    True_56130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 14), 'True')
    # Getting the type of 'colorup' (line 1116)
    colorup_56131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 20), 'colorup')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1116, 13), dict_56129, (True_56130, colorup_56131))
    # Adding element type (key, value) (line 1116)
    # Getting the type of 'False' (line 1116)
    False_56132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 29), 'False')
    # Getting the type of 'colordown' (line 1116)
    colordown_56133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 36), 'colordown')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1116, 13), dict_56129, (False_56132, colordown_56133))
    
    # Assigning a type to the variable 'colord' (line 1116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 4), 'colord', dict_56129)
    
    # Assigning a ListComp to a Name (line 1117):
    
    # Assigning a ListComp to a Name (line 1117):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1118)
    # Processing the call arguments (line 1118)
    # Getting the type of 'opens' (line 1118)
    opens_56148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 37), 'opens', False)
    # Getting the type of 'closes' (line 1118)
    closes_56149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 44), 'closes', False)
    # Processing the call keyword arguments (line 1118)
    kwargs_56150 = {}
    # Getting the type of 'zip' (line 1118)
    zip_56147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 33), 'zip', False)
    # Calling zip(args, kwargs) (line 1118)
    zip_call_result_56151 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 33), zip_56147, *[opens_56148, closes_56149], **kwargs_56150)
    
    comprehension_56152 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1117, 14), zip_call_result_56151)
    # Assigning a type to the variable 'open' (line 1117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1117, 14), 'open', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1117, 14), comprehension_56152))
    # Assigning a type to the variable 'close' (line 1117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1117, 14), 'close', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1117, 14), comprehension_56152))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'open' (line 1119)
    open_56140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 17), 'open')
    int_56141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 25), 'int')
    # Applying the binary operator '!=' (line 1119)
    result_ne_56142 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 17), '!=', open_56140, int_56141)
    
    
    # Getting the type of 'close' (line 1119)
    close_56143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 32), 'close')
    int_56144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 41), 'int')
    # Applying the binary operator '!=' (line 1119)
    result_ne_56145 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 32), '!=', close_56143, int_56144)
    
    # Applying the binary operator 'and' (line 1119)
    result_and_keyword_56146 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 17), 'and', result_ne_56142, result_ne_56145)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'open' (line 1117)
    open_56134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 21), 'open')
    # Getting the type of 'close' (line 1117)
    close_56135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 28), 'close')
    # Applying the binary operator '<' (line 1117)
    result_lt_56136 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 21), '<', open_56134, close_56135)
    
    # Getting the type of 'colord' (line 1117)
    colord_56137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 14), 'colord')
    # Obtaining the member '__getitem__' of a type (line 1117)
    getitem___56138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1117, 14), colord_56137, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1117)
    subscript_call_result_56139 = invoke(stypy.reporting.localization.Localization(__file__, 1117, 14), getitem___56138, result_lt_56136)
    
    list_56153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1117, 14), list_56153, subscript_call_result_56139)
    # Assigning a type to the variable 'colors' (line 1117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1117, 4), 'colors', list_56153)
    
    # Assigning a Tuple to a Name (line 1121):
    
    # Assigning a Tuple to a Name (line 1121):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1121)
    tuple_56154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1121)
    # Adding element type (line 1121)
    int_56155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1121, 12), tuple_56154, int_56155)
    
    # Assigning a type to the variable 'useAA' (line 1121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1121, 4), 'useAA', tuple_56154)
    
    # Assigning a Tuple to a Name (line 1122):
    
    # Assigning a Tuple to a Name (line 1122):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1122)
    tuple_56156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1122, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1122)
    # Adding element type (line 1122)
    float_56157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1122, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1122, 9), tuple_56156, float_56157)
    
    # Assigning a type to the variable 'lw' (line 1122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1122, 4), 'lw', tuple_56156)
    
    # Assigning a Call to a Name (line 1123):
    
    # Assigning a Call to a Name (line 1123):
    
    # Call to LineCollection(...): (line 1123)
    # Processing the call arguments (line 1123)
    # Getting the type of 'rangeSegments' (line 1123)
    rangeSegments_56159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 37), 'rangeSegments', False)
    # Processing the call keyword arguments (line 1123)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1124)
    tuple_56160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1124)
    # Adding element type (line 1124)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1124)
    tuple_56161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1124)
    # Adding element type (line 1124)
    int_56162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 46), tuple_56161, int_56162)
    # Adding element type (line 1124)
    int_56163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 46), tuple_56161, int_56163)
    # Adding element type (line 1124)
    int_56164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 46), tuple_56161, int_56164)
    # Adding element type (line 1124)
    int_56165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 46), tuple_56161, int_56165)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 45), tuple_56160, tuple_56161)
    
    keyword_56166 = tuple_56160
    # Getting the type of 'lw' (line 1125)
    lw_56167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 48), 'lw', False)
    keyword_56168 = lw_56167
    # Getting the type of 'useAA' (line 1126)
    useAA_56169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 50), 'useAA', False)
    keyword_56170 = useAA_56169
    kwargs_56171 = {'colors': keyword_56166, 'antialiaseds': keyword_56170, 'linewidths': keyword_56168}
    # Getting the type of 'LineCollection' (line 1123)
    LineCollection_56158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 22), 'LineCollection', False)
    # Calling LineCollection(args, kwargs) (line 1123)
    LineCollection_call_result_56172 = invoke(stypy.reporting.localization.Localization(__file__, 1123, 22), LineCollection_56158, *[rangeSegments_56159], **kwargs_56171)
    
    # Assigning a type to the variable 'rangeCollection' (line 1123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1123, 4), 'rangeCollection', LineCollection_call_result_56172)
    
    # Assigning a Call to a Name (line 1129):
    
    # Assigning a Call to a Name (line 1129):
    
    # Call to PolyCollection(...): (line 1129)
    # Processing the call arguments (line 1129)
    # Getting the type of 'barVerts' (line 1129)
    barVerts_56174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 35), 'barVerts', False)
    # Processing the call keyword arguments (line 1129)
    # Getting the type of 'colors' (line 1130)
    colors_56175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 46), 'colors', False)
    keyword_56176 = colors_56175
    
    # Obtaining an instance of the builtin type 'tuple' (line 1131)
    tuple_56177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1131)
    # Adding element type (line 1131)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1131)
    tuple_56178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1131)
    # Adding element type (line 1131)
    int_56179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1131, 48), tuple_56178, int_56179)
    # Adding element type (line 1131)
    int_56180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1131, 48), tuple_56178, int_56180)
    # Adding element type (line 1131)
    int_56181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1131, 48), tuple_56178, int_56181)
    # Adding element type (line 1131)
    int_56182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1131, 48), tuple_56178, int_56182)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1131, 47), tuple_56177, tuple_56178)
    
    keyword_56183 = tuple_56177
    # Getting the type of 'useAA' (line 1132)
    useAA_56184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 48), 'useAA', False)
    keyword_56185 = useAA_56184
    # Getting the type of 'lw' (line 1133)
    lw_56186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 46), 'lw', False)
    keyword_56187 = lw_56186
    kwargs_56188 = {'edgecolors': keyword_56183, 'antialiaseds': keyword_56185, 'facecolors': keyword_56176, 'linewidths': keyword_56187}
    # Getting the type of 'PolyCollection' (line 1129)
    PolyCollection_56173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 20), 'PolyCollection', False)
    # Calling PolyCollection(args, kwargs) (line 1129)
    PolyCollection_call_result_56189 = invoke(stypy.reporting.localization.Localization(__file__, 1129, 20), PolyCollection_56173, *[barVerts_56174], **kwargs_56188)
    
    # Assigning a type to the variable 'barCollection' (line 1129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1129, 4), 'barCollection', PolyCollection_call_result_56189)
    
    # Assigning a Tuple to a Tuple (line 1136):
    
    # Assigning a Num to a Name (line 1136):
    int_56190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 17), 'int')
    # Assigning a type to the variable 'tuple_assignment_54320' (line 1136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1136, 4), 'tuple_assignment_54320', int_56190)
    
    # Assigning a Call to a Name (line 1136):
    
    # Call to len(...): (line 1136)
    # Processing the call arguments (line 1136)
    # Getting the type of 'rangeSegments' (line 1136)
    rangeSegments_56192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 24), 'rangeSegments', False)
    # Processing the call keyword arguments (line 1136)
    kwargs_56193 = {}
    # Getting the type of 'len' (line 1136)
    len_56191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 20), 'len', False)
    # Calling len(args, kwargs) (line 1136)
    len_call_result_56194 = invoke(stypy.reporting.localization.Localization(__file__, 1136, 20), len_56191, *[rangeSegments_56192], **kwargs_56193)
    
    # Assigning a type to the variable 'tuple_assignment_54321' (line 1136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1136, 4), 'tuple_assignment_54321', len_call_result_56194)
    
    # Assigning a Name to a Name (line 1136):
    # Getting the type of 'tuple_assignment_54320' (line 1136)
    tuple_assignment_54320_56195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 4), 'tuple_assignment_54320')
    # Assigning a type to the variable 'minx' (line 1136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1136, 4), 'minx', tuple_assignment_54320_56195)
    
    # Assigning a Name to a Name (line 1136):
    # Getting the type of 'tuple_assignment_54321' (line 1136)
    tuple_assignment_54321_56196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 4), 'tuple_assignment_54321')
    # Assigning a type to the variable 'maxx' (line 1136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1136, 10), 'maxx', tuple_assignment_54321_56196)
    
    # Assigning a Call to a Name (line 1137):
    
    # Assigning a Call to a Name (line 1137):
    
    # Call to min(...): (line 1137)
    # Processing the call arguments (line 1137)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'lows' (line 1137)
    lows_56202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 31), 'lows', False)
    comprehension_56203 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 16), lows_56202)
    # Assigning a type to the variable 'low' (line 1137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1137, 16), 'low', comprehension_56203)
    
    # Getting the type of 'low' (line 1137)
    low_56199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 39), 'low', False)
    int_56200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 46), 'int')
    # Applying the binary operator '!=' (line 1137)
    result_ne_56201 = python_operator(stypy.reporting.localization.Localization(__file__, 1137, 39), '!=', low_56199, int_56200)
    
    # Getting the type of 'low' (line 1137)
    low_56198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 16), 'low', False)
    list_56204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 16), list_56204, low_56198)
    # Processing the call keyword arguments (line 1137)
    kwargs_56205 = {}
    # Getting the type of 'min' (line 1137)
    min_56197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 11), 'min', False)
    # Calling min(args, kwargs) (line 1137)
    min_call_result_56206 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 11), min_56197, *[list_56204], **kwargs_56205)
    
    # Assigning a type to the variable 'miny' (line 1137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1137, 4), 'miny', min_call_result_56206)
    
    # Assigning a Call to a Name (line 1138):
    
    # Assigning a Call to a Name (line 1138):
    
    # Call to max(...): (line 1138)
    # Processing the call arguments (line 1138)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'highs' (line 1138)
    highs_56212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 33), 'highs', False)
    comprehension_56213 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1138, 16), highs_56212)
    # Assigning a type to the variable 'high' (line 1138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1138, 16), 'high', comprehension_56213)
    
    # Getting the type of 'high' (line 1138)
    high_56209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 42), 'high', False)
    int_56210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1138, 50), 'int')
    # Applying the binary operator '!=' (line 1138)
    result_ne_56211 = python_operator(stypy.reporting.localization.Localization(__file__, 1138, 42), '!=', high_56209, int_56210)
    
    # Getting the type of 'high' (line 1138)
    high_56208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 16), 'high', False)
    list_56214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1138, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1138, 16), list_56214, high_56208)
    # Processing the call keyword arguments (line 1138)
    kwargs_56215 = {}
    # Getting the type of 'max' (line 1138)
    max_56207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 11), 'max', False)
    # Calling max(args, kwargs) (line 1138)
    max_call_result_56216 = invoke(stypy.reporting.localization.Localization(__file__, 1138, 11), max_56207, *[list_56214], **kwargs_56215)
    
    # Assigning a type to the variable 'maxy' (line 1138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1138, 4), 'maxy', max_call_result_56216)
    
    # Assigning a Tuple to a Name (line 1140):
    
    # Assigning a Tuple to a Name (line 1140):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1140)
    tuple_56217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1140, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1140)
    # Adding element type (line 1140)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1140)
    tuple_56218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1140, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1140)
    # Adding element type (line 1140)
    # Getting the type of 'minx' (line 1140)
    minx_56219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 15), 'minx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1140, 15), tuple_56218, minx_56219)
    # Adding element type (line 1140)
    # Getting the type of 'miny' (line 1140)
    miny_56220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 21), 'miny')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1140, 15), tuple_56218, miny_56220)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1140, 14), tuple_56217, tuple_56218)
    # Adding element type (line 1140)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1140)
    tuple_56221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1140, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1140)
    # Adding element type (line 1140)
    # Getting the type of 'maxx' (line 1140)
    maxx_56222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 29), 'maxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1140, 29), tuple_56221, maxx_56222)
    # Adding element type (line 1140)
    # Getting the type of 'maxy' (line 1140)
    maxy_56223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 35), 'maxy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1140, 29), tuple_56221, maxy_56223)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1140, 14), tuple_56217, tuple_56221)
    
    # Assigning a type to the variable 'corners' (line 1140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1140, 4), 'corners', tuple_56217)
    
    # Call to update_datalim(...): (line 1141)
    # Processing the call arguments (line 1141)
    # Getting the type of 'corners' (line 1141)
    corners_56226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 22), 'corners', False)
    # Processing the call keyword arguments (line 1141)
    kwargs_56227 = {}
    # Getting the type of 'ax' (line 1141)
    ax_56224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 4), 'ax', False)
    # Obtaining the member 'update_datalim' of a type (line 1141)
    update_datalim_56225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1141, 4), ax_56224, 'update_datalim')
    # Calling update_datalim(args, kwargs) (line 1141)
    update_datalim_call_result_56228 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 4), update_datalim_56225, *[corners_56226], **kwargs_56227)
    
    
    # Call to autoscale_view(...): (line 1142)
    # Processing the call keyword arguments (line 1142)
    kwargs_56231 = {}
    # Getting the type of 'ax' (line 1142)
    ax_56229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 4), 'ax', False)
    # Obtaining the member 'autoscale_view' of a type (line 1142)
    autoscale_view_56230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1142, 4), ax_56229, 'autoscale_view')
    # Calling autoscale_view(args, kwargs) (line 1142)
    autoscale_view_call_result_56232 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 4), autoscale_view_56230, *[], **kwargs_56231)
    
    
    # Call to add_collection(...): (line 1145)
    # Processing the call arguments (line 1145)
    # Getting the type of 'rangeCollection' (line 1145)
    rangeCollection_56235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 22), 'rangeCollection', False)
    # Processing the call keyword arguments (line 1145)
    kwargs_56236 = {}
    # Getting the type of 'ax' (line 1145)
    ax_56233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 1145)
    add_collection_56234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1145, 4), ax_56233, 'add_collection')
    # Calling add_collection(args, kwargs) (line 1145)
    add_collection_call_result_56237 = invoke(stypy.reporting.localization.Localization(__file__, 1145, 4), add_collection_56234, *[rangeCollection_56235], **kwargs_56236)
    
    
    # Call to add_collection(...): (line 1146)
    # Processing the call arguments (line 1146)
    # Getting the type of 'barCollection' (line 1146)
    barCollection_56240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 22), 'barCollection', False)
    # Processing the call keyword arguments (line 1146)
    kwargs_56241 = {}
    # Getting the type of 'ax' (line 1146)
    ax_56238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 1146)
    add_collection_56239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 4), ax_56238, 'add_collection')
    # Calling add_collection(args, kwargs) (line 1146)
    add_collection_call_result_56242 = invoke(stypy.reporting.localization.Localization(__file__, 1146, 4), add_collection_56239, *[barCollection_56240], **kwargs_56241)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1147)
    tuple_56243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1147, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1147)
    # Adding element type (line 1147)
    # Getting the type of 'rangeCollection' (line 1147)
    rangeCollection_56244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 11), 'rangeCollection')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1147, 11), tuple_56243, rangeCollection_56244)
    # Adding element type (line 1147)
    # Getting the type of 'barCollection' (line 1147)
    barCollection_56245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 28), 'barCollection')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1147, 11), tuple_56243, barCollection_56245)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1147, 4), 'stypy_return_type', tuple_56243)
    
    # ################# End of 'candlestick2_ohlc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'candlestick2_ohlc' in the type store
    # Getting the type of 'stypy_return_type' (line 1062)
    stypy_return_type_56246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_56246)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'candlestick2_ohlc'
    return stypy_return_type_56246

# Assigning a type to the variable 'candlestick2_ohlc' (line 1062)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1062, 0), 'candlestick2_ohlc', candlestick2_ohlc)

@norecursion
def volume_overlay(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_56247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 27), 'unicode', u'k')
    unicode_56248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 42), 'unicode', u'r')
    int_56249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1152, 25), 'int')
    float_56250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1152, 34), 'float')
    defaults = [unicode_56247, unicode_56248, int_56249, float_56250]
    # Create a new context for function 'volume_overlay'
    module_type_store = module_type_store.open_function_context('volume_overlay', 1150, 0, False)
    
    # Passed parameters checking function
    volume_overlay.stypy_localization = localization
    volume_overlay.stypy_type_of_self = None
    volume_overlay.stypy_type_store = module_type_store
    volume_overlay.stypy_function_name = 'volume_overlay'
    volume_overlay.stypy_param_names_list = ['ax', 'opens', 'closes', 'volumes', 'colorup', 'colordown', 'width', 'alpha']
    volume_overlay.stypy_varargs_param_name = None
    volume_overlay.stypy_kwargs_param_name = None
    volume_overlay.stypy_call_defaults = defaults
    volume_overlay.stypy_call_varargs = varargs
    volume_overlay.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'volume_overlay', ['ax', 'opens', 'closes', 'volumes', 'colorup', 'colordown', 'width', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'volume_overlay', localization, ['ax', 'opens', 'closes', 'volumes', 'colorup', 'colordown', 'width', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'volume_overlay(...)' code ##################

    unicode_56251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1181, (-1)), 'unicode', u'Add a volume overlay to the current axes.  The opens and closes\n    are used to determine the color of the bar.  -1 is missing.  If a\n    value is missing on one it must be missing on all\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    opens : sequence\n        a sequence of opens\n    closes : sequence\n        a sequence of closes\n    volumes : sequence\n        a sequence of volumes\n    width : int\n        the bar width in points\n    colorup : color\n        the color of the lines where close >= open\n    colordown : color\n        the color of the lines where close <  open\n    alpha : float\n        bar transparency\n\n    Returns\n    -------\n    ret : `barCollection`\n        The `barrCollection` added to the axes\n\n    ')
    
    # Assigning a Call to a Name (line 1183):
    
    # Assigning a Call to a Name (line 1183):
    
    # Call to to_rgba(...): (line 1183)
    # Processing the call arguments (line 1183)
    # Getting the type of 'colorup' (line 1183)
    colorup_56254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 30), 'colorup', False)
    # Getting the type of 'alpha' (line 1183)
    alpha_56255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 39), 'alpha', False)
    # Processing the call keyword arguments (line 1183)
    kwargs_56256 = {}
    # Getting the type of 'mcolors' (line 1183)
    mcolors_56252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 14), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 1183)
    to_rgba_56253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1183, 14), mcolors_56252, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 1183)
    to_rgba_call_result_56257 = invoke(stypy.reporting.localization.Localization(__file__, 1183, 14), to_rgba_56253, *[colorup_56254, alpha_56255], **kwargs_56256)
    
    # Assigning a type to the variable 'colorup' (line 1183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1183, 4), 'colorup', to_rgba_call_result_56257)
    
    # Assigning a Call to a Name (line 1184):
    
    # Assigning a Call to a Name (line 1184):
    
    # Call to to_rgba(...): (line 1184)
    # Processing the call arguments (line 1184)
    # Getting the type of 'colordown' (line 1184)
    colordown_56260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 32), 'colordown', False)
    # Getting the type of 'alpha' (line 1184)
    alpha_56261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 43), 'alpha', False)
    # Processing the call keyword arguments (line 1184)
    kwargs_56262 = {}
    # Getting the type of 'mcolors' (line 1184)
    mcolors_56258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 16), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 1184)
    to_rgba_56259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1184, 16), mcolors_56258, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 1184)
    to_rgba_call_result_56263 = invoke(stypy.reporting.localization.Localization(__file__, 1184, 16), to_rgba_56259, *[colordown_56260, alpha_56261], **kwargs_56262)
    
    # Assigning a type to the variable 'colordown' (line 1184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'colordown', to_rgba_call_result_56263)
    
    # Assigning a Dict to a Name (line 1185):
    
    # Assigning a Dict to a Name (line 1185):
    
    # Obtaining an instance of the builtin type 'dict' (line 1185)
    dict_56264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1185)
    # Adding element type (key, value) (line 1185)
    # Getting the type of 'True' (line 1185)
    True_56265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 14), 'True')
    # Getting the type of 'colorup' (line 1185)
    colorup_56266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 20), 'colorup')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 13), dict_56264, (True_56265, colorup_56266))
    # Adding element type (key, value) (line 1185)
    # Getting the type of 'False' (line 1185)
    False_56267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 29), 'False')
    # Getting the type of 'colordown' (line 1185)
    colordown_56268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 36), 'colordown')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1185, 13), dict_56264, (False_56267, colordown_56268))
    
    # Assigning a type to the variable 'colord' (line 1185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'colord', dict_56264)
    
    # Assigning a ListComp to a Name (line 1186):
    
    # Assigning a ListComp to a Name (line 1186):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1187)
    # Processing the call arguments (line 1187)
    # Getting the type of 'opens' (line 1187)
    opens_56283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 37), 'opens', False)
    # Getting the type of 'closes' (line 1187)
    closes_56284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 44), 'closes', False)
    # Processing the call keyword arguments (line 1187)
    kwargs_56285 = {}
    # Getting the type of 'zip' (line 1187)
    zip_56282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 33), 'zip', False)
    # Calling zip(args, kwargs) (line 1187)
    zip_call_result_56286 = invoke(stypy.reporting.localization.Localization(__file__, 1187, 33), zip_56282, *[opens_56283, closes_56284], **kwargs_56285)
    
    comprehension_56287 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1186, 14), zip_call_result_56286)
    # Assigning a type to the variable 'open' (line 1186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 14), 'open', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1186, 14), comprehension_56287))
    # Assigning a type to the variable 'close' (line 1186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 14), 'close', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1186, 14), comprehension_56287))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'open' (line 1188)
    open_56275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 17), 'open')
    int_56276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1188, 25), 'int')
    # Applying the binary operator '!=' (line 1188)
    result_ne_56277 = python_operator(stypy.reporting.localization.Localization(__file__, 1188, 17), '!=', open_56275, int_56276)
    
    
    # Getting the type of 'close' (line 1188)
    close_56278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 32), 'close')
    int_56279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1188, 41), 'int')
    # Applying the binary operator '!=' (line 1188)
    result_ne_56280 = python_operator(stypy.reporting.localization.Localization(__file__, 1188, 32), '!=', close_56278, int_56279)
    
    # Applying the binary operator 'and' (line 1188)
    result_and_keyword_56281 = python_operator(stypy.reporting.localization.Localization(__file__, 1188, 17), 'and', result_ne_56277, result_ne_56280)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'open' (line 1186)
    open_56269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 21), 'open')
    # Getting the type of 'close' (line 1186)
    close_56270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 28), 'close')
    # Applying the binary operator '<' (line 1186)
    result_lt_56271 = python_operator(stypy.reporting.localization.Localization(__file__, 1186, 21), '<', open_56269, close_56270)
    
    # Getting the type of 'colord' (line 1186)
    colord_56272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 14), 'colord')
    # Obtaining the member '__getitem__' of a type (line 1186)
    getitem___56273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1186, 14), colord_56272, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1186)
    subscript_call_result_56274 = invoke(stypy.reporting.localization.Localization(__file__, 1186, 14), getitem___56273, result_lt_56271)
    
    list_56288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1186, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1186, 14), list_56288, subscript_call_result_56274)
    # Assigning a type to the variable 'colors' (line 1186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 4), 'colors', list_56288)
    
    # Assigning a BinOp to a Name (line 1190):
    
    # Assigning a BinOp to a Name (line 1190):
    # Getting the type of 'width' (line 1190)
    width_56289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 12), 'width')
    float_56290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1190, 20), 'float')
    # Applying the binary operator 'div' (line 1190)
    result_div_56291 = python_operator(stypy.reporting.localization.Localization(__file__, 1190, 12), 'div', width_56289, float_56290)
    
    # Assigning a type to the variable 'delta' (line 1190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1190, 4), 'delta', result_div_56291)
    
    # Assigning a ListComp to a Name (line 1191):
    
    # Assigning a ListComp to a Name (line 1191):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 1192)
    # Processing the call arguments (line 1192)
    # Getting the type of 'volumes' (line 1192)
    volumes_56317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 34), 'volumes', False)
    # Processing the call keyword arguments (line 1192)
    kwargs_56318 = {}
    # Getting the type of 'enumerate' (line 1192)
    enumerate_56316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 24), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1192)
    enumerate_call_result_56319 = invoke(stypy.reporting.localization.Localization(__file__, 1192, 24), enumerate_56316, *[volumes_56317], **kwargs_56318)
    
    comprehension_56320 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 12), enumerate_call_result_56319)
    # Assigning a type to the variable 'i' (line 1191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1191, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 12), comprehension_56320))
    # Assigning a type to the variable 'v' (line 1191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1191, 12), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 12), comprehension_56320))
    
    # Getting the type of 'v' (line 1193)
    v_56313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 15), 'v')
    int_56314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1193, 20), 'int')
    # Applying the binary operator '!=' (line 1193)
    result_ne_56315 = python_operator(stypy.reporting.localization.Localization(__file__, 1193, 15), '!=', v_56313, int_56314)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1191)
    tuple_56292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1191)
    # Adding element type (line 1191)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1191)
    tuple_56293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1191)
    # Adding element type (line 1191)
    # Getting the type of 'i' (line 1191)
    i_56294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 14), 'i')
    # Getting the type of 'delta' (line 1191)
    delta_56295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 18), 'delta')
    # Applying the binary operator '-' (line 1191)
    result_sub_56296 = python_operator(stypy.reporting.localization.Localization(__file__, 1191, 14), '-', i_56294, delta_56295)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 14), tuple_56293, result_sub_56296)
    # Adding element type (line 1191)
    int_56297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 14), tuple_56293, int_56297)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 13), tuple_56292, tuple_56293)
    # Adding element type (line 1191)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1191)
    tuple_56298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1191)
    # Adding element type (line 1191)
    # Getting the type of 'i' (line 1191)
    i_56299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 30), 'i')
    # Getting the type of 'delta' (line 1191)
    delta_56300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 34), 'delta')
    # Applying the binary operator '-' (line 1191)
    result_sub_56301 = python_operator(stypy.reporting.localization.Localization(__file__, 1191, 30), '-', i_56299, delta_56300)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 30), tuple_56298, result_sub_56301)
    # Adding element type (line 1191)
    # Getting the type of 'v' (line 1191)
    v_56302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 41), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 30), tuple_56298, v_56302)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 13), tuple_56292, tuple_56298)
    # Adding element type (line 1191)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1191)
    tuple_56303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1191)
    # Adding element type (line 1191)
    # Getting the type of 'i' (line 1191)
    i_56304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 46), 'i')
    # Getting the type of 'delta' (line 1191)
    delta_56305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 50), 'delta')
    # Applying the binary operator '+' (line 1191)
    result_add_56306 = python_operator(stypy.reporting.localization.Localization(__file__, 1191, 46), '+', i_56304, delta_56305)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 46), tuple_56303, result_add_56306)
    # Adding element type (line 1191)
    # Getting the type of 'v' (line 1191)
    v_56307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 57), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 46), tuple_56303, v_56307)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 13), tuple_56292, tuple_56303)
    # Adding element type (line 1191)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1191)
    tuple_56308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 62), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1191)
    # Adding element type (line 1191)
    # Getting the type of 'i' (line 1191)
    i_56309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 62), 'i')
    # Getting the type of 'delta' (line 1191)
    delta_56310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 66), 'delta')
    # Applying the binary operator '+' (line 1191)
    result_add_56311 = python_operator(stypy.reporting.localization.Localization(__file__, 1191, 62), '+', i_56309, delta_56310)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 62), tuple_56308, result_add_56311)
    # Adding element type (line 1191)
    int_56312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 73), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 62), tuple_56308, int_56312)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 13), tuple_56292, tuple_56308)
    
    list_56321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 12), list_56321, tuple_56292)
    # Assigning a type to the variable 'bars' (line 1191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1191, 4), 'bars', list_56321)
    
    # Assigning a Call to a Name (line 1195):
    
    # Assigning a Call to a Name (line 1195):
    
    # Call to PolyCollection(...): (line 1195)
    # Processing the call arguments (line 1195)
    # Getting the type of 'bars' (line 1195)
    bars_56323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 35), 'bars', False)
    # Processing the call keyword arguments (line 1195)
    # Getting the type of 'colors' (line 1196)
    colors_56324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 46), 'colors', False)
    keyword_56325 = colors_56324
    
    # Obtaining an instance of the builtin type 'tuple' (line 1197)
    tuple_56326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1197)
    # Adding element type (line 1197)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1197)
    tuple_56327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1197)
    # Adding element type (line 1197)
    int_56328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1197, 48), tuple_56327, int_56328)
    # Adding element type (line 1197)
    int_56329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1197, 48), tuple_56327, int_56329)
    # Adding element type (line 1197)
    int_56330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1197, 48), tuple_56327, int_56330)
    # Adding element type (line 1197)
    int_56331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1197, 48), tuple_56327, int_56331)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1197, 47), tuple_56326, tuple_56327)
    
    keyword_56332 = tuple_56326
    
    # Obtaining an instance of the builtin type 'tuple' (line 1198)
    tuple_56333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1198, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1198)
    # Adding element type (line 1198)
    int_56334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1198, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1198, 49), tuple_56333, int_56334)
    
    keyword_56335 = tuple_56333
    
    # Obtaining an instance of the builtin type 'tuple' (line 1199)
    tuple_56336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1199, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1199)
    # Adding element type (line 1199)
    float_56337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1199, 47), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1199, 47), tuple_56336, float_56337)
    
    keyword_56338 = tuple_56336
    kwargs_56339 = {'edgecolors': keyword_56332, 'antialiaseds': keyword_56335, 'facecolors': keyword_56325, 'linewidths': keyword_56338}
    # Getting the type of 'PolyCollection' (line 1195)
    PolyCollection_56322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 20), 'PolyCollection', False)
    # Calling PolyCollection(args, kwargs) (line 1195)
    PolyCollection_call_result_56340 = invoke(stypy.reporting.localization.Localization(__file__, 1195, 20), PolyCollection_56322, *[bars_56323], **kwargs_56339)
    
    # Assigning a type to the variable 'barCollection' (line 1195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1195, 4), 'barCollection', PolyCollection_call_result_56340)
    
    # Call to add_collection(...): (line 1202)
    # Processing the call arguments (line 1202)
    # Getting the type of 'barCollection' (line 1202)
    barCollection_56343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 22), 'barCollection', False)
    # Processing the call keyword arguments (line 1202)
    kwargs_56344 = {}
    # Getting the type of 'ax' (line 1202)
    ax_56341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 1202)
    add_collection_56342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1202, 4), ax_56341, 'add_collection')
    # Calling add_collection(args, kwargs) (line 1202)
    add_collection_call_result_56345 = invoke(stypy.reporting.localization.Localization(__file__, 1202, 4), add_collection_56342, *[barCollection_56343], **kwargs_56344)
    
    
    # Assigning a Tuple to a Name (line 1203):
    
    # Assigning a Tuple to a Name (line 1203):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1203)
    tuple_56346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1203)
    # Adding element type (line 1203)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1203)
    tuple_56347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1203)
    # Adding element type (line 1203)
    int_56348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1203, 15), tuple_56347, int_56348)
    # Adding element type (line 1203)
    int_56349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1203, 15), tuple_56347, int_56349)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1203, 14), tuple_56346, tuple_56347)
    # Adding element type (line 1203)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1203)
    tuple_56350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1203)
    # Adding element type (line 1203)
    
    # Call to len(...): (line 1203)
    # Processing the call arguments (line 1203)
    # Getting the type of 'bars' (line 1203)
    bars_56352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 27), 'bars', False)
    # Processing the call keyword arguments (line 1203)
    kwargs_56353 = {}
    # Getting the type of 'len' (line 1203)
    len_56351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 23), 'len', False)
    # Calling len(args, kwargs) (line 1203)
    len_call_result_56354 = invoke(stypy.reporting.localization.Localization(__file__, 1203, 23), len_56351, *[bars_56352], **kwargs_56353)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1203, 23), tuple_56350, len_call_result_56354)
    # Adding element type (line 1203)
    
    # Call to max(...): (line 1203)
    # Processing the call arguments (line 1203)
    # Getting the type of 'volumes' (line 1203)
    volumes_56356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 38), 'volumes', False)
    # Processing the call keyword arguments (line 1203)
    kwargs_56357 = {}
    # Getting the type of 'max' (line 1203)
    max_56355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 34), 'max', False)
    # Calling max(args, kwargs) (line 1203)
    max_call_result_56358 = invoke(stypy.reporting.localization.Localization(__file__, 1203, 34), max_56355, *[volumes_56356], **kwargs_56357)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1203, 23), tuple_56350, max_call_result_56358)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1203, 14), tuple_56346, tuple_56350)
    
    # Assigning a type to the variable 'corners' (line 1203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1203, 4), 'corners', tuple_56346)
    
    # Call to update_datalim(...): (line 1204)
    # Processing the call arguments (line 1204)
    # Getting the type of 'corners' (line 1204)
    corners_56361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 22), 'corners', False)
    # Processing the call keyword arguments (line 1204)
    kwargs_56362 = {}
    # Getting the type of 'ax' (line 1204)
    ax_56359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 4), 'ax', False)
    # Obtaining the member 'update_datalim' of a type (line 1204)
    update_datalim_56360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1204, 4), ax_56359, 'update_datalim')
    # Calling update_datalim(args, kwargs) (line 1204)
    update_datalim_call_result_56363 = invoke(stypy.reporting.localization.Localization(__file__, 1204, 4), update_datalim_56360, *[corners_56361], **kwargs_56362)
    
    
    # Call to autoscale_view(...): (line 1205)
    # Processing the call keyword arguments (line 1205)
    kwargs_56366 = {}
    # Getting the type of 'ax' (line 1205)
    ax_56364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 4), 'ax', False)
    # Obtaining the member 'autoscale_view' of a type (line 1205)
    autoscale_view_56365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 4), ax_56364, 'autoscale_view')
    # Calling autoscale_view(args, kwargs) (line 1205)
    autoscale_view_call_result_56367 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 4), autoscale_view_56365, *[], **kwargs_56366)
    
    # Getting the type of 'barCollection' (line 1208)
    barCollection_56368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 11), 'barCollection')
    # Assigning a type to the variable 'stypy_return_type' (line 1208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1208, 4), 'stypy_return_type', barCollection_56368)
    
    # ################# End of 'volume_overlay(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'volume_overlay' in the type store
    # Getting the type of 'stypy_return_type' (line 1150)
    stypy_return_type_56369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_56369)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'volume_overlay'
    return stypy_return_type_56369

# Assigning a type to the variable 'volume_overlay' (line 1150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1150, 0), 'volume_overlay', volume_overlay)

@norecursion
def volume_overlay2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_56370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1212, 28), 'unicode', u'k')
    unicode_56371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1212, 43), 'unicode', u'r')
    int_56372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 26), 'int')
    float_56373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 35), 'float')
    defaults = [unicode_56370, unicode_56371, int_56372, float_56373]
    # Create a new context for function 'volume_overlay2'
    module_type_store = module_type_store.open_function_context('volume_overlay2', 1211, 0, False)
    
    # Passed parameters checking function
    volume_overlay2.stypy_localization = localization
    volume_overlay2.stypy_type_of_self = None
    volume_overlay2.stypy_type_store = module_type_store
    volume_overlay2.stypy_function_name = 'volume_overlay2'
    volume_overlay2.stypy_param_names_list = ['ax', 'closes', 'volumes', 'colorup', 'colordown', 'width', 'alpha']
    volume_overlay2.stypy_varargs_param_name = None
    volume_overlay2.stypy_kwargs_param_name = None
    volume_overlay2.stypy_call_defaults = defaults
    volume_overlay2.stypy_call_varargs = varargs
    volume_overlay2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'volume_overlay2', ['ax', 'closes', 'volumes', 'colorup', 'colordown', 'width', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'volume_overlay2', localization, ['ax', 'closes', 'volumes', 'colorup', 'colordown', 'width', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'volume_overlay2(...)' code ##################

    unicode_56374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1245, (-1)), 'unicode', u'\n    Add a volume overlay to the current axes.  The closes are used to\n    determine the color of the bar.  -1 is missing.  If a value is\n    missing on one it must be missing on all\n\n    nb: first point is not displayed - it is used only for choosing the\n    right color\n\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    closes : sequence\n        a sequence of closes\n    volumes : sequence\n        a sequence of volumes\n    width : int\n        the bar width in points\n    colorup : color\n        the color of the lines where close >= open\n    colordown : color\n        the color of the lines where close <  open\n    alpha : float\n        bar transparency\n\n    Returns\n    -------\n    ret : `barCollection`\n        The `barrCollection` added to the axes\n\n    ')
    
    # Call to volume_overlay(...): (line 1247)
    # Processing the call arguments (line 1247)
    # Getting the type of 'ax' (line 1247)
    ax_56376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 26), 'ax', False)
    
    # Obtaining the type of the subscript
    int_56377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1247, 38), 'int')
    slice_56378 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1247, 30), None, int_56377, None)
    # Getting the type of 'closes' (line 1247)
    closes_56379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 30), 'closes', False)
    # Obtaining the member '__getitem__' of a type (line 1247)
    getitem___56380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 30), closes_56379, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1247)
    subscript_call_result_56381 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 30), getitem___56380, slice_56378)
    
    
    # Obtaining the type of the subscript
    int_56382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1247, 50), 'int')
    slice_56383 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1247, 43), int_56382, None, None)
    # Getting the type of 'closes' (line 1247)
    closes_56384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 43), 'closes', False)
    # Obtaining the member '__getitem__' of a type (line 1247)
    getitem___56385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 43), closes_56384, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1247)
    subscript_call_result_56386 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 43), getitem___56385, slice_56383)
    
    
    # Obtaining the type of the subscript
    int_56387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1247, 63), 'int')
    slice_56388 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1247, 55), int_56387, None, None)
    # Getting the type of 'volumes' (line 1247)
    volumes_56389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 55), 'volumes', False)
    # Obtaining the member '__getitem__' of a type (line 1247)
    getitem___56390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 55), volumes_56389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1247)
    subscript_call_result_56391 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 55), getitem___56390, slice_56388)
    
    # Getting the type of 'colorup' (line 1248)
    colorup_56392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 26), 'colorup', False)
    # Getting the type of 'colordown' (line 1248)
    colordown_56393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 35), 'colordown', False)
    # Getting the type of 'width' (line 1248)
    width_56394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 46), 'width', False)
    # Getting the type of 'alpha' (line 1248)
    alpha_56395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 53), 'alpha', False)
    # Processing the call keyword arguments (line 1247)
    kwargs_56396 = {}
    # Getting the type of 'volume_overlay' (line 1247)
    volume_overlay_56375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 11), 'volume_overlay', False)
    # Calling volume_overlay(args, kwargs) (line 1247)
    volume_overlay_call_result_56397 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 11), volume_overlay_56375, *[ax_56376, subscript_call_result_56381, subscript_call_result_56386, subscript_call_result_56391, colorup_56392, colordown_56393, width_56394, alpha_56395], **kwargs_56396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1247, 4), 'stypy_return_type', volume_overlay_call_result_56397)
    
    # ################# End of 'volume_overlay2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'volume_overlay2' in the type store
    # Getting the type of 'stypy_return_type' (line 1211)
    stypy_return_type_56398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_56398)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'volume_overlay2'
    return stypy_return_type_56398

# Assigning a type to the variable 'volume_overlay2' (line 1211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 0), 'volume_overlay2', volume_overlay2)

@norecursion
def volume_overlay3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_56399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1252, 28), 'unicode', u'k')
    unicode_56400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1252, 43), 'unicode', u'r')
    int_56401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1253, 26), 'int')
    float_56402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1253, 35), 'float')
    defaults = [unicode_56399, unicode_56400, int_56401, float_56402]
    # Create a new context for function 'volume_overlay3'
    module_type_store = module_type_store.open_function_context('volume_overlay3', 1251, 0, False)
    
    # Passed parameters checking function
    volume_overlay3.stypy_localization = localization
    volume_overlay3.stypy_type_of_self = None
    volume_overlay3.stypy_type_store = module_type_store
    volume_overlay3.stypy_function_name = 'volume_overlay3'
    volume_overlay3.stypy_param_names_list = ['ax', 'quotes', 'colorup', 'colordown', 'width', 'alpha']
    volume_overlay3.stypy_varargs_param_name = None
    volume_overlay3.stypy_kwargs_param_name = None
    volume_overlay3.stypy_call_defaults = defaults
    volume_overlay3.stypy_call_varargs = varargs
    volume_overlay3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'volume_overlay3', ['ax', 'quotes', 'colorup', 'colordown', 'width', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'volume_overlay3', localization, ['ax', 'quotes', 'colorup', 'colordown', 'width', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'volume_overlay3(...)' code ##################

    unicode_56403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1279, (-1)), 'unicode', u'Add a volume overlay to the current axes.  quotes is a list of (d,\n    open, high, low, close, volume) and close-open is used to\n    determine the color of the bar\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    quotes : sequence of (time, open, high, low, close, ...) sequences\n        data to plot.  time must be in float date format - see date2num\n    width : int\n        the bar width in points\n    colorup : color\n        the color of the lines where close1 >= close0\n    colordown : color\n        the color of the lines where close1 <  close0\n    alpha : float\n         bar transparency\n\n    Returns\n    -------\n    ret : `barCollection`\n        The `barrCollection` added to the axes\n\n\n    ')
    
    # Assigning a Call to a Name (line 1281):
    
    # Assigning a Call to a Name (line 1281):
    
    # Call to to_rgba(...): (line 1281)
    # Processing the call arguments (line 1281)
    # Getting the type of 'colorup' (line 1281)
    colorup_56406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1281, 30), 'colorup', False)
    # Getting the type of 'alpha' (line 1281)
    alpha_56407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1281, 39), 'alpha', False)
    # Processing the call keyword arguments (line 1281)
    kwargs_56408 = {}
    # Getting the type of 'mcolors' (line 1281)
    mcolors_56404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1281, 14), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 1281)
    to_rgba_56405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1281, 14), mcolors_56404, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 1281)
    to_rgba_call_result_56409 = invoke(stypy.reporting.localization.Localization(__file__, 1281, 14), to_rgba_56405, *[colorup_56406, alpha_56407], **kwargs_56408)
    
    # Assigning a type to the variable 'colorup' (line 1281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1281, 4), 'colorup', to_rgba_call_result_56409)
    
    # Assigning a Call to a Name (line 1282):
    
    # Assigning a Call to a Name (line 1282):
    
    # Call to to_rgba(...): (line 1282)
    # Processing the call arguments (line 1282)
    # Getting the type of 'colordown' (line 1282)
    colordown_56412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 32), 'colordown', False)
    # Getting the type of 'alpha' (line 1282)
    alpha_56413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 43), 'alpha', False)
    # Processing the call keyword arguments (line 1282)
    kwargs_56414 = {}
    # Getting the type of 'mcolors' (line 1282)
    mcolors_56410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 16), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 1282)
    to_rgba_56411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1282, 16), mcolors_56410, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 1282)
    to_rgba_call_result_56415 = invoke(stypy.reporting.localization.Localization(__file__, 1282, 16), to_rgba_56411, *[colordown_56412, alpha_56413], **kwargs_56414)
    
    # Assigning a type to the variable 'colordown' (line 1282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1282, 4), 'colordown', to_rgba_call_result_56415)
    
    # Assigning a Dict to a Name (line 1283):
    
    # Assigning a Dict to a Name (line 1283):
    
    # Obtaining an instance of the builtin type 'dict' (line 1283)
    dict_56416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1283, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1283)
    # Adding element type (key, value) (line 1283)
    # Getting the type of 'True' (line 1283)
    True_56417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 14), 'True')
    # Getting the type of 'colorup' (line 1283)
    colorup_56418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 20), 'colorup')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1283, 13), dict_56416, (True_56417, colorup_56418))
    # Adding element type (key, value) (line 1283)
    # Getting the type of 'False' (line 1283)
    False_56419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 29), 'False')
    # Getting the type of 'colordown' (line 1283)
    colordown_56420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 36), 'colordown')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1283, 13), dict_56416, (False_56419, colordown_56420))
    
    # Assigning a type to the variable 'colord' (line 1283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1283, 4), 'colord', dict_56416)
    
    # Assigning a Call to a Tuple (line 1285):
    
    # Assigning a Call to a Name:
    
    # Call to list(...): (line 1285)
    # Processing the call arguments (line 1285)
    
    # Call to zip(...): (line 1285)
    # Getting the type of 'quotes' (line 1285)
    quotes_56423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 59), 'quotes', False)
    # Processing the call keyword arguments (line 1285)
    kwargs_56424 = {}
    # Getting the type of 'zip' (line 1285)
    zip_56422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 54), 'zip', False)
    # Calling zip(args, kwargs) (line 1285)
    zip_call_result_56425 = invoke(stypy.reporting.localization.Localization(__file__, 1285, 54), zip_56422, *[quotes_56423], **kwargs_56424)
    
    # Processing the call keyword arguments (line 1285)
    kwargs_56426 = {}
    # Getting the type of 'list' (line 1285)
    list_56421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 49), 'list', False)
    # Calling list(args, kwargs) (line 1285)
    list_call_result_56427 = invoke(stypy.reporting.localization.Localization(__file__, 1285, 49), list_56421, *[zip_call_result_56425], **kwargs_56426)
    
    # Assigning a type to the variable 'call_assignment_54322' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54322', list_call_result_56427)
    
    # Assigning a Call to a Name (line 1285):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_56430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1285, 4), 'int')
    # Processing the call keyword arguments
    kwargs_56431 = {}
    # Getting the type of 'call_assignment_54322' (line 1285)
    call_assignment_54322_56428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54322', False)
    # Obtaining the member '__getitem__' of a type (line 1285)
    getitem___56429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1285, 4), call_assignment_54322_56428, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_56432 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56429, *[int_56430], **kwargs_56431)
    
    # Assigning a type to the variable 'call_assignment_54323' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54323', getitem___call_result_56432)
    
    # Assigning a Name to a Name (line 1285):
    # Getting the type of 'call_assignment_54323' (line 1285)
    call_assignment_54323_56433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54323')
    # Assigning a type to the variable 'dates' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'dates', call_assignment_54323_56433)
    
    # Assigning a Call to a Name (line 1285):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_56436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1285, 4), 'int')
    # Processing the call keyword arguments
    kwargs_56437 = {}
    # Getting the type of 'call_assignment_54322' (line 1285)
    call_assignment_54322_56434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54322', False)
    # Obtaining the member '__getitem__' of a type (line 1285)
    getitem___56435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1285, 4), call_assignment_54322_56434, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_56438 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56435, *[int_56436], **kwargs_56437)
    
    # Assigning a type to the variable 'call_assignment_54324' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54324', getitem___call_result_56438)
    
    # Assigning a Name to a Name (line 1285):
    # Getting the type of 'call_assignment_54324' (line 1285)
    call_assignment_54324_56439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54324')
    # Assigning a type to the variable 'opens' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 11), 'opens', call_assignment_54324_56439)
    
    # Assigning a Call to a Name (line 1285):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_56442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1285, 4), 'int')
    # Processing the call keyword arguments
    kwargs_56443 = {}
    # Getting the type of 'call_assignment_54322' (line 1285)
    call_assignment_54322_56440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54322', False)
    # Obtaining the member '__getitem__' of a type (line 1285)
    getitem___56441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1285, 4), call_assignment_54322_56440, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_56444 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56441, *[int_56442], **kwargs_56443)
    
    # Assigning a type to the variable 'call_assignment_54325' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54325', getitem___call_result_56444)
    
    # Assigning a Name to a Name (line 1285):
    # Getting the type of 'call_assignment_54325' (line 1285)
    call_assignment_54325_56445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54325')
    # Assigning a type to the variable 'highs' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 18), 'highs', call_assignment_54325_56445)
    
    # Assigning a Call to a Name (line 1285):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_56448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1285, 4), 'int')
    # Processing the call keyword arguments
    kwargs_56449 = {}
    # Getting the type of 'call_assignment_54322' (line 1285)
    call_assignment_54322_56446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54322', False)
    # Obtaining the member '__getitem__' of a type (line 1285)
    getitem___56447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1285, 4), call_assignment_54322_56446, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_56450 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56447, *[int_56448], **kwargs_56449)
    
    # Assigning a type to the variable 'call_assignment_54326' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54326', getitem___call_result_56450)
    
    # Assigning a Name to a Name (line 1285):
    # Getting the type of 'call_assignment_54326' (line 1285)
    call_assignment_54326_56451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54326')
    # Assigning a type to the variable 'lows' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 25), 'lows', call_assignment_54326_56451)
    
    # Assigning a Call to a Name (line 1285):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_56454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1285, 4), 'int')
    # Processing the call keyword arguments
    kwargs_56455 = {}
    # Getting the type of 'call_assignment_54322' (line 1285)
    call_assignment_54322_56452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54322', False)
    # Obtaining the member '__getitem__' of a type (line 1285)
    getitem___56453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1285, 4), call_assignment_54322_56452, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_56456 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56453, *[int_56454], **kwargs_56455)
    
    # Assigning a type to the variable 'call_assignment_54327' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54327', getitem___call_result_56456)
    
    # Assigning a Name to a Name (line 1285):
    # Getting the type of 'call_assignment_54327' (line 1285)
    call_assignment_54327_56457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54327')
    # Assigning a type to the variable 'closes' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 31), 'closes', call_assignment_54327_56457)
    
    # Assigning a Call to a Name (line 1285):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_56460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1285, 4), 'int')
    # Processing the call keyword arguments
    kwargs_56461 = {}
    # Getting the type of 'call_assignment_54322' (line 1285)
    call_assignment_54322_56458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54322', False)
    # Obtaining the member '__getitem__' of a type (line 1285)
    getitem___56459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1285, 4), call_assignment_54322_56458, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_56462 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___56459, *[int_56460], **kwargs_56461)
    
    # Assigning a type to the variable 'call_assignment_54328' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54328', getitem___call_result_56462)
    
    # Assigning a Name to a Name (line 1285):
    # Getting the type of 'call_assignment_54328' (line 1285)
    call_assignment_54328_56463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1285, 4), 'call_assignment_54328')
    # Assigning a type to the variable 'volumes' (line 1285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1285, 39), 'volumes', call_assignment_54328_56463)
    
    # Assigning a ListComp to a Name (line 1286):
    
    # Assigning a ListComp to a Name (line 1286):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1287)
    # Processing the call arguments (line 1287)
    
    # Obtaining the type of the subscript
    int_56478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1287, 48), 'int')
    slice_56479 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1287, 40), None, int_56478, None)
    # Getting the type of 'closes' (line 1287)
    closes_56480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 40), 'closes', False)
    # Obtaining the member '__getitem__' of a type (line 1287)
    getitem___56481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 40), closes_56480, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1287)
    subscript_call_result_56482 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 40), getitem___56481, slice_56479)
    
    
    # Obtaining the type of the subscript
    int_56483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1287, 60), 'int')
    slice_56484 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1287, 53), int_56483, None, None)
    # Getting the type of 'closes' (line 1287)
    closes_56485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 53), 'closes', False)
    # Obtaining the member '__getitem__' of a type (line 1287)
    getitem___56486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 53), closes_56485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1287)
    subscript_call_result_56487 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 53), getitem___56486, slice_56484)
    
    # Processing the call keyword arguments (line 1287)
    kwargs_56488 = {}
    # Getting the type of 'zip' (line 1287)
    zip_56477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 36), 'zip', False)
    # Calling zip(args, kwargs) (line 1287)
    zip_call_result_56489 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 36), zip_56477, *[subscript_call_result_56482, subscript_call_result_56487], **kwargs_56488)
    
    comprehension_56490 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1286, 14), zip_call_result_56489)
    # Assigning a type to the variable 'close0' (line 1286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1286, 14), 'close0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1286, 14), comprehension_56490))
    # Assigning a type to the variable 'close1' (line 1286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1286, 14), 'close1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1286, 14), comprehension_56490))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'close0' (line 1288)
    close0_56470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 17), 'close0')
    int_56471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1288, 27), 'int')
    # Applying the binary operator '!=' (line 1288)
    result_ne_56472 = python_operator(stypy.reporting.localization.Localization(__file__, 1288, 17), '!=', close0_56470, int_56471)
    
    
    # Getting the type of 'close1' (line 1288)
    close1_56473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 34), 'close1')
    int_56474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1288, 44), 'int')
    # Applying the binary operator '!=' (line 1288)
    result_ne_56475 = python_operator(stypy.reporting.localization.Localization(__file__, 1288, 34), '!=', close1_56473, int_56474)
    
    # Applying the binary operator 'and' (line 1288)
    result_and_keyword_56476 = python_operator(stypy.reporting.localization.Localization(__file__, 1288, 17), 'and', result_ne_56472, result_ne_56475)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'close1' (line 1286)
    close1_56464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 21), 'close1')
    # Getting the type of 'close0' (line 1286)
    close0_56465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 31), 'close0')
    # Applying the binary operator '>=' (line 1286)
    result_ge_56466 = python_operator(stypy.reporting.localization.Localization(__file__, 1286, 21), '>=', close1_56464, close0_56465)
    
    # Getting the type of 'colord' (line 1286)
    colord_56467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 14), 'colord')
    # Obtaining the member '__getitem__' of a type (line 1286)
    getitem___56468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1286, 14), colord_56467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1286)
    subscript_call_result_56469 = invoke(stypy.reporting.localization.Localization(__file__, 1286, 14), getitem___56468, result_ge_56466)
    
    list_56491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1286, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1286, 14), list_56491, subscript_call_result_56469)
    # Assigning a type to the variable 'colors' (line 1286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1286, 4), 'colors', list_56491)
    
    # Call to insert(...): (line 1289)
    # Processing the call arguments (line 1289)
    int_56494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 18), 'int')
    
    # Obtaining the type of the subscript
    
    
    # Obtaining the type of the subscript
    int_56495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 35), 'int')
    # Getting the type of 'closes' (line 1289)
    closes_56496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 28), 'closes', False)
    # Obtaining the member '__getitem__' of a type (line 1289)
    getitem___56497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 28), closes_56496, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1289)
    subscript_call_result_56498 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 28), getitem___56497, int_56495)
    
    
    # Obtaining the type of the subscript
    int_56499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 47), 'int')
    # Getting the type of 'opens' (line 1289)
    opens_56500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 41), 'opens', False)
    # Obtaining the member '__getitem__' of a type (line 1289)
    getitem___56501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 41), opens_56500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1289)
    subscript_call_result_56502 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 41), getitem___56501, int_56499)
    
    # Applying the binary operator '>=' (line 1289)
    result_ge_56503 = python_operator(stypy.reporting.localization.Localization(__file__, 1289, 28), '>=', subscript_call_result_56498, subscript_call_result_56502)
    
    # Getting the type of 'colord' (line 1289)
    colord_56504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 21), 'colord', False)
    # Obtaining the member '__getitem__' of a type (line 1289)
    getitem___56505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 21), colord_56504, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1289)
    subscript_call_result_56506 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 21), getitem___56505, result_ge_56503)
    
    # Processing the call keyword arguments (line 1289)
    kwargs_56507 = {}
    # Getting the type of 'colors' (line 1289)
    colors_56492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'colors', False)
    # Obtaining the member 'insert' of a type (line 1289)
    insert_56493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 4), colors_56492, 'insert')
    # Calling insert(args, kwargs) (line 1289)
    insert_call_result_56508 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 4), insert_56493, *[int_56494, subscript_call_result_56506], **kwargs_56507)
    
    
    # Assigning a BinOp to a Name (line 1291):
    
    # Assigning a BinOp to a Name (line 1291):
    # Getting the type of 'width' (line 1291)
    width_56509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 12), 'width')
    float_56510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 20), 'float')
    # Applying the binary operator 'div' (line 1291)
    result_div_56511 = python_operator(stypy.reporting.localization.Localization(__file__, 1291, 12), 'div', width_56509, float_56510)
    
    # Assigning a type to the variable 'right' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 4), 'right', result_div_56511)
    
    # Assigning a BinOp to a Name (line 1292):
    
    # Assigning a BinOp to a Name (line 1292):
    
    # Getting the type of 'width' (line 1292)
    width_56512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 12), 'width')
    # Applying the 'usub' unary operator (line 1292)
    result___neg___56513 = python_operator(stypy.reporting.localization.Localization(__file__, 1292, 11), 'usub', width_56512)
    
    float_56514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 20), 'float')
    # Applying the binary operator 'div' (line 1292)
    result_div_56515 = python_operator(stypy.reporting.localization.Localization(__file__, 1292, 11), 'div', result___neg___56513, float_56514)
    
    # Assigning a type to the variable 'left' (line 1292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 4), 'left', result_div_56515)
    
    # Assigning a ListComp to a Name (line 1294):
    
    # Assigning a ListComp to a Name (line 1294):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'quotes' (line 1295)
    quotes_56529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 53), 'quotes')
    comprehension_56530 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 12), quotes_56529)
    # Assigning a type to the variable 'd' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 12), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 12), comprehension_56530))
    # Assigning a type to the variable 'open' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 12), 'open', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 12), comprehension_56530))
    # Assigning a type to the variable 'high' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 12), 'high', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 12), comprehension_56530))
    # Assigning a type to the variable 'low' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 12), 'low', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 12), comprehension_56530))
    # Assigning a type to the variable 'close' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 12), 'close', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 12), comprehension_56530))
    # Assigning a type to the variable 'volume' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 12), 'volume', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 12), comprehension_56530))
    
    # Obtaining an instance of the builtin type 'tuple' (line 1294)
    tuple_56516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1294)
    # Adding element type (line 1294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1294)
    tuple_56517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1294)
    # Adding element type (line 1294)
    # Getting the type of 'left' (line 1294)
    left_56518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 14), 'left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 14), tuple_56517, left_56518)
    # Adding element type (line 1294)
    int_56519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 14), tuple_56517, int_56519)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 13), tuple_56516, tuple_56517)
    # Adding element type (line 1294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1294)
    tuple_56520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1294)
    # Adding element type (line 1294)
    # Getting the type of 'left' (line 1294)
    left_56521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 25), 'left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 25), tuple_56520, left_56521)
    # Adding element type (line 1294)
    # Getting the type of 'volume' (line 1294)
    volume_56522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 31), 'volume')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 25), tuple_56520, volume_56522)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 13), tuple_56516, tuple_56520)
    # Adding element type (line 1294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1294)
    tuple_56523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1294)
    # Adding element type (line 1294)
    # Getting the type of 'right' (line 1294)
    right_56524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 41), 'right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 41), tuple_56523, right_56524)
    # Adding element type (line 1294)
    # Getting the type of 'volume' (line 1294)
    volume_56525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 48), 'volume')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 41), tuple_56523, volume_56525)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 13), tuple_56516, tuple_56523)
    # Adding element type (line 1294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1294)
    tuple_56526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 58), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1294)
    # Adding element type (line 1294)
    # Getting the type of 'right' (line 1294)
    right_56527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 58), 'right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 58), tuple_56526, right_56527)
    # Adding element type (line 1294)
    int_56528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 58), tuple_56526, int_56528)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 13), tuple_56516, tuple_56526)
    
    list_56531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1294, 12), list_56531, tuple_56516)
    # Assigning a type to the variable 'bars' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'bars', list_56531)
    
    # Assigning a BinOp to a Name (line 1297):
    
    # Assigning a BinOp to a Name (line 1297):
    # Getting the type of 'ax' (line 1297)
    ax_56532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 9), 'ax')
    # Obtaining the member 'figure' of a type (line 1297)
    figure_56533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1297, 9), ax_56532, 'figure')
    # Obtaining the member 'dpi' of a type (line 1297)
    dpi_56534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1297, 9), figure_56533, 'dpi')
    float_56535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1297, 26), 'float')
    float_56536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1297, 32), 'float')
    # Applying the binary operator 'div' (line 1297)
    result_div_56537 = python_operator(stypy.reporting.localization.Localization(__file__, 1297, 26), 'div', float_56535, float_56536)
    
    # Applying the binary operator '*' (line 1297)
    result_mul_56538 = python_operator(stypy.reporting.localization.Localization(__file__, 1297, 9), '*', dpi_56534, result_div_56537)
    
    # Assigning a type to the variable 'sx' (line 1297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1297, 4), 'sx', result_mul_56538)
    
    # Assigning a BinOp to a Name (line 1298):
    
    # Assigning a BinOp to a Name (line 1298):
    # Getting the type of 'ax' (line 1298)
    ax_56539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 9), 'ax')
    # Obtaining the member 'bbox' of a type (line 1298)
    bbox_56540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 9), ax_56539, 'bbox')
    # Obtaining the member 'height' of a type (line 1298)
    height_56541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 9), bbox_56540, 'height')
    # Getting the type of 'ax' (line 1298)
    ax_56542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 26), 'ax')
    # Obtaining the member 'viewLim' of a type (line 1298)
    viewLim_56543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 26), ax_56542, 'viewLim')
    # Obtaining the member 'height' of a type (line 1298)
    height_56544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 26), viewLim_56543, 'height')
    # Applying the binary operator 'div' (line 1298)
    result_div_56545 = python_operator(stypy.reporting.localization.Localization(__file__, 1298, 9), 'div', height_56541, height_56544)
    
    # Assigning a type to the variable 'sy' (line 1298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1298, 4), 'sy', result_div_56545)
    
    # Assigning a Call to a Name (line 1300):
    
    # Assigning a Call to a Name (line 1300):
    
    # Call to scale(...): (line 1300)
    # Processing the call arguments (line 1300)
    # Getting the type of 'sx' (line 1300)
    sx_56550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 36), 'sx', False)
    # Getting the type of 'sy' (line 1300)
    sy_56551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 40), 'sy', False)
    # Processing the call keyword arguments (line 1300)
    kwargs_56552 = {}
    
    # Call to Affine2D(...): (line 1300)
    # Processing the call keyword arguments (line 1300)
    kwargs_56547 = {}
    # Getting the type of 'Affine2D' (line 1300)
    Affine2D_56546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 19), 'Affine2D', False)
    # Calling Affine2D(args, kwargs) (line 1300)
    Affine2D_call_result_56548 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 19), Affine2D_56546, *[], **kwargs_56547)
    
    # Obtaining the member 'scale' of a type (line 1300)
    scale_56549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 19), Affine2D_call_result_56548, 'scale')
    # Calling scale(args, kwargs) (line 1300)
    scale_call_result_56553 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 19), scale_56549, *[sx_56550, sy_56551], **kwargs_56552)
    
    # Assigning a type to the variable 'barTransform' (line 1300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1300, 4), 'barTransform', scale_call_result_56553)
    
    # Assigning a ListComp to a Name (line 1302):
    
    # Assigning a ListComp to a Name (line 1302):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'quotes' (line 1302)
    quotes_56555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 56), 'quotes')
    comprehension_56556 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 13), quotes_56555)
    # Assigning a type to the variable 'd' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 13), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 13), comprehension_56556))
    # Assigning a type to the variable 'open' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 13), 'open', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 13), comprehension_56556))
    # Assigning a type to the variable 'high' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 13), 'high', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 13), comprehension_56556))
    # Assigning a type to the variable 'low' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 13), 'low', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 13), comprehension_56556))
    # Assigning a type to the variable 'close' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 13), 'close', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 13), comprehension_56556))
    # Assigning a type to the variable 'volume' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 13), 'volume', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 13), comprehension_56556))
    # Getting the type of 'd' (line 1302)
    d_56554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 13), 'd')
    list_56557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1302, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1302, 13), list_56557, d_56554)
    # Assigning a type to the variable 'dates' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 4), 'dates', list_56557)
    
    # Assigning a ListComp to a Name (line 1303):
    
    # Assigning a ListComp to a Name (line 1303):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'dates' (line 1303)
    dates_56561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 35), 'dates')
    comprehension_56562 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1303, 19), dates_56561)
    # Assigning a type to the variable 'd' (line 1303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1303, 19), 'd', comprehension_56562)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1303)
    tuple_56558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1303, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1303)
    # Adding element type (line 1303)
    # Getting the type of 'd' (line 1303)
    d_56559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 20), 'd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1303, 20), tuple_56558, d_56559)
    # Adding element type (line 1303)
    int_56560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1303, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1303, 20), tuple_56558, int_56560)
    
    list_56563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1303, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1303, 19), list_56563, tuple_56558)
    # Assigning a type to the variable 'offsetsBars' (line 1303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1303, 4), 'offsetsBars', list_56563)
    
    # Assigning a Tuple to a Name (line 1305):
    
    # Assigning a Tuple to a Name (line 1305):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1305)
    tuple_56564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1305, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1305)
    # Adding element type (line 1305)
    int_56565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1305, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1305, 12), tuple_56564, int_56565)
    
    # Assigning a type to the variable 'useAA' (line 1305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1305, 4), 'useAA', tuple_56564)
    
    # Assigning a Tuple to a Name (line 1306):
    
    # Assigning a Tuple to a Name (line 1306):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1306)
    tuple_56566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1306, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1306)
    # Adding element type (line 1306)
    float_56567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1306, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1306, 9), tuple_56566, float_56567)
    
    # Assigning a type to the variable 'lw' (line 1306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1306, 4), 'lw', tuple_56566)
    
    # Assigning a Call to a Name (line 1307):
    
    # Assigning a Call to a Name (line 1307):
    
    # Call to PolyCollection(...): (line 1307)
    # Processing the call arguments (line 1307)
    # Getting the type of 'bars' (line 1307)
    bars_56569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1307, 35), 'bars', False)
    # Processing the call keyword arguments (line 1307)
    # Getting the type of 'colors' (line 1308)
    colors_56570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 46), 'colors', False)
    keyword_56571 = colors_56570
    
    # Obtaining an instance of the builtin type 'tuple' (line 1309)
    tuple_56572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1309, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1309)
    # Adding element type (line 1309)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1309)
    tuple_56573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1309, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1309)
    # Adding element type (line 1309)
    int_56574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1309, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1309, 48), tuple_56573, int_56574)
    # Adding element type (line 1309)
    int_56575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1309, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1309, 48), tuple_56573, int_56575)
    # Adding element type (line 1309)
    int_56576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1309, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1309, 48), tuple_56573, int_56576)
    # Adding element type (line 1309)
    int_56577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1309, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1309, 48), tuple_56573, int_56577)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1309, 47), tuple_56572, tuple_56573)
    
    keyword_56578 = tuple_56572
    # Getting the type of 'useAA' (line 1310)
    useAA_56579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1310, 48), 'useAA', False)
    keyword_56580 = useAA_56579
    # Getting the type of 'lw' (line 1311)
    lw_56581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1311, 46), 'lw', False)
    keyword_56582 = lw_56581
    # Getting the type of 'offsetsBars' (line 1312)
    offsetsBars_56583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1312, 43), 'offsetsBars', False)
    keyword_56584 = offsetsBars_56583
    # Getting the type of 'ax' (line 1313)
    ax_56585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1313, 47), 'ax', False)
    # Obtaining the member 'transData' of a type (line 1313)
    transData_56586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1313, 47), ax_56585, 'transData')
    keyword_56587 = transData_56586
    kwargs_56588 = {'edgecolors': keyword_56578, 'antialiaseds': keyword_56580, 'offsets': keyword_56584, 'linewidths': keyword_56582, 'transOffset': keyword_56587, 'facecolors': keyword_56571}
    # Getting the type of 'PolyCollection' (line 1307)
    PolyCollection_56568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1307, 20), 'PolyCollection', False)
    # Calling PolyCollection(args, kwargs) (line 1307)
    PolyCollection_call_result_56589 = invoke(stypy.reporting.localization.Localization(__file__, 1307, 20), PolyCollection_56568, *[bars_56569], **kwargs_56588)
    
    # Assigning a type to the variable 'barCollection' (line 1307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1307, 4), 'barCollection', PolyCollection_call_result_56589)
    
    # Call to set_transform(...): (line 1315)
    # Processing the call arguments (line 1315)
    # Getting the type of 'barTransform' (line 1315)
    barTransform_56592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 32), 'barTransform', False)
    # Processing the call keyword arguments (line 1315)
    kwargs_56593 = {}
    # Getting the type of 'barCollection' (line 1315)
    barCollection_56590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 4), 'barCollection', False)
    # Obtaining the member 'set_transform' of a type (line 1315)
    set_transform_56591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1315, 4), barCollection_56590, 'set_transform')
    # Calling set_transform(args, kwargs) (line 1315)
    set_transform_call_result_56594 = invoke(stypy.reporting.localization.Localization(__file__, 1315, 4), set_transform_56591, *[barTransform_56592], **kwargs_56593)
    
    
    # Assigning a Tuple to a Tuple (line 1317):
    
    # Assigning a Call to a Name (line 1317):
    
    # Call to min(...): (line 1317)
    # Processing the call arguments (line 1317)
    # Getting the type of 'dates' (line 1317)
    dates_56596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 23), 'dates', False)
    # Processing the call keyword arguments (line 1317)
    kwargs_56597 = {}
    # Getting the type of 'min' (line 1317)
    min_56595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 19), 'min', False)
    # Calling min(args, kwargs) (line 1317)
    min_call_result_56598 = invoke(stypy.reporting.localization.Localization(__file__, 1317, 19), min_56595, *[dates_56596], **kwargs_56597)
    
    # Assigning a type to the variable 'tuple_assignment_54329' (line 1317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1317, 4), 'tuple_assignment_54329', min_call_result_56598)
    
    # Assigning a Call to a Name (line 1317):
    
    # Call to max(...): (line 1317)
    # Processing the call arguments (line 1317)
    # Getting the type of 'dates' (line 1317)
    dates_56600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 35), 'dates', False)
    # Processing the call keyword arguments (line 1317)
    kwargs_56601 = {}
    # Getting the type of 'max' (line 1317)
    max_56599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 31), 'max', False)
    # Calling max(args, kwargs) (line 1317)
    max_call_result_56602 = invoke(stypy.reporting.localization.Localization(__file__, 1317, 31), max_56599, *[dates_56600], **kwargs_56601)
    
    # Assigning a type to the variable 'tuple_assignment_54330' (line 1317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1317, 4), 'tuple_assignment_54330', max_call_result_56602)
    
    # Assigning a Name to a Name (line 1317):
    # Getting the type of 'tuple_assignment_54329' (line 1317)
    tuple_assignment_54329_56603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 4), 'tuple_assignment_54329')
    # Assigning a type to the variable 'minpy' (line 1317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1317, 4), 'minpy', tuple_assignment_54329_56603)
    
    # Assigning a Name to a Name (line 1317):
    # Getting the type of 'tuple_assignment_54330' (line 1317)
    tuple_assignment_54330_56604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 4), 'tuple_assignment_54330')
    # Assigning a type to the variable 'maxx' (line 1317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1317, 11), 'maxx', tuple_assignment_54330_56604)
    
    # Assigning a Num to a Name (line 1318):
    
    # Assigning a Num to a Name (line 1318):
    int_56605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1318, 11), 'int')
    # Assigning a type to the variable 'miny' (line 1318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1318, 4), 'miny', int_56605)
    
    # Assigning a Call to a Name (line 1319):
    
    # Assigning a Call to a Name (line 1319):
    
    # Call to max(...): (line 1319)
    # Processing the call arguments (line 1319)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'quotes' (line 1319)
    quotes_56608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1319, 64), 'quotes', False)
    comprehension_56609 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1319, 16), quotes_56608)
    # Assigning a type to the variable 'd' (line 1319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1319, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1319, 16), comprehension_56609))
    # Assigning a type to the variable 'open' (line 1319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1319, 16), 'open', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1319, 16), comprehension_56609))
    # Assigning a type to the variable 'high' (line 1319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1319, 16), 'high', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1319, 16), comprehension_56609))
    # Assigning a type to the variable 'low' (line 1319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1319, 16), 'low', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1319, 16), comprehension_56609))
    # Assigning a type to the variable 'close' (line 1319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1319, 16), 'close', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1319, 16), comprehension_56609))
    # Assigning a type to the variable 'volume' (line 1319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1319, 16), 'volume', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1319, 16), comprehension_56609))
    # Getting the type of 'volume' (line 1319)
    volume_56607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1319, 16), 'volume', False)
    list_56610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1319, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1319, 16), list_56610, volume_56607)
    # Processing the call keyword arguments (line 1319)
    kwargs_56611 = {}
    # Getting the type of 'max' (line 1319)
    max_56606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1319, 11), 'max', False)
    # Calling max(args, kwargs) (line 1319)
    max_call_result_56612 = invoke(stypy.reporting.localization.Localization(__file__, 1319, 11), max_56606, *[list_56610], **kwargs_56611)
    
    # Assigning a type to the variable 'maxy' (line 1319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1319, 4), 'maxy', max_call_result_56612)
    
    # Assigning a Tuple to a Name (line 1320):
    
    # Assigning a Tuple to a Name (line 1320):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1320)
    tuple_56613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1320)
    # Adding element type (line 1320)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1320)
    tuple_56614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1320)
    # Adding element type (line 1320)
    # Getting the type of 'minpy' (line 1320)
    minpy_56615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 15), 'minpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 15), tuple_56614, minpy_56615)
    # Adding element type (line 1320)
    # Getting the type of 'miny' (line 1320)
    miny_56616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 22), 'miny')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 15), tuple_56614, miny_56616)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 14), tuple_56613, tuple_56614)
    # Adding element type (line 1320)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1320)
    tuple_56617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1320)
    # Adding element type (line 1320)
    # Getting the type of 'maxx' (line 1320)
    maxx_56618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 30), 'maxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 30), tuple_56617, maxx_56618)
    # Adding element type (line 1320)
    # Getting the type of 'maxy' (line 1320)
    maxy_56619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 36), 'maxy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 30), tuple_56617, maxy_56619)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 14), tuple_56613, tuple_56617)
    
    # Assigning a type to the variable 'corners' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'corners', tuple_56613)
    
    # Call to update_datalim(...): (line 1321)
    # Processing the call arguments (line 1321)
    # Getting the type of 'corners' (line 1321)
    corners_56622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 22), 'corners', False)
    # Processing the call keyword arguments (line 1321)
    kwargs_56623 = {}
    # Getting the type of 'ax' (line 1321)
    ax_56620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1321, 4), 'ax', False)
    # Obtaining the member 'update_datalim' of a type (line 1321)
    update_datalim_56621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1321, 4), ax_56620, 'update_datalim')
    # Calling update_datalim(args, kwargs) (line 1321)
    update_datalim_call_result_56624 = invoke(stypy.reporting.localization.Localization(__file__, 1321, 4), update_datalim_56621, *[corners_56622], **kwargs_56623)
    
    
    # Call to add_collection(...): (line 1325)
    # Processing the call arguments (line 1325)
    # Getting the type of 'barCollection' (line 1325)
    barCollection_56627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 22), 'barCollection', False)
    # Processing the call keyword arguments (line 1325)
    kwargs_56628 = {}
    # Getting the type of 'ax' (line 1325)
    ax_56625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 1325)
    add_collection_56626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1325, 4), ax_56625, 'add_collection')
    # Calling add_collection(args, kwargs) (line 1325)
    add_collection_call_result_56629 = invoke(stypy.reporting.localization.Localization(__file__, 1325, 4), add_collection_56626, *[barCollection_56627], **kwargs_56628)
    
    
    # Call to autoscale_view(...): (line 1326)
    # Processing the call keyword arguments (line 1326)
    kwargs_56632 = {}
    # Getting the type of 'ax' (line 1326)
    ax_56630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1326, 4), 'ax', False)
    # Obtaining the member 'autoscale_view' of a type (line 1326)
    autoscale_view_56631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1326, 4), ax_56630, 'autoscale_view')
    # Calling autoscale_view(args, kwargs) (line 1326)
    autoscale_view_call_result_56633 = invoke(stypy.reporting.localization.Localization(__file__, 1326, 4), autoscale_view_56631, *[], **kwargs_56632)
    
    # Getting the type of 'barCollection' (line 1328)
    barCollection_56634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 11), 'barCollection')
    # Assigning a type to the variable 'stypy_return_type' (line 1328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1328, 4), 'stypy_return_type', barCollection_56634)
    
    # ################# End of 'volume_overlay3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'volume_overlay3' in the type store
    # Getting the type of 'stypy_return_type' (line 1251)
    stypy_return_type_56635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_56635)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'volume_overlay3'
    return stypy_return_type_56635

# Assigning a type to the variable 'volume_overlay3' (line 1251)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1251, 0), 'volume_overlay3', volume_overlay3)

@norecursion
def index_bar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    unicode_56636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1332, 24), 'unicode', u'b')
    unicode_56637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1332, 39), 'unicode', u'l')
    int_56638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1333, 20), 'int')
    float_56639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1333, 29), 'float')
    defaults = [unicode_56636, unicode_56637, int_56638, float_56639]
    # Create a new context for function 'index_bar'
    module_type_store = module_type_store.open_function_context('index_bar', 1331, 0, False)
    
    # Passed parameters checking function
    index_bar.stypy_localization = localization
    index_bar.stypy_type_of_self = None
    index_bar.stypy_type_store = module_type_store
    index_bar.stypy_function_name = 'index_bar'
    index_bar.stypy_param_names_list = ['ax', 'vals', 'facecolor', 'edgecolor', 'width', 'alpha']
    index_bar.stypy_varargs_param_name = None
    index_bar.stypy_kwargs_param_name = None
    index_bar.stypy_call_defaults = defaults
    index_bar.stypy_call_varargs = varargs
    index_bar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'index_bar', ['ax', 'vals', 'facecolor', 'edgecolor', 'width', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'index_bar', localization, ['ax', 'vals', 'facecolor', 'edgecolor', 'width', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'index_bar(...)' code ##################

    unicode_56640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, (-1)), 'unicode', u'Add a bar collection graph with height vals (-1 is missing).\n\n    Parameters\n    ----------\n    ax : `Axes`\n        an Axes instance to plot to\n    vals : sequence\n        a sequence of values\n    facecolor : color\n        the color of the bar face\n    edgecolor : color\n        the color of the bar edges\n    width : int\n        the bar width in points\n    alpha : float\n       bar transparency\n\n    Returns\n    -------\n    ret : `barCollection`\n        The `barrCollection` added to the axes\n\n    ')
    
    # Assigning a Tuple to a Name (line 1358):
    
    # Assigning a Tuple to a Name (line 1358):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1358)
    tuple_56641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1358)
    # Adding element type (line 1358)
    
    # Call to to_rgba(...): (line 1358)
    # Processing the call arguments (line 1358)
    # Getting the type of 'facecolor' (line 1358)
    facecolor_56644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 34), 'facecolor', False)
    # Getting the type of 'alpha' (line 1358)
    alpha_56645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 45), 'alpha', False)
    # Processing the call keyword arguments (line 1358)
    kwargs_56646 = {}
    # Getting the type of 'mcolors' (line 1358)
    mcolors_56642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 18), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 1358)
    to_rgba_56643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 18), mcolors_56642, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 1358)
    to_rgba_call_result_56647 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 18), to_rgba_56643, *[facecolor_56644, alpha_56645], **kwargs_56646)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1358, 18), tuple_56641, to_rgba_call_result_56647)
    
    # Assigning a type to the variable 'facecolors' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'facecolors', tuple_56641)
    
    # Assigning a Tuple to a Name (line 1359):
    
    # Assigning a Tuple to a Name (line 1359):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1359)
    tuple_56648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1359)
    # Adding element type (line 1359)
    
    # Call to to_rgba(...): (line 1359)
    # Processing the call arguments (line 1359)
    # Getting the type of 'edgecolor' (line 1359)
    edgecolor_56651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 34), 'edgecolor', False)
    # Getting the type of 'alpha' (line 1359)
    alpha_56652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 45), 'alpha', False)
    # Processing the call keyword arguments (line 1359)
    kwargs_56653 = {}
    # Getting the type of 'mcolors' (line 1359)
    mcolors_56649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 18), 'mcolors', False)
    # Obtaining the member 'to_rgba' of a type (line 1359)
    to_rgba_56650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1359, 18), mcolors_56649, 'to_rgba')
    # Calling to_rgba(args, kwargs) (line 1359)
    to_rgba_call_result_56654 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 18), to_rgba_56650, *[edgecolor_56651, alpha_56652], **kwargs_56653)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 18), tuple_56648, to_rgba_call_result_56654)
    
    # Assigning a type to the variable 'edgecolors' (line 1359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 4), 'edgecolors', tuple_56648)
    
    # Assigning a BinOp to a Name (line 1361):
    
    # Assigning a BinOp to a Name (line 1361):
    # Getting the type of 'width' (line 1361)
    width_56655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 12), 'width')
    float_56656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1361, 20), 'float')
    # Applying the binary operator 'div' (line 1361)
    result_div_56657 = python_operator(stypy.reporting.localization.Localization(__file__, 1361, 12), 'div', width_56655, float_56656)
    
    # Assigning a type to the variable 'right' (line 1361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1361, 4), 'right', result_div_56657)
    
    # Assigning a BinOp to a Name (line 1362):
    
    # Assigning a BinOp to a Name (line 1362):
    
    # Getting the type of 'width' (line 1362)
    width_56658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 12), 'width')
    # Applying the 'usub' unary operator (line 1362)
    result___neg___56659 = python_operator(stypy.reporting.localization.Localization(__file__, 1362, 11), 'usub', width_56658)
    
    float_56660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 20), 'float')
    # Applying the binary operator 'div' (line 1362)
    result_div_56661 = python_operator(stypy.reporting.localization.Localization(__file__, 1362, 11), 'div', result___neg___56659, float_56660)
    
    # Assigning a type to the variable 'left' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 4), 'left', result_div_56661)
    
    # Assigning a ListComp to a Name (line 1364):
    
    # Assigning a ListComp to a Name (line 1364):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'vals' (line 1365)
    vals_56678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 21), 'vals')
    comprehension_56679 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 12), vals_56678)
    # Assigning a type to the variable 'v' (line 1364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1364, 12), 'v', comprehension_56679)
    
    # Getting the type of 'v' (line 1365)
    v_56675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 29), 'v')
    int_56676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 34), 'int')
    # Applying the binary operator '!=' (line 1365)
    result_ne_56677 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 29), '!=', v_56675, int_56676)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1364)
    tuple_56662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1364)
    # Adding element type (line 1364)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1364)
    tuple_56663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1364)
    # Adding element type (line 1364)
    # Getting the type of 'left' (line 1364)
    left_56664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 14), 'left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 14), tuple_56663, left_56664)
    # Adding element type (line 1364)
    int_56665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 14), tuple_56663, int_56665)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 13), tuple_56662, tuple_56663)
    # Adding element type (line 1364)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1364)
    tuple_56666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1364)
    # Adding element type (line 1364)
    # Getting the type of 'left' (line 1364)
    left_56667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 25), 'left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 25), tuple_56666, left_56667)
    # Adding element type (line 1364)
    # Getting the type of 'v' (line 1364)
    v_56668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 31), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 25), tuple_56666, v_56668)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 13), tuple_56662, tuple_56666)
    # Adding element type (line 1364)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1364)
    tuple_56669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1364)
    # Adding element type (line 1364)
    # Getting the type of 'right' (line 1364)
    right_56670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 36), 'right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 36), tuple_56669, right_56670)
    # Adding element type (line 1364)
    # Getting the type of 'v' (line 1364)
    v_56671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 43), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 36), tuple_56669, v_56671)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 13), tuple_56662, tuple_56669)
    # Adding element type (line 1364)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1364)
    tuple_56672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1364)
    # Adding element type (line 1364)
    # Getting the type of 'right' (line 1364)
    right_56673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 48), 'right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 48), tuple_56672, right_56673)
    # Adding element type (line 1364)
    int_56674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 48), tuple_56672, int_56674)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 13), tuple_56662, tuple_56672)
    
    list_56680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1364, 12), list_56680, tuple_56662)
    # Assigning a type to the variable 'bars' (line 1364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1364, 4), 'bars', list_56680)
    
    # Assigning a BinOp to a Name (line 1367):
    
    # Assigning a BinOp to a Name (line 1367):
    # Getting the type of 'ax' (line 1367)
    ax_56681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 9), 'ax')
    # Obtaining the member 'figure' of a type (line 1367)
    figure_56682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1367, 9), ax_56681, 'figure')
    # Obtaining the member 'dpi' of a type (line 1367)
    dpi_56683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1367, 9), figure_56682, 'dpi')
    float_56684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 26), 'float')
    float_56685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 32), 'float')
    # Applying the binary operator 'div' (line 1367)
    result_div_56686 = python_operator(stypy.reporting.localization.Localization(__file__, 1367, 26), 'div', float_56684, float_56685)
    
    # Applying the binary operator '*' (line 1367)
    result_mul_56687 = python_operator(stypy.reporting.localization.Localization(__file__, 1367, 9), '*', dpi_56683, result_div_56686)
    
    # Assigning a type to the variable 'sx' (line 1367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1367, 4), 'sx', result_mul_56687)
    
    # Assigning a BinOp to a Name (line 1368):
    
    # Assigning a BinOp to a Name (line 1368):
    # Getting the type of 'ax' (line 1368)
    ax_56688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 9), 'ax')
    # Obtaining the member 'bbox' of a type (line 1368)
    bbox_56689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 9), ax_56688, 'bbox')
    # Obtaining the member 'height' of a type (line 1368)
    height_56690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 9), bbox_56689, 'height')
    # Getting the type of 'ax' (line 1368)
    ax_56691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 26), 'ax')
    # Obtaining the member 'viewLim' of a type (line 1368)
    viewLim_56692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 26), ax_56691, 'viewLim')
    # Obtaining the member 'height' of a type (line 1368)
    height_56693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 26), viewLim_56692, 'height')
    # Applying the binary operator 'div' (line 1368)
    result_div_56694 = python_operator(stypy.reporting.localization.Localization(__file__, 1368, 9), 'div', height_56690, height_56693)
    
    # Assigning a type to the variable 'sy' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'sy', result_div_56694)
    
    # Assigning a Call to a Name (line 1370):
    
    # Assigning a Call to a Name (line 1370):
    
    # Call to scale(...): (line 1370)
    # Processing the call arguments (line 1370)
    # Getting the type of 'sx' (line 1370)
    sx_56699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 36), 'sx', False)
    # Getting the type of 'sy' (line 1370)
    sy_56700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 40), 'sy', False)
    # Processing the call keyword arguments (line 1370)
    kwargs_56701 = {}
    
    # Call to Affine2D(...): (line 1370)
    # Processing the call keyword arguments (line 1370)
    kwargs_56696 = {}
    # Getting the type of 'Affine2D' (line 1370)
    Affine2D_56695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 19), 'Affine2D', False)
    # Calling Affine2D(args, kwargs) (line 1370)
    Affine2D_call_result_56697 = invoke(stypy.reporting.localization.Localization(__file__, 1370, 19), Affine2D_56695, *[], **kwargs_56696)
    
    # Obtaining the member 'scale' of a type (line 1370)
    scale_56698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1370, 19), Affine2D_call_result_56697, 'scale')
    # Calling scale(args, kwargs) (line 1370)
    scale_call_result_56702 = invoke(stypy.reporting.localization.Localization(__file__, 1370, 19), scale_56698, *[sx_56699, sy_56700], **kwargs_56701)
    
    # Assigning a type to the variable 'barTransform' (line 1370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1370, 4), 'barTransform', scale_call_result_56702)
    
    # Assigning a ListComp to a Name (line 1372):
    
    # Assigning a ListComp to a Name (line 1372):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 1372)
    # Processing the call arguments (line 1372)
    # Getting the type of 'vals' (line 1372)
    vals_56710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 48), 'vals', False)
    # Processing the call keyword arguments (line 1372)
    kwargs_56711 = {}
    # Getting the type of 'enumerate' (line 1372)
    enumerate_56709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 38), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1372)
    enumerate_call_result_56712 = invoke(stypy.reporting.localization.Localization(__file__, 1372, 38), enumerate_56709, *[vals_56710], **kwargs_56711)
    
    comprehension_56713 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1372, 19), enumerate_call_result_56712)
    # Assigning a type to the variable 'i' (line 1372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1372, 19), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1372, 19), comprehension_56713))
    # Assigning a type to the variable 'v' (line 1372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1372, 19), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1372, 19), comprehension_56713))
    
    # Getting the type of 'v' (line 1372)
    v_56706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 57), 'v')
    int_56707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, 62), 'int')
    # Applying the binary operator '!=' (line 1372)
    result_ne_56708 = python_operator(stypy.reporting.localization.Localization(__file__, 1372, 57), '!=', v_56706, int_56707)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1372)
    tuple_56703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1372)
    # Adding element type (line 1372)
    # Getting the type of 'i' (line 1372)
    i_56704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 20), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1372, 20), tuple_56703, i_56704)
    # Adding element type (line 1372)
    int_56705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1372, 20), tuple_56703, int_56705)
    
    list_56714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1372, 19), list_56714, tuple_56703)
    # Assigning a type to the variable 'offsetsBars' (line 1372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1372, 4), 'offsetsBars', list_56714)
    
    # Assigning a Call to a Name (line 1374):
    
    # Assigning a Call to a Name (line 1374):
    
    # Call to PolyCollection(...): (line 1374)
    # Processing the call arguments (line 1374)
    # Getting the type of 'bars' (line 1374)
    bars_56716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 35), 'bars', False)
    # Processing the call keyword arguments (line 1374)
    # Getting the type of 'facecolors' (line 1375)
    facecolors_56717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1375, 46), 'facecolors', False)
    keyword_56718 = facecolors_56717
    # Getting the type of 'edgecolors' (line 1376)
    edgecolors_56719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 46), 'edgecolors', False)
    keyword_56720 = edgecolors_56719
    
    # Obtaining an instance of the builtin type 'tuple' (line 1377)
    tuple_56721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1377, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1377)
    # Adding element type (line 1377)
    int_56722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1377, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1377, 49), tuple_56721, int_56722)
    
    keyword_56723 = tuple_56721
    
    # Obtaining an instance of the builtin type 'tuple' (line 1378)
    tuple_56724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1378, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1378)
    # Adding element type (line 1378)
    float_56725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1378, 47), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1378, 47), tuple_56724, float_56725)
    
    keyword_56726 = tuple_56724
    # Getting the type of 'offsetsBars' (line 1379)
    offsetsBars_56727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1379, 43), 'offsetsBars', False)
    keyword_56728 = offsetsBars_56727
    # Getting the type of 'ax' (line 1380)
    ax_56729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 47), 'ax', False)
    # Obtaining the member 'transData' of a type (line 1380)
    transData_56730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1380, 47), ax_56729, 'transData')
    keyword_56731 = transData_56730
    kwargs_56732 = {'edgecolors': keyword_56720, 'antialiaseds': keyword_56723, 'offsets': keyword_56728, 'linewidths': keyword_56726, 'transOffset': keyword_56731, 'facecolors': keyword_56718}
    # Getting the type of 'PolyCollection' (line 1374)
    PolyCollection_56715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 20), 'PolyCollection', False)
    # Calling PolyCollection(args, kwargs) (line 1374)
    PolyCollection_call_result_56733 = invoke(stypy.reporting.localization.Localization(__file__, 1374, 20), PolyCollection_56715, *[bars_56716], **kwargs_56732)
    
    # Assigning a type to the variable 'barCollection' (line 1374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1374, 4), 'barCollection', PolyCollection_call_result_56733)
    
    # Call to set_transform(...): (line 1382)
    # Processing the call arguments (line 1382)
    # Getting the type of 'barTransform' (line 1382)
    barTransform_56736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 32), 'barTransform', False)
    # Processing the call keyword arguments (line 1382)
    kwargs_56737 = {}
    # Getting the type of 'barCollection' (line 1382)
    barCollection_56734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 4), 'barCollection', False)
    # Obtaining the member 'set_transform' of a type (line 1382)
    set_transform_56735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1382, 4), barCollection_56734, 'set_transform')
    # Calling set_transform(args, kwargs) (line 1382)
    set_transform_call_result_56738 = invoke(stypy.reporting.localization.Localization(__file__, 1382, 4), set_transform_56735, *[barTransform_56736], **kwargs_56737)
    
    
    # Assigning a Tuple to a Tuple (line 1384):
    
    # Assigning a Num to a Name (line 1384):
    int_56739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1384, 19), 'int')
    # Assigning a type to the variable 'tuple_assignment_54331' (line 1384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1384, 4), 'tuple_assignment_54331', int_56739)
    
    # Assigning a Call to a Name (line 1384):
    
    # Call to len(...): (line 1384)
    # Processing the call arguments (line 1384)
    # Getting the type of 'offsetsBars' (line 1384)
    offsetsBars_56741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 26), 'offsetsBars', False)
    # Processing the call keyword arguments (line 1384)
    kwargs_56742 = {}
    # Getting the type of 'len' (line 1384)
    len_56740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 22), 'len', False)
    # Calling len(args, kwargs) (line 1384)
    len_call_result_56743 = invoke(stypy.reporting.localization.Localization(__file__, 1384, 22), len_56740, *[offsetsBars_56741], **kwargs_56742)
    
    # Assigning a type to the variable 'tuple_assignment_54332' (line 1384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1384, 4), 'tuple_assignment_54332', len_call_result_56743)
    
    # Assigning a Name to a Name (line 1384):
    # Getting the type of 'tuple_assignment_54331' (line 1384)
    tuple_assignment_54331_56744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 4), 'tuple_assignment_54331')
    # Assigning a type to the variable 'minpy' (line 1384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1384, 4), 'minpy', tuple_assignment_54331_56744)
    
    # Assigning a Name to a Name (line 1384):
    # Getting the type of 'tuple_assignment_54332' (line 1384)
    tuple_assignment_54332_56745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1384, 4), 'tuple_assignment_54332')
    # Assigning a type to the variable 'maxx' (line 1384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1384, 11), 'maxx', tuple_assignment_54332_56745)
    
    # Assigning a Num to a Name (line 1385):
    
    # Assigning a Num to a Name (line 1385):
    int_56746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1385, 11), 'int')
    # Assigning a type to the variable 'miny' (line 1385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1385, 4), 'miny', int_56746)
    
    # Assigning a Call to a Name (line 1386):
    
    # Assigning a Call to a Name (line 1386):
    
    # Call to max(...): (line 1386)
    # Processing the call arguments (line 1386)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'vals' (line 1386)
    vals_56752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 27), 'vals', False)
    comprehension_56753 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1386, 16), vals_56752)
    # Assigning a type to the variable 'v' (line 1386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1386, 16), 'v', comprehension_56753)
    
    # Getting the type of 'v' (line 1386)
    v_56749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 35), 'v', False)
    int_56750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1386, 40), 'int')
    # Applying the binary operator '!=' (line 1386)
    result_ne_56751 = python_operator(stypy.reporting.localization.Localization(__file__, 1386, 35), '!=', v_56749, int_56750)
    
    # Getting the type of 'v' (line 1386)
    v_56748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 16), 'v', False)
    list_56754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1386, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1386, 16), list_56754, v_56748)
    # Processing the call keyword arguments (line 1386)
    kwargs_56755 = {}
    # Getting the type of 'max' (line 1386)
    max_56747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 11), 'max', False)
    # Calling max(args, kwargs) (line 1386)
    max_call_result_56756 = invoke(stypy.reporting.localization.Localization(__file__, 1386, 11), max_56747, *[list_56754], **kwargs_56755)
    
    # Assigning a type to the variable 'maxy' (line 1386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1386, 4), 'maxy', max_call_result_56756)
    
    # Assigning a Tuple to a Name (line 1387):
    
    # Assigning a Tuple to a Name (line 1387):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1387)
    tuple_56757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1387, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1387)
    # Adding element type (line 1387)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1387)
    tuple_56758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1387, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1387)
    # Adding element type (line 1387)
    # Getting the type of 'minpy' (line 1387)
    minpy_56759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 15), 'minpy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1387, 15), tuple_56758, minpy_56759)
    # Adding element type (line 1387)
    # Getting the type of 'miny' (line 1387)
    miny_56760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 22), 'miny')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1387, 15), tuple_56758, miny_56760)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1387, 14), tuple_56757, tuple_56758)
    # Adding element type (line 1387)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1387)
    tuple_56761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1387, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1387)
    # Adding element type (line 1387)
    # Getting the type of 'maxx' (line 1387)
    maxx_56762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 30), 'maxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1387, 30), tuple_56761, maxx_56762)
    # Adding element type (line 1387)
    # Getting the type of 'maxy' (line 1387)
    maxy_56763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 36), 'maxy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1387, 30), tuple_56761, maxy_56763)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1387, 14), tuple_56757, tuple_56761)
    
    # Assigning a type to the variable 'corners' (line 1387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1387, 4), 'corners', tuple_56757)
    
    # Call to update_datalim(...): (line 1388)
    # Processing the call arguments (line 1388)
    # Getting the type of 'corners' (line 1388)
    corners_56766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 22), 'corners', False)
    # Processing the call keyword arguments (line 1388)
    kwargs_56767 = {}
    # Getting the type of 'ax' (line 1388)
    ax_56764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'ax', False)
    # Obtaining the member 'update_datalim' of a type (line 1388)
    update_datalim_56765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1388, 4), ax_56764, 'update_datalim')
    # Calling update_datalim(args, kwargs) (line 1388)
    update_datalim_call_result_56768 = invoke(stypy.reporting.localization.Localization(__file__, 1388, 4), update_datalim_56765, *[corners_56766], **kwargs_56767)
    
    
    # Call to autoscale_view(...): (line 1389)
    # Processing the call keyword arguments (line 1389)
    kwargs_56771 = {}
    # Getting the type of 'ax' (line 1389)
    ax_56769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'ax', False)
    # Obtaining the member 'autoscale_view' of a type (line 1389)
    autoscale_view_56770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 4), ax_56769, 'autoscale_view')
    # Calling autoscale_view(args, kwargs) (line 1389)
    autoscale_view_call_result_56772 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 4), autoscale_view_56770, *[], **kwargs_56771)
    
    
    # Call to add_collection(...): (line 1392)
    # Processing the call arguments (line 1392)
    # Getting the type of 'barCollection' (line 1392)
    barCollection_56775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 22), 'barCollection', False)
    # Processing the call keyword arguments (line 1392)
    kwargs_56776 = {}
    # Getting the type of 'ax' (line 1392)
    ax_56773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'ax', False)
    # Obtaining the member 'add_collection' of a type (line 1392)
    add_collection_56774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 4), ax_56773, 'add_collection')
    # Calling add_collection(args, kwargs) (line 1392)
    add_collection_call_result_56777 = invoke(stypy.reporting.localization.Localization(__file__, 1392, 4), add_collection_56774, *[barCollection_56775], **kwargs_56776)
    
    # Getting the type of 'barCollection' (line 1393)
    barCollection_56778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 11), 'barCollection')
    # Assigning a type to the variable 'stypy_return_type' (line 1393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1393, 4), 'stypy_return_type', barCollection_56778)
    
    # ################# End of 'index_bar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'index_bar' in the type store
    # Getting the type of 'stypy_return_type' (line 1331)
    stypy_return_type_56779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_56779)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'index_bar'
    return stypy_return_type_56779

# Assigning a type to the variable 'index_bar' (line 1331)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1331, 0), 'index_bar', index_bar)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
