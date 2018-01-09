
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: import datetime
3: import calendar
4: 
5: import operator
6: from math import copysign
7: 
8: from six import integer_types
9: from warnings import warn
10: 
11: from ._common import weekday
12: 
13: MO, TU, WE, TH, FR, SA, SU = weekdays = tuple(weekday(x) for x in range(7))
14: 
15: __all__ = ["relativedelta", "MO", "TU", "WE", "TH", "FR", "SA", "SU"]
16: 
17: 
18: class relativedelta(object):
19:     '''
20:     The relativedelta type is based on the specification of the excellent
21:     work done by M.-A. Lemburg in his
22:     `mx.DateTime <http://www.egenix.com/files/python/mxDateTime.html>`_ extension.
23:     However, notice that this type does *NOT* implement the same algorithm as
24:     his work. Do *NOT* expect it to behave like mx.DateTime's counterpart.
25: 
26:     There are two different ways to build a relativedelta instance. The
27:     first one is passing it two date/datetime classes::
28: 
29:         relativedelta(datetime1, datetime2)
30: 
31:     The second one is passing it any number of the following keyword arguments::
32: 
33:         relativedelta(arg1=x,arg2=y,arg3=z...)
34: 
35:         year, month, day, hour, minute, second, microsecond:
36:             Absolute information (argument is singular); adding or subtracting a
37:             relativedelta with absolute information does not perform an aritmetic
38:             operation, but rather REPLACES the corresponding value in the
39:             original datetime with the value(s) in relativedelta.
40: 
41:         years, months, weeks, days, hours, minutes, seconds, microseconds:
42:             Relative information, may be negative (argument is plural); adding
43:             or subtracting a relativedelta with relative information performs
44:             the corresponding aritmetic operation on the original datetime value
45:             with the information in the relativedelta.
46: 
47:         weekday:
48:             One of the weekday instances (MO, TU, etc). These instances may
49:             receive a parameter N, specifying the Nth weekday, which could
50:             be positive or negative (like MO(+1) or MO(-2). Not specifying
51:             it is the same as specifying +1. You can also use an integer,
52:             where 0=MO.
53: 
54:         leapdays:
55:             Will add given days to the date found, if year is a leap
56:             year, and the date found is post 28 of february.
57: 
58:         yearday, nlyearday:
59:             Set the yearday or the non-leap year day (jump leap days).
60:             These are converted to day/month/leapdays information.
61: 
62:     Here is the behavior of operations with relativedelta:
63: 
64:     1. Calculate the absolute year, using the 'year' argument, or the
65:        original datetime year, if the argument is not present.
66: 
67:     2. Add the relative 'years' argument to the absolute year.
68: 
69:     3. Do steps 1 and 2 for month/months.
70: 
71:     4. Calculate the absolute day, using the 'day' argument, or the
72:        original datetime day, if the argument is not present. Then,
73:        subtract from the day until it fits in the year and month
74:        found after their operations.
75: 
76:     5. Add the relative 'days' argument to the absolute day. Notice
77:        that the 'weeks' argument is multiplied by 7 and added to
78:        'days'.
79: 
80:     6. Do steps 1 and 2 for hour/hours, minute/minutes, second/seconds,
81:        microsecond/microseconds.
82: 
83:     7. If the 'weekday' argument is present, calculate the weekday,
84:        with the given (wday, nth) tuple. wday is the index of the
85:        weekday (0-6, 0=Mon), and nth is the number of weeks to add
86:        forward or backward, depending on its signal. Notice that if
87:        the calculated date is already Monday, for example, using
88:        (0, 1) or (0, -1) won't change the day.
89:     '''
90: 
91:     def __init__(self, dt1=None, dt2=None,
92:                  years=0, months=0, days=0, leapdays=0, weeks=0,
93:                  hours=0, minutes=0, seconds=0, microseconds=0,
94:                  year=None, month=None, day=None, weekday=None,
95:                  yearday=None, nlyearday=None,
96:                  hour=None, minute=None, second=None, microsecond=None):
97: 
98:         # Check for non-integer values in integer-only quantities
99:         if any(x is not None and x != int(x) for x in (years, months)):
100:             raise ValueError("Non-integer years and months are "
101:                              "ambiguous and not currently supported.")
102: 
103:         if dt1 and dt2:
104:             # datetime is a subclass of date. So both must be date
105:             if not (isinstance(dt1, datetime.date) and
106:                     isinstance(dt2, datetime.date)):
107:                 raise TypeError("relativedelta only diffs datetime/date")
108: 
109:             # We allow two dates, or two datetimes, so we coerce them to be
110:             # of the same type
111:             if (isinstance(dt1, datetime.datetime) !=
112:                     isinstance(dt2, datetime.datetime)):
113:                 if not isinstance(dt1, datetime.datetime):
114:                     dt1 = datetime.datetime.fromordinal(dt1.toordinal())
115:                 elif not isinstance(dt2, datetime.datetime):
116:                     dt2 = datetime.datetime.fromordinal(dt2.toordinal())
117: 
118:             self.years = 0
119:             self.months = 0
120:             self.days = 0
121:             self.leapdays = 0
122:             self.hours = 0
123:             self.minutes = 0
124:             self.seconds = 0
125:             self.microseconds = 0
126:             self.year = None
127:             self.month = None
128:             self.day = None
129:             self.weekday = None
130:             self.hour = None
131:             self.minute = None
132:             self.second = None
133:             self.microsecond = None
134:             self._has_time = 0
135: 
136:             # Get year / month delta between the two
137:             months = (dt1.year - dt2.year) * 12 + (dt1.month - dt2.month)
138:             self._set_months(months)
139: 
140:             # Remove the year/month delta so the timedelta is just well-defined
141:             # time units (seconds, days and microseconds)
142:             dtm = self.__radd__(dt2)
143: 
144:             # If we've overshot our target, make an adjustment
145:             if dt1 < dt2:
146:                 compare = operator.gt
147:                 increment = 1
148:             else:
149:                 compare = operator.lt
150:                 increment = -1
151: 
152:             while compare(dt1, dtm):
153:                 months += increment
154:                 self._set_months(months)
155:                 dtm = self.__radd__(dt2)
156: 
157:             # Get the timedelta between the "months-adjusted" date and dt1
158:             delta = dt1 - dtm
159:             self.seconds = delta.seconds + delta.days * 86400
160:             self.microseconds = delta.microseconds
161:         else:
162:             # Relative information
163:             self.years = years
164:             self.months = months
165:             self.days = days + weeks * 7
166:             self.leapdays = leapdays
167:             self.hours = hours
168:             self.minutes = minutes
169:             self.seconds = seconds
170:             self.microseconds = microseconds
171: 
172:             # Absolute information
173:             self.year = year
174:             self.month = month
175:             self.day = day
176:             self.hour = hour
177:             self.minute = minute
178:             self.second = second
179:             self.microsecond = microsecond
180: 
181:             if any(x is not None and int(x) != x
182:                    for x in (year, month, day, hour,
183:                              minute, second, microsecond)):
184:                 # For now we'll deprecate floats - later it'll be an error.
185:                 warn("Non-integer value passed as absolute information. " +
186:                      "This is not a well-defined condition and will raise " +
187:                      "errors in future versions.", DeprecationWarning)
188: 
189:             if isinstance(weekday, integer_types):
190:                 self.weekday = weekdays[weekday]
191:             else:
192:                 self.weekday = weekday
193: 
194:             yday = 0
195:             if nlyearday:
196:                 yday = nlyearday
197:             elif yearday:
198:                 yday = yearday
199:                 if yearday > 59:
200:                     self.leapdays = -1
201:             if yday:
202:                 ydayidx = [31, 59, 90, 120, 151, 181, 212,
203:                            243, 273, 304, 334, 366]
204:                 for idx, ydays in enumerate(ydayidx):
205:                     if yday <= ydays:
206:                         self.month = idx+1
207:                         if idx == 0:
208:                             self.day = yday
209:                         else:
210:                             self.day = yday-ydayidx[idx-1]
211:                         break
212:                 else:
213:                     raise ValueError("invalid year day (%d)" % yday)
214: 
215:         self._fix()
216: 
217:     def _fix(self):
218:         if abs(self.microseconds) > 999999:
219:             s = _sign(self.microseconds)
220:             div, mod = divmod(self.microseconds * s, 1000000)
221:             self.microseconds = mod * s
222:             self.seconds += div * s
223:         if abs(self.seconds) > 59:
224:             s = _sign(self.seconds)
225:             div, mod = divmod(self.seconds * s, 60)
226:             self.seconds = mod * s
227:             self.minutes += div * s
228:         if abs(self.minutes) > 59:
229:             s = _sign(self.minutes)
230:             div, mod = divmod(self.minutes * s, 60)
231:             self.minutes = mod * s
232:             self.hours += div * s
233:         if abs(self.hours) > 23:
234:             s = _sign(self.hours)
235:             div, mod = divmod(self.hours * s, 24)
236:             self.hours = mod * s
237:             self.days += div * s
238:         if abs(self.months) > 11:
239:             s = _sign(self.months)
240:             div, mod = divmod(self.months * s, 12)
241:             self.months = mod * s
242:             self.years += div * s
243:         if (self.hours or self.minutes or self.seconds or self.microseconds
244:                 or self.hour is not None or self.minute is not None or
245:                 self.second is not None or self.microsecond is not None):
246:             self._has_time = 1
247:         else:
248:             self._has_time = 0
249: 
250:     @property
251:     def weeks(self):
252:         return self.days // 7
253: 
254:     @weeks.setter
255:     def weeks(self, value):
256:         self.days = self.days - (self.weeks * 7) + value * 7
257: 
258:     def _set_months(self, months):
259:         self.months = months
260:         if abs(self.months) > 11:
261:             s = _sign(self.months)
262:             div, mod = divmod(self.months * s, 12)
263:             self.months = mod * s
264:             self.years = div * s
265:         else:
266:             self.years = 0
267: 
268:     def normalized(self):
269:         '''
270:         Return a version of this object represented entirely using integer
271:         values for the relative attributes.
272: 
273:         >>> relativedelta(days=1.5, hours=2).normalized()
274:         relativedelta(days=1, hours=14)
275: 
276:         :return:
277:             Returns a :class:`dateutil.relativedelta.relativedelta` object.
278:         '''
279:         # Cascade remainders down (rounding each to roughly nearest microsecond)
280:         days = int(self.days)
281: 
282:         hours_f = round(self.hours + 24 * (self.days - days), 11)
283:         hours = int(hours_f)
284: 
285:         minutes_f = round(self.minutes + 60 * (hours_f - hours), 10)
286:         minutes = int(minutes_f)
287: 
288:         seconds_f = round(self.seconds + 60 * (minutes_f - minutes), 8)
289:         seconds = int(seconds_f)
290: 
291:         microseconds = round(self.microseconds + 1e6 * (seconds_f - seconds))
292: 
293:         # Constructor carries overflow back up with call to _fix()
294:         return self.__class__(years=self.years, months=self.months,
295:                               days=days, hours=hours, minutes=minutes,
296:                               seconds=seconds, microseconds=microseconds,
297:                               leapdays=self.leapdays, year=self.year,
298:                               month=self.month, day=self.day,
299:                               weekday=self.weekday, hour=self.hour,
300:                               minute=self.minute, second=self.second,
301:                               microsecond=self.microsecond)
302: 
303:     def __add__(self, other):
304:         if isinstance(other, relativedelta):
305:             return self.__class__(years=other.years + self.years,
306:                                  months=other.months + self.months,
307:                                  days=other.days + self.days,
308:                                  hours=other.hours + self.hours,
309:                                  minutes=other.minutes + self.minutes,
310:                                  seconds=other.seconds + self.seconds,
311:                                  microseconds=(other.microseconds +
312:                                                self.microseconds),
313:                                  leapdays=other.leapdays or self.leapdays,
314:                                  year=(other.year if other.year is not None
315:                                        else self.year),
316:                                  month=(other.month if other.month is not None
317:                                         else self.month),
318:                                  day=(other.day if other.day is not None
319:                                       else self.day),
320:                                  weekday=(other.weekday if other.weekday is not None
321:                                           else self.weekday),
322:                                  hour=(other.hour if other.hour is not None
323:                                        else self.hour),
324:                                  minute=(other.minute if other.minute is not None
325:                                          else self.minute),
326:                                  second=(other.second if other.second is not None
327:                                          else self.second),
328:                                  microsecond=(other.microsecond if other.microsecond
329:                                               is not None else
330:                                               self.microsecond))
331:         if isinstance(other, datetime.timedelta):
332:             return self.__class__(years=self.years,
333:                                   months=self.months,
334:                                   days=self.days + other.days,
335:                                   hours=self.hours,
336:                                   minutes=self.minutes,
337:                                   seconds=self.seconds + other.seconds,
338:                                   microseconds=self.microseconds + other.microseconds,
339:                                   leapdays=self.leapdays,
340:                                   year=self.year,
341:                                   month=self.month,
342:                                   day=self.day,
343:                                   weekday=self.weekday,
344:                                   hour=self.hour,
345:                                   minute=self.minute,
346:                                   second=self.second,
347:                                   microsecond=self.microsecond)
348:         if not isinstance(other, datetime.date):
349:             return NotImplemented
350:         elif self._has_time and not isinstance(other, datetime.datetime):
351:             other = datetime.datetime.fromordinal(other.toordinal())
352:         year = (self.year or other.year)+self.years
353:         month = self.month or other.month
354:         if self.months:
355:             assert 1 <= abs(self.months) <= 12
356:             month += self.months
357:             if month > 12:
358:                 year += 1
359:                 month -= 12
360:             elif month < 1:
361:                 year -= 1
362:                 month += 12
363:         day = min(calendar.monthrange(year, month)[1],
364:                   self.day or other.day)
365:         repl = {"year": year, "month": month, "day": day}
366:         for attr in ["hour", "minute", "second", "microsecond"]:
367:             value = getattr(self, attr)
368:             if value is not None:
369:                 repl[attr] = value
370:         days = self.days
371:         if self.leapdays and month > 2 and calendar.isleap(year):
372:             days += self.leapdays
373:         ret = (other.replace(**repl)
374:                + datetime.timedelta(days=days,
375:                                     hours=self.hours,
376:                                     minutes=self.minutes,
377:                                     seconds=self.seconds,
378:                                     microseconds=self.microseconds))
379:         if self.weekday:
380:             weekday, nth = self.weekday.weekday, self.weekday.n or 1
381:             jumpdays = (abs(nth) - 1) * 7
382:             if nth > 0:
383:                 jumpdays += (7 - ret.weekday() + weekday) % 7
384:             else:
385:                 jumpdays += (ret.weekday() - weekday) % 7
386:                 jumpdays *= -1
387:             ret += datetime.timedelta(days=jumpdays)
388:         return ret
389: 
390:     def __radd__(self, other):
391:         return self.__add__(other)
392: 
393:     def __rsub__(self, other):
394:         return self.__neg__().__radd__(other)
395: 
396:     def __sub__(self, other):
397:         if not isinstance(other, relativedelta):
398:             return NotImplemented   # In case the other object defines __rsub__
399:         return self.__class__(years=self.years - other.years,
400:                              months=self.months - other.months,
401:                              days=self.days - other.days,
402:                              hours=self.hours - other.hours,
403:                              minutes=self.minutes - other.minutes,
404:                              seconds=self.seconds - other.seconds,
405:                              microseconds=self.microseconds - other.microseconds,
406:                              leapdays=self.leapdays or other.leapdays,
407:                              year=(self.year if self.year is not None
408:                                    else other.year),
409:                              month=(self.month if self.month is not None else
410:                                     other.month),
411:                              day=(self.day if self.day is not None else
412:                                   other.day),
413:                              weekday=(self.weekday if self.weekday is not None else
414:                                       other.weekday),
415:                              hour=(self.hour if self.hour is not None else
416:                                    other.hour),
417:                              minute=(self.minute if self.minute is not None else
418:                                      other.minute),
419:                              second=(self.second if self.second is not None else
420:                                      other.second),
421:                              microsecond=(self.microsecond if self.microsecond
422:                                           is not None else
423:                                           other.microsecond))
424: 
425:     def __neg__(self):
426:         return self.__class__(years=-self.years,
427:                              months=-self.months,
428:                              days=-self.days,
429:                              hours=-self.hours,
430:                              minutes=-self.minutes,
431:                              seconds=-self.seconds,
432:                              microseconds=-self.microseconds,
433:                              leapdays=self.leapdays,
434:                              year=self.year,
435:                              month=self.month,
436:                              day=self.day,
437:                              weekday=self.weekday,
438:                              hour=self.hour,
439:                              minute=self.minute,
440:                              second=self.second,
441:                              microsecond=self.microsecond)
442: 
443:     def __bool__(self):
444:         return not (not self.years and
445:                     not self.months and
446:                     not self.days and
447:                     not self.hours and
448:                     not self.minutes and
449:                     not self.seconds and
450:                     not self.microseconds and
451:                     not self.leapdays and
452:                     self.year is None and
453:                     self.month is None and
454:                     self.day is None and
455:                     self.weekday is None and
456:                     self.hour is None and
457:                     self.minute is None and
458:                     self.second is None and
459:                     self.microsecond is None)
460:     # Compatibility with Python 2.x
461:     __nonzero__ = __bool__
462: 
463:     def __mul__(self, other):
464:         try:
465:             f = float(other)
466:         except TypeError:
467:             return NotImplemented
468: 
469:         return self.__class__(years=int(self.years * f),
470:                              months=int(self.months * f),
471:                              days=int(self.days * f),
472:                              hours=int(self.hours * f),
473:                              minutes=int(self.minutes * f),
474:                              seconds=int(self.seconds * f),
475:                              microseconds=int(self.microseconds * f),
476:                              leapdays=self.leapdays,
477:                              year=self.year,
478:                              month=self.month,
479:                              day=self.day,
480:                              weekday=self.weekday,
481:                              hour=self.hour,
482:                              minute=self.minute,
483:                              second=self.second,
484:                              microsecond=self.microsecond)
485: 
486:     __rmul__ = __mul__
487: 
488:     def __eq__(self, other):
489:         if not isinstance(other, relativedelta):
490:             return NotImplemented
491:         if self.weekday or other.weekday:
492:             if not self.weekday or not other.weekday:
493:                 return False
494:             if self.weekday.weekday != other.weekday.weekday:
495:                 return False
496:             n1, n2 = self.weekday.n, other.weekday.n
497:             if n1 != n2 and not ((not n1 or n1 == 1) and (not n2 or n2 == 1)):
498:                 return False
499:         return (self.years == other.years and
500:                 self.months == other.months and
501:                 self.days == other.days and
502:                 self.hours == other.hours and
503:                 self.minutes == other.minutes and
504:                 self.seconds == other.seconds and
505:                 self.microseconds == other.microseconds and
506:                 self.leapdays == other.leapdays and
507:                 self.year == other.year and
508:                 self.month == other.month and
509:                 self.day == other.day and
510:                 self.hour == other.hour and
511:                 self.minute == other.minute and
512:                 self.second == other.second and
513:                 self.microsecond == other.microsecond)
514: 
515:     __hash__ = None
516: 
517:     def __ne__(self, other):
518:         return not self.__eq__(other)
519: 
520:     def __div__(self, other):
521:         try:
522:             reciprocal = 1 / float(other)
523:         except TypeError:
524:             return NotImplemented
525: 
526:         return self.__mul__(reciprocal)
527: 
528:     __truediv__ = __div__
529: 
530:     def __repr__(self):
531:         l = []
532:         for attr in ["years", "months", "days", "leapdays",
533:                      "hours", "minutes", "seconds", "microseconds"]:
534:             value = getattr(self, attr)
535:             if value:
536:                 l.append("{attr}={value:+g}".format(attr=attr, value=value))
537:         for attr in ["year", "month", "day", "weekday",
538:                      "hour", "minute", "second", "microsecond"]:
539:             value = getattr(self, attr)
540:             if value is not None:
541:                 l.append("{attr}={value}".format(attr=attr, value=repr(value)))
542:         return "{classname}({attrs})".format(classname=self.__class__.__name__,
543:                                              attrs=", ".join(l))
544: 
545: 
546: def _sign(x):
547:     return int(copysign(1, x))
548: 
549: # vim:ts=4:sw=4:et
550: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import datetime' statement (line 2)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import calendar' statement (line 3)
import calendar

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'calendar', calendar, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import operator' statement (line 5)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from math import copysign' statement (line 6)
try:
    from math import copysign

except:
    copysign = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'math', None, module_type_store, ['copysign'], [copysign])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from six import integer_types' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/')
import_313106 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six')

if (type(import_313106) is not StypyTypeError):

    if (import_313106 != 'pyd_module'):
        __import__(import_313106)
        sys_modules_313107 = sys.modules[import_313106]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', sys_modules_313107.module_type_store, module_type_store, ['integer_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_313107, sys_modules_313107.module_type_store, module_type_store)
    else:
        from six import integer_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', None, module_type_store, ['integer_types'], [integer_types])

else:
    # Assigning a type to the variable 'six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', import_313106)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from warnings import warn' statement (line 9)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from dateutil._common import weekday' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/')
import_313108 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'dateutil._common')

if (type(import_313108) is not StypyTypeError):

    if (import_313108 != 'pyd_module'):
        __import__(import_313108)
        sys_modules_313109 = sys.modules[import_313108]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'dateutil._common', sys_modules_313109.module_type_store, module_type_store, ['weekday'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_313109, sys_modules_313109.module_type_store, module_type_store)
    else:
        from dateutil._common import weekday

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'dateutil._common', None, module_type_store, ['weekday'], [weekday])

else:
    # Assigning a type to the variable 'dateutil._common' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'dateutil._common', import_313108)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/')


# Multiple assignment of 2 elements.

# Assigning a Call to a Name (line 13):

# Call to tuple(...): (line 13)
# Processing the call arguments (line 13)
# Calculating generator expression
module_type_store = module_type_store.open_function_context('list comprehension expression', 13, 46, True)
# Calculating comprehension expression

# Call to range(...): (line 13)
# Processing the call arguments (line 13)
int_313116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 72), 'int')
# Processing the call keyword arguments (line 13)
kwargs_313117 = {}
# Getting the type of 'range' (line 13)
range_313115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 66), 'range', False)
# Calling range(args, kwargs) (line 13)
range_call_result_313118 = invoke(stypy.reporting.localization.Localization(__file__, 13, 66), range_313115, *[int_313116], **kwargs_313117)

comprehension_313119 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 46), range_call_result_313118)
# Assigning a type to the variable 'x' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 46), 'x', comprehension_313119)

# Call to weekday(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'x' (line 13)
x_313112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 54), 'x', False)
# Processing the call keyword arguments (line 13)
kwargs_313113 = {}
# Getting the type of 'weekday' (line 13)
weekday_313111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 46), 'weekday', False)
# Calling weekday(args, kwargs) (line 13)
weekday_call_result_313114 = invoke(stypy.reporting.localization.Localization(__file__, 13, 46), weekday_313111, *[x_313112], **kwargs_313113)

list_313120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 46), 'list')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 46), list_313120, weekday_call_result_313114)
# Processing the call keyword arguments (line 13)
kwargs_313121 = {}
# Getting the type of 'tuple' (line 13)
tuple_313110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 40), 'tuple', False)
# Calling tuple(args, kwargs) (line 13)
tuple_call_result_313122 = invoke(stypy.reporting.localization.Localization(__file__, 13, 40), tuple_313110, *[list_313120], **kwargs_313121)

# Assigning a type to the variable 'weekdays' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'weekdays', tuple_call_result_313122)

# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript
int_313123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 0), 'int')
# Getting the type of 'weekdays' (line 13)
weekdays_313124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'weekdays')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___313125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), weekdays_313124, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_313126 = invoke(stypy.reporting.localization.Localization(__file__, 13, 0), getitem___313125, int_313123)

# Assigning a type to the variable 'tuple_var_assignment_313077' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313077', subscript_call_result_313126)

# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript
int_313127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 0), 'int')
# Getting the type of 'weekdays' (line 13)
weekdays_313128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'weekdays')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___313129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), weekdays_313128, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_313130 = invoke(stypy.reporting.localization.Localization(__file__, 13, 0), getitem___313129, int_313127)

# Assigning a type to the variable 'tuple_var_assignment_313078' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313078', subscript_call_result_313130)

# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript
int_313131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 0), 'int')
# Getting the type of 'weekdays' (line 13)
weekdays_313132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'weekdays')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___313133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), weekdays_313132, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_313134 = invoke(stypy.reporting.localization.Localization(__file__, 13, 0), getitem___313133, int_313131)

# Assigning a type to the variable 'tuple_var_assignment_313079' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313079', subscript_call_result_313134)

# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript
int_313135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 0), 'int')
# Getting the type of 'weekdays' (line 13)
weekdays_313136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'weekdays')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___313137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), weekdays_313136, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_313138 = invoke(stypy.reporting.localization.Localization(__file__, 13, 0), getitem___313137, int_313135)

# Assigning a type to the variable 'tuple_var_assignment_313080' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313080', subscript_call_result_313138)

# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript
int_313139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 0), 'int')
# Getting the type of 'weekdays' (line 13)
weekdays_313140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'weekdays')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___313141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), weekdays_313140, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_313142 = invoke(stypy.reporting.localization.Localization(__file__, 13, 0), getitem___313141, int_313139)

# Assigning a type to the variable 'tuple_var_assignment_313081' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313081', subscript_call_result_313142)

# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript
int_313143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 0), 'int')
# Getting the type of 'weekdays' (line 13)
weekdays_313144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'weekdays')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___313145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), weekdays_313144, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_313146 = invoke(stypy.reporting.localization.Localization(__file__, 13, 0), getitem___313145, int_313143)

# Assigning a type to the variable 'tuple_var_assignment_313082' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313082', subscript_call_result_313146)

# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript
int_313147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 0), 'int')
# Getting the type of 'weekdays' (line 13)
weekdays_313148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'weekdays')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___313149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), weekdays_313148, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_313150 = invoke(stypy.reporting.localization.Localization(__file__, 13, 0), getitem___313149, int_313147)

# Assigning a type to the variable 'tuple_var_assignment_313083' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313083', subscript_call_result_313150)

# Assigning a Name to a Name (line 13):
# Getting the type of 'tuple_var_assignment_313077' (line 13)
tuple_var_assignment_313077_313151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313077')
# Assigning a type to the variable 'MO' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'MO', tuple_var_assignment_313077_313151)

# Assigning a Name to a Name (line 13):
# Getting the type of 'tuple_var_assignment_313078' (line 13)
tuple_var_assignment_313078_313152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313078')
# Assigning a type to the variable 'TU' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'TU', tuple_var_assignment_313078_313152)

# Assigning a Name to a Name (line 13):
# Getting the type of 'tuple_var_assignment_313079' (line 13)
tuple_var_assignment_313079_313153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313079')
# Assigning a type to the variable 'WE' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'WE', tuple_var_assignment_313079_313153)

# Assigning a Name to a Name (line 13):
# Getting the type of 'tuple_var_assignment_313080' (line 13)
tuple_var_assignment_313080_313154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313080')
# Assigning a type to the variable 'TH' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'TH', tuple_var_assignment_313080_313154)

# Assigning a Name to a Name (line 13):
# Getting the type of 'tuple_var_assignment_313081' (line 13)
tuple_var_assignment_313081_313155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313081')
# Assigning a type to the variable 'FR' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'FR', tuple_var_assignment_313081_313155)

# Assigning a Name to a Name (line 13):
# Getting the type of 'tuple_var_assignment_313082' (line 13)
tuple_var_assignment_313082_313156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313082')
# Assigning a type to the variable 'SA' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'SA', tuple_var_assignment_313082_313156)

# Assigning a Name to a Name (line 13):
# Getting the type of 'tuple_var_assignment_313083' (line 13)
tuple_var_assignment_313083_313157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_var_assignment_313083')
# Assigning a type to the variable 'SU' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'SU', tuple_var_assignment_313083_313157)

# Assigning a List to a Name (line 15):

# Assigning a List to a Name (line 15):
__all__ = ['relativedelta', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU']
module_type_store.set_exportable_members(['relativedelta', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU'])

# Obtaining an instance of the builtin type 'list' (line 15)
list_313158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_313159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'relativedelta')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_313158, str_313159)
# Adding element type (line 15)
str_313160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 28), 'str', 'MO')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_313158, str_313160)
# Adding element type (line 15)
str_313161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'str', 'TU')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_313158, str_313161)
# Adding element type (line 15)
str_313162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 40), 'str', 'WE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_313158, str_313162)
# Adding element type (line 15)
str_313163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 46), 'str', 'TH')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_313158, str_313163)
# Adding element type (line 15)
str_313164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 52), 'str', 'FR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_313158, str_313164)
# Adding element type (line 15)
str_313165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 58), 'str', 'SA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_313158, str_313165)
# Adding element type (line 15)
str_313166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 64), 'str', 'SU')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_313158, str_313166)

# Assigning a type to the variable '__all__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__all__', list_313158)
# Declaration of the 'relativedelta' class

class relativedelta(object, ):
    str_313167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', "\n    The relativedelta type is based on the specification of the excellent\n    work done by M.-A. Lemburg in his\n    `mx.DateTime <http://www.egenix.com/files/python/mxDateTime.html>`_ extension.\n    However, notice that this type does *NOT* implement the same algorithm as\n    his work. Do *NOT* expect it to behave like mx.DateTime's counterpart.\n\n    There are two different ways to build a relativedelta instance. The\n    first one is passing it two date/datetime classes::\n\n        relativedelta(datetime1, datetime2)\n\n    The second one is passing it any number of the following keyword arguments::\n\n        relativedelta(arg1=x,arg2=y,arg3=z...)\n\n        year, month, day, hour, minute, second, microsecond:\n            Absolute information (argument is singular); adding or subtracting a\n            relativedelta with absolute information does not perform an aritmetic\n            operation, but rather REPLACES the corresponding value in the\n            original datetime with the value(s) in relativedelta.\n\n        years, months, weeks, days, hours, minutes, seconds, microseconds:\n            Relative information, may be negative (argument is plural); adding\n            or subtracting a relativedelta with relative information performs\n            the corresponding aritmetic operation on the original datetime value\n            with the information in the relativedelta.\n\n        weekday:\n            One of the weekday instances (MO, TU, etc). These instances may\n            receive a parameter N, specifying the Nth weekday, which could\n            be positive or negative (like MO(+1) or MO(-2). Not specifying\n            it is the same as specifying +1. You can also use an integer,\n            where 0=MO.\n\n        leapdays:\n            Will add given days to the date found, if year is a leap\n            year, and the date found is post 28 of february.\n\n        yearday, nlyearday:\n            Set the yearday or the non-leap year day (jump leap days).\n            These are converted to day/month/leapdays information.\n\n    Here is the behavior of operations with relativedelta:\n\n    1. Calculate the absolute year, using the 'year' argument, or the\n       original datetime year, if the argument is not present.\n\n    2. Add the relative 'years' argument to the absolute year.\n\n    3. Do steps 1 and 2 for month/months.\n\n    4. Calculate the absolute day, using the 'day' argument, or the\n       original datetime day, if the argument is not present. Then,\n       subtract from the day until it fits in the year and month\n       found after their operations.\n\n    5. Add the relative 'days' argument to the absolute day. Notice\n       that the 'weeks' argument is multiplied by 7 and added to\n       'days'.\n\n    6. Do steps 1 and 2 for hour/hours, minute/minutes, second/seconds,\n       microsecond/microseconds.\n\n    7. If the 'weekday' argument is present, calculate the weekday,\n       with the given (wday, nth) tuple. wday is the index of the\n       weekday (0-6, 0=Mon), and nth is the number of weeks to add\n       forward or backward, depending on its signal. Notice that if\n       the calculated date is already Monday, for example, using\n       (0, 1) or (0, -1) won't change the day.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 91)
        None_313168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'None')
        # Getting the type of 'None' (line 91)
        None_313169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'None')
        int_313170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'int')
        int_313171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 33), 'int')
        int_313172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 41), 'int')
        int_313173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 53), 'int')
        int_313174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 62), 'int')
        int_313175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 23), 'int')
        int_313176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 34), 'int')
        int_313177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 45), 'int')
        int_313178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 61), 'int')
        # Getting the type of 'None' (line 94)
        None_313179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 22), 'None')
        # Getting the type of 'None' (line 94)
        None_313180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 34), 'None')
        # Getting the type of 'None' (line 94)
        None_313181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), 'None')
        # Getting the type of 'None' (line 94)
        None_313182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 58), 'None')
        # Getting the type of 'None' (line 95)
        None_313183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'None')
        # Getting the type of 'None' (line 95)
        None_313184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 41), 'None')
        # Getting the type of 'None' (line 96)
        None_313185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'None')
        # Getting the type of 'None' (line 96)
        None_313186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 35), 'None')
        # Getting the type of 'None' (line 96)
        None_313187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 48), 'None')
        # Getting the type of 'None' (line 96)
        None_313188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 66), 'None')
        defaults = [None_313168, None_313169, int_313170, int_313171, int_313172, int_313173, int_313174, int_313175, int_313176, int_313177, int_313178, None_313179, None_313180, None_313181, None_313182, None_313183, None_313184, None_313185, None_313186, None_313187, None_313188]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__init__', ['dt1', 'dt2', 'years', 'months', 'days', 'leapdays', 'weeks', 'hours', 'minutes', 'seconds', 'microseconds', 'year', 'month', 'day', 'weekday', 'yearday', 'nlyearday', 'hour', 'minute', 'second', 'microsecond'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dt1', 'dt2', 'years', 'months', 'days', 'leapdays', 'weeks', 'hours', 'minutes', 'seconds', 'microseconds', 'year', 'month', 'day', 'weekday', 'yearday', 'nlyearday', 'hour', 'minute', 'second', 'microsecond'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Call to any(...): (line 99)
        # Processing the call arguments (line 99)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 99, 15, True)
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_313200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        # Getting the type of 'years' (line 99)
        years_313201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 55), 'years', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 55), tuple_313200, years_313201)
        # Adding element type (line 99)
        # Getting the type of 'months' (line 99)
        months_313202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 62), 'months', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 55), tuple_313200, months_313202)
        
        comprehension_313203 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 15), tuple_313200)
        # Assigning a type to the variable 'x' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'x', comprehension_313203)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 99)
        x_313190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'x', False)
        # Getting the type of 'None' (line 99)
        None_313191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'None', False)
        # Applying the binary operator 'isnot' (line 99)
        result_is_not_313192 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 15), 'isnot', x_313190, None_313191)
        
        
        # Getting the type of 'x' (line 99)
        x_313193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'x', False)
        
        # Call to int(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'x' (line 99)
        x_313195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 42), 'x', False)
        # Processing the call keyword arguments (line 99)
        kwargs_313196 = {}
        # Getting the type of 'int' (line 99)
        int_313194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'int', False)
        # Calling int(args, kwargs) (line 99)
        int_call_result_313197 = invoke(stypy.reporting.localization.Localization(__file__, 99, 38), int_313194, *[x_313195], **kwargs_313196)
        
        # Applying the binary operator '!=' (line 99)
        result_ne_313198 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 33), '!=', x_313193, int_call_result_313197)
        
        # Applying the binary operator 'and' (line 99)
        result_and_keyword_313199 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 15), 'and', result_is_not_313192, result_ne_313198)
        
        list_313204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 15), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 15), list_313204, result_and_keyword_313199)
        # Processing the call keyword arguments (line 99)
        kwargs_313205 = {}
        # Getting the type of 'any' (line 99)
        any_313189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'any', False)
        # Calling any(args, kwargs) (line 99)
        any_call_result_313206 = invoke(stypy.reporting.localization.Localization(__file__, 99, 11), any_313189, *[list_313204], **kwargs_313205)
        
        # Testing the type of an if condition (line 99)
        if_condition_313207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), any_call_result_313206)
        # Assigning a type to the variable 'if_condition_313207' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_313207', if_condition_313207)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 100)
        # Processing the call arguments (line 100)
        str_313209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'str', 'Non-integer years and months are ambiguous and not currently supported.')
        # Processing the call keyword arguments (line 100)
        kwargs_313210 = {}
        # Getting the type of 'ValueError' (line 100)
        ValueError_313208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 100)
        ValueError_call_result_313211 = invoke(stypy.reporting.localization.Localization(__file__, 100, 18), ValueError_313208, *[str_313209], **kwargs_313210)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 100, 12), ValueError_call_result_313211, 'raise parameter', BaseException)
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'dt1' (line 103)
        dt1_313212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'dt1')
        # Getting the type of 'dt2' (line 103)
        dt2_313213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'dt2')
        # Applying the binary operator 'and' (line 103)
        result_and_keyword_313214 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), 'and', dt1_313212, dt2_313213)
        
        # Testing the type of an if condition (line 103)
        if_condition_313215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), result_and_keyword_313214)
        # Assigning a type to the variable 'if_condition_313215' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_313215', if_condition_313215)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'dt1' (line 105)
        dt1_313217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'dt1', False)
        # Getting the type of 'datetime' (line 105)
        datetime_313218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 36), 'datetime', False)
        # Obtaining the member 'date' of a type (line 105)
        date_313219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 36), datetime_313218, 'date')
        # Processing the call keyword arguments (line 105)
        kwargs_313220 = {}
        # Getting the type of 'isinstance' (line 105)
        isinstance_313216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 105)
        isinstance_call_result_313221 = invoke(stypy.reporting.localization.Localization(__file__, 105, 20), isinstance_313216, *[dt1_313217, date_313219], **kwargs_313220)
        
        
        # Call to isinstance(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'dt2' (line 106)
        dt2_313223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 31), 'dt2', False)
        # Getting the type of 'datetime' (line 106)
        datetime_313224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'datetime', False)
        # Obtaining the member 'date' of a type (line 106)
        date_313225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 36), datetime_313224, 'date')
        # Processing the call keyword arguments (line 106)
        kwargs_313226 = {}
        # Getting the type of 'isinstance' (line 106)
        isinstance_313222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 106)
        isinstance_call_result_313227 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), isinstance_313222, *[dt2_313223, date_313225], **kwargs_313226)
        
        # Applying the binary operator 'and' (line 105)
        result_and_keyword_313228 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 20), 'and', isinstance_call_result_313221, isinstance_call_result_313227)
        
        # Applying the 'not' unary operator (line 105)
        result_not__313229 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 15), 'not', result_and_keyword_313228)
        
        # Testing the type of an if condition (line 105)
        if_condition_313230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 12), result_not__313229)
        # Assigning a type to the variable 'if_condition_313230' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'if_condition_313230', if_condition_313230)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 107)
        # Processing the call arguments (line 107)
        str_313232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 32), 'str', 'relativedelta only diffs datetime/date')
        # Processing the call keyword arguments (line 107)
        kwargs_313233 = {}
        # Getting the type of 'TypeError' (line 107)
        TypeError_313231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 107)
        TypeError_call_result_313234 = invoke(stypy.reporting.localization.Localization(__file__, 107, 22), TypeError_313231, *[str_313232], **kwargs_313233)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 107, 16), TypeError_call_result_313234, 'raise parameter', BaseException)
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to isinstance(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'dt1' (line 111)
        dt1_313236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'dt1', False)
        # Getting the type of 'datetime' (line 111)
        datetime_313237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 32), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 111)
        datetime_313238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 32), datetime_313237, 'datetime')
        # Processing the call keyword arguments (line 111)
        kwargs_313239 = {}
        # Getting the type of 'isinstance' (line 111)
        isinstance_313235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 111)
        isinstance_call_result_313240 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), isinstance_313235, *[dt1_313236, datetime_313238], **kwargs_313239)
        
        
        # Call to isinstance(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'dt2' (line 112)
        dt2_313242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'dt2', False)
        # Getting the type of 'datetime' (line 112)
        datetime_313243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 112)
        datetime_313244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 36), datetime_313243, 'datetime')
        # Processing the call keyword arguments (line 112)
        kwargs_313245 = {}
        # Getting the type of 'isinstance' (line 112)
        isinstance_313241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 112)
        isinstance_call_result_313246 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), isinstance_313241, *[dt2_313242, datetime_313244], **kwargs_313245)
        
        # Applying the binary operator '!=' (line 111)
        result_ne_313247 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 16), '!=', isinstance_call_result_313240, isinstance_call_result_313246)
        
        # Testing the type of an if condition (line 111)
        if_condition_313248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), result_ne_313247)
        # Assigning a type to the variable 'if_condition_313248' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_313248', if_condition_313248)
        # SSA begins for if statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to isinstance(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'dt1' (line 113)
        dt1_313250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 34), 'dt1', False)
        # Getting the type of 'datetime' (line 113)
        datetime_313251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 39), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 113)
        datetime_313252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 39), datetime_313251, 'datetime')
        # Processing the call keyword arguments (line 113)
        kwargs_313253 = {}
        # Getting the type of 'isinstance' (line 113)
        isinstance_313249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 113)
        isinstance_call_result_313254 = invoke(stypy.reporting.localization.Localization(__file__, 113, 23), isinstance_313249, *[dt1_313250, datetime_313252], **kwargs_313253)
        
        # Applying the 'not' unary operator (line 113)
        result_not__313255 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 19), 'not', isinstance_call_result_313254)
        
        # Testing the type of an if condition (line 113)
        if_condition_313256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 16), result_not__313255)
        # Assigning a type to the variable 'if_condition_313256' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'if_condition_313256', if_condition_313256)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to fromordinal(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to toordinal(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_313262 = {}
        # Getting the type of 'dt1' (line 114)
        dt1_313260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 56), 'dt1', False)
        # Obtaining the member 'toordinal' of a type (line 114)
        toordinal_313261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 56), dt1_313260, 'toordinal')
        # Calling toordinal(args, kwargs) (line 114)
        toordinal_call_result_313263 = invoke(stypy.reporting.localization.Localization(__file__, 114, 56), toordinal_313261, *[], **kwargs_313262)
        
        # Processing the call keyword arguments (line 114)
        kwargs_313264 = {}
        # Getting the type of 'datetime' (line 114)
        datetime_313257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 114)
        datetime_313258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 26), datetime_313257, 'datetime')
        # Obtaining the member 'fromordinal' of a type (line 114)
        fromordinal_313259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 26), datetime_313258, 'fromordinal')
        # Calling fromordinal(args, kwargs) (line 114)
        fromordinal_call_result_313265 = invoke(stypy.reporting.localization.Localization(__file__, 114, 26), fromordinal_313259, *[toordinal_call_result_313263], **kwargs_313264)
        
        # Assigning a type to the variable 'dt1' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'dt1', fromordinal_call_result_313265)
        # SSA branch for the else part of an if statement (line 113)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to isinstance(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'dt2' (line 115)
        dt2_313267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 36), 'dt2', False)
        # Getting the type of 'datetime' (line 115)
        datetime_313268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 115)
        datetime_313269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 41), datetime_313268, 'datetime')
        # Processing the call keyword arguments (line 115)
        kwargs_313270 = {}
        # Getting the type of 'isinstance' (line 115)
        isinstance_313266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 115)
        isinstance_call_result_313271 = invoke(stypy.reporting.localization.Localization(__file__, 115, 25), isinstance_313266, *[dt2_313267, datetime_313269], **kwargs_313270)
        
        # Applying the 'not' unary operator (line 115)
        result_not__313272 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 21), 'not', isinstance_call_result_313271)
        
        # Testing the type of an if condition (line 115)
        if_condition_313273 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 21), result_not__313272)
        # Assigning a type to the variable 'if_condition_313273' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), 'if_condition_313273', if_condition_313273)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to fromordinal(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Call to toordinal(...): (line 116)
        # Processing the call keyword arguments (line 116)
        kwargs_313279 = {}
        # Getting the type of 'dt2' (line 116)
        dt2_313277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 56), 'dt2', False)
        # Obtaining the member 'toordinal' of a type (line 116)
        toordinal_313278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 56), dt2_313277, 'toordinal')
        # Calling toordinal(args, kwargs) (line 116)
        toordinal_call_result_313280 = invoke(stypy.reporting.localization.Localization(__file__, 116, 56), toordinal_313278, *[], **kwargs_313279)
        
        # Processing the call keyword arguments (line 116)
        kwargs_313281 = {}
        # Getting the type of 'datetime' (line 116)
        datetime_313274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 116)
        datetime_313275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), datetime_313274, 'datetime')
        # Obtaining the member 'fromordinal' of a type (line 116)
        fromordinal_313276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), datetime_313275, 'fromordinal')
        # Calling fromordinal(args, kwargs) (line 116)
        fromordinal_call_result_313282 = invoke(stypy.reporting.localization.Localization(__file__, 116, 26), fromordinal_313276, *[toordinal_call_result_313280], **kwargs_313281)
        
        # Assigning a type to the variable 'dt2' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'dt2', fromordinal_call_result_313282)
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 111)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Attribute (line 118):
        
        # Assigning a Num to a Attribute (line 118):
        int_313283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'int')
        # Getting the type of 'self' (line 118)
        self_313284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'self')
        # Setting the type of the member 'years' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), self_313284, 'years', int_313283)
        
        # Assigning a Num to a Attribute (line 119):
        
        # Assigning a Num to a Attribute (line 119):
        int_313285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 26), 'int')
        # Getting the type of 'self' (line 119)
        self_313286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'self')
        # Setting the type of the member 'months' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), self_313286, 'months', int_313285)
        
        # Assigning a Num to a Attribute (line 120):
        
        # Assigning a Num to a Attribute (line 120):
        int_313287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 24), 'int')
        # Getting the type of 'self' (line 120)
        self_313288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'self')
        # Setting the type of the member 'days' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), self_313288, 'days', int_313287)
        
        # Assigning a Num to a Attribute (line 121):
        
        # Assigning a Num to a Attribute (line 121):
        int_313289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 28), 'int')
        # Getting the type of 'self' (line 121)
        self_313290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'self')
        # Setting the type of the member 'leapdays' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), self_313290, 'leapdays', int_313289)
        
        # Assigning a Num to a Attribute (line 122):
        
        # Assigning a Num to a Attribute (line 122):
        int_313291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 25), 'int')
        # Getting the type of 'self' (line 122)
        self_313292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'self')
        # Setting the type of the member 'hours' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), self_313292, 'hours', int_313291)
        
        # Assigning a Num to a Attribute (line 123):
        
        # Assigning a Num to a Attribute (line 123):
        int_313293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 27), 'int')
        # Getting the type of 'self' (line 123)
        self_313294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'self')
        # Setting the type of the member 'minutes' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), self_313294, 'minutes', int_313293)
        
        # Assigning a Num to a Attribute (line 124):
        
        # Assigning a Num to a Attribute (line 124):
        int_313295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 27), 'int')
        # Getting the type of 'self' (line 124)
        self_313296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'self')
        # Setting the type of the member 'seconds' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), self_313296, 'seconds', int_313295)
        
        # Assigning a Num to a Attribute (line 125):
        
        # Assigning a Num to a Attribute (line 125):
        int_313297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 32), 'int')
        # Getting the type of 'self' (line 125)
        self_313298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'self')
        # Setting the type of the member 'microseconds' of a type (line 125)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), self_313298, 'microseconds', int_313297)
        
        # Assigning a Name to a Attribute (line 126):
        
        # Assigning a Name to a Attribute (line 126):
        # Getting the type of 'None' (line 126)
        None_313299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'None')
        # Getting the type of 'self' (line 126)
        self_313300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'self')
        # Setting the type of the member 'year' of a type (line 126)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), self_313300, 'year', None_313299)
        
        # Assigning a Name to a Attribute (line 127):
        
        # Assigning a Name to a Attribute (line 127):
        # Getting the type of 'None' (line 127)
        None_313301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 'None')
        # Getting the type of 'self' (line 127)
        self_313302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self')
        # Setting the type of the member 'month' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), self_313302, 'month', None_313301)
        
        # Assigning a Name to a Attribute (line 128):
        
        # Assigning a Name to a Attribute (line 128):
        # Getting the type of 'None' (line 128)
        None_313303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'None')
        # Getting the type of 'self' (line 128)
        self_313304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'self')
        # Setting the type of the member 'day' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), self_313304, 'day', None_313303)
        
        # Assigning a Name to a Attribute (line 129):
        
        # Assigning a Name to a Attribute (line 129):
        # Getting the type of 'None' (line 129)
        None_313305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'None')
        # Getting the type of 'self' (line 129)
        self_313306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'self')
        # Setting the type of the member 'weekday' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), self_313306, 'weekday', None_313305)
        
        # Assigning a Name to a Attribute (line 130):
        
        # Assigning a Name to a Attribute (line 130):
        # Getting the type of 'None' (line 130)
        None_313307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'None')
        # Getting the type of 'self' (line 130)
        self_313308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'self')
        # Setting the type of the member 'hour' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), self_313308, 'hour', None_313307)
        
        # Assigning a Name to a Attribute (line 131):
        
        # Assigning a Name to a Attribute (line 131):
        # Getting the type of 'None' (line 131)
        None_313309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'None')
        # Getting the type of 'self' (line 131)
        self_313310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'self')
        # Setting the type of the member 'minute' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), self_313310, 'minute', None_313309)
        
        # Assigning a Name to a Attribute (line 132):
        
        # Assigning a Name to a Attribute (line 132):
        # Getting the type of 'None' (line 132)
        None_313311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 26), 'None')
        # Getting the type of 'self' (line 132)
        self_313312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'self')
        # Setting the type of the member 'second' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), self_313312, 'second', None_313311)
        
        # Assigning a Name to a Attribute (line 133):
        
        # Assigning a Name to a Attribute (line 133):
        # Getting the type of 'None' (line 133)
        None_313313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 31), 'None')
        # Getting the type of 'self' (line 133)
        self_313314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'self')
        # Setting the type of the member 'microsecond' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), self_313314, 'microsecond', None_313313)
        
        # Assigning a Num to a Attribute (line 134):
        
        # Assigning a Num to a Attribute (line 134):
        int_313315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 29), 'int')
        # Getting the type of 'self' (line 134)
        self_313316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'self')
        # Setting the type of the member '_has_time' of a type (line 134)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), self_313316, '_has_time', int_313315)
        
        # Assigning a BinOp to a Name (line 137):
        
        # Assigning a BinOp to a Name (line 137):
        # Getting the type of 'dt1' (line 137)
        dt1_313317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 22), 'dt1')
        # Obtaining the member 'year' of a type (line 137)
        year_313318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 22), dt1_313317, 'year')
        # Getting the type of 'dt2' (line 137)
        dt2_313319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'dt2')
        # Obtaining the member 'year' of a type (line 137)
        year_313320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 33), dt2_313319, 'year')
        # Applying the binary operator '-' (line 137)
        result_sub_313321 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 22), '-', year_313318, year_313320)
        
        int_313322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 45), 'int')
        # Applying the binary operator '*' (line 137)
        result_mul_313323 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 21), '*', result_sub_313321, int_313322)
        
        # Getting the type of 'dt1' (line 137)
        dt1_313324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 51), 'dt1')
        # Obtaining the member 'month' of a type (line 137)
        month_313325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 51), dt1_313324, 'month')
        # Getting the type of 'dt2' (line 137)
        dt2_313326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 63), 'dt2')
        # Obtaining the member 'month' of a type (line 137)
        month_313327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 63), dt2_313326, 'month')
        # Applying the binary operator '-' (line 137)
        result_sub_313328 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 51), '-', month_313325, month_313327)
        
        # Applying the binary operator '+' (line 137)
        result_add_313329 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 21), '+', result_mul_313323, result_sub_313328)
        
        # Assigning a type to the variable 'months' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'months', result_add_313329)
        
        # Call to _set_months(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'months' (line 138)
        months_313332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'months', False)
        # Processing the call keyword arguments (line 138)
        kwargs_313333 = {}
        # Getting the type of 'self' (line 138)
        self_313330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'self', False)
        # Obtaining the member '_set_months' of a type (line 138)
        _set_months_313331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), self_313330, '_set_months')
        # Calling _set_months(args, kwargs) (line 138)
        _set_months_call_result_313334 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), _set_months_313331, *[months_313332], **kwargs_313333)
        
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to __radd__(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'dt2' (line 142)
        dt2_313337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 32), 'dt2', False)
        # Processing the call keyword arguments (line 142)
        kwargs_313338 = {}
        # Getting the type of 'self' (line 142)
        self_313335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'self', False)
        # Obtaining the member '__radd__' of a type (line 142)
        radd___313336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 18), self_313335, '__radd__')
        # Calling __radd__(args, kwargs) (line 142)
        radd___call_result_313339 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), radd___313336, *[dt2_313337], **kwargs_313338)
        
        # Assigning a type to the variable 'dtm' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'dtm', radd___call_result_313339)
        
        
        # Getting the type of 'dt1' (line 145)
        dt1_313340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'dt1')
        # Getting the type of 'dt2' (line 145)
        dt2_313341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 21), 'dt2')
        # Applying the binary operator '<' (line 145)
        result_lt_313342 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 15), '<', dt1_313340, dt2_313341)
        
        # Testing the type of an if condition (line 145)
        if_condition_313343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 12), result_lt_313342)
        # Assigning a type to the variable 'if_condition_313343' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'if_condition_313343', if_condition_313343)
        # SSA begins for if statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 146):
        
        # Assigning a Attribute to a Name (line 146):
        # Getting the type of 'operator' (line 146)
        operator_313344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'operator')
        # Obtaining the member 'gt' of a type (line 146)
        gt_313345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 26), operator_313344, 'gt')
        # Assigning a type to the variable 'compare' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'compare', gt_313345)
        
        # Assigning a Num to a Name (line 147):
        
        # Assigning a Num to a Name (line 147):
        int_313346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 28), 'int')
        # Assigning a type to the variable 'increment' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'increment', int_313346)
        # SSA branch for the else part of an if statement (line 145)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 149):
        
        # Assigning a Attribute to a Name (line 149):
        # Getting the type of 'operator' (line 149)
        operator_313347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'operator')
        # Obtaining the member 'lt' of a type (line 149)
        lt_313348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 26), operator_313347, 'lt')
        # Assigning a type to the variable 'compare' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'compare', lt_313348)
        
        # Assigning a Num to a Name (line 150):
        
        # Assigning a Num to a Name (line 150):
        int_313349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 28), 'int')
        # Assigning a type to the variable 'increment' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'increment', int_313349)
        # SSA join for if statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to compare(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'dt1' (line 152)
        dt1_313351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 26), 'dt1', False)
        # Getting the type of 'dtm' (line 152)
        dtm_313352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'dtm', False)
        # Processing the call keyword arguments (line 152)
        kwargs_313353 = {}
        # Getting the type of 'compare' (line 152)
        compare_313350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'compare', False)
        # Calling compare(args, kwargs) (line 152)
        compare_call_result_313354 = invoke(stypy.reporting.localization.Localization(__file__, 152, 18), compare_313350, *[dt1_313351, dtm_313352], **kwargs_313353)
        
        # Testing the type of an if condition (line 152)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 12), compare_call_result_313354)
        # SSA begins for while statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'months' (line 153)
        months_313355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'months')
        # Getting the type of 'increment' (line 153)
        increment_313356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 26), 'increment')
        # Applying the binary operator '+=' (line 153)
        result_iadd_313357 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 16), '+=', months_313355, increment_313356)
        # Assigning a type to the variable 'months' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'months', result_iadd_313357)
        
        
        # Call to _set_months(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'months' (line 154)
        months_313360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 33), 'months', False)
        # Processing the call keyword arguments (line 154)
        kwargs_313361 = {}
        # Getting the type of 'self' (line 154)
        self_313358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'self', False)
        # Obtaining the member '_set_months' of a type (line 154)
        _set_months_313359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 16), self_313358, '_set_months')
        # Calling _set_months(args, kwargs) (line 154)
        _set_months_call_result_313362 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), _set_months_313359, *[months_313360], **kwargs_313361)
        
        
        # Assigning a Call to a Name (line 155):
        
        # Assigning a Call to a Name (line 155):
        
        # Call to __radd__(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'dt2' (line 155)
        dt2_313365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 36), 'dt2', False)
        # Processing the call keyword arguments (line 155)
        kwargs_313366 = {}
        # Getting the type of 'self' (line 155)
        self_313363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'self', False)
        # Obtaining the member '__radd__' of a type (line 155)
        radd___313364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 22), self_313363, '__radd__')
        # Calling __radd__(args, kwargs) (line 155)
        radd___call_result_313367 = invoke(stypy.reporting.localization.Localization(__file__, 155, 22), radd___313364, *[dt2_313365], **kwargs_313366)
        
        # Assigning a type to the variable 'dtm' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'dtm', radd___call_result_313367)
        # SSA join for while statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 158):
        
        # Assigning a BinOp to a Name (line 158):
        # Getting the type of 'dt1' (line 158)
        dt1_313368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'dt1')
        # Getting the type of 'dtm' (line 158)
        dtm_313369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'dtm')
        # Applying the binary operator '-' (line 158)
        result_sub_313370 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 20), '-', dt1_313368, dtm_313369)
        
        # Assigning a type to the variable 'delta' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'delta', result_sub_313370)
        
        # Assigning a BinOp to a Attribute (line 159):
        
        # Assigning a BinOp to a Attribute (line 159):
        # Getting the type of 'delta' (line 159)
        delta_313371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 27), 'delta')
        # Obtaining the member 'seconds' of a type (line 159)
        seconds_313372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 27), delta_313371, 'seconds')
        # Getting the type of 'delta' (line 159)
        delta_313373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 43), 'delta')
        # Obtaining the member 'days' of a type (line 159)
        days_313374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 43), delta_313373, 'days')
        int_313375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 56), 'int')
        # Applying the binary operator '*' (line 159)
        result_mul_313376 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 43), '*', days_313374, int_313375)
        
        # Applying the binary operator '+' (line 159)
        result_add_313377 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 27), '+', seconds_313372, result_mul_313376)
        
        # Getting the type of 'self' (line 159)
        self_313378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'self')
        # Setting the type of the member 'seconds' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), self_313378, 'seconds', result_add_313377)
        
        # Assigning a Attribute to a Attribute (line 160):
        
        # Assigning a Attribute to a Attribute (line 160):
        # Getting the type of 'delta' (line 160)
        delta_313379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'delta')
        # Obtaining the member 'microseconds' of a type (line 160)
        microseconds_313380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 32), delta_313379, 'microseconds')
        # Getting the type of 'self' (line 160)
        self_313381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'self')
        # Setting the type of the member 'microseconds' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), self_313381, 'microseconds', microseconds_313380)
        # SSA branch for the else part of an if statement (line 103)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 163):
        
        # Assigning a Name to a Attribute (line 163):
        # Getting the type of 'years' (line 163)
        years_313382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'years')
        # Getting the type of 'self' (line 163)
        self_313383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'self')
        # Setting the type of the member 'years' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), self_313383, 'years', years_313382)
        
        # Assigning a Name to a Attribute (line 164):
        
        # Assigning a Name to a Attribute (line 164):
        # Getting the type of 'months' (line 164)
        months_313384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'months')
        # Getting the type of 'self' (line 164)
        self_313385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'self')
        # Setting the type of the member 'months' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), self_313385, 'months', months_313384)
        
        # Assigning a BinOp to a Attribute (line 165):
        
        # Assigning a BinOp to a Attribute (line 165):
        # Getting the type of 'days' (line 165)
        days_313386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'days')
        # Getting the type of 'weeks' (line 165)
        weeks_313387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 31), 'weeks')
        int_313388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 39), 'int')
        # Applying the binary operator '*' (line 165)
        result_mul_313389 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 31), '*', weeks_313387, int_313388)
        
        # Applying the binary operator '+' (line 165)
        result_add_313390 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 24), '+', days_313386, result_mul_313389)
        
        # Getting the type of 'self' (line 165)
        self_313391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'self')
        # Setting the type of the member 'days' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), self_313391, 'days', result_add_313390)
        
        # Assigning a Name to a Attribute (line 166):
        
        # Assigning a Name to a Attribute (line 166):
        # Getting the type of 'leapdays' (line 166)
        leapdays_313392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'leapdays')
        # Getting the type of 'self' (line 166)
        self_313393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'self')
        # Setting the type of the member 'leapdays' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), self_313393, 'leapdays', leapdays_313392)
        
        # Assigning a Name to a Attribute (line 167):
        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'hours' (line 167)
        hours_313394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'hours')
        # Getting the type of 'self' (line 167)
        self_313395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'self')
        # Setting the type of the member 'hours' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), self_313395, 'hours', hours_313394)
        
        # Assigning a Name to a Attribute (line 168):
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of 'minutes' (line 168)
        minutes_313396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'minutes')
        # Getting the type of 'self' (line 168)
        self_313397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'self')
        # Setting the type of the member 'minutes' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), self_313397, 'minutes', minutes_313396)
        
        # Assigning a Name to a Attribute (line 169):
        
        # Assigning a Name to a Attribute (line 169):
        # Getting the type of 'seconds' (line 169)
        seconds_313398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 27), 'seconds')
        # Getting the type of 'self' (line 169)
        self_313399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'self')
        # Setting the type of the member 'seconds' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), self_313399, 'seconds', seconds_313398)
        
        # Assigning a Name to a Attribute (line 170):
        
        # Assigning a Name to a Attribute (line 170):
        # Getting the type of 'microseconds' (line 170)
        microseconds_313400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 32), 'microseconds')
        # Getting the type of 'self' (line 170)
        self_313401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'self')
        # Setting the type of the member 'microseconds' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 12), self_313401, 'microseconds', microseconds_313400)
        
        # Assigning a Name to a Attribute (line 173):
        
        # Assigning a Name to a Attribute (line 173):
        # Getting the type of 'year' (line 173)
        year_313402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'year')
        # Getting the type of 'self' (line 173)
        self_313403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'self')
        # Setting the type of the member 'year' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 12), self_313403, 'year', year_313402)
        
        # Assigning a Name to a Attribute (line 174):
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'month' (line 174)
        month_313404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'month')
        # Getting the type of 'self' (line 174)
        self_313405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'self')
        # Setting the type of the member 'month' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 12), self_313405, 'month', month_313404)
        
        # Assigning a Name to a Attribute (line 175):
        
        # Assigning a Name to a Attribute (line 175):
        # Getting the type of 'day' (line 175)
        day_313406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'day')
        # Getting the type of 'self' (line 175)
        self_313407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'self')
        # Setting the type of the member 'day' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), self_313407, 'day', day_313406)
        
        # Assigning a Name to a Attribute (line 176):
        
        # Assigning a Name to a Attribute (line 176):
        # Getting the type of 'hour' (line 176)
        hour_313408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'hour')
        # Getting the type of 'self' (line 176)
        self_313409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'self')
        # Setting the type of the member 'hour' of a type (line 176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), self_313409, 'hour', hour_313408)
        
        # Assigning a Name to a Attribute (line 177):
        
        # Assigning a Name to a Attribute (line 177):
        # Getting the type of 'minute' (line 177)
        minute_313410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 26), 'minute')
        # Getting the type of 'self' (line 177)
        self_313411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'self')
        # Setting the type of the member 'minute' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), self_313411, 'minute', minute_313410)
        
        # Assigning a Name to a Attribute (line 178):
        
        # Assigning a Name to a Attribute (line 178):
        # Getting the type of 'second' (line 178)
        second_313412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'second')
        # Getting the type of 'self' (line 178)
        self_313413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'self')
        # Setting the type of the member 'second' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), self_313413, 'second', second_313412)
        
        # Assigning a Name to a Attribute (line 179):
        
        # Assigning a Name to a Attribute (line 179):
        # Getting the type of 'microsecond' (line 179)
        microsecond_313414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'microsecond')
        # Getting the type of 'self' (line 179)
        self_313415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'self')
        # Setting the type of the member 'microsecond' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), self_313415, 'microsecond', microsecond_313414)
        
        
        # Call to any(...): (line 181)
        # Processing the call arguments (line 181)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 181, 19, True)
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 182)
        tuple_313427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 182)
        # Adding element type (line 182)
        # Getting the type of 'year' (line 182)
        year_313428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 29), 'year', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 29), tuple_313427, year_313428)
        # Adding element type (line 182)
        # Getting the type of 'month' (line 182)
        month_313429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 35), 'month', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 29), tuple_313427, month_313429)
        # Adding element type (line 182)
        # Getting the type of 'day' (line 182)
        day_313430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 42), 'day', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 29), tuple_313427, day_313430)
        # Adding element type (line 182)
        # Getting the type of 'hour' (line 182)
        hour_313431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 47), 'hour', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 29), tuple_313427, hour_313431)
        # Adding element type (line 182)
        # Getting the type of 'minute' (line 183)
        minute_313432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 29), 'minute', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 29), tuple_313427, minute_313432)
        # Adding element type (line 182)
        # Getting the type of 'second' (line 183)
        second_313433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 37), 'second', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 29), tuple_313427, second_313433)
        # Adding element type (line 182)
        # Getting the type of 'microsecond' (line 183)
        microsecond_313434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 45), 'microsecond', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 29), tuple_313427, microsecond_313434)
        
        comprehension_313435 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 19), tuple_313427)
        # Assigning a type to the variable 'x' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'x', comprehension_313435)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 181)
        x_313417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'x', False)
        # Getting the type of 'None' (line 181)
        None_313418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'None', False)
        # Applying the binary operator 'isnot' (line 181)
        result_is_not_313419 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 19), 'isnot', x_313417, None_313418)
        
        
        
        # Call to int(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'x' (line 181)
        x_313421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 41), 'x', False)
        # Processing the call keyword arguments (line 181)
        kwargs_313422 = {}
        # Getting the type of 'int' (line 181)
        int_313420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 37), 'int', False)
        # Calling int(args, kwargs) (line 181)
        int_call_result_313423 = invoke(stypy.reporting.localization.Localization(__file__, 181, 37), int_313420, *[x_313421], **kwargs_313422)
        
        # Getting the type of 'x' (line 181)
        x_313424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 47), 'x', False)
        # Applying the binary operator '!=' (line 181)
        result_ne_313425 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 37), '!=', int_call_result_313423, x_313424)
        
        # Applying the binary operator 'and' (line 181)
        result_and_keyword_313426 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 19), 'and', result_is_not_313419, result_ne_313425)
        
        list_313436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 19), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 19), list_313436, result_and_keyword_313426)
        # Processing the call keyword arguments (line 181)
        kwargs_313437 = {}
        # Getting the type of 'any' (line 181)
        any_313416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'any', False)
        # Calling any(args, kwargs) (line 181)
        any_call_result_313438 = invoke(stypy.reporting.localization.Localization(__file__, 181, 15), any_313416, *[list_313436], **kwargs_313437)
        
        # Testing the type of an if condition (line 181)
        if_condition_313439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 12), any_call_result_313438)
        # Assigning a type to the variable 'if_condition_313439' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'if_condition_313439', if_condition_313439)
        # SSA begins for if statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 185)
        # Processing the call arguments (line 185)
        str_313441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 21), 'str', 'Non-integer value passed as absolute information. ')
        str_313442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 21), 'str', 'This is not a well-defined condition and will raise ')
        # Applying the binary operator '+' (line 185)
        result_add_313443 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 21), '+', str_313441, str_313442)
        
        str_313444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 21), 'str', 'errors in future versions.')
        # Applying the binary operator '+' (line 186)
        result_add_313445 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 76), '+', result_add_313443, str_313444)
        
        # Getting the type of 'DeprecationWarning' (line 187)
        DeprecationWarning_313446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 51), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 185)
        kwargs_313447 = {}
        # Getting the type of 'warn' (line 185)
        warn_313440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'warn', False)
        # Calling warn(args, kwargs) (line 185)
        warn_call_result_313448 = invoke(stypy.reporting.localization.Localization(__file__, 185, 16), warn_313440, *[result_add_313445, DeprecationWarning_313446], **kwargs_313447)
        
        # SSA join for if statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'weekday' (line 189)
        weekday_313450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 26), 'weekday', False)
        # Getting the type of 'integer_types' (line 189)
        integer_types_313451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 35), 'integer_types', False)
        # Processing the call keyword arguments (line 189)
        kwargs_313452 = {}
        # Getting the type of 'isinstance' (line 189)
        isinstance_313449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 189)
        isinstance_call_result_313453 = invoke(stypy.reporting.localization.Localization(__file__, 189, 15), isinstance_313449, *[weekday_313450, integer_types_313451], **kwargs_313452)
        
        # Testing the type of an if condition (line 189)
        if_condition_313454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 12), isinstance_call_result_313453)
        # Assigning a type to the variable 'if_condition_313454' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'if_condition_313454', if_condition_313454)
        # SSA begins for if statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Attribute (line 190):
        
        # Assigning a Subscript to a Attribute (line 190):
        
        # Obtaining the type of the subscript
        # Getting the type of 'weekday' (line 190)
        weekday_313455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 40), 'weekday')
        # Getting the type of 'weekdays' (line 190)
        weekdays_313456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 31), 'weekdays')
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___313457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 31), weekdays_313456, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_313458 = invoke(stypy.reporting.localization.Localization(__file__, 190, 31), getitem___313457, weekday_313455)
        
        # Getting the type of 'self' (line 190)
        self_313459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'self')
        # Setting the type of the member 'weekday' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), self_313459, 'weekday', subscript_call_result_313458)
        # SSA branch for the else part of an if statement (line 189)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 192):
        
        # Assigning a Name to a Attribute (line 192):
        # Getting the type of 'weekday' (line 192)
        weekday_313460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 31), 'weekday')
        # Getting the type of 'self' (line 192)
        self_313461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'self')
        # Setting the type of the member 'weekday' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 16), self_313461, 'weekday', weekday_313460)
        # SSA join for if statement (line 189)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 194):
        
        # Assigning a Num to a Name (line 194):
        int_313462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 19), 'int')
        # Assigning a type to the variable 'yday' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'yday', int_313462)
        
        # Getting the type of 'nlyearday' (line 195)
        nlyearday_313463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'nlyearday')
        # Testing the type of an if condition (line 195)
        if_condition_313464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 12), nlyearday_313463)
        # Assigning a type to the variable 'if_condition_313464' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'if_condition_313464', if_condition_313464)
        # SSA begins for if statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 196):
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'nlyearday' (line 196)
        nlyearday_313465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'nlyearday')
        # Assigning a type to the variable 'yday' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'yday', nlyearday_313465)
        # SSA branch for the else part of an if statement (line 195)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'yearday' (line 197)
        yearday_313466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 'yearday')
        # Testing the type of an if condition (line 197)
        if_condition_313467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 17), yearday_313466)
        # Assigning a type to the variable 'if_condition_313467' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 'if_condition_313467', if_condition_313467)
        # SSA begins for if statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 198):
        
        # Assigning a Name to a Name (line 198):
        # Getting the type of 'yearday' (line 198)
        yearday_313468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'yearday')
        # Assigning a type to the variable 'yday' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'yday', yearday_313468)
        
        
        # Getting the type of 'yearday' (line 199)
        yearday_313469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'yearday')
        int_313470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'int')
        # Applying the binary operator '>' (line 199)
        result_gt_313471 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 19), '>', yearday_313469, int_313470)
        
        # Testing the type of an if condition (line 199)
        if_condition_313472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 16), result_gt_313471)
        # Assigning a type to the variable 'if_condition_313472' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'if_condition_313472', if_condition_313472)
        # SSA begins for if statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 200):
        
        # Assigning a Num to a Attribute (line 200):
        int_313473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 36), 'int')
        # Getting the type of 'self' (line 200)
        self_313474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'self')
        # Setting the type of the member 'leapdays' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 20), self_313474, 'leapdays', int_313473)
        # SSA join for if statement (line 199)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 197)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 195)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'yday' (line 201)
        yday_313475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'yday')
        # Testing the type of an if condition (line 201)
        if_condition_313476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 12), yday_313475)
        # Assigning a type to the variable 'if_condition_313476' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'if_condition_313476', if_condition_313476)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 202):
        
        # Assigning a List to a Name (line 202):
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_313477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        int_313478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313478)
        # Adding element type (line 202)
        int_313479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313479)
        # Adding element type (line 202)
        int_313480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313480)
        # Adding element type (line 202)
        int_313481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313481)
        # Adding element type (line 202)
        int_313482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313482)
        # Adding element type (line 202)
        int_313483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313483)
        # Adding element type (line 202)
        int_313484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313484)
        # Adding element type (line 202)
        int_313485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313485)
        # Adding element type (line 202)
        int_313486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313486)
        # Adding element type (line 202)
        int_313487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313487)
        # Adding element type (line 202)
        int_313488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313488)
        # Adding element type (line 202)
        int_313489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_313477, int_313489)
        
        # Assigning a type to the variable 'ydayidx' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'ydayidx', list_313477)
        
        
        # Call to enumerate(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'ydayidx' (line 204)
        ydayidx_313491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 44), 'ydayidx', False)
        # Processing the call keyword arguments (line 204)
        kwargs_313492 = {}
        # Getting the type of 'enumerate' (line 204)
        enumerate_313490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 34), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 204)
        enumerate_call_result_313493 = invoke(stypy.reporting.localization.Localization(__file__, 204, 34), enumerate_313490, *[ydayidx_313491], **kwargs_313492)
        
        # Testing the type of a for loop iterable (line 204)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 204, 16), enumerate_call_result_313493)
        # Getting the type of the for loop variable (line 204)
        for_loop_var_313494 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 204, 16), enumerate_call_result_313493)
        # Assigning a type to the variable 'idx' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'idx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 16), for_loop_var_313494))
        # Assigning a type to the variable 'ydays' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'ydays', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 16), for_loop_var_313494))
        # SSA begins for a for statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'yday' (line 205)
        yday_313495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'yday')
        # Getting the type of 'ydays' (line 205)
        ydays_313496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 31), 'ydays')
        # Applying the binary operator '<=' (line 205)
        result_le_313497 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 23), '<=', yday_313495, ydays_313496)
        
        # Testing the type of an if condition (line 205)
        if_condition_313498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 20), result_le_313497)
        # Assigning a type to the variable 'if_condition_313498' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'if_condition_313498', if_condition_313498)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Attribute (line 206):
        
        # Assigning a BinOp to a Attribute (line 206):
        # Getting the type of 'idx' (line 206)
        idx_313499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 37), 'idx')
        int_313500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 41), 'int')
        # Applying the binary operator '+' (line 206)
        result_add_313501 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 37), '+', idx_313499, int_313500)
        
        # Getting the type of 'self' (line 206)
        self_313502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'self')
        # Setting the type of the member 'month' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 24), self_313502, 'month', result_add_313501)
        
        
        # Getting the type of 'idx' (line 207)
        idx_313503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'idx')
        int_313504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 34), 'int')
        # Applying the binary operator '==' (line 207)
        result_eq_313505 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 27), '==', idx_313503, int_313504)
        
        # Testing the type of an if condition (line 207)
        if_condition_313506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 24), result_eq_313505)
        # Assigning a type to the variable 'if_condition_313506' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), 'if_condition_313506', if_condition_313506)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 208):
        
        # Assigning a Name to a Attribute (line 208):
        # Getting the type of 'yday' (line 208)
        yday_313507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 39), 'yday')
        # Getting the type of 'self' (line 208)
        self_313508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 28), 'self')
        # Setting the type of the member 'day' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 28), self_313508, 'day', yday_313507)
        # SSA branch for the else part of an if statement (line 207)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Attribute (line 210):
        
        # Assigning a BinOp to a Attribute (line 210):
        # Getting the type of 'yday' (line 210)
        yday_313509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 39), 'yday')
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 210)
        idx_313510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 52), 'idx')
        int_313511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 56), 'int')
        # Applying the binary operator '-' (line 210)
        result_sub_313512 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 52), '-', idx_313510, int_313511)
        
        # Getting the type of 'ydayidx' (line 210)
        ydayidx_313513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 44), 'ydayidx')
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___313514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 44), ydayidx_313513, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_313515 = invoke(stypy.reporting.localization.Localization(__file__, 210, 44), getitem___313514, result_sub_313512)
        
        # Applying the binary operator '-' (line 210)
        result_sub_313516 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 39), '-', yday_313509, subscript_call_result_313515)
        
        # Getting the type of 'self' (line 210)
        self_313517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 28), 'self')
        # Setting the type of the member 'day' of a type (line 210)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 28), self_313517, 'day', result_sub_313516)
        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 204)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to ValueError(...): (line 213)
        # Processing the call arguments (line 213)
        str_313519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 37), 'str', 'invalid year day (%d)')
        # Getting the type of 'yday' (line 213)
        yday_313520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 63), 'yday', False)
        # Applying the binary operator '%' (line 213)
        result_mod_313521 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 37), '%', str_313519, yday_313520)
        
        # Processing the call keyword arguments (line 213)
        kwargs_313522 = {}
        # Getting the type of 'ValueError' (line 213)
        ValueError_313518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 213)
        ValueError_call_result_313523 = invoke(stypy.reporting.localization.Localization(__file__, 213, 26), ValueError_313518, *[result_mod_313521], **kwargs_313522)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 213, 20), ValueError_call_result_313523, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _fix(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_313526 = {}
        # Getting the type of 'self' (line 215)
        self_313524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'self', False)
        # Obtaining the member '_fix' of a type (line 215)
        _fix_313525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), self_313524, '_fix')
        # Calling _fix(args, kwargs) (line 215)
        _fix_call_result_313527 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), _fix_313525, *[], **kwargs_313526)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _fix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fix'
        module_type_store = module_type_store.open_function_context('_fix', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta._fix.__dict__.__setitem__('stypy_localization', localization)
        relativedelta._fix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta._fix.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta._fix.__dict__.__setitem__('stypy_function_name', 'relativedelta._fix')
        relativedelta._fix.__dict__.__setitem__('stypy_param_names_list', [])
        relativedelta._fix.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta._fix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta._fix.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta._fix.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta._fix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta._fix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta._fix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fix(...)' code ##################

        
        
        
        # Call to abs(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'self' (line 218)
        self_313529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 218)
        microseconds_313530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 15), self_313529, 'microseconds')
        # Processing the call keyword arguments (line 218)
        kwargs_313531 = {}
        # Getting the type of 'abs' (line 218)
        abs_313528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'abs', False)
        # Calling abs(args, kwargs) (line 218)
        abs_call_result_313532 = invoke(stypy.reporting.localization.Localization(__file__, 218, 11), abs_313528, *[microseconds_313530], **kwargs_313531)
        
        int_313533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 36), 'int')
        # Applying the binary operator '>' (line 218)
        result_gt_313534 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), '>', abs_call_result_313532, int_313533)
        
        # Testing the type of an if condition (line 218)
        if_condition_313535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), result_gt_313534)
        # Assigning a type to the variable 'if_condition_313535' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_313535', if_condition_313535)
        # SSA begins for if statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to _sign(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'self' (line 219)
        self_313537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 22), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 219)
        microseconds_313538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 22), self_313537, 'microseconds')
        # Processing the call keyword arguments (line 219)
        kwargs_313539 = {}
        # Getting the type of '_sign' (line 219)
        _sign_313536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), '_sign', False)
        # Calling _sign(args, kwargs) (line 219)
        _sign_call_result_313540 = invoke(stypy.reporting.localization.Localization(__file__, 219, 16), _sign_313536, *[microseconds_313538], **kwargs_313539)
        
        # Assigning a type to the variable 's' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 's', _sign_call_result_313540)
        
        # Assigning a Call to a Tuple (line 220):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'self' (line 220)
        self_313542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 220)
        microseconds_313543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 30), self_313542, 'microseconds')
        # Getting the type of 's' (line 220)
        s_313544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 50), 's', False)
        # Applying the binary operator '*' (line 220)
        result_mul_313545 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 30), '*', microseconds_313543, s_313544)
        
        int_313546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 53), 'int')
        # Processing the call keyword arguments (line 220)
        kwargs_313547 = {}
        # Getting the type of 'divmod' (line 220)
        divmod_313541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'divmod', False)
        # Calling divmod(args, kwargs) (line 220)
        divmod_call_result_313548 = invoke(stypy.reporting.localization.Localization(__file__, 220, 23), divmod_313541, *[result_mul_313545, int_313546], **kwargs_313547)
        
        # Assigning a type to the variable 'call_assignment_313084' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'call_assignment_313084', divmod_call_result_313548)
        
        # Assigning a Call to a Name (line 220):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313552 = {}
        # Getting the type of 'call_assignment_313084' (line 220)
        call_assignment_313084_313549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'call_assignment_313084', False)
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___313550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), call_assignment_313084_313549, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313553 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313550, *[int_313551], **kwargs_313552)
        
        # Assigning a type to the variable 'call_assignment_313085' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'call_assignment_313085', getitem___call_result_313553)
        
        # Assigning a Name to a Name (line 220):
        # Getting the type of 'call_assignment_313085' (line 220)
        call_assignment_313085_313554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'call_assignment_313085')
        # Assigning a type to the variable 'div' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'div', call_assignment_313085_313554)
        
        # Assigning a Call to a Name (line 220):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313558 = {}
        # Getting the type of 'call_assignment_313084' (line 220)
        call_assignment_313084_313555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'call_assignment_313084', False)
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___313556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), call_assignment_313084_313555, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313559 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313556, *[int_313557], **kwargs_313558)
        
        # Assigning a type to the variable 'call_assignment_313086' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'call_assignment_313086', getitem___call_result_313559)
        
        # Assigning a Name to a Name (line 220):
        # Getting the type of 'call_assignment_313086' (line 220)
        call_assignment_313086_313560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'call_assignment_313086')
        # Assigning a type to the variable 'mod' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 17), 'mod', call_assignment_313086_313560)
        
        # Assigning a BinOp to a Attribute (line 221):
        
        # Assigning a BinOp to a Attribute (line 221):
        # Getting the type of 'mod' (line 221)
        mod_313561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'mod')
        # Getting the type of 's' (line 221)
        s_313562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 38), 's')
        # Applying the binary operator '*' (line 221)
        result_mul_313563 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 32), '*', mod_313561, s_313562)
        
        # Getting the type of 'self' (line 221)
        self_313564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'self')
        # Setting the type of the member 'microseconds' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), self_313564, 'microseconds', result_mul_313563)
        
        # Getting the type of 'self' (line 222)
        self_313565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'self')
        # Obtaining the member 'seconds' of a type (line 222)
        seconds_313566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), self_313565, 'seconds')
        # Getting the type of 'div' (line 222)
        div_313567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'div')
        # Getting the type of 's' (line 222)
        s_313568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 34), 's')
        # Applying the binary operator '*' (line 222)
        result_mul_313569 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 28), '*', div_313567, s_313568)
        
        # Applying the binary operator '+=' (line 222)
        result_iadd_313570 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 12), '+=', seconds_313566, result_mul_313569)
        # Getting the type of 'self' (line 222)
        self_313571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'self')
        # Setting the type of the member 'seconds' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), self_313571, 'seconds', result_iadd_313570)
        
        # SSA join for if statement (line 218)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to abs(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'self' (line 223)
        self_313573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'self', False)
        # Obtaining the member 'seconds' of a type (line 223)
        seconds_313574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), self_313573, 'seconds')
        # Processing the call keyword arguments (line 223)
        kwargs_313575 = {}
        # Getting the type of 'abs' (line 223)
        abs_313572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'abs', False)
        # Calling abs(args, kwargs) (line 223)
        abs_call_result_313576 = invoke(stypy.reporting.localization.Localization(__file__, 223, 11), abs_313572, *[seconds_313574], **kwargs_313575)
        
        int_313577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 31), 'int')
        # Applying the binary operator '>' (line 223)
        result_gt_313578 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 11), '>', abs_call_result_313576, int_313577)
        
        # Testing the type of an if condition (line 223)
        if_condition_313579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), result_gt_313578)
        # Assigning a type to the variable 'if_condition_313579' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_313579', if_condition_313579)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to _sign(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'self' (line 224)
        self_313581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 22), 'self', False)
        # Obtaining the member 'seconds' of a type (line 224)
        seconds_313582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 22), self_313581, 'seconds')
        # Processing the call keyword arguments (line 224)
        kwargs_313583 = {}
        # Getting the type of '_sign' (line 224)
        _sign_313580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), '_sign', False)
        # Calling _sign(args, kwargs) (line 224)
        _sign_call_result_313584 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), _sign_313580, *[seconds_313582], **kwargs_313583)
        
        # Assigning a type to the variable 's' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 's', _sign_call_result_313584)
        
        # Assigning a Call to a Tuple (line 225):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'self' (line 225)
        self_313586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 30), 'self', False)
        # Obtaining the member 'seconds' of a type (line 225)
        seconds_313587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 30), self_313586, 'seconds')
        # Getting the type of 's' (line 225)
        s_313588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 45), 's', False)
        # Applying the binary operator '*' (line 225)
        result_mul_313589 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 30), '*', seconds_313587, s_313588)
        
        int_313590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 48), 'int')
        # Processing the call keyword arguments (line 225)
        kwargs_313591 = {}
        # Getting the type of 'divmod' (line 225)
        divmod_313585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'divmod', False)
        # Calling divmod(args, kwargs) (line 225)
        divmod_call_result_313592 = invoke(stypy.reporting.localization.Localization(__file__, 225, 23), divmod_313585, *[result_mul_313589, int_313590], **kwargs_313591)
        
        # Assigning a type to the variable 'call_assignment_313087' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_313087', divmod_call_result_313592)
        
        # Assigning a Call to a Name (line 225):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313596 = {}
        # Getting the type of 'call_assignment_313087' (line 225)
        call_assignment_313087_313593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_313087', False)
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___313594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), call_assignment_313087_313593, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313597 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313594, *[int_313595], **kwargs_313596)
        
        # Assigning a type to the variable 'call_assignment_313088' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_313088', getitem___call_result_313597)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'call_assignment_313088' (line 225)
        call_assignment_313088_313598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_313088')
        # Assigning a type to the variable 'div' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'div', call_assignment_313088_313598)
        
        # Assigning a Call to a Name (line 225):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313602 = {}
        # Getting the type of 'call_assignment_313087' (line 225)
        call_assignment_313087_313599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_313087', False)
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___313600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), call_assignment_313087_313599, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313603 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313600, *[int_313601], **kwargs_313602)
        
        # Assigning a type to the variable 'call_assignment_313089' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_313089', getitem___call_result_313603)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'call_assignment_313089' (line 225)
        call_assignment_313089_313604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'call_assignment_313089')
        # Assigning a type to the variable 'mod' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), 'mod', call_assignment_313089_313604)
        
        # Assigning a BinOp to a Attribute (line 226):
        
        # Assigning a BinOp to a Attribute (line 226):
        # Getting the type of 'mod' (line 226)
        mod_313605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'mod')
        # Getting the type of 's' (line 226)
        s_313606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 33), 's')
        # Applying the binary operator '*' (line 226)
        result_mul_313607 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 27), '*', mod_313605, s_313606)
        
        # Getting the type of 'self' (line 226)
        self_313608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'self')
        # Setting the type of the member 'seconds' of a type (line 226)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), self_313608, 'seconds', result_mul_313607)
        
        # Getting the type of 'self' (line 227)
        self_313609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'self')
        # Obtaining the member 'minutes' of a type (line 227)
        minutes_313610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), self_313609, 'minutes')
        # Getting the type of 'div' (line 227)
        div_313611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 28), 'div')
        # Getting the type of 's' (line 227)
        s_313612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 's')
        # Applying the binary operator '*' (line 227)
        result_mul_313613 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 28), '*', div_313611, s_313612)
        
        # Applying the binary operator '+=' (line 227)
        result_iadd_313614 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 12), '+=', minutes_313610, result_mul_313613)
        # Getting the type of 'self' (line 227)
        self_313615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'self')
        # Setting the type of the member 'minutes' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), self_313615, 'minutes', result_iadd_313614)
        
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to abs(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'self' (line 228)
        self_313617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'self', False)
        # Obtaining the member 'minutes' of a type (line 228)
        minutes_313618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 15), self_313617, 'minutes')
        # Processing the call keyword arguments (line 228)
        kwargs_313619 = {}
        # Getting the type of 'abs' (line 228)
        abs_313616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'abs', False)
        # Calling abs(args, kwargs) (line 228)
        abs_call_result_313620 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), abs_313616, *[minutes_313618], **kwargs_313619)
        
        int_313621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 31), 'int')
        # Applying the binary operator '>' (line 228)
        result_gt_313622 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 11), '>', abs_call_result_313620, int_313621)
        
        # Testing the type of an if condition (line 228)
        if_condition_313623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 8), result_gt_313622)
        # Assigning a type to the variable 'if_condition_313623' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'if_condition_313623', if_condition_313623)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to _sign(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'self' (line 229)
        self_313625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 22), 'self', False)
        # Obtaining the member 'minutes' of a type (line 229)
        minutes_313626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 22), self_313625, 'minutes')
        # Processing the call keyword arguments (line 229)
        kwargs_313627 = {}
        # Getting the type of '_sign' (line 229)
        _sign_313624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), '_sign', False)
        # Calling _sign(args, kwargs) (line 229)
        _sign_call_result_313628 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), _sign_313624, *[minutes_313626], **kwargs_313627)
        
        # Assigning a type to the variable 's' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 's', _sign_call_result_313628)
        
        # Assigning a Call to a Tuple (line 230):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'self' (line 230)
        self_313630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 30), 'self', False)
        # Obtaining the member 'minutes' of a type (line 230)
        minutes_313631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 30), self_313630, 'minutes')
        # Getting the type of 's' (line 230)
        s_313632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 's', False)
        # Applying the binary operator '*' (line 230)
        result_mul_313633 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 30), '*', minutes_313631, s_313632)
        
        int_313634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 48), 'int')
        # Processing the call keyword arguments (line 230)
        kwargs_313635 = {}
        # Getting the type of 'divmod' (line 230)
        divmod_313629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'divmod', False)
        # Calling divmod(args, kwargs) (line 230)
        divmod_call_result_313636 = invoke(stypy.reporting.localization.Localization(__file__, 230, 23), divmod_313629, *[result_mul_313633, int_313634], **kwargs_313635)
        
        # Assigning a type to the variable 'call_assignment_313090' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_313090', divmod_call_result_313636)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313640 = {}
        # Getting the type of 'call_assignment_313090' (line 230)
        call_assignment_313090_313637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_313090', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___313638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), call_assignment_313090_313637, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313641 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313638, *[int_313639], **kwargs_313640)
        
        # Assigning a type to the variable 'call_assignment_313091' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_313091', getitem___call_result_313641)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_313091' (line 230)
        call_assignment_313091_313642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_313091')
        # Assigning a type to the variable 'div' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'div', call_assignment_313091_313642)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313646 = {}
        # Getting the type of 'call_assignment_313090' (line 230)
        call_assignment_313090_313643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_313090', False)
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___313644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), call_assignment_313090_313643, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313647 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313644, *[int_313645], **kwargs_313646)
        
        # Assigning a type to the variable 'call_assignment_313092' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_313092', getitem___call_result_313647)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'call_assignment_313092' (line 230)
        call_assignment_313092_313648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'call_assignment_313092')
        # Assigning a type to the variable 'mod' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 17), 'mod', call_assignment_313092_313648)
        
        # Assigning a BinOp to a Attribute (line 231):
        
        # Assigning a BinOp to a Attribute (line 231):
        # Getting the type of 'mod' (line 231)
        mod_313649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'mod')
        # Getting the type of 's' (line 231)
        s_313650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 33), 's')
        # Applying the binary operator '*' (line 231)
        result_mul_313651 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 27), '*', mod_313649, s_313650)
        
        # Getting the type of 'self' (line 231)
        self_313652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'self')
        # Setting the type of the member 'minutes' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), self_313652, 'minutes', result_mul_313651)
        
        # Getting the type of 'self' (line 232)
        self_313653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self')
        # Obtaining the member 'hours' of a type (line 232)
        hours_313654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_313653, 'hours')
        # Getting the type of 'div' (line 232)
        div_313655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 26), 'div')
        # Getting the type of 's' (line 232)
        s_313656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 's')
        # Applying the binary operator '*' (line 232)
        result_mul_313657 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 26), '*', div_313655, s_313656)
        
        # Applying the binary operator '+=' (line 232)
        result_iadd_313658 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 12), '+=', hours_313654, result_mul_313657)
        # Getting the type of 'self' (line 232)
        self_313659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self')
        # Setting the type of the member 'hours' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_313659, 'hours', result_iadd_313658)
        
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to abs(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'self' (line 233)
        self_313661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'self', False)
        # Obtaining the member 'hours' of a type (line 233)
        hours_313662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 15), self_313661, 'hours')
        # Processing the call keyword arguments (line 233)
        kwargs_313663 = {}
        # Getting the type of 'abs' (line 233)
        abs_313660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'abs', False)
        # Calling abs(args, kwargs) (line 233)
        abs_call_result_313664 = invoke(stypy.reporting.localization.Localization(__file__, 233, 11), abs_313660, *[hours_313662], **kwargs_313663)
        
        int_313665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'int')
        # Applying the binary operator '>' (line 233)
        result_gt_313666 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 11), '>', abs_call_result_313664, int_313665)
        
        # Testing the type of an if condition (line 233)
        if_condition_313667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 8), result_gt_313666)
        # Assigning a type to the variable 'if_condition_313667' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'if_condition_313667', if_condition_313667)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Call to _sign(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'self' (line 234)
        self_313669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 22), 'self', False)
        # Obtaining the member 'hours' of a type (line 234)
        hours_313670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 22), self_313669, 'hours')
        # Processing the call keyword arguments (line 234)
        kwargs_313671 = {}
        # Getting the type of '_sign' (line 234)
        _sign_313668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), '_sign', False)
        # Calling _sign(args, kwargs) (line 234)
        _sign_call_result_313672 = invoke(stypy.reporting.localization.Localization(__file__, 234, 16), _sign_313668, *[hours_313670], **kwargs_313671)
        
        # Assigning a type to the variable 's' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 's', _sign_call_result_313672)
        
        # Assigning a Call to a Tuple (line 235):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'self' (line 235)
        self_313674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), 'self', False)
        # Obtaining the member 'hours' of a type (line 235)
        hours_313675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 30), self_313674, 'hours')
        # Getting the type of 's' (line 235)
        s_313676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 43), 's', False)
        # Applying the binary operator '*' (line 235)
        result_mul_313677 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 30), '*', hours_313675, s_313676)
        
        int_313678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 46), 'int')
        # Processing the call keyword arguments (line 235)
        kwargs_313679 = {}
        # Getting the type of 'divmod' (line 235)
        divmod_313673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'divmod', False)
        # Calling divmod(args, kwargs) (line 235)
        divmod_call_result_313680 = invoke(stypy.reporting.localization.Localization(__file__, 235, 23), divmod_313673, *[result_mul_313677, int_313678], **kwargs_313679)
        
        # Assigning a type to the variable 'call_assignment_313093' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_313093', divmod_call_result_313680)
        
        # Assigning a Call to a Name (line 235):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313684 = {}
        # Getting the type of 'call_assignment_313093' (line 235)
        call_assignment_313093_313681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_313093', False)
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___313682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), call_assignment_313093_313681, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313685 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313682, *[int_313683], **kwargs_313684)
        
        # Assigning a type to the variable 'call_assignment_313094' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_313094', getitem___call_result_313685)
        
        # Assigning a Name to a Name (line 235):
        # Getting the type of 'call_assignment_313094' (line 235)
        call_assignment_313094_313686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_313094')
        # Assigning a type to the variable 'div' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'div', call_assignment_313094_313686)
        
        # Assigning a Call to a Name (line 235):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313690 = {}
        # Getting the type of 'call_assignment_313093' (line 235)
        call_assignment_313093_313687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_313093', False)
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___313688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), call_assignment_313093_313687, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313691 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313688, *[int_313689], **kwargs_313690)
        
        # Assigning a type to the variable 'call_assignment_313095' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_313095', getitem___call_result_313691)
        
        # Assigning a Name to a Name (line 235):
        # Getting the type of 'call_assignment_313095' (line 235)
        call_assignment_313095_313692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'call_assignment_313095')
        # Assigning a type to the variable 'mod' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 17), 'mod', call_assignment_313095_313692)
        
        # Assigning a BinOp to a Attribute (line 236):
        
        # Assigning a BinOp to a Attribute (line 236):
        # Getting the type of 'mod' (line 236)
        mod_313693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 25), 'mod')
        # Getting the type of 's' (line 236)
        s_313694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 31), 's')
        # Applying the binary operator '*' (line 236)
        result_mul_313695 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 25), '*', mod_313693, s_313694)
        
        # Getting the type of 'self' (line 236)
        self_313696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'self')
        # Setting the type of the member 'hours' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), self_313696, 'hours', result_mul_313695)
        
        # Getting the type of 'self' (line 237)
        self_313697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self')
        # Obtaining the member 'days' of a type (line 237)
        days_313698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_313697, 'days')
        # Getting the type of 'div' (line 237)
        div_313699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 25), 'div')
        # Getting the type of 's' (line 237)
        s_313700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 31), 's')
        # Applying the binary operator '*' (line 237)
        result_mul_313701 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 25), '*', div_313699, s_313700)
        
        # Applying the binary operator '+=' (line 237)
        result_iadd_313702 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 12), '+=', days_313698, result_mul_313701)
        # Getting the type of 'self' (line 237)
        self_313703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self')
        # Setting the type of the member 'days' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_313703, 'days', result_iadd_313702)
        
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to abs(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'self' (line 238)
        self_313705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'self', False)
        # Obtaining the member 'months' of a type (line 238)
        months_313706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 15), self_313705, 'months')
        # Processing the call keyword arguments (line 238)
        kwargs_313707 = {}
        # Getting the type of 'abs' (line 238)
        abs_313704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'abs', False)
        # Calling abs(args, kwargs) (line 238)
        abs_call_result_313708 = invoke(stypy.reporting.localization.Localization(__file__, 238, 11), abs_313704, *[months_313706], **kwargs_313707)
        
        int_313709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 30), 'int')
        # Applying the binary operator '>' (line 238)
        result_gt_313710 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 11), '>', abs_call_result_313708, int_313709)
        
        # Testing the type of an if condition (line 238)
        if_condition_313711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), result_gt_313710)
        # Assigning a type to the variable 'if_condition_313711' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_313711', if_condition_313711)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 239):
        
        # Assigning a Call to a Name (line 239):
        
        # Call to _sign(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'self' (line 239)
        self_313713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 22), 'self', False)
        # Obtaining the member 'months' of a type (line 239)
        months_313714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 22), self_313713, 'months')
        # Processing the call keyword arguments (line 239)
        kwargs_313715 = {}
        # Getting the type of '_sign' (line 239)
        _sign_313712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), '_sign', False)
        # Calling _sign(args, kwargs) (line 239)
        _sign_call_result_313716 = invoke(stypy.reporting.localization.Localization(__file__, 239, 16), _sign_313712, *[months_313714], **kwargs_313715)
        
        # Assigning a type to the variable 's' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 's', _sign_call_result_313716)
        
        # Assigning a Call to a Tuple (line 240):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'self' (line 240)
        self_313718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 30), 'self', False)
        # Obtaining the member 'months' of a type (line 240)
        months_313719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 30), self_313718, 'months')
        # Getting the type of 's' (line 240)
        s_313720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 's', False)
        # Applying the binary operator '*' (line 240)
        result_mul_313721 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 30), '*', months_313719, s_313720)
        
        int_313722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 47), 'int')
        # Processing the call keyword arguments (line 240)
        kwargs_313723 = {}
        # Getting the type of 'divmod' (line 240)
        divmod_313717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'divmod', False)
        # Calling divmod(args, kwargs) (line 240)
        divmod_call_result_313724 = invoke(stypy.reporting.localization.Localization(__file__, 240, 23), divmod_313717, *[result_mul_313721, int_313722], **kwargs_313723)
        
        # Assigning a type to the variable 'call_assignment_313096' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'call_assignment_313096', divmod_call_result_313724)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313728 = {}
        # Getting the type of 'call_assignment_313096' (line 240)
        call_assignment_313096_313725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'call_assignment_313096', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___313726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), call_assignment_313096_313725, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313729 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313726, *[int_313727], **kwargs_313728)
        
        # Assigning a type to the variable 'call_assignment_313097' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'call_assignment_313097', getitem___call_result_313729)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'call_assignment_313097' (line 240)
        call_assignment_313097_313730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'call_assignment_313097')
        # Assigning a type to the variable 'div' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'div', call_assignment_313097_313730)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313734 = {}
        # Getting the type of 'call_assignment_313096' (line 240)
        call_assignment_313096_313731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'call_assignment_313096', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___313732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), call_assignment_313096_313731, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313735 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313732, *[int_313733], **kwargs_313734)
        
        # Assigning a type to the variable 'call_assignment_313098' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'call_assignment_313098', getitem___call_result_313735)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'call_assignment_313098' (line 240)
        call_assignment_313098_313736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'call_assignment_313098')
        # Assigning a type to the variable 'mod' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 17), 'mod', call_assignment_313098_313736)
        
        # Assigning a BinOp to a Attribute (line 241):
        
        # Assigning a BinOp to a Attribute (line 241):
        # Getting the type of 'mod' (line 241)
        mod_313737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'mod')
        # Getting the type of 's' (line 241)
        s_313738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 32), 's')
        # Applying the binary operator '*' (line 241)
        result_mul_313739 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 26), '*', mod_313737, s_313738)
        
        # Getting the type of 'self' (line 241)
        self_313740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'self')
        # Setting the type of the member 'months' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), self_313740, 'months', result_mul_313739)
        
        # Getting the type of 'self' (line 242)
        self_313741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'self')
        # Obtaining the member 'years' of a type (line 242)
        years_313742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), self_313741, 'years')
        # Getting the type of 'div' (line 242)
        div_313743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 26), 'div')
        # Getting the type of 's' (line 242)
        s_313744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 32), 's')
        # Applying the binary operator '*' (line 242)
        result_mul_313745 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 26), '*', div_313743, s_313744)
        
        # Applying the binary operator '+=' (line 242)
        result_iadd_313746 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 12), '+=', years_313742, result_mul_313745)
        # Getting the type of 'self' (line 242)
        self_313747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'self')
        # Setting the type of the member 'years' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), self_313747, 'years', result_iadd_313746)
        
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 243)
        self_313748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self')
        # Obtaining the member 'hours' of a type (line 243)
        hours_313749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), self_313748, 'hours')
        # Getting the type of 'self' (line 243)
        self_313750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 26), 'self')
        # Obtaining the member 'minutes' of a type (line 243)
        minutes_313751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 26), self_313750, 'minutes')
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_313752 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), 'or', hours_313749, minutes_313751)
        # Getting the type of 'self' (line 243)
        self_313753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 42), 'self')
        # Obtaining the member 'seconds' of a type (line 243)
        seconds_313754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 42), self_313753, 'seconds')
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_313755 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), 'or', result_or_keyword_313752, seconds_313754)
        # Getting the type of 'self' (line 243)
        self_313756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 58), 'self')
        # Obtaining the member 'microseconds' of a type (line 243)
        microseconds_313757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 58), self_313756, 'microseconds')
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_313758 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), 'or', result_or_keyword_313755, microseconds_313757)
        
        # Getting the type of 'self' (line 244)
        self_313759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'self')
        # Obtaining the member 'hour' of a type (line 244)
        hour_313760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 19), self_313759, 'hour')
        # Getting the type of 'None' (line 244)
        None_313761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 36), 'None')
        # Applying the binary operator 'isnot' (line 244)
        result_is_not_313762 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 19), 'isnot', hour_313760, None_313761)
        
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_313763 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), 'or', result_or_keyword_313758, result_is_not_313762)
        
        # Getting the type of 'self' (line 244)
        self_313764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 44), 'self')
        # Obtaining the member 'minute' of a type (line 244)
        minute_313765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 44), self_313764, 'minute')
        # Getting the type of 'None' (line 244)
        None_313766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 63), 'None')
        # Applying the binary operator 'isnot' (line 244)
        result_is_not_313767 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 44), 'isnot', minute_313765, None_313766)
        
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_313768 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), 'or', result_or_keyword_313763, result_is_not_313767)
        
        # Getting the type of 'self' (line 245)
        self_313769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'self')
        # Obtaining the member 'second' of a type (line 245)
        second_313770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 16), self_313769, 'second')
        # Getting the type of 'None' (line 245)
        None_313771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 35), 'None')
        # Applying the binary operator 'isnot' (line 245)
        result_is_not_313772 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 16), 'isnot', second_313770, None_313771)
        
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_313773 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), 'or', result_or_keyword_313768, result_is_not_313772)
        
        # Getting the type of 'self' (line 245)
        self_313774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 43), 'self')
        # Obtaining the member 'microsecond' of a type (line 245)
        microsecond_313775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 43), self_313774, 'microsecond')
        # Getting the type of 'None' (line 245)
        None_313776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 67), 'None')
        # Applying the binary operator 'isnot' (line 245)
        result_is_not_313777 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 43), 'isnot', microsecond_313775, None_313776)
        
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_313778 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), 'or', result_or_keyword_313773, result_is_not_313777)
        
        # Testing the type of an if condition (line 243)
        if_condition_313779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), result_or_keyword_313778)
        # Assigning a type to the variable 'if_condition_313779' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_313779', if_condition_313779)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 246):
        
        # Assigning a Num to a Attribute (line 246):
        int_313780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 29), 'int')
        # Getting the type of 'self' (line 246)
        self_313781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'self')
        # Setting the type of the member '_has_time' of a type (line 246)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), self_313781, '_has_time', int_313780)
        # SSA branch for the else part of an if statement (line 243)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Attribute (line 248):
        
        # Assigning a Num to a Attribute (line 248):
        int_313782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 29), 'int')
        # Getting the type of 'self' (line 248)
        self_313783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'self')
        # Setting the type of the member '_has_time' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), self_313783, '_has_time', int_313782)
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_fix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fix' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_313784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_313784)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fix'
        return stypy_return_type_313784


    @norecursion
    def weeks(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'weeks'
        module_type_store = module_type_store.open_function_context('weeks', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.weeks.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.weeks.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.weeks.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.weeks.__dict__.__setitem__('stypy_function_name', 'relativedelta.weeks')
        relativedelta.weeks.__dict__.__setitem__('stypy_param_names_list', [])
        relativedelta.weeks.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.weeks.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.weeks.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.weeks.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.weeks.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.weeks.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.weeks', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'weeks', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'weeks(...)' code ##################

        # Getting the type of 'self' (line 252)
        self_313785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'self')
        # Obtaining the member 'days' of a type (line 252)
        days_313786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 15), self_313785, 'days')
        int_313787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 28), 'int')
        # Applying the binary operator '//' (line 252)
        result_floordiv_313788 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 15), '//', days_313786, int_313787)
        
        # Assigning a type to the variable 'stypy_return_type' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'stypy_return_type', result_floordiv_313788)
        
        # ################# End of 'weeks(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'weeks' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_313789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_313789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'weeks'
        return stypy_return_type_313789


    @norecursion
    def weeks(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'weeks'
        module_type_store = module_type_store.open_function_context('weeks', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.weeks.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.weeks.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.weeks.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.weeks.__dict__.__setitem__('stypy_function_name', 'relativedelta.weeks')
        relativedelta.weeks.__dict__.__setitem__('stypy_param_names_list', ['value'])
        relativedelta.weeks.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.weeks.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.weeks.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.weeks.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.weeks.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.weeks.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.weeks', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'weeks', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'weeks(...)' code ##################

        
        # Assigning a BinOp to a Attribute (line 256):
        
        # Assigning a BinOp to a Attribute (line 256):
        # Getting the type of 'self' (line 256)
        self_313790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'self')
        # Obtaining the member 'days' of a type (line 256)
        days_313791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 20), self_313790, 'days')
        # Getting the type of 'self' (line 256)
        self_313792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 33), 'self')
        # Obtaining the member 'weeks' of a type (line 256)
        weeks_313793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 33), self_313792, 'weeks')
        int_313794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 46), 'int')
        # Applying the binary operator '*' (line 256)
        result_mul_313795 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 33), '*', weeks_313793, int_313794)
        
        # Applying the binary operator '-' (line 256)
        result_sub_313796 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 20), '-', days_313791, result_mul_313795)
        
        # Getting the type of 'value' (line 256)
        value_313797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 51), 'value')
        int_313798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 59), 'int')
        # Applying the binary operator '*' (line 256)
        result_mul_313799 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 51), '*', value_313797, int_313798)
        
        # Applying the binary operator '+' (line 256)
        result_add_313800 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 49), '+', result_sub_313796, result_mul_313799)
        
        # Getting the type of 'self' (line 256)
        self_313801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'self')
        # Setting the type of the member 'days' of a type (line 256)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), self_313801, 'days', result_add_313800)
        
        # ################# End of 'weeks(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'weeks' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_313802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_313802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'weeks'
        return stypy_return_type_313802


    @norecursion
    def _set_months(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_months'
        module_type_store = module_type_store.open_function_context('_set_months', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta._set_months.__dict__.__setitem__('stypy_localization', localization)
        relativedelta._set_months.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta._set_months.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta._set_months.__dict__.__setitem__('stypy_function_name', 'relativedelta._set_months')
        relativedelta._set_months.__dict__.__setitem__('stypy_param_names_list', ['months'])
        relativedelta._set_months.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta._set_months.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta._set_months.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta._set_months.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta._set_months.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta._set_months.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta._set_months', ['months'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_months', localization, ['months'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_months(...)' code ##################

        
        # Assigning a Name to a Attribute (line 259):
        
        # Assigning a Name to a Attribute (line 259):
        # Getting the type of 'months' (line 259)
        months_313803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 22), 'months')
        # Getting the type of 'self' (line 259)
        self_313804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self')
        # Setting the type of the member 'months' of a type (line 259)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_313804, 'months', months_313803)
        
        
        
        # Call to abs(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'self' (line 260)
        self_313806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'self', False)
        # Obtaining the member 'months' of a type (line 260)
        months_313807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), self_313806, 'months')
        # Processing the call keyword arguments (line 260)
        kwargs_313808 = {}
        # Getting the type of 'abs' (line 260)
        abs_313805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'abs', False)
        # Calling abs(args, kwargs) (line 260)
        abs_call_result_313809 = invoke(stypy.reporting.localization.Localization(__file__, 260, 11), abs_313805, *[months_313807], **kwargs_313808)
        
        int_313810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 30), 'int')
        # Applying the binary operator '>' (line 260)
        result_gt_313811 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 11), '>', abs_call_result_313809, int_313810)
        
        # Testing the type of an if condition (line 260)
        if_condition_313812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 8), result_gt_313811)
        # Assigning a type to the variable 'if_condition_313812' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'if_condition_313812', if_condition_313812)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 261):
        
        # Assigning a Call to a Name (line 261):
        
        # Call to _sign(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'self' (line 261)
        self_313814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'self', False)
        # Obtaining the member 'months' of a type (line 261)
        months_313815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 22), self_313814, 'months')
        # Processing the call keyword arguments (line 261)
        kwargs_313816 = {}
        # Getting the type of '_sign' (line 261)
        _sign_313813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), '_sign', False)
        # Calling _sign(args, kwargs) (line 261)
        _sign_call_result_313817 = invoke(stypy.reporting.localization.Localization(__file__, 261, 16), _sign_313813, *[months_313815], **kwargs_313816)
        
        # Assigning a type to the variable 's' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 's', _sign_call_result_313817)
        
        # Assigning a Call to a Tuple (line 262):
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'self' (line 262)
        self_313819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 30), 'self', False)
        # Obtaining the member 'months' of a type (line 262)
        months_313820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 30), self_313819, 'months')
        # Getting the type of 's' (line 262)
        s_313821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 44), 's', False)
        # Applying the binary operator '*' (line 262)
        result_mul_313822 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 30), '*', months_313820, s_313821)
        
        int_313823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 47), 'int')
        # Processing the call keyword arguments (line 262)
        kwargs_313824 = {}
        # Getting the type of 'divmod' (line 262)
        divmod_313818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'divmod', False)
        # Calling divmod(args, kwargs) (line 262)
        divmod_call_result_313825 = invoke(stypy.reporting.localization.Localization(__file__, 262, 23), divmod_313818, *[result_mul_313822, int_313823], **kwargs_313824)
        
        # Assigning a type to the variable 'call_assignment_313099' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'call_assignment_313099', divmod_call_result_313825)
        
        # Assigning a Call to a Name (line 262):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313829 = {}
        # Getting the type of 'call_assignment_313099' (line 262)
        call_assignment_313099_313826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'call_assignment_313099', False)
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___313827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 12), call_assignment_313099_313826, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313830 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313827, *[int_313828], **kwargs_313829)
        
        # Assigning a type to the variable 'call_assignment_313100' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'call_assignment_313100', getitem___call_result_313830)
        
        # Assigning a Name to a Name (line 262):
        # Getting the type of 'call_assignment_313100' (line 262)
        call_assignment_313100_313831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'call_assignment_313100')
        # Assigning a type to the variable 'div' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'div', call_assignment_313100_313831)
        
        # Assigning a Call to a Name (line 262):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_313834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 12), 'int')
        # Processing the call keyword arguments
        kwargs_313835 = {}
        # Getting the type of 'call_assignment_313099' (line 262)
        call_assignment_313099_313832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'call_assignment_313099', False)
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___313833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 12), call_assignment_313099_313832, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_313836 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___313833, *[int_313834], **kwargs_313835)
        
        # Assigning a type to the variable 'call_assignment_313101' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'call_assignment_313101', getitem___call_result_313836)
        
        # Assigning a Name to a Name (line 262):
        # Getting the type of 'call_assignment_313101' (line 262)
        call_assignment_313101_313837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'call_assignment_313101')
        # Assigning a type to the variable 'mod' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 17), 'mod', call_assignment_313101_313837)
        
        # Assigning a BinOp to a Attribute (line 263):
        
        # Assigning a BinOp to a Attribute (line 263):
        # Getting the type of 'mod' (line 263)
        mod_313838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 26), 'mod')
        # Getting the type of 's' (line 263)
        s_313839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 32), 's')
        # Applying the binary operator '*' (line 263)
        result_mul_313840 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 26), '*', mod_313838, s_313839)
        
        # Getting the type of 'self' (line 263)
        self_313841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'self')
        # Setting the type of the member 'months' of a type (line 263)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), self_313841, 'months', result_mul_313840)
        
        # Assigning a BinOp to a Attribute (line 264):
        
        # Assigning a BinOp to a Attribute (line 264):
        # Getting the type of 'div' (line 264)
        div_313842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 25), 'div')
        # Getting the type of 's' (line 264)
        s_313843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 31), 's')
        # Applying the binary operator '*' (line 264)
        result_mul_313844 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 25), '*', div_313842, s_313843)
        
        # Getting the type of 'self' (line 264)
        self_313845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'self')
        # Setting the type of the member 'years' of a type (line 264)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), self_313845, 'years', result_mul_313844)
        # SSA branch for the else part of an if statement (line 260)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Attribute (line 266):
        
        # Assigning a Num to a Attribute (line 266):
        int_313846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 25), 'int')
        # Getting the type of 'self' (line 266)
        self_313847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'self')
        # Setting the type of the member 'years' of a type (line 266)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), self_313847, 'years', int_313846)
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_set_months(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_months' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_313848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_313848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_months'
        return stypy_return_type_313848


    @norecursion
    def normalized(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'normalized'
        module_type_store = module_type_store.open_function_context('normalized', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.normalized.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.normalized.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.normalized.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.normalized.__dict__.__setitem__('stypy_function_name', 'relativedelta.normalized')
        relativedelta.normalized.__dict__.__setitem__('stypy_param_names_list', [])
        relativedelta.normalized.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.normalized.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.normalized.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.normalized.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.normalized.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.normalized.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.normalized', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'normalized', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'normalized(...)' code ##################

        str_313849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, (-1)), 'str', '\n        Return a version of this object represented entirely using integer\n        values for the relative attributes.\n\n        >>> relativedelta(days=1.5, hours=2).normalized()\n        relativedelta(days=1, hours=14)\n\n        :return:\n            Returns a :class:`dateutil.relativedelta.relativedelta` object.\n        ')
        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to int(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'self' (line 280)
        self_313851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'self', False)
        # Obtaining the member 'days' of a type (line 280)
        days_313852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 19), self_313851, 'days')
        # Processing the call keyword arguments (line 280)
        kwargs_313853 = {}
        # Getting the type of 'int' (line 280)
        int_313850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'int', False)
        # Calling int(args, kwargs) (line 280)
        int_call_result_313854 = invoke(stypy.reporting.localization.Localization(__file__, 280, 15), int_313850, *[days_313852], **kwargs_313853)
        
        # Assigning a type to the variable 'days' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'days', int_call_result_313854)
        
        # Assigning a Call to a Name (line 282):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to round(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'self' (line 282)
        self_313856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'self', False)
        # Obtaining the member 'hours' of a type (line 282)
        hours_313857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), self_313856, 'hours')
        int_313858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 37), 'int')
        # Getting the type of 'self' (line 282)
        self_313859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 43), 'self', False)
        # Obtaining the member 'days' of a type (line 282)
        days_313860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 43), self_313859, 'days')
        # Getting the type of 'days' (line 282)
        days_313861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 55), 'days', False)
        # Applying the binary operator '-' (line 282)
        result_sub_313862 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 43), '-', days_313860, days_313861)
        
        # Applying the binary operator '*' (line 282)
        result_mul_313863 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 37), '*', int_313858, result_sub_313862)
        
        # Applying the binary operator '+' (line 282)
        result_add_313864 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 24), '+', hours_313857, result_mul_313863)
        
        int_313865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 62), 'int')
        # Processing the call keyword arguments (line 282)
        kwargs_313866 = {}
        # Getting the type of 'round' (line 282)
        round_313855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'round', False)
        # Calling round(args, kwargs) (line 282)
        round_call_result_313867 = invoke(stypy.reporting.localization.Localization(__file__, 282, 18), round_313855, *[result_add_313864, int_313865], **kwargs_313866)
        
        # Assigning a type to the variable 'hours_f' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'hours_f', round_call_result_313867)
        
        # Assigning a Call to a Name (line 283):
        
        # Assigning a Call to a Name (line 283):
        
        # Call to int(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'hours_f' (line 283)
        hours_f_313869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'hours_f', False)
        # Processing the call keyword arguments (line 283)
        kwargs_313870 = {}
        # Getting the type of 'int' (line 283)
        int_313868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'int', False)
        # Calling int(args, kwargs) (line 283)
        int_call_result_313871 = invoke(stypy.reporting.localization.Localization(__file__, 283, 16), int_313868, *[hours_f_313869], **kwargs_313870)
        
        # Assigning a type to the variable 'hours' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'hours', int_call_result_313871)
        
        # Assigning a Call to a Name (line 285):
        
        # Assigning a Call to a Name (line 285):
        
        # Call to round(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'self' (line 285)
        self_313873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 26), 'self', False)
        # Obtaining the member 'minutes' of a type (line 285)
        minutes_313874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 26), self_313873, 'minutes')
        int_313875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 41), 'int')
        # Getting the type of 'hours_f' (line 285)
        hours_f_313876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 47), 'hours_f', False)
        # Getting the type of 'hours' (line 285)
        hours_313877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 57), 'hours', False)
        # Applying the binary operator '-' (line 285)
        result_sub_313878 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 47), '-', hours_f_313876, hours_313877)
        
        # Applying the binary operator '*' (line 285)
        result_mul_313879 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 41), '*', int_313875, result_sub_313878)
        
        # Applying the binary operator '+' (line 285)
        result_add_313880 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 26), '+', minutes_313874, result_mul_313879)
        
        int_313881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 65), 'int')
        # Processing the call keyword arguments (line 285)
        kwargs_313882 = {}
        # Getting the type of 'round' (line 285)
        round_313872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'round', False)
        # Calling round(args, kwargs) (line 285)
        round_call_result_313883 = invoke(stypy.reporting.localization.Localization(__file__, 285, 20), round_313872, *[result_add_313880, int_313881], **kwargs_313882)
        
        # Assigning a type to the variable 'minutes_f' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'minutes_f', round_call_result_313883)
        
        # Assigning a Call to a Name (line 286):
        
        # Assigning a Call to a Name (line 286):
        
        # Call to int(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'minutes_f' (line 286)
        minutes_f_313885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'minutes_f', False)
        # Processing the call keyword arguments (line 286)
        kwargs_313886 = {}
        # Getting the type of 'int' (line 286)
        int_313884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'int', False)
        # Calling int(args, kwargs) (line 286)
        int_call_result_313887 = invoke(stypy.reporting.localization.Localization(__file__, 286, 18), int_313884, *[minutes_f_313885], **kwargs_313886)
        
        # Assigning a type to the variable 'minutes' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'minutes', int_call_result_313887)
        
        # Assigning a Call to a Name (line 288):
        
        # Assigning a Call to a Name (line 288):
        
        # Call to round(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'self' (line 288)
        self_313889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 26), 'self', False)
        # Obtaining the member 'seconds' of a type (line 288)
        seconds_313890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 26), self_313889, 'seconds')
        int_313891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 41), 'int')
        # Getting the type of 'minutes_f' (line 288)
        minutes_f_313892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 47), 'minutes_f', False)
        # Getting the type of 'minutes' (line 288)
        minutes_313893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 59), 'minutes', False)
        # Applying the binary operator '-' (line 288)
        result_sub_313894 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 47), '-', minutes_f_313892, minutes_313893)
        
        # Applying the binary operator '*' (line 288)
        result_mul_313895 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 41), '*', int_313891, result_sub_313894)
        
        # Applying the binary operator '+' (line 288)
        result_add_313896 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 26), '+', seconds_313890, result_mul_313895)
        
        int_313897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 69), 'int')
        # Processing the call keyword arguments (line 288)
        kwargs_313898 = {}
        # Getting the type of 'round' (line 288)
        round_313888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 20), 'round', False)
        # Calling round(args, kwargs) (line 288)
        round_call_result_313899 = invoke(stypy.reporting.localization.Localization(__file__, 288, 20), round_313888, *[result_add_313896, int_313897], **kwargs_313898)
        
        # Assigning a type to the variable 'seconds_f' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'seconds_f', round_call_result_313899)
        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to int(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'seconds_f' (line 289)
        seconds_f_313901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 'seconds_f', False)
        # Processing the call keyword arguments (line 289)
        kwargs_313902 = {}
        # Getting the type of 'int' (line 289)
        int_313900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 18), 'int', False)
        # Calling int(args, kwargs) (line 289)
        int_call_result_313903 = invoke(stypy.reporting.localization.Localization(__file__, 289, 18), int_313900, *[seconds_f_313901], **kwargs_313902)
        
        # Assigning a type to the variable 'seconds' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'seconds', int_call_result_313903)
        
        # Assigning a Call to a Name (line 291):
        
        # Assigning a Call to a Name (line 291):
        
        # Call to round(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'self' (line 291)
        self_313905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 29), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 291)
        microseconds_313906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 29), self_313905, 'microseconds')
        float_313907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 49), 'float')
        # Getting the type of 'seconds_f' (line 291)
        seconds_f_313908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 56), 'seconds_f', False)
        # Getting the type of 'seconds' (line 291)
        seconds_313909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 68), 'seconds', False)
        # Applying the binary operator '-' (line 291)
        result_sub_313910 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 56), '-', seconds_f_313908, seconds_313909)
        
        # Applying the binary operator '*' (line 291)
        result_mul_313911 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 49), '*', float_313907, result_sub_313910)
        
        # Applying the binary operator '+' (line 291)
        result_add_313912 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 29), '+', microseconds_313906, result_mul_313911)
        
        # Processing the call keyword arguments (line 291)
        kwargs_313913 = {}
        # Getting the type of 'round' (line 291)
        round_313904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 23), 'round', False)
        # Calling round(args, kwargs) (line 291)
        round_call_result_313914 = invoke(stypy.reporting.localization.Localization(__file__, 291, 23), round_313904, *[result_add_313912], **kwargs_313913)
        
        # Assigning a type to the variable 'microseconds' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'microseconds', round_call_result_313914)
        
        # Call to __class__(...): (line 294)
        # Processing the call keyword arguments (line 294)
        # Getting the type of 'self' (line 294)
        self_313917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 36), 'self', False)
        # Obtaining the member 'years' of a type (line 294)
        years_313918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 36), self_313917, 'years')
        keyword_313919 = years_313918
        # Getting the type of 'self' (line 294)
        self_313920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 55), 'self', False)
        # Obtaining the member 'months' of a type (line 294)
        months_313921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 55), self_313920, 'months')
        keyword_313922 = months_313921
        # Getting the type of 'days' (line 295)
        days_313923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'days', False)
        keyword_313924 = days_313923
        # Getting the type of 'hours' (line 295)
        hours_313925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 47), 'hours', False)
        keyword_313926 = hours_313925
        # Getting the type of 'minutes' (line 295)
        minutes_313927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 62), 'minutes', False)
        keyword_313928 = minutes_313927
        # Getting the type of 'seconds' (line 296)
        seconds_313929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 38), 'seconds', False)
        keyword_313930 = seconds_313929
        # Getting the type of 'microseconds' (line 296)
        microseconds_313931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 60), 'microseconds', False)
        keyword_313932 = microseconds_313931
        # Getting the type of 'self' (line 297)
        self_313933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 39), 'self', False)
        # Obtaining the member 'leapdays' of a type (line 297)
        leapdays_313934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 39), self_313933, 'leapdays')
        keyword_313935 = leapdays_313934
        # Getting the type of 'self' (line 297)
        self_313936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 59), 'self', False)
        # Obtaining the member 'year' of a type (line 297)
        year_313937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 59), self_313936, 'year')
        keyword_313938 = year_313937
        # Getting the type of 'self' (line 298)
        self_313939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 36), 'self', False)
        # Obtaining the member 'month' of a type (line 298)
        month_313940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 36), self_313939, 'month')
        keyword_313941 = month_313940
        # Getting the type of 'self' (line 298)
        self_313942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 52), 'self', False)
        # Obtaining the member 'day' of a type (line 298)
        day_313943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 52), self_313942, 'day')
        keyword_313944 = day_313943
        # Getting the type of 'self' (line 299)
        self_313945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 38), 'self', False)
        # Obtaining the member 'weekday' of a type (line 299)
        weekday_313946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 38), self_313945, 'weekday')
        keyword_313947 = weekday_313946
        # Getting the type of 'self' (line 299)
        self_313948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 57), 'self', False)
        # Obtaining the member 'hour' of a type (line 299)
        hour_313949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 57), self_313948, 'hour')
        keyword_313950 = hour_313949
        # Getting the type of 'self' (line 300)
        self_313951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 37), 'self', False)
        # Obtaining the member 'minute' of a type (line 300)
        minute_313952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 37), self_313951, 'minute')
        keyword_313953 = minute_313952
        # Getting the type of 'self' (line 300)
        self_313954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 57), 'self', False)
        # Obtaining the member 'second' of a type (line 300)
        second_313955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 57), self_313954, 'second')
        keyword_313956 = second_313955
        # Getting the type of 'self' (line 301)
        self_313957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 42), 'self', False)
        # Obtaining the member 'microsecond' of a type (line 301)
        microsecond_313958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 42), self_313957, 'microsecond')
        keyword_313959 = microsecond_313958
        kwargs_313960 = {'hour': keyword_313950, 'seconds': keyword_313930, 'months': keyword_313922, 'year': keyword_313938, 'days': keyword_313924, 'years': keyword_313919, 'hours': keyword_313926, 'second': keyword_313956, 'microsecond': keyword_313959, 'month': keyword_313941, 'microseconds': keyword_313932, 'leapdays': keyword_313935, 'minutes': keyword_313928, 'day': keyword_313944, 'minute': keyword_313953, 'weekday': keyword_313947}
        # Getting the type of 'self' (line 294)
        self_313915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 294)
        class___313916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), self_313915, '__class__')
        # Calling __class__(args, kwargs) (line 294)
        class___call_result_313961 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), class___313916, *[], **kwargs_313960)
        
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'stypy_return_type', class___call_result_313961)
        
        # ################# End of 'normalized(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'normalized' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_313962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_313962)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'normalized'
        return stypy_return_type_313962


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.__add__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.__add__.__dict__.__setitem__('stypy_function_name', 'relativedelta.__add__')
        relativedelta.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        relativedelta.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__add__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to isinstance(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'other' (line 304)
        other_313964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 22), 'other', False)
        # Getting the type of 'relativedelta' (line 304)
        relativedelta_313965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'relativedelta', False)
        # Processing the call keyword arguments (line 304)
        kwargs_313966 = {}
        # Getting the type of 'isinstance' (line 304)
        isinstance_313963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 304)
        isinstance_call_result_313967 = invoke(stypy.reporting.localization.Localization(__file__, 304, 11), isinstance_313963, *[other_313964, relativedelta_313965], **kwargs_313966)
        
        # Testing the type of an if condition (line 304)
        if_condition_313968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 8), isinstance_call_result_313967)
        # Assigning a type to the variable 'if_condition_313968' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'if_condition_313968', if_condition_313968)
        # SSA begins for if statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __class__(...): (line 305)
        # Processing the call keyword arguments (line 305)
        # Getting the type of 'other' (line 305)
        other_313971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 40), 'other', False)
        # Obtaining the member 'years' of a type (line 305)
        years_313972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 40), other_313971, 'years')
        # Getting the type of 'self' (line 305)
        self_313973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 54), 'self', False)
        # Obtaining the member 'years' of a type (line 305)
        years_313974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 54), self_313973, 'years')
        # Applying the binary operator '+' (line 305)
        result_add_313975 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 40), '+', years_313972, years_313974)
        
        keyword_313976 = result_add_313975
        # Getting the type of 'other' (line 306)
        other_313977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 40), 'other', False)
        # Obtaining the member 'months' of a type (line 306)
        months_313978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 40), other_313977, 'months')
        # Getting the type of 'self' (line 306)
        self_313979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 55), 'self', False)
        # Obtaining the member 'months' of a type (line 306)
        months_313980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 55), self_313979, 'months')
        # Applying the binary operator '+' (line 306)
        result_add_313981 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 40), '+', months_313978, months_313980)
        
        keyword_313982 = result_add_313981
        # Getting the type of 'other' (line 307)
        other_313983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 38), 'other', False)
        # Obtaining the member 'days' of a type (line 307)
        days_313984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 38), other_313983, 'days')
        # Getting the type of 'self' (line 307)
        self_313985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 51), 'self', False)
        # Obtaining the member 'days' of a type (line 307)
        days_313986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 51), self_313985, 'days')
        # Applying the binary operator '+' (line 307)
        result_add_313987 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 38), '+', days_313984, days_313986)
        
        keyword_313988 = result_add_313987
        # Getting the type of 'other' (line 308)
        other_313989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 39), 'other', False)
        # Obtaining the member 'hours' of a type (line 308)
        hours_313990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 39), other_313989, 'hours')
        # Getting the type of 'self' (line 308)
        self_313991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 53), 'self', False)
        # Obtaining the member 'hours' of a type (line 308)
        hours_313992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 53), self_313991, 'hours')
        # Applying the binary operator '+' (line 308)
        result_add_313993 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 39), '+', hours_313990, hours_313992)
        
        keyword_313994 = result_add_313993
        # Getting the type of 'other' (line 309)
        other_313995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 41), 'other', False)
        # Obtaining the member 'minutes' of a type (line 309)
        minutes_313996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 41), other_313995, 'minutes')
        # Getting the type of 'self' (line 309)
        self_313997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 57), 'self', False)
        # Obtaining the member 'minutes' of a type (line 309)
        minutes_313998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 57), self_313997, 'minutes')
        # Applying the binary operator '+' (line 309)
        result_add_313999 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 41), '+', minutes_313996, minutes_313998)
        
        keyword_314000 = result_add_313999
        # Getting the type of 'other' (line 310)
        other_314001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 41), 'other', False)
        # Obtaining the member 'seconds' of a type (line 310)
        seconds_314002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 41), other_314001, 'seconds')
        # Getting the type of 'self' (line 310)
        self_314003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 57), 'self', False)
        # Obtaining the member 'seconds' of a type (line 310)
        seconds_314004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 57), self_314003, 'seconds')
        # Applying the binary operator '+' (line 310)
        result_add_314005 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 41), '+', seconds_314002, seconds_314004)
        
        keyword_314006 = result_add_314005
        # Getting the type of 'other' (line 311)
        other_314007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 47), 'other', False)
        # Obtaining the member 'microseconds' of a type (line 311)
        microseconds_314008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 47), other_314007, 'microseconds')
        # Getting the type of 'self' (line 312)
        self_314009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 47), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 312)
        microseconds_314010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 47), self_314009, 'microseconds')
        # Applying the binary operator '+' (line 311)
        result_add_314011 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 47), '+', microseconds_314008, microseconds_314010)
        
        keyword_314012 = result_add_314011
        
        # Evaluating a boolean operation
        # Getting the type of 'other' (line 313)
        other_314013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 42), 'other', False)
        # Obtaining the member 'leapdays' of a type (line 313)
        leapdays_314014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 42), other_314013, 'leapdays')
        # Getting the type of 'self' (line 313)
        self_314015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 60), 'self', False)
        # Obtaining the member 'leapdays' of a type (line 313)
        leapdays_314016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 60), self_314015, 'leapdays')
        # Applying the binary operator 'or' (line 313)
        result_or_keyword_314017 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 42), 'or', leapdays_314014, leapdays_314016)
        
        keyword_314018 = result_or_keyword_314017
        
        
        # Getting the type of 'other' (line 314)
        other_314019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 53), 'other', False)
        # Obtaining the member 'year' of a type (line 314)
        year_314020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 53), other_314019, 'year')
        # Getting the type of 'None' (line 314)
        None_314021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 71), 'None', False)
        # Applying the binary operator 'isnot' (line 314)
        result_is_not_314022 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 53), 'isnot', year_314020, None_314021)
        
        # Testing the type of an if expression (line 314)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 39), result_is_not_314022)
        # SSA begins for if expression (line 314)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'other' (line 314)
        other_314023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 39), 'other', False)
        # Obtaining the member 'year' of a type (line 314)
        year_314024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 39), other_314023, 'year')
        # SSA branch for the else part of an if expression (line 314)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'self' (line 315)
        self_314025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 44), 'self', False)
        # Obtaining the member 'year' of a type (line 315)
        year_314026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 44), self_314025, 'year')
        # SSA join for if expression (line 314)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314027 = union_type.UnionType.add(year_314024, year_314026)
        
        keyword_314028 = if_exp_314027
        
        
        # Getting the type of 'other' (line 316)
        other_314029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 55), 'other', False)
        # Obtaining the member 'month' of a type (line 316)
        month_314030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 55), other_314029, 'month')
        # Getting the type of 'None' (line 316)
        None_314031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 74), 'None', False)
        # Applying the binary operator 'isnot' (line 316)
        result_is_not_314032 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 55), 'isnot', month_314030, None_314031)
        
        # Testing the type of an if expression (line 316)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 40), result_is_not_314032)
        # SSA begins for if expression (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'other' (line 316)
        other_314033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 40), 'other', False)
        # Obtaining the member 'month' of a type (line 316)
        month_314034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 40), other_314033, 'month')
        # SSA branch for the else part of an if expression (line 316)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'self' (line 317)
        self_314035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 45), 'self', False)
        # Obtaining the member 'month' of a type (line 317)
        month_314036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 45), self_314035, 'month')
        # SSA join for if expression (line 316)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314037 = union_type.UnionType.add(month_314034, month_314036)
        
        keyword_314038 = if_exp_314037
        
        
        # Getting the type of 'other' (line 318)
        other_314039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 51), 'other', False)
        # Obtaining the member 'day' of a type (line 318)
        day_314040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 51), other_314039, 'day')
        # Getting the type of 'None' (line 318)
        None_314041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 68), 'None', False)
        # Applying the binary operator 'isnot' (line 318)
        result_is_not_314042 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 51), 'isnot', day_314040, None_314041)
        
        # Testing the type of an if expression (line 318)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 38), result_is_not_314042)
        # SSA begins for if expression (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'other' (line 318)
        other_314043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 38), 'other', False)
        # Obtaining the member 'day' of a type (line 318)
        day_314044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 38), other_314043, 'day')
        # SSA branch for the else part of an if expression (line 318)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'self' (line 319)
        self_314045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 43), 'self', False)
        # Obtaining the member 'day' of a type (line 319)
        day_314046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 43), self_314045, 'day')
        # SSA join for if expression (line 318)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314047 = union_type.UnionType.add(day_314044, day_314046)
        
        keyword_314048 = if_exp_314047
        
        
        # Getting the type of 'other' (line 320)
        other_314049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 59), 'other', False)
        # Obtaining the member 'weekday' of a type (line 320)
        weekday_314050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 59), other_314049, 'weekday')
        # Getting the type of 'None' (line 320)
        None_314051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 80), 'None', False)
        # Applying the binary operator 'isnot' (line 320)
        result_is_not_314052 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 59), 'isnot', weekday_314050, None_314051)
        
        # Testing the type of an if expression (line 320)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 42), result_is_not_314052)
        # SSA begins for if expression (line 320)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'other' (line 320)
        other_314053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 42), 'other', False)
        # Obtaining the member 'weekday' of a type (line 320)
        weekday_314054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 42), other_314053, 'weekday')
        # SSA branch for the else part of an if expression (line 320)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'self' (line 321)
        self_314055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 47), 'self', False)
        # Obtaining the member 'weekday' of a type (line 321)
        weekday_314056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 47), self_314055, 'weekday')
        # SSA join for if expression (line 320)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314057 = union_type.UnionType.add(weekday_314054, weekday_314056)
        
        keyword_314058 = if_exp_314057
        
        
        # Getting the type of 'other' (line 322)
        other_314059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 53), 'other', False)
        # Obtaining the member 'hour' of a type (line 322)
        hour_314060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 53), other_314059, 'hour')
        # Getting the type of 'None' (line 322)
        None_314061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 71), 'None', False)
        # Applying the binary operator 'isnot' (line 322)
        result_is_not_314062 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 53), 'isnot', hour_314060, None_314061)
        
        # Testing the type of an if expression (line 322)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 39), result_is_not_314062)
        # SSA begins for if expression (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'other' (line 322)
        other_314063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 39), 'other', False)
        # Obtaining the member 'hour' of a type (line 322)
        hour_314064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 39), other_314063, 'hour')
        # SSA branch for the else part of an if expression (line 322)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'self' (line 323)
        self_314065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 44), 'self', False)
        # Obtaining the member 'hour' of a type (line 323)
        hour_314066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 44), self_314065, 'hour')
        # SSA join for if expression (line 322)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314067 = union_type.UnionType.add(hour_314064, hour_314066)
        
        keyword_314068 = if_exp_314067
        
        
        # Getting the type of 'other' (line 324)
        other_314069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 57), 'other', False)
        # Obtaining the member 'minute' of a type (line 324)
        minute_314070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 57), other_314069, 'minute')
        # Getting the type of 'None' (line 324)
        None_314071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 77), 'None', False)
        # Applying the binary operator 'isnot' (line 324)
        result_is_not_314072 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 57), 'isnot', minute_314070, None_314071)
        
        # Testing the type of an if expression (line 324)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 41), result_is_not_314072)
        # SSA begins for if expression (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'other' (line 324)
        other_314073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 41), 'other', False)
        # Obtaining the member 'minute' of a type (line 324)
        minute_314074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 41), other_314073, 'minute')
        # SSA branch for the else part of an if expression (line 324)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'self' (line 325)
        self_314075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 46), 'self', False)
        # Obtaining the member 'minute' of a type (line 325)
        minute_314076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 46), self_314075, 'minute')
        # SSA join for if expression (line 324)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314077 = union_type.UnionType.add(minute_314074, minute_314076)
        
        keyword_314078 = if_exp_314077
        
        
        # Getting the type of 'other' (line 326)
        other_314079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 57), 'other', False)
        # Obtaining the member 'second' of a type (line 326)
        second_314080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 57), other_314079, 'second')
        # Getting the type of 'None' (line 326)
        None_314081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 77), 'None', False)
        # Applying the binary operator 'isnot' (line 326)
        result_is_not_314082 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 57), 'isnot', second_314080, None_314081)
        
        # Testing the type of an if expression (line 326)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 41), result_is_not_314082)
        # SSA begins for if expression (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'other' (line 326)
        other_314083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 41), 'other', False)
        # Obtaining the member 'second' of a type (line 326)
        second_314084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 41), other_314083, 'second')
        # SSA branch for the else part of an if expression (line 326)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'self' (line 327)
        self_314085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 46), 'self', False)
        # Obtaining the member 'second' of a type (line 327)
        second_314086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 46), self_314085, 'second')
        # SSA join for if expression (line 326)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314087 = union_type.UnionType.add(second_314084, second_314086)
        
        keyword_314088 = if_exp_314087
        
        
        # Getting the type of 'other' (line 328)
        other_314089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 67), 'other', False)
        # Obtaining the member 'microsecond' of a type (line 328)
        microsecond_314090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 67), other_314089, 'microsecond')
        # Getting the type of 'None' (line 329)
        None_314091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 53), 'None', False)
        # Applying the binary operator 'isnot' (line 328)
        result_is_not_314092 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 67), 'isnot', microsecond_314090, None_314091)
        
        # Testing the type of an if expression (line 328)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 46), result_is_not_314092)
        # SSA begins for if expression (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'other' (line 328)
        other_314093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 46), 'other', False)
        # Obtaining the member 'microsecond' of a type (line 328)
        microsecond_314094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 46), other_314093, 'microsecond')
        # SSA branch for the else part of an if expression (line 328)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'self' (line 330)
        self_314095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 46), 'self', False)
        # Obtaining the member 'microsecond' of a type (line 330)
        microsecond_314096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 46), self_314095, 'microsecond')
        # SSA join for if expression (line 328)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314097 = union_type.UnionType.add(microsecond_314094, microsecond_314096)
        
        keyword_314098 = if_exp_314097
        kwargs_314099 = {'hour': keyword_314068, 'seconds': keyword_314006, 'months': keyword_313982, 'year': keyword_314028, 'days': keyword_313988, 'years': keyword_313976, 'hours': keyword_313994, 'second': keyword_314088, 'microsecond': keyword_314098, 'month': keyword_314038, 'microseconds': keyword_314012, 'leapdays': keyword_314018, 'minutes': keyword_314000, 'day': keyword_314048, 'minute': keyword_314078, 'weekday': keyword_314058}
        # Getting the type of 'self' (line 305)
        self_313969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'self', False)
        # Obtaining the member '__class__' of a type (line 305)
        class___313970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 19), self_313969, '__class__')
        # Calling __class__(args, kwargs) (line 305)
        class___call_result_314100 = invoke(stypy.reporting.localization.Localization(__file__, 305, 19), class___313970, *[], **kwargs_314099)
        
        # Assigning a type to the variable 'stypy_return_type' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'stypy_return_type', class___call_result_314100)
        # SSA join for if statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'other' (line 331)
        other_314102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 22), 'other', False)
        # Getting the type of 'datetime' (line 331)
        datetime_314103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 29), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 331)
        timedelta_314104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 29), datetime_314103, 'timedelta')
        # Processing the call keyword arguments (line 331)
        kwargs_314105 = {}
        # Getting the type of 'isinstance' (line 331)
        isinstance_314101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 331)
        isinstance_call_result_314106 = invoke(stypy.reporting.localization.Localization(__file__, 331, 11), isinstance_314101, *[other_314102, timedelta_314104], **kwargs_314105)
        
        # Testing the type of an if condition (line 331)
        if_condition_314107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), isinstance_call_result_314106)
        # Assigning a type to the variable 'if_condition_314107' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'if_condition_314107', if_condition_314107)
        # SSA begins for if statement (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __class__(...): (line 332)
        # Processing the call keyword arguments (line 332)
        # Getting the type of 'self' (line 332)
        self_314110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 40), 'self', False)
        # Obtaining the member 'years' of a type (line 332)
        years_314111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 40), self_314110, 'years')
        keyword_314112 = years_314111
        # Getting the type of 'self' (line 333)
        self_314113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 41), 'self', False)
        # Obtaining the member 'months' of a type (line 333)
        months_314114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 41), self_314113, 'months')
        keyword_314115 = months_314114
        # Getting the type of 'self' (line 334)
        self_314116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 39), 'self', False)
        # Obtaining the member 'days' of a type (line 334)
        days_314117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 39), self_314116, 'days')
        # Getting the type of 'other' (line 334)
        other_314118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 51), 'other', False)
        # Obtaining the member 'days' of a type (line 334)
        days_314119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 51), other_314118, 'days')
        # Applying the binary operator '+' (line 334)
        result_add_314120 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 39), '+', days_314117, days_314119)
        
        keyword_314121 = result_add_314120
        # Getting the type of 'self' (line 335)
        self_314122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 40), 'self', False)
        # Obtaining the member 'hours' of a type (line 335)
        hours_314123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 40), self_314122, 'hours')
        keyword_314124 = hours_314123
        # Getting the type of 'self' (line 336)
        self_314125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 42), 'self', False)
        # Obtaining the member 'minutes' of a type (line 336)
        minutes_314126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 42), self_314125, 'minutes')
        keyword_314127 = minutes_314126
        # Getting the type of 'self' (line 337)
        self_314128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 42), 'self', False)
        # Obtaining the member 'seconds' of a type (line 337)
        seconds_314129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 42), self_314128, 'seconds')
        # Getting the type of 'other' (line 337)
        other_314130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 57), 'other', False)
        # Obtaining the member 'seconds' of a type (line 337)
        seconds_314131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 57), other_314130, 'seconds')
        # Applying the binary operator '+' (line 337)
        result_add_314132 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 42), '+', seconds_314129, seconds_314131)
        
        keyword_314133 = result_add_314132
        # Getting the type of 'self' (line 338)
        self_314134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 47), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 338)
        microseconds_314135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 47), self_314134, 'microseconds')
        # Getting the type of 'other' (line 338)
        other_314136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 67), 'other', False)
        # Obtaining the member 'microseconds' of a type (line 338)
        microseconds_314137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 67), other_314136, 'microseconds')
        # Applying the binary operator '+' (line 338)
        result_add_314138 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 47), '+', microseconds_314135, microseconds_314137)
        
        keyword_314139 = result_add_314138
        # Getting the type of 'self' (line 339)
        self_314140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 43), 'self', False)
        # Obtaining the member 'leapdays' of a type (line 339)
        leapdays_314141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 43), self_314140, 'leapdays')
        keyword_314142 = leapdays_314141
        # Getting the type of 'self' (line 340)
        self_314143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 39), 'self', False)
        # Obtaining the member 'year' of a type (line 340)
        year_314144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 39), self_314143, 'year')
        keyword_314145 = year_314144
        # Getting the type of 'self' (line 341)
        self_314146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 40), 'self', False)
        # Obtaining the member 'month' of a type (line 341)
        month_314147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 40), self_314146, 'month')
        keyword_314148 = month_314147
        # Getting the type of 'self' (line 342)
        self_314149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 38), 'self', False)
        # Obtaining the member 'day' of a type (line 342)
        day_314150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 38), self_314149, 'day')
        keyword_314151 = day_314150
        # Getting the type of 'self' (line 343)
        self_314152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 42), 'self', False)
        # Obtaining the member 'weekday' of a type (line 343)
        weekday_314153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 42), self_314152, 'weekday')
        keyword_314154 = weekday_314153
        # Getting the type of 'self' (line 344)
        self_314155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 39), 'self', False)
        # Obtaining the member 'hour' of a type (line 344)
        hour_314156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 39), self_314155, 'hour')
        keyword_314157 = hour_314156
        # Getting the type of 'self' (line 345)
        self_314158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 41), 'self', False)
        # Obtaining the member 'minute' of a type (line 345)
        minute_314159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 41), self_314158, 'minute')
        keyword_314160 = minute_314159
        # Getting the type of 'self' (line 346)
        self_314161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 41), 'self', False)
        # Obtaining the member 'second' of a type (line 346)
        second_314162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 41), self_314161, 'second')
        keyword_314163 = second_314162
        # Getting the type of 'self' (line 347)
        self_314164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 46), 'self', False)
        # Obtaining the member 'microsecond' of a type (line 347)
        microsecond_314165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 46), self_314164, 'microsecond')
        keyword_314166 = microsecond_314165
        kwargs_314167 = {'hour': keyword_314157, 'seconds': keyword_314133, 'months': keyword_314115, 'year': keyword_314145, 'days': keyword_314121, 'years': keyword_314112, 'hours': keyword_314124, 'second': keyword_314163, 'microsecond': keyword_314166, 'month': keyword_314148, 'microseconds': keyword_314139, 'leapdays': keyword_314142, 'minutes': keyword_314127, 'day': keyword_314151, 'minute': keyword_314160, 'weekday': keyword_314154}
        # Getting the type of 'self' (line 332)
        self_314108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'self', False)
        # Obtaining the member '__class__' of a type (line 332)
        class___314109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), self_314108, '__class__')
        # Calling __class__(args, kwargs) (line 332)
        class___call_result_314168 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), class___314109, *[], **kwargs_314167)
        
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'stypy_return_type', class___call_result_314168)
        # SSA join for if statement (line 331)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to isinstance(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'other' (line 348)
        other_314170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'other', False)
        # Getting the type of 'datetime' (line 348)
        datetime_314171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 33), 'datetime', False)
        # Obtaining the member 'date' of a type (line 348)
        date_314172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 33), datetime_314171, 'date')
        # Processing the call keyword arguments (line 348)
        kwargs_314173 = {}
        # Getting the type of 'isinstance' (line 348)
        isinstance_314169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 348)
        isinstance_call_result_314174 = invoke(stypy.reporting.localization.Localization(__file__, 348, 15), isinstance_314169, *[other_314170, date_314172], **kwargs_314173)
        
        # Applying the 'not' unary operator (line 348)
        result_not__314175 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 11), 'not', isinstance_call_result_314174)
        
        # Testing the type of an if condition (line 348)
        if_condition_314176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 8), result_not__314175)
        # Assigning a type to the variable 'if_condition_314176' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'if_condition_314176', if_condition_314176)
        # SSA begins for if statement (line 348)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 349)
        NotImplemented_314177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'stypy_return_type', NotImplemented_314177)
        # SSA branch for the else part of an if statement (line 348)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 350)
        self_314178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 'self')
        # Obtaining the member '_has_time' of a type (line 350)
        _has_time_314179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 13), self_314178, '_has_time')
        
        
        # Call to isinstance(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'other' (line 350)
        other_314181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 47), 'other', False)
        # Getting the type of 'datetime' (line 350)
        datetime_314182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 54), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 350)
        datetime_314183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 54), datetime_314182, 'datetime')
        # Processing the call keyword arguments (line 350)
        kwargs_314184 = {}
        # Getting the type of 'isinstance' (line 350)
        isinstance_314180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 36), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 350)
        isinstance_call_result_314185 = invoke(stypy.reporting.localization.Localization(__file__, 350, 36), isinstance_314180, *[other_314181, datetime_314183], **kwargs_314184)
        
        # Applying the 'not' unary operator (line 350)
        result_not__314186 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 32), 'not', isinstance_call_result_314185)
        
        # Applying the binary operator 'and' (line 350)
        result_and_keyword_314187 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 13), 'and', _has_time_314179, result_not__314186)
        
        # Testing the type of an if condition (line 350)
        if_condition_314188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 13), result_and_keyword_314187)
        # Assigning a type to the variable 'if_condition_314188' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 'if_condition_314188', if_condition_314188)
        # SSA begins for if statement (line 350)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to fromordinal(...): (line 351)
        # Processing the call arguments (line 351)
        
        # Call to toordinal(...): (line 351)
        # Processing the call keyword arguments (line 351)
        kwargs_314194 = {}
        # Getting the type of 'other' (line 351)
        other_314192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 50), 'other', False)
        # Obtaining the member 'toordinal' of a type (line 351)
        toordinal_314193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 50), other_314192, 'toordinal')
        # Calling toordinal(args, kwargs) (line 351)
        toordinal_call_result_314195 = invoke(stypy.reporting.localization.Localization(__file__, 351, 50), toordinal_314193, *[], **kwargs_314194)
        
        # Processing the call keyword arguments (line 351)
        kwargs_314196 = {}
        # Getting the type of 'datetime' (line 351)
        datetime_314189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 20), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 351)
        datetime_314190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 20), datetime_314189, 'datetime')
        # Obtaining the member 'fromordinal' of a type (line 351)
        fromordinal_314191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 20), datetime_314190, 'fromordinal')
        # Calling fromordinal(args, kwargs) (line 351)
        fromordinal_call_result_314197 = invoke(stypy.reporting.localization.Localization(__file__, 351, 20), fromordinal_314191, *[toordinal_call_result_314195], **kwargs_314196)
        
        # Assigning a type to the variable 'other' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'other', fromordinal_call_result_314197)
        # SSA join for if statement (line 350)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 348)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 352):
        
        # Assigning a BinOp to a Name (line 352):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 352)
        self_314198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'self')
        # Obtaining the member 'year' of a type (line 352)
        year_314199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 16), self_314198, 'year')
        # Getting the type of 'other' (line 352)
        other_314200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 29), 'other')
        # Obtaining the member 'year' of a type (line 352)
        year_314201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 29), other_314200, 'year')
        # Applying the binary operator 'or' (line 352)
        result_or_keyword_314202 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 16), 'or', year_314199, year_314201)
        
        # Getting the type of 'self' (line 352)
        self_314203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 41), 'self')
        # Obtaining the member 'years' of a type (line 352)
        years_314204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 41), self_314203, 'years')
        # Applying the binary operator '+' (line 352)
        result_add_314205 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 15), '+', result_or_keyword_314202, years_314204)
        
        # Assigning a type to the variable 'year' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'year', result_add_314205)
        
        # Assigning a BoolOp to a Name (line 353):
        
        # Assigning a BoolOp to a Name (line 353):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 353)
        self_314206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'self')
        # Obtaining the member 'month' of a type (line 353)
        month_314207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), self_314206, 'month')
        # Getting the type of 'other' (line 353)
        other_314208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'other')
        # Obtaining the member 'month' of a type (line 353)
        month_314209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 30), other_314208, 'month')
        # Applying the binary operator 'or' (line 353)
        result_or_keyword_314210 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 16), 'or', month_314207, month_314209)
        
        # Assigning a type to the variable 'month' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'month', result_or_keyword_314210)
        
        # Getting the type of 'self' (line 354)
        self_314211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'self')
        # Obtaining the member 'months' of a type (line 354)
        months_314212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 11), self_314211, 'months')
        # Testing the type of an if condition (line 354)
        if_condition_314213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 8), months_314212)
        # Assigning a type to the variable 'if_condition_314213' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'if_condition_314213', if_condition_314213)
        # SSA begins for if statement (line 354)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Evaluating assert statement condition
        
        int_314214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 19), 'int')
        
        # Call to abs(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'self' (line 355)
        self_314216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 28), 'self', False)
        # Obtaining the member 'months' of a type (line 355)
        months_314217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 28), self_314216, 'months')
        # Processing the call keyword arguments (line 355)
        kwargs_314218 = {}
        # Getting the type of 'abs' (line 355)
        abs_314215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'abs', False)
        # Calling abs(args, kwargs) (line 355)
        abs_call_result_314219 = invoke(stypy.reporting.localization.Localization(__file__, 355, 24), abs_314215, *[months_314217], **kwargs_314218)
        
        # Applying the binary operator '<=' (line 355)
        result_le_314220 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 19), '<=', int_314214, abs_call_result_314219)
        int_314221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 44), 'int')
        # Applying the binary operator '<=' (line 355)
        result_le_314222 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 19), '<=', abs_call_result_314219, int_314221)
        # Applying the binary operator '&' (line 355)
        result_and__314223 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 19), '&', result_le_314220, result_le_314222)
        
        
        # Getting the type of 'month' (line 356)
        month_314224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'month')
        # Getting the type of 'self' (line 356)
        self_314225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 21), 'self')
        # Obtaining the member 'months' of a type (line 356)
        months_314226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 21), self_314225, 'months')
        # Applying the binary operator '+=' (line 356)
        result_iadd_314227 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 12), '+=', month_314224, months_314226)
        # Assigning a type to the variable 'month' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'month', result_iadd_314227)
        
        
        
        # Getting the type of 'month' (line 357)
        month_314228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 15), 'month')
        int_314229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 23), 'int')
        # Applying the binary operator '>' (line 357)
        result_gt_314230 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 15), '>', month_314228, int_314229)
        
        # Testing the type of an if condition (line 357)
        if_condition_314231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 12), result_gt_314230)
        # Assigning a type to the variable 'if_condition_314231' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'if_condition_314231', if_condition_314231)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'year' (line 358)
        year_314232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'year')
        int_314233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 24), 'int')
        # Applying the binary operator '+=' (line 358)
        result_iadd_314234 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 16), '+=', year_314232, int_314233)
        # Assigning a type to the variable 'year' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'year', result_iadd_314234)
        
        
        # Getting the type of 'month' (line 359)
        month_314235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'month')
        int_314236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 25), 'int')
        # Applying the binary operator '-=' (line 359)
        result_isub_314237 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 16), '-=', month_314235, int_314236)
        # Assigning a type to the variable 'month' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'month', result_isub_314237)
        
        # SSA branch for the else part of an if statement (line 357)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'month' (line 360)
        month_314238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 17), 'month')
        int_314239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 25), 'int')
        # Applying the binary operator '<' (line 360)
        result_lt_314240 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 17), '<', month_314238, int_314239)
        
        # Testing the type of an if condition (line 360)
        if_condition_314241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 17), result_lt_314240)
        # Assigning a type to the variable 'if_condition_314241' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 17), 'if_condition_314241', if_condition_314241)
        # SSA begins for if statement (line 360)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'year' (line 361)
        year_314242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'year')
        int_314243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 24), 'int')
        # Applying the binary operator '-=' (line 361)
        result_isub_314244 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 16), '-=', year_314242, int_314243)
        # Assigning a type to the variable 'year' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'year', result_isub_314244)
        
        
        # Getting the type of 'month' (line 362)
        month_314245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'month')
        int_314246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 25), 'int')
        # Applying the binary operator '+=' (line 362)
        result_iadd_314247 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 16), '+=', month_314245, int_314246)
        # Assigning a type to the variable 'month' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'month', result_iadd_314247)
        
        # SSA join for if statement (line 360)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 354)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 363):
        
        # Assigning a Call to a Name (line 363):
        
        # Call to min(...): (line 363)
        # Processing the call arguments (line 363)
        
        # Obtaining the type of the subscript
        int_314249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 51), 'int')
        
        # Call to monthrange(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'year' (line 363)
        year_314252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 38), 'year', False)
        # Getting the type of 'month' (line 363)
        month_314253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 44), 'month', False)
        # Processing the call keyword arguments (line 363)
        kwargs_314254 = {}
        # Getting the type of 'calendar' (line 363)
        calendar_314250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 18), 'calendar', False)
        # Obtaining the member 'monthrange' of a type (line 363)
        monthrange_314251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 18), calendar_314250, 'monthrange')
        # Calling monthrange(args, kwargs) (line 363)
        monthrange_call_result_314255 = invoke(stypy.reporting.localization.Localization(__file__, 363, 18), monthrange_314251, *[year_314252, month_314253], **kwargs_314254)
        
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___314256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 18), monthrange_call_result_314255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_314257 = invoke(stypy.reporting.localization.Localization(__file__, 363, 18), getitem___314256, int_314249)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 364)
        self_314258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 18), 'self', False)
        # Obtaining the member 'day' of a type (line 364)
        day_314259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 18), self_314258, 'day')
        # Getting the type of 'other' (line 364)
        other_314260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'other', False)
        # Obtaining the member 'day' of a type (line 364)
        day_314261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 30), other_314260, 'day')
        # Applying the binary operator 'or' (line 364)
        result_or_keyword_314262 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 18), 'or', day_314259, day_314261)
        
        # Processing the call keyword arguments (line 363)
        kwargs_314263 = {}
        # Getting the type of 'min' (line 363)
        min_314248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 14), 'min', False)
        # Calling min(args, kwargs) (line 363)
        min_call_result_314264 = invoke(stypy.reporting.localization.Localization(__file__, 363, 14), min_314248, *[subscript_call_result_314257, result_or_keyword_314262], **kwargs_314263)
        
        # Assigning a type to the variable 'day' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'day', min_call_result_314264)
        
        # Assigning a Dict to a Name (line 365):
        
        # Assigning a Dict to a Name (line 365):
        
        # Obtaining an instance of the builtin type 'dict' (line 365)
        dict_314265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 365)
        # Adding element type (key, value) (line 365)
        str_314266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 16), 'str', 'year')
        # Getting the type of 'year' (line 365)
        year_314267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 24), 'year')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 15), dict_314265, (str_314266, year_314267))
        # Adding element type (key, value) (line 365)
        str_314268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 30), 'str', 'month')
        # Getting the type of 'month' (line 365)
        month_314269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 39), 'month')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 15), dict_314265, (str_314268, month_314269))
        # Adding element type (key, value) (line 365)
        str_314270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 46), 'str', 'day')
        # Getting the type of 'day' (line 365)
        day_314271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 53), 'day')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 15), dict_314265, (str_314270, day_314271))
        
        # Assigning a type to the variable 'repl' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'repl', dict_314265)
        
        
        # Obtaining an instance of the builtin type 'list' (line 366)
        list_314272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 366)
        # Adding element type (line 366)
        str_314273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 21), 'str', 'hour')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 20), list_314272, str_314273)
        # Adding element type (line 366)
        str_314274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 29), 'str', 'minute')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 20), list_314272, str_314274)
        # Adding element type (line 366)
        str_314275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 39), 'str', 'second')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 20), list_314272, str_314275)
        # Adding element type (line 366)
        str_314276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 49), 'str', 'microsecond')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 20), list_314272, str_314276)
        
        # Testing the type of a for loop iterable (line 366)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 366, 8), list_314272)
        # Getting the type of the for loop variable (line 366)
        for_loop_var_314277 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 366, 8), list_314272)
        # Assigning a type to the variable 'attr' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'attr', for_loop_var_314277)
        # SSA begins for a for statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 367):
        
        # Assigning a Call to a Name (line 367):
        
        # Call to getattr(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'self' (line 367)
        self_314279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 28), 'self', False)
        # Getting the type of 'attr' (line 367)
        attr_314280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 34), 'attr', False)
        # Processing the call keyword arguments (line 367)
        kwargs_314281 = {}
        # Getting the type of 'getattr' (line 367)
        getattr_314278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 367)
        getattr_call_result_314282 = invoke(stypy.reporting.localization.Localization(__file__, 367, 20), getattr_314278, *[self_314279, attr_314280], **kwargs_314281)
        
        # Assigning a type to the variable 'value' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'value', getattr_call_result_314282)
        
        # Type idiom detected: calculating its left and rigth part (line 368)
        # Getting the type of 'value' (line 368)
        value_314283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'value')
        # Getting the type of 'None' (line 368)
        None_314284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 28), 'None')
        
        (may_be_314285, more_types_in_union_314286) = may_not_be_none(value_314283, None_314284)

        if may_be_314285:

            if more_types_in_union_314286:
                # Runtime conditional SSA (line 368)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Subscript (line 369):
            
            # Assigning a Name to a Subscript (line 369):
            # Getting the type of 'value' (line 369)
            value_314287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 29), 'value')
            # Getting the type of 'repl' (line 369)
            repl_314288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'repl')
            # Getting the type of 'attr' (line 369)
            attr_314289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'attr')
            # Storing an element on a container (line 369)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 16), repl_314288, (attr_314289, value_314287))

            if more_types_in_union_314286:
                # SSA join for if statement (line 368)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 370):
        
        # Assigning a Attribute to a Name (line 370):
        # Getting the type of 'self' (line 370)
        self_314290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'self')
        # Obtaining the member 'days' of a type (line 370)
        days_314291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 15), self_314290, 'days')
        # Assigning a type to the variable 'days' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'days', days_314291)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 371)
        self_314292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'self')
        # Obtaining the member 'leapdays' of a type (line 371)
        leapdays_314293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 11), self_314292, 'leapdays')
        
        # Getting the type of 'month' (line 371)
        month_314294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 29), 'month')
        int_314295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 37), 'int')
        # Applying the binary operator '>' (line 371)
        result_gt_314296 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 29), '>', month_314294, int_314295)
        
        # Applying the binary operator 'and' (line 371)
        result_and_keyword_314297 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 11), 'and', leapdays_314293, result_gt_314296)
        
        # Call to isleap(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'year' (line 371)
        year_314300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 59), 'year', False)
        # Processing the call keyword arguments (line 371)
        kwargs_314301 = {}
        # Getting the type of 'calendar' (line 371)
        calendar_314298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 43), 'calendar', False)
        # Obtaining the member 'isleap' of a type (line 371)
        isleap_314299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 43), calendar_314298, 'isleap')
        # Calling isleap(args, kwargs) (line 371)
        isleap_call_result_314302 = invoke(stypy.reporting.localization.Localization(__file__, 371, 43), isleap_314299, *[year_314300], **kwargs_314301)
        
        # Applying the binary operator 'and' (line 371)
        result_and_keyword_314303 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 11), 'and', result_and_keyword_314297, isleap_call_result_314302)
        
        # Testing the type of an if condition (line 371)
        if_condition_314304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 8), result_and_keyword_314303)
        # Assigning a type to the variable 'if_condition_314304' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'if_condition_314304', if_condition_314304)
        # SSA begins for if statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'days' (line 372)
        days_314305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'days')
        # Getting the type of 'self' (line 372)
        self_314306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'self')
        # Obtaining the member 'leapdays' of a type (line 372)
        leapdays_314307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 20), self_314306, 'leapdays')
        # Applying the binary operator '+=' (line 372)
        result_iadd_314308 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 12), '+=', days_314305, leapdays_314307)
        # Assigning a type to the variable 'days' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'days', result_iadd_314308)
        
        # SSA join for if statement (line 371)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 373):
        
        # Assigning a BinOp to a Name (line 373):
        
        # Call to replace(...): (line 373)
        # Processing the call keyword arguments (line 373)
        # Getting the type of 'repl' (line 373)
        repl_314311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 31), 'repl', False)
        kwargs_314312 = {'repl_314311': repl_314311}
        # Getting the type of 'other' (line 373)
        other_314309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'other', False)
        # Obtaining the member 'replace' of a type (line 373)
        replace_314310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), other_314309, 'replace')
        # Calling replace(args, kwargs) (line 373)
        replace_call_result_314313 = invoke(stypy.reporting.localization.Localization(__file__, 373, 15), replace_314310, *[], **kwargs_314312)
        
        
        # Call to timedelta(...): (line 374)
        # Processing the call keyword arguments (line 374)
        # Getting the type of 'days' (line 374)
        days_314316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 41), 'days', False)
        keyword_314317 = days_314316
        # Getting the type of 'self' (line 375)
        self_314318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 42), 'self', False)
        # Obtaining the member 'hours' of a type (line 375)
        hours_314319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 42), self_314318, 'hours')
        keyword_314320 = hours_314319
        # Getting the type of 'self' (line 376)
        self_314321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 44), 'self', False)
        # Obtaining the member 'minutes' of a type (line 376)
        minutes_314322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 44), self_314321, 'minutes')
        keyword_314323 = minutes_314322
        # Getting the type of 'self' (line 377)
        self_314324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 44), 'self', False)
        # Obtaining the member 'seconds' of a type (line 377)
        seconds_314325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 44), self_314324, 'seconds')
        keyword_314326 = seconds_314325
        # Getting the type of 'self' (line 378)
        self_314327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 49), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 378)
        microseconds_314328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 49), self_314327, 'microseconds')
        keyword_314329 = microseconds_314328
        kwargs_314330 = {'hours': keyword_314320, 'seconds': keyword_314326, 'minutes': keyword_314323, 'days': keyword_314317, 'microseconds': keyword_314329}
        # Getting the type of 'datetime' (line 374)
        datetime_314314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 17), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 374)
        timedelta_314315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 17), datetime_314314, 'timedelta')
        # Calling timedelta(args, kwargs) (line 374)
        timedelta_call_result_314331 = invoke(stypy.reporting.localization.Localization(__file__, 374, 17), timedelta_314315, *[], **kwargs_314330)
        
        # Applying the binary operator '+' (line 373)
        result_add_314332 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 15), '+', replace_call_result_314313, timedelta_call_result_314331)
        
        # Assigning a type to the variable 'ret' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'ret', result_add_314332)
        
        # Getting the type of 'self' (line 379)
        self_314333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'self')
        # Obtaining the member 'weekday' of a type (line 379)
        weekday_314334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 11), self_314333, 'weekday')
        # Testing the type of an if condition (line 379)
        if_condition_314335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 8), weekday_314334)
        # Assigning a type to the variable 'if_condition_314335' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'if_condition_314335', if_condition_314335)
        # SSA begins for if statement (line 379)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 380):
        
        # Assigning a Attribute to a Name (line 380):
        # Getting the type of 'self' (line 380)
        self_314336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 27), 'self')
        # Obtaining the member 'weekday' of a type (line 380)
        weekday_314337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 27), self_314336, 'weekday')
        # Obtaining the member 'weekday' of a type (line 380)
        weekday_314338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 27), weekday_314337, 'weekday')
        # Assigning a type to the variable 'tuple_assignment_313102' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'tuple_assignment_313102', weekday_314338)
        
        # Assigning a BoolOp to a Name (line 380):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 380)
        self_314339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 49), 'self')
        # Obtaining the member 'weekday' of a type (line 380)
        weekday_314340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 49), self_314339, 'weekday')
        # Obtaining the member 'n' of a type (line 380)
        n_314341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 49), weekday_314340, 'n')
        int_314342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 67), 'int')
        # Applying the binary operator 'or' (line 380)
        result_or_keyword_314343 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 49), 'or', n_314341, int_314342)
        
        # Assigning a type to the variable 'tuple_assignment_313103' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'tuple_assignment_313103', result_or_keyword_314343)
        
        # Assigning a Name to a Name (line 380):
        # Getting the type of 'tuple_assignment_313102' (line 380)
        tuple_assignment_313102_314344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'tuple_assignment_313102')
        # Assigning a type to the variable 'weekday' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'weekday', tuple_assignment_313102_314344)
        
        # Assigning a Name to a Name (line 380):
        # Getting the type of 'tuple_assignment_313103' (line 380)
        tuple_assignment_313103_314345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'tuple_assignment_313103')
        # Assigning a type to the variable 'nth' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'nth', tuple_assignment_313103_314345)
        
        # Assigning a BinOp to a Name (line 381):
        
        # Assigning a BinOp to a Name (line 381):
        
        # Call to abs(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'nth' (line 381)
        nth_314347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 28), 'nth', False)
        # Processing the call keyword arguments (line 381)
        kwargs_314348 = {}
        # Getting the type of 'abs' (line 381)
        abs_314346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 24), 'abs', False)
        # Calling abs(args, kwargs) (line 381)
        abs_call_result_314349 = invoke(stypy.reporting.localization.Localization(__file__, 381, 24), abs_314346, *[nth_314347], **kwargs_314348)
        
        int_314350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 35), 'int')
        # Applying the binary operator '-' (line 381)
        result_sub_314351 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 24), '-', abs_call_result_314349, int_314350)
        
        int_314352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 40), 'int')
        # Applying the binary operator '*' (line 381)
        result_mul_314353 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 23), '*', result_sub_314351, int_314352)
        
        # Assigning a type to the variable 'jumpdays' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'jumpdays', result_mul_314353)
        
        
        # Getting the type of 'nth' (line 382)
        nth_314354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 15), 'nth')
        int_314355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 21), 'int')
        # Applying the binary operator '>' (line 382)
        result_gt_314356 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 15), '>', nth_314354, int_314355)
        
        # Testing the type of an if condition (line 382)
        if_condition_314357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 12), result_gt_314356)
        # Assigning a type to the variable 'if_condition_314357' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'if_condition_314357', if_condition_314357)
        # SSA begins for if statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'jumpdays' (line 383)
        jumpdays_314358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'jumpdays')
        int_314359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 29), 'int')
        
        # Call to weekday(...): (line 383)
        # Processing the call keyword arguments (line 383)
        kwargs_314362 = {}
        # Getting the type of 'ret' (line 383)
        ret_314360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 33), 'ret', False)
        # Obtaining the member 'weekday' of a type (line 383)
        weekday_314361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 33), ret_314360, 'weekday')
        # Calling weekday(args, kwargs) (line 383)
        weekday_call_result_314363 = invoke(stypy.reporting.localization.Localization(__file__, 383, 33), weekday_314361, *[], **kwargs_314362)
        
        # Applying the binary operator '-' (line 383)
        result_sub_314364 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 29), '-', int_314359, weekday_call_result_314363)
        
        # Getting the type of 'weekday' (line 383)
        weekday_314365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 49), 'weekday')
        # Applying the binary operator '+' (line 383)
        result_add_314366 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 47), '+', result_sub_314364, weekday_314365)
        
        int_314367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 60), 'int')
        # Applying the binary operator '%' (line 383)
        result_mod_314368 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 28), '%', result_add_314366, int_314367)
        
        # Applying the binary operator '+=' (line 383)
        result_iadd_314369 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 16), '+=', jumpdays_314358, result_mod_314368)
        # Assigning a type to the variable 'jumpdays' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'jumpdays', result_iadd_314369)
        
        # SSA branch for the else part of an if statement (line 382)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'jumpdays' (line 385)
        jumpdays_314370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'jumpdays')
        
        # Call to weekday(...): (line 385)
        # Processing the call keyword arguments (line 385)
        kwargs_314373 = {}
        # Getting the type of 'ret' (line 385)
        ret_314371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 29), 'ret', False)
        # Obtaining the member 'weekday' of a type (line 385)
        weekday_314372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 29), ret_314371, 'weekday')
        # Calling weekday(args, kwargs) (line 385)
        weekday_call_result_314374 = invoke(stypy.reporting.localization.Localization(__file__, 385, 29), weekday_314372, *[], **kwargs_314373)
        
        # Getting the type of 'weekday' (line 385)
        weekday_314375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 45), 'weekday')
        # Applying the binary operator '-' (line 385)
        result_sub_314376 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 29), '-', weekday_call_result_314374, weekday_314375)
        
        int_314377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 56), 'int')
        # Applying the binary operator '%' (line 385)
        result_mod_314378 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 28), '%', result_sub_314376, int_314377)
        
        # Applying the binary operator '+=' (line 385)
        result_iadd_314379 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 16), '+=', jumpdays_314370, result_mod_314378)
        # Assigning a type to the variable 'jumpdays' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 16), 'jumpdays', result_iadd_314379)
        
        
        # Getting the type of 'jumpdays' (line 386)
        jumpdays_314380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'jumpdays')
        int_314381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 28), 'int')
        # Applying the binary operator '*=' (line 386)
        result_imul_314382 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 16), '*=', jumpdays_314380, int_314381)
        # Assigning a type to the variable 'jumpdays' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'jumpdays', result_imul_314382)
        
        # SSA join for if statement (line 382)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'ret' (line 387)
        ret_314383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'ret')
        
        # Call to timedelta(...): (line 387)
        # Processing the call keyword arguments (line 387)
        # Getting the type of 'jumpdays' (line 387)
        jumpdays_314386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 43), 'jumpdays', False)
        keyword_314387 = jumpdays_314386
        kwargs_314388 = {'days': keyword_314387}
        # Getting the type of 'datetime' (line 387)
        datetime_314384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 19), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 387)
        timedelta_314385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 19), datetime_314384, 'timedelta')
        # Calling timedelta(args, kwargs) (line 387)
        timedelta_call_result_314389 = invoke(stypy.reporting.localization.Localization(__file__, 387, 19), timedelta_314385, *[], **kwargs_314388)
        
        # Applying the binary operator '+=' (line 387)
        result_iadd_314390 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 12), '+=', ret_314383, timedelta_call_result_314389)
        # Assigning a type to the variable 'ret' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'ret', result_iadd_314390)
        
        # SSA join for if statement (line 379)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'ret' (line 388)
        ret_314391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type', ret_314391)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_314392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314392)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_314392


    @norecursion
    def __radd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__radd__'
        module_type_store = module_type_store.open_function_context('__radd__', 390, 4, False)
        # Assigning a type to the variable 'self' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.__radd__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.__radd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.__radd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.__radd__.__dict__.__setitem__('stypy_function_name', 'relativedelta.__radd__')
        relativedelta.__radd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        relativedelta.__radd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.__radd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.__radd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.__radd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.__radd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.__radd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__radd__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __add__(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'other' (line 391)
        other_314395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 28), 'other', False)
        # Processing the call keyword arguments (line 391)
        kwargs_314396 = {}
        # Getting the type of 'self' (line 391)
        self_314393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 15), 'self', False)
        # Obtaining the member '__add__' of a type (line 391)
        add___314394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 15), self_314393, '__add__')
        # Calling __add__(args, kwargs) (line 391)
        add___call_result_314397 = invoke(stypy.reporting.localization.Localization(__file__, 391, 15), add___314394, *[other_314395], **kwargs_314396)
        
        # Assigning a type to the variable 'stypy_return_type' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'stypy_return_type', add___call_result_314397)
        
        # ################# End of '__radd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__radd__' in the type store
        # Getting the type of 'stypy_return_type' (line 390)
        stypy_return_type_314398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314398)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__radd__'
        return stypy_return_type_314398


    @norecursion
    def __rsub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rsub__'
        module_type_store = module_type_store.open_function_context('__rsub__', 393, 4, False)
        # Assigning a type to the variable 'self' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.__rsub__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.__rsub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.__rsub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.__rsub__.__dict__.__setitem__('stypy_function_name', 'relativedelta.__rsub__')
        relativedelta.__rsub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        relativedelta.__rsub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.__rsub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.__rsub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.__rsub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.__rsub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.__rsub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__rsub__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __radd__(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'other' (line 394)
        other_314404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 39), 'other', False)
        # Processing the call keyword arguments (line 394)
        kwargs_314405 = {}
        
        # Call to __neg__(...): (line 394)
        # Processing the call keyword arguments (line 394)
        kwargs_314401 = {}
        # Getting the type of 'self' (line 394)
        self_314399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 15), 'self', False)
        # Obtaining the member '__neg__' of a type (line 394)
        neg___314400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 15), self_314399, '__neg__')
        # Calling __neg__(args, kwargs) (line 394)
        neg___call_result_314402 = invoke(stypy.reporting.localization.Localization(__file__, 394, 15), neg___314400, *[], **kwargs_314401)
        
        # Obtaining the member '__radd__' of a type (line 394)
        radd___314403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 15), neg___call_result_314402, '__radd__')
        # Calling __radd__(args, kwargs) (line 394)
        radd___call_result_314406 = invoke(stypy.reporting.localization.Localization(__file__, 394, 15), radd___314403, *[other_314404], **kwargs_314405)
        
        # Assigning a type to the variable 'stypy_return_type' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'stypy_return_type', radd___call_result_314406)
        
        # ################# End of '__rsub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rsub__' in the type store
        # Getting the type of 'stypy_return_type' (line 393)
        stypy_return_type_314407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rsub__'
        return stypy_return_type_314407


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 396, 4, False)
        # Assigning a type to the variable 'self' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.__sub__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.__sub__.__dict__.__setitem__('stypy_function_name', 'relativedelta.__sub__')
        relativedelta.__sub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        relativedelta.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__sub__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Call to isinstance(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'other' (line 397)
        other_314409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 26), 'other', False)
        # Getting the type of 'relativedelta' (line 397)
        relativedelta_314410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 33), 'relativedelta', False)
        # Processing the call keyword arguments (line 397)
        kwargs_314411 = {}
        # Getting the type of 'isinstance' (line 397)
        isinstance_314408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 397)
        isinstance_call_result_314412 = invoke(stypy.reporting.localization.Localization(__file__, 397, 15), isinstance_314408, *[other_314409, relativedelta_314410], **kwargs_314411)
        
        # Applying the 'not' unary operator (line 397)
        result_not__314413 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 11), 'not', isinstance_call_result_314412)
        
        # Testing the type of an if condition (line 397)
        if_condition_314414 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 8), result_not__314413)
        # Assigning a type to the variable 'if_condition_314414' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'if_condition_314414', if_condition_314414)
        # SSA begins for if statement (line 397)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 398)
        NotImplemented_314415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'stypy_return_type', NotImplemented_314415)
        # SSA join for if statement (line 397)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 399)
        # Processing the call keyword arguments (line 399)
        # Getting the type of 'self' (line 399)
        self_314418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 36), 'self', False)
        # Obtaining the member 'years' of a type (line 399)
        years_314419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 36), self_314418, 'years')
        # Getting the type of 'other' (line 399)
        other_314420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 49), 'other', False)
        # Obtaining the member 'years' of a type (line 399)
        years_314421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 49), other_314420, 'years')
        # Applying the binary operator '-' (line 399)
        result_sub_314422 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 36), '-', years_314419, years_314421)
        
        keyword_314423 = result_sub_314422
        # Getting the type of 'self' (line 400)
        self_314424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 36), 'self', False)
        # Obtaining the member 'months' of a type (line 400)
        months_314425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 36), self_314424, 'months')
        # Getting the type of 'other' (line 400)
        other_314426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 50), 'other', False)
        # Obtaining the member 'months' of a type (line 400)
        months_314427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 50), other_314426, 'months')
        # Applying the binary operator '-' (line 400)
        result_sub_314428 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 36), '-', months_314425, months_314427)
        
        keyword_314429 = result_sub_314428
        # Getting the type of 'self' (line 401)
        self_314430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 34), 'self', False)
        # Obtaining the member 'days' of a type (line 401)
        days_314431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 34), self_314430, 'days')
        # Getting the type of 'other' (line 401)
        other_314432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 46), 'other', False)
        # Obtaining the member 'days' of a type (line 401)
        days_314433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 46), other_314432, 'days')
        # Applying the binary operator '-' (line 401)
        result_sub_314434 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 34), '-', days_314431, days_314433)
        
        keyword_314435 = result_sub_314434
        # Getting the type of 'self' (line 402)
        self_314436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 35), 'self', False)
        # Obtaining the member 'hours' of a type (line 402)
        hours_314437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 35), self_314436, 'hours')
        # Getting the type of 'other' (line 402)
        other_314438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 48), 'other', False)
        # Obtaining the member 'hours' of a type (line 402)
        hours_314439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 48), other_314438, 'hours')
        # Applying the binary operator '-' (line 402)
        result_sub_314440 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 35), '-', hours_314437, hours_314439)
        
        keyword_314441 = result_sub_314440
        # Getting the type of 'self' (line 403)
        self_314442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 37), 'self', False)
        # Obtaining the member 'minutes' of a type (line 403)
        minutes_314443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 37), self_314442, 'minutes')
        # Getting the type of 'other' (line 403)
        other_314444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 52), 'other', False)
        # Obtaining the member 'minutes' of a type (line 403)
        minutes_314445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 52), other_314444, 'minutes')
        # Applying the binary operator '-' (line 403)
        result_sub_314446 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 37), '-', minutes_314443, minutes_314445)
        
        keyword_314447 = result_sub_314446
        # Getting the type of 'self' (line 404)
        self_314448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 37), 'self', False)
        # Obtaining the member 'seconds' of a type (line 404)
        seconds_314449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 37), self_314448, 'seconds')
        # Getting the type of 'other' (line 404)
        other_314450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 52), 'other', False)
        # Obtaining the member 'seconds' of a type (line 404)
        seconds_314451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 52), other_314450, 'seconds')
        # Applying the binary operator '-' (line 404)
        result_sub_314452 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 37), '-', seconds_314449, seconds_314451)
        
        keyword_314453 = result_sub_314452
        # Getting the type of 'self' (line 405)
        self_314454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 42), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 405)
        microseconds_314455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 42), self_314454, 'microseconds')
        # Getting the type of 'other' (line 405)
        other_314456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 62), 'other', False)
        # Obtaining the member 'microseconds' of a type (line 405)
        microseconds_314457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 62), other_314456, 'microseconds')
        # Applying the binary operator '-' (line 405)
        result_sub_314458 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 42), '-', microseconds_314455, microseconds_314457)
        
        keyword_314459 = result_sub_314458
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 406)
        self_314460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 38), 'self', False)
        # Obtaining the member 'leapdays' of a type (line 406)
        leapdays_314461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 38), self_314460, 'leapdays')
        # Getting the type of 'other' (line 406)
        other_314462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 55), 'other', False)
        # Obtaining the member 'leapdays' of a type (line 406)
        leapdays_314463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 55), other_314462, 'leapdays')
        # Applying the binary operator 'or' (line 406)
        result_or_keyword_314464 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 38), 'or', leapdays_314461, leapdays_314463)
        
        keyword_314465 = result_or_keyword_314464
        
        
        # Getting the type of 'self' (line 407)
        self_314466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'self', False)
        # Obtaining the member 'year' of a type (line 407)
        year_314467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 48), self_314466, 'year')
        # Getting the type of 'None' (line 407)
        None_314468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 65), 'None', False)
        # Applying the binary operator 'isnot' (line 407)
        result_is_not_314469 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 48), 'isnot', year_314467, None_314468)
        
        # Testing the type of an if expression (line 407)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 35), result_is_not_314469)
        # SSA begins for if expression (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 407)
        self_314470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 35), 'self', False)
        # Obtaining the member 'year' of a type (line 407)
        year_314471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 35), self_314470, 'year')
        # SSA branch for the else part of an if expression (line 407)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'other' (line 408)
        other_314472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'other', False)
        # Obtaining the member 'year' of a type (line 408)
        year_314473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 40), other_314472, 'year')
        # SSA join for if expression (line 407)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314474 = union_type.UnionType.add(year_314471, year_314473)
        
        keyword_314475 = if_exp_314474
        
        
        # Getting the type of 'self' (line 409)
        self_314476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 50), 'self', False)
        # Obtaining the member 'month' of a type (line 409)
        month_314477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 50), self_314476, 'month')
        # Getting the type of 'None' (line 409)
        None_314478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 68), 'None', False)
        # Applying the binary operator 'isnot' (line 409)
        result_is_not_314479 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 50), 'isnot', month_314477, None_314478)
        
        # Testing the type of an if expression (line 409)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 36), result_is_not_314479)
        # SSA begins for if expression (line 409)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 409)
        self_314480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 36), 'self', False)
        # Obtaining the member 'month' of a type (line 409)
        month_314481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 36), self_314480, 'month')
        # SSA branch for the else part of an if expression (line 409)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'other' (line 410)
        other_314482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 36), 'other', False)
        # Obtaining the member 'month' of a type (line 410)
        month_314483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 36), other_314482, 'month')
        # SSA join for if expression (line 409)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314484 = union_type.UnionType.add(month_314481, month_314483)
        
        keyword_314485 = if_exp_314484
        
        
        # Getting the type of 'self' (line 411)
        self_314486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 46), 'self', False)
        # Obtaining the member 'day' of a type (line 411)
        day_314487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 46), self_314486, 'day')
        # Getting the type of 'None' (line 411)
        None_314488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 62), 'None', False)
        # Applying the binary operator 'isnot' (line 411)
        result_is_not_314489 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 46), 'isnot', day_314487, None_314488)
        
        # Testing the type of an if expression (line 411)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 34), result_is_not_314489)
        # SSA begins for if expression (line 411)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 411)
        self_314490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 34), 'self', False)
        # Obtaining the member 'day' of a type (line 411)
        day_314491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 34), self_314490, 'day')
        # SSA branch for the else part of an if expression (line 411)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'other' (line 412)
        other_314492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 34), 'other', False)
        # Obtaining the member 'day' of a type (line 412)
        day_314493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 34), other_314492, 'day')
        # SSA join for if expression (line 411)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314494 = union_type.UnionType.add(day_314491, day_314493)
        
        keyword_314495 = if_exp_314494
        
        
        # Getting the type of 'self' (line 413)
        self_314496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 54), 'self', False)
        # Obtaining the member 'weekday' of a type (line 413)
        weekday_314497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 54), self_314496, 'weekday')
        # Getting the type of 'None' (line 413)
        None_314498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 74), 'None', False)
        # Applying the binary operator 'isnot' (line 413)
        result_is_not_314499 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 54), 'isnot', weekday_314497, None_314498)
        
        # Testing the type of an if expression (line 413)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 38), result_is_not_314499)
        # SSA begins for if expression (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 413)
        self_314500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 38), 'self', False)
        # Obtaining the member 'weekday' of a type (line 413)
        weekday_314501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 38), self_314500, 'weekday')
        # SSA branch for the else part of an if expression (line 413)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'other' (line 414)
        other_314502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 38), 'other', False)
        # Obtaining the member 'weekday' of a type (line 414)
        weekday_314503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 38), other_314502, 'weekday')
        # SSA join for if expression (line 413)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314504 = union_type.UnionType.add(weekday_314501, weekday_314503)
        
        keyword_314505 = if_exp_314504
        
        
        # Getting the type of 'self' (line 415)
        self_314506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 48), 'self', False)
        # Obtaining the member 'hour' of a type (line 415)
        hour_314507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 48), self_314506, 'hour')
        # Getting the type of 'None' (line 415)
        None_314508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 65), 'None', False)
        # Applying the binary operator 'isnot' (line 415)
        result_is_not_314509 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 48), 'isnot', hour_314507, None_314508)
        
        # Testing the type of an if expression (line 415)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 35), result_is_not_314509)
        # SSA begins for if expression (line 415)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 415)
        self_314510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 35), 'self', False)
        # Obtaining the member 'hour' of a type (line 415)
        hour_314511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 35), self_314510, 'hour')
        # SSA branch for the else part of an if expression (line 415)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'other' (line 416)
        other_314512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 35), 'other', False)
        # Obtaining the member 'hour' of a type (line 416)
        hour_314513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 35), other_314512, 'hour')
        # SSA join for if expression (line 415)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314514 = union_type.UnionType.add(hour_314511, hour_314513)
        
        keyword_314515 = if_exp_314514
        
        
        # Getting the type of 'self' (line 417)
        self_314516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 52), 'self', False)
        # Obtaining the member 'minute' of a type (line 417)
        minute_314517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 52), self_314516, 'minute')
        # Getting the type of 'None' (line 417)
        None_314518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 71), 'None', False)
        # Applying the binary operator 'isnot' (line 417)
        result_is_not_314519 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 52), 'isnot', minute_314517, None_314518)
        
        # Testing the type of an if expression (line 417)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 37), result_is_not_314519)
        # SSA begins for if expression (line 417)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 417)
        self_314520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 37), 'self', False)
        # Obtaining the member 'minute' of a type (line 417)
        minute_314521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 37), self_314520, 'minute')
        # SSA branch for the else part of an if expression (line 417)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'other' (line 418)
        other_314522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 37), 'other', False)
        # Obtaining the member 'minute' of a type (line 418)
        minute_314523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 37), other_314522, 'minute')
        # SSA join for if expression (line 417)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314524 = union_type.UnionType.add(minute_314521, minute_314523)
        
        keyword_314525 = if_exp_314524
        
        
        # Getting the type of 'self' (line 419)
        self_314526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 52), 'self', False)
        # Obtaining the member 'second' of a type (line 419)
        second_314527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 52), self_314526, 'second')
        # Getting the type of 'None' (line 419)
        None_314528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 71), 'None', False)
        # Applying the binary operator 'isnot' (line 419)
        result_is_not_314529 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 52), 'isnot', second_314527, None_314528)
        
        # Testing the type of an if expression (line 419)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 37), result_is_not_314529)
        # SSA begins for if expression (line 419)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 419)
        self_314530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 37), 'self', False)
        # Obtaining the member 'second' of a type (line 419)
        second_314531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 37), self_314530, 'second')
        # SSA branch for the else part of an if expression (line 419)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'other' (line 420)
        other_314532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 37), 'other', False)
        # Obtaining the member 'second' of a type (line 420)
        second_314533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 37), other_314532, 'second')
        # SSA join for if expression (line 419)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314534 = union_type.UnionType.add(second_314531, second_314533)
        
        keyword_314535 = if_exp_314534
        
        
        # Getting the type of 'self' (line 421)
        self_314536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 62), 'self', False)
        # Obtaining the member 'microsecond' of a type (line 421)
        microsecond_314537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 62), self_314536, 'microsecond')
        # Getting the type of 'None' (line 422)
        None_314538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 49), 'None', False)
        # Applying the binary operator 'isnot' (line 421)
        result_is_not_314539 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 62), 'isnot', microsecond_314537, None_314538)
        
        # Testing the type of an if expression (line 421)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 421, 42), result_is_not_314539)
        # SSA begins for if expression (line 421)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 421)
        self_314540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 42), 'self', False)
        # Obtaining the member 'microsecond' of a type (line 421)
        microsecond_314541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 42), self_314540, 'microsecond')
        # SSA branch for the else part of an if expression (line 421)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'other' (line 423)
        other_314542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 42), 'other', False)
        # Obtaining the member 'microsecond' of a type (line 423)
        microsecond_314543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 42), other_314542, 'microsecond')
        # SSA join for if expression (line 421)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_314544 = union_type.UnionType.add(microsecond_314541, microsecond_314543)
        
        keyword_314545 = if_exp_314544
        kwargs_314546 = {'hour': keyword_314515, 'seconds': keyword_314453, 'months': keyword_314429, 'year': keyword_314475, 'days': keyword_314435, 'years': keyword_314423, 'hours': keyword_314441, 'second': keyword_314535, 'microsecond': keyword_314545, 'month': keyword_314485, 'microseconds': keyword_314459, 'leapdays': keyword_314465, 'minutes': keyword_314447, 'day': keyword_314495, 'minute': keyword_314525, 'weekday': keyword_314505}
        # Getting the type of 'self' (line 399)
        self_314416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 399)
        class___314417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 15), self_314416, '__class__')
        # Calling __class__(args, kwargs) (line 399)
        class___call_result_314547 = invoke(stypy.reporting.localization.Localization(__file__, 399, 15), class___314417, *[], **kwargs_314546)
        
        # Assigning a type to the variable 'stypy_return_type' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'stypy_return_type', class___call_result_314547)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 396)
        stypy_return_type_314548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314548)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_314548


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.__neg__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.__neg__.__dict__.__setitem__('stypy_function_name', 'relativedelta.__neg__')
        relativedelta.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        relativedelta.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__neg__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __class__(...): (line 426)
        # Processing the call keyword arguments (line 426)
        
        # Getting the type of 'self' (line 426)
        self_314551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 37), 'self', False)
        # Obtaining the member 'years' of a type (line 426)
        years_314552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 37), self_314551, 'years')
        # Applying the 'usub' unary operator (line 426)
        result___neg___314553 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 36), 'usub', years_314552)
        
        keyword_314554 = result___neg___314553
        
        # Getting the type of 'self' (line 427)
        self_314555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 37), 'self', False)
        # Obtaining the member 'months' of a type (line 427)
        months_314556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 37), self_314555, 'months')
        # Applying the 'usub' unary operator (line 427)
        result___neg___314557 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 36), 'usub', months_314556)
        
        keyword_314558 = result___neg___314557
        
        # Getting the type of 'self' (line 428)
        self_314559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 35), 'self', False)
        # Obtaining the member 'days' of a type (line 428)
        days_314560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 35), self_314559, 'days')
        # Applying the 'usub' unary operator (line 428)
        result___neg___314561 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 34), 'usub', days_314560)
        
        keyword_314562 = result___neg___314561
        
        # Getting the type of 'self' (line 429)
        self_314563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 36), 'self', False)
        # Obtaining the member 'hours' of a type (line 429)
        hours_314564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 36), self_314563, 'hours')
        # Applying the 'usub' unary operator (line 429)
        result___neg___314565 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 35), 'usub', hours_314564)
        
        keyword_314566 = result___neg___314565
        
        # Getting the type of 'self' (line 430)
        self_314567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 38), 'self', False)
        # Obtaining the member 'minutes' of a type (line 430)
        minutes_314568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 38), self_314567, 'minutes')
        # Applying the 'usub' unary operator (line 430)
        result___neg___314569 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 37), 'usub', minutes_314568)
        
        keyword_314570 = result___neg___314569
        
        # Getting the type of 'self' (line 431)
        self_314571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 38), 'self', False)
        # Obtaining the member 'seconds' of a type (line 431)
        seconds_314572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 38), self_314571, 'seconds')
        # Applying the 'usub' unary operator (line 431)
        result___neg___314573 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 37), 'usub', seconds_314572)
        
        keyword_314574 = result___neg___314573
        
        # Getting the type of 'self' (line 432)
        self_314575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 43), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 432)
        microseconds_314576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 43), self_314575, 'microseconds')
        # Applying the 'usub' unary operator (line 432)
        result___neg___314577 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 42), 'usub', microseconds_314576)
        
        keyword_314578 = result___neg___314577
        # Getting the type of 'self' (line 433)
        self_314579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 38), 'self', False)
        # Obtaining the member 'leapdays' of a type (line 433)
        leapdays_314580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 38), self_314579, 'leapdays')
        keyword_314581 = leapdays_314580
        # Getting the type of 'self' (line 434)
        self_314582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 34), 'self', False)
        # Obtaining the member 'year' of a type (line 434)
        year_314583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 34), self_314582, 'year')
        keyword_314584 = year_314583
        # Getting the type of 'self' (line 435)
        self_314585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 35), 'self', False)
        # Obtaining the member 'month' of a type (line 435)
        month_314586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 35), self_314585, 'month')
        keyword_314587 = month_314586
        # Getting the type of 'self' (line 436)
        self_314588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 33), 'self', False)
        # Obtaining the member 'day' of a type (line 436)
        day_314589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 33), self_314588, 'day')
        keyword_314590 = day_314589
        # Getting the type of 'self' (line 437)
        self_314591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 37), 'self', False)
        # Obtaining the member 'weekday' of a type (line 437)
        weekday_314592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 37), self_314591, 'weekday')
        keyword_314593 = weekday_314592
        # Getting the type of 'self' (line 438)
        self_314594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 34), 'self', False)
        # Obtaining the member 'hour' of a type (line 438)
        hour_314595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 34), self_314594, 'hour')
        keyword_314596 = hour_314595
        # Getting the type of 'self' (line 439)
        self_314597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 36), 'self', False)
        # Obtaining the member 'minute' of a type (line 439)
        minute_314598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 36), self_314597, 'minute')
        keyword_314599 = minute_314598
        # Getting the type of 'self' (line 440)
        self_314600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 36), 'self', False)
        # Obtaining the member 'second' of a type (line 440)
        second_314601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 36), self_314600, 'second')
        keyword_314602 = second_314601
        # Getting the type of 'self' (line 441)
        self_314603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 41), 'self', False)
        # Obtaining the member 'microsecond' of a type (line 441)
        microsecond_314604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 41), self_314603, 'microsecond')
        keyword_314605 = microsecond_314604
        kwargs_314606 = {'hour': keyword_314596, 'seconds': keyword_314574, 'months': keyword_314558, 'year': keyword_314584, 'days': keyword_314562, 'years': keyword_314554, 'hours': keyword_314566, 'second': keyword_314602, 'microsecond': keyword_314605, 'month': keyword_314587, 'microseconds': keyword_314578, 'leapdays': keyword_314581, 'minutes': keyword_314570, 'day': keyword_314590, 'minute': keyword_314599, 'weekday': keyword_314593}
        # Getting the type of 'self' (line 426)
        self_314549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 426)
        class___314550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 15), self_314549, '__class__')
        # Calling __class__(args, kwargs) (line 426)
        class___call_result_314607 = invoke(stypy.reporting.localization.Localization(__file__, 426, 15), class___314550, *[], **kwargs_314606)
        
        # Assigning a type to the variable 'stypy_return_type' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'stypy_return_type', class___call_result_314607)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_314608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_314608


    @norecursion
    def __bool__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__bool__'
        module_type_store = module_type_store.open_function_context('__bool__', 443, 4, False)
        # Assigning a type to the variable 'self' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.__bool__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.__bool__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.__bool__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.__bool__.__dict__.__setitem__('stypy_function_name', 'relativedelta.__bool__')
        relativedelta.__bool__.__dict__.__setitem__('stypy_param_names_list', [])
        relativedelta.__bool__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.__bool__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.__bool__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.__bool__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.__bool__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.__bool__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__bool__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__bool__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__bool__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 444)
        self_314609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 24), 'self')
        # Obtaining the member 'years' of a type (line 444)
        years_314610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 24), self_314609, 'years')
        # Applying the 'not' unary operator (line 444)
        result_not__314611 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'not', years_314610)
        
        
        # Getting the type of 'self' (line 445)
        self_314612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 24), 'self')
        # Obtaining the member 'months' of a type (line 445)
        months_314613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 24), self_314612, 'months')
        # Applying the 'not' unary operator (line 445)
        result_not__314614 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 20), 'not', months_314613)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314615 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_not__314611, result_not__314614)
        
        # Getting the type of 'self' (line 446)
        self_314616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 24), 'self')
        # Obtaining the member 'days' of a type (line 446)
        days_314617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 24), self_314616, 'days')
        # Applying the 'not' unary operator (line 446)
        result_not__314618 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 20), 'not', days_314617)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314619 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314615, result_not__314618)
        
        # Getting the type of 'self' (line 447)
        self_314620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'self')
        # Obtaining the member 'hours' of a type (line 447)
        hours_314621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 24), self_314620, 'hours')
        # Applying the 'not' unary operator (line 447)
        result_not__314622 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 20), 'not', hours_314621)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314623 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314619, result_not__314622)
        
        # Getting the type of 'self' (line 448)
        self_314624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'self')
        # Obtaining the member 'minutes' of a type (line 448)
        minutes_314625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 24), self_314624, 'minutes')
        # Applying the 'not' unary operator (line 448)
        result_not__314626 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 20), 'not', minutes_314625)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314627 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314623, result_not__314626)
        
        # Getting the type of 'self' (line 449)
        self_314628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 24), 'self')
        # Obtaining the member 'seconds' of a type (line 449)
        seconds_314629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 24), self_314628, 'seconds')
        # Applying the 'not' unary operator (line 449)
        result_not__314630 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 20), 'not', seconds_314629)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314631 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314627, result_not__314630)
        
        # Getting the type of 'self' (line 450)
        self_314632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'self')
        # Obtaining the member 'microseconds' of a type (line 450)
        microseconds_314633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 24), self_314632, 'microseconds')
        # Applying the 'not' unary operator (line 450)
        result_not__314634 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 20), 'not', microseconds_314633)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314635 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314631, result_not__314634)
        
        # Getting the type of 'self' (line 451)
        self_314636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 24), 'self')
        # Obtaining the member 'leapdays' of a type (line 451)
        leapdays_314637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), self_314636, 'leapdays')
        # Applying the 'not' unary operator (line 451)
        result_not__314638 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 20), 'not', leapdays_314637)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314639 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314635, result_not__314638)
        
        # Getting the type of 'self' (line 452)
        self_314640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 20), 'self')
        # Obtaining the member 'year' of a type (line 452)
        year_314641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 20), self_314640, 'year')
        # Getting the type of 'None' (line 452)
        None_314642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 33), 'None')
        # Applying the binary operator 'is' (line 452)
        result_is__314643 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 20), 'is', year_314641, None_314642)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314644 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314639, result_is__314643)
        
        # Getting the type of 'self' (line 453)
        self_314645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 20), 'self')
        # Obtaining the member 'month' of a type (line 453)
        month_314646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 20), self_314645, 'month')
        # Getting the type of 'None' (line 453)
        None_314647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'None')
        # Applying the binary operator 'is' (line 453)
        result_is__314648 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 20), 'is', month_314646, None_314647)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314649 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314644, result_is__314648)
        
        # Getting the type of 'self' (line 454)
        self_314650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 20), 'self')
        # Obtaining the member 'day' of a type (line 454)
        day_314651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 20), self_314650, 'day')
        # Getting the type of 'None' (line 454)
        None_314652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 32), 'None')
        # Applying the binary operator 'is' (line 454)
        result_is__314653 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 20), 'is', day_314651, None_314652)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314654 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314649, result_is__314653)
        
        # Getting the type of 'self' (line 455)
        self_314655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 20), 'self')
        # Obtaining the member 'weekday' of a type (line 455)
        weekday_314656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 20), self_314655, 'weekday')
        # Getting the type of 'None' (line 455)
        None_314657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 36), 'None')
        # Applying the binary operator 'is' (line 455)
        result_is__314658 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 20), 'is', weekday_314656, None_314657)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314659 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314654, result_is__314658)
        
        # Getting the type of 'self' (line 456)
        self_314660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 20), 'self')
        # Obtaining the member 'hour' of a type (line 456)
        hour_314661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 20), self_314660, 'hour')
        # Getting the type of 'None' (line 456)
        None_314662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 33), 'None')
        # Applying the binary operator 'is' (line 456)
        result_is__314663 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 20), 'is', hour_314661, None_314662)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314664 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314659, result_is__314663)
        
        # Getting the type of 'self' (line 457)
        self_314665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 20), 'self')
        # Obtaining the member 'minute' of a type (line 457)
        minute_314666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 20), self_314665, 'minute')
        # Getting the type of 'None' (line 457)
        None_314667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 35), 'None')
        # Applying the binary operator 'is' (line 457)
        result_is__314668 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 20), 'is', minute_314666, None_314667)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314669 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314664, result_is__314668)
        
        # Getting the type of 'self' (line 458)
        self_314670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 20), 'self')
        # Obtaining the member 'second' of a type (line 458)
        second_314671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 20), self_314670, 'second')
        # Getting the type of 'None' (line 458)
        None_314672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 35), 'None')
        # Applying the binary operator 'is' (line 458)
        result_is__314673 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 20), 'is', second_314671, None_314672)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314674 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314669, result_is__314673)
        
        # Getting the type of 'self' (line 459)
        self_314675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 20), 'self')
        # Obtaining the member 'microsecond' of a type (line 459)
        microsecond_314676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 20), self_314675, 'microsecond')
        # Getting the type of 'None' (line 459)
        None_314677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 40), 'None')
        # Applying the binary operator 'is' (line 459)
        result_is__314678 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 20), 'is', microsecond_314676, None_314677)
        
        # Applying the binary operator 'and' (line 444)
        result_and_keyword_314679 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 20), 'and', result_and_keyword_314674, result_is__314678)
        
        # Applying the 'not' unary operator (line 444)
        result_not__314680 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 15), 'not', result_and_keyword_314679)
        
        # Assigning a type to the variable 'stypy_return_type' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'stypy_return_type', result_not__314680)
        
        # ################# End of '__bool__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__bool__' in the type store
        # Getting the type of 'stypy_return_type' (line 443)
        stypy_return_type_314681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314681)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__bool__'
        return stypy_return_type_314681

    
    # Assigning a Name to a Name (line 461):

    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 463, 4, False)
        # Assigning a type to the variable 'self' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.__mul__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.__mul__.__dict__.__setitem__('stypy_function_name', 'relativedelta.__mul__')
        relativedelta.__mul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        relativedelta.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__mul__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # SSA begins for try-except statement (line 464)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 465):
        
        # Assigning a Call to a Name (line 465):
        
        # Call to float(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'other' (line 465)
        other_314683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 22), 'other', False)
        # Processing the call keyword arguments (line 465)
        kwargs_314684 = {}
        # Getting the type of 'float' (line 465)
        float_314682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'float', False)
        # Calling float(args, kwargs) (line 465)
        float_call_result_314685 = invoke(stypy.reporting.localization.Localization(__file__, 465, 16), float_314682, *[other_314683], **kwargs_314684)
        
        # Assigning a type to the variable 'f' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'f', float_call_result_314685)
        # SSA branch for the except part of a try statement (line 464)
        # SSA branch for the except 'TypeError' branch of a try statement (line 464)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 467)
        NotImplemented_314686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'stypy_return_type', NotImplemented_314686)
        # SSA join for try-except statement (line 464)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 469)
        # Processing the call keyword arguments (line 469)
        
        # Call to int(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'self' (line 469)
        self_314690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 40), 'self', False)
        # Obtaining the member 'years' of a type (line 469)
        years_314691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 40), self_314690, 'years')
        # Getting the type of 'f' (line 469)
        f_314692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 53), 'f', False)
        # Applying the binary operator '*' (line 469)
        result_mul_314693 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 40), '*', years_314691, f_314692)
        
        # Processing the call keyword arguments (line 469)
        kwargs_314694 = {}
        # Getting the type of 'int' (line 469)
        int_314689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 36), 'int', False)
        # Calling int(args, kwargs) (line 469)
        int_call_result_314695 = invoke(stypy.reporting.localization.Localization(__file__, 469, 36), int_314689, *[result_mul_314693], **kwargs_314694)
        
        keyword_314696 = int_call_result_314695
        
        # Call to int(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'self' (line 470)
        self_314698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 40), 'self', False)
        # Obtaining the member 'months' of a type (line 470)
        months_314699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 40), self_314698, 'months')
        # Getting the type of 'f' (line 470)
        f_314700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 54), 'f', False)
        # Applying the binary operator '*' (line 470)
        result_mul_314701 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 40), '*', months_314699, f_314700)
        
        # Processing the call keyword arguments (line 470)
        kwargs_314702 = {}
        # Getting the type of 'int' (line 470)
        int_314697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 36), 'int', False)
        # Calling int(args, kwargs) (line 470)
        int_call_result_314703 = invoke(stypy.reporting.localization.Localization(__file__, 470, 36), int_314697, *[result_mul_314701], **kwargs_314702)
        
        keyword_314704 = int_call_result_314703
        
        # Call to int(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'self' (line 471)
        self_314706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 38), 'self', False)
        # Obtaining the member 'days' of a type (line 471)
        days_314707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 38), self_314706, 'days')
        # Getting the type of 'f' (line 471)
        f_314708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 50), 'f', False)
        # Applying the binary operator '*' (line 471)
        result_mul_314709 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 38), '*', days_314707, f_314708)
        
        # Processing the call keyword arguments (line 471)
        kwargs_314710 = {}
        # Getting the type of 'int' (line 471)
        int_314705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 34), 'int', False)
        # Calling int(args, kwargs) (line 471)
        int_call_result_314711 = invoke(stypy.reporting.localization.Localization(__file__, 471, 34), int_314705, *[result_mul_314709], **kwargs_314710)
        
        keyword_314712 = int_call_result_314711
        
        # Call to int(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'self' (line 472)
        self_314714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 39), 'self', False)
        # Obtaining the member 'hours' of a type (line 472)
        hours_314715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 39), self_314714, 'hours')
        # Getting the type of 'f' (line 472)
        f_314716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 52), 'f', False)
        # Applying the binary operator '*' (line 472)
        result_mul_314717 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 39), '*', hours_314715, f_314716)
        
        # Processing the call keyword arguments (line 472)
        kwargs_314718 = {}
        # Getting the type of 'int' (line 472)
        int_314713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 35), 'int', False)
        # Calling int(args, kwargs) (line 472)
        int_call_result_314719 = invoke(stypy.reporting.localization.Localization(__file__, 472, 35), int_314713, *[result_mul_314717], **kwargs_314718)
        
        keyword_314720 = int_call_result_314719
        
        # Call to int(...): (line 473)
        # Processing the call arguments (line 473)
        # Getting the type of 'self' (line 473)
        self_314722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 41), 'self', False)
        # Obtaining the member 'minutes' of a type (line 473)
        minutes_314723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 41), self_314722, 'minutes')
        # Getting the type of 'f' (line 473)
        f_314724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 56), 'f', False)
        # Applying the binary operator '*' (line 473)
        result_mul_314725 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 41), '*', minutes_314723, f_314724)
        
        # Processing the call keyword arguments (line 473)
        kwargs_314726 = {}
        # Getting the type of 'int' (line 473)
        int_314721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 37), 'int', False)
        # Calling int(args, kwargs) (line 473)
        int_call_result_314727 = invoke(stypy.reporting.localization.Localization(__file__, 473, 37), int_314721, *[result_mul_314725], **kwargs_314726)
        
        keyword_314728 = int_call_result_314727
        
        # Call to int(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'self' (line 474)
        self_314730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 41), 'self', False)
        # Obtaining the member 'seconds' of a type (line 474)
        seconds_314731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 41), self_314730, 'seconds')
        # Getting the type of 'f' (line 474)
        f_314732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 56), 'f', False)
        # Applying the binary operator '*' (line 474)
        result_mul_314733 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 41), '*', seconds_314731, f_314732)
        
        # Processing the call keyword arguments (line 474)
        kwargs_314734 = {}
        # Getting the type of 'int' (line 474)
        int_314729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 37), 'int', False)
        # Calling int(args, kwargs) (line 474)
        int_call_result_314735 = invoke(stypy.reporting.localization.Localization(__file__, 474, 37), int_314729, *[result_mul_314733], **kwargs_314734)
        
        keyword_314736 = int_call_result_314735
        
        # Call to int(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'self' (line 475)
        self_314738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 46), 'self', False)
        # Obtaining the member 'microseconds' of a type (line 475)
        microseconds_314739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 46), self_314738, 'microseconds')
        # Getting the type of 'f' (line 475)
        f_314740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 66), 'f', False)
        # Applying the binary operator '*' (line 475)
        result_mul_314741 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 46), '*', microseconds_314739, f_314740)
        
        # Processing the call keyword arguments (line 475)
        kwargs_314742 = {}
        # Getting the type of 'int' (line 475)
        int_314737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 42), 'int', False)
        # Calling int(args, kwargs) (line 475)
        int_call_result_314743 = invoke(stypy.reporting.localization.Localization(__file__, 475, 42), int_314737, *[result_mul_314741], **kwargs_314742)
        
        keyword_314744 = int_call_result_314743
        # Getting the type of 'self' (line 476)
        self_314745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 38), 'self', False)
        # Obtaining the member 'leapdays' of a type (line 476)
        leapdays_314746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 38), self_314745, 'leapdays')
        keyword_314747 = leapdays_314746
        # Getting the type of 'self' (line 477)
        self_314748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 34), 'self', False)
        # Obtaining the member 'year' of a type (line 477)
        year_314749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 34), self_314748, 'year')
        keyword_314750 = year_314749
        # Getting the type of 'self' (line 478)
        self_314751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 35), 'self', False)
        # Obtaining the member 'month' of a type (line 478)
        month_314752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 35), self_314751, 'month')
        keyword_314753 = month_314752
        # Getting the type of 'self' (line 479)
        self_314754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 33), 'self', False)
        # Obtaining the member 'day' of a type (line 479)
        day_314755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 33), self_314754, 'day')
        keyword_314756 = day_314755
        # Getting the type of 'self' (line 480)
        self_314757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 37), 'self', False)
        # Obtaining the member 'weekday' of a type (line 480)
        weekday_314758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 37), self_314757, 'weekday')
        keyword_314759 = weekday_314758
        # Getting the type of 'self' (line 481)
        self_314760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 34), 'self', False)
        # Obtaining the member 'hour' of a type (line 481)
        hour_314761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 34), self_314760, 'hour')
        keyword_314762 = hour_314761
        # Getting the type of 'self' (line 482)
        self_314763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 36), 'self', False)
        # Obtaining the member 'minute' of a type (line 482)
        minute_314764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 36), self_314763, 'minute')
        keyword_314765 = minute_314764
        # Getting the type of 'self' (line 483)
        self_314766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 36), 'self', False)
        # Obtaining the member 'second' of a type (line 483)
        second_314767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 36), self_314766, 'second')
        keyword_314768 = second_314767
        # Getting the type of 'self' (line 484)
        self_314769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 41), 'self', False)
        # Obtaining the member 'microsecond' of a type (line 484)
        microsecond_314770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 41), self_314769, 'microsecond')
        keyword_314771 = microsecond_314770
        kwargs_314772 = {'hour': keyword_314762, 'seconds': keyword_314736, 'months': keyword_314704, 'year': keyword_314750, 'days': keyword_314712, 'years': keyword_314696, 'hours': keyword_314720, 'second': keyword_314768, 'microsecond': keyword_314771, 'month': keyword_314753, 'microseconds': keyword_314744, 'leapdays': keyword_314747, 'minutes': keyword_314728, 'day': keyword_314756, 'minute': keyword_314765, 'weekday': keyword_314759}
        # Getting the type of 'self' (line 469)
        self_314687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 469)
        class___314688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 15), self_314687, '__class__')
        # Calling __class__(args, kwargs) (line 469)
        class___call_result_314773 = invoke(stypy.reporting.localization.Localization(__file__, 469, 15), class___314688, *[], **kwargs_314772)
        
        # Assigning a type to the variable 'stypy_return_type' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'stypy_return_type', class___call_result_314773)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 463)
        stypy_return_type_314774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314774)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_314774

    
    # Assigning a Name to a Name (line 486):

    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 488, 4, False)
        # Assigning a type to the variable 'self' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'relativedelta.stypy__eq__')
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Call to isinstance(...): (line 489)
        # Processing the call arguments (line 489)
        # Getting the type of 'other' (line 489)
        other_314776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 26), 'other', False)
        # Getting the type of 'relativedelta' (line 489)
        relativedelta_314777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 33), 'relativedelta', False)
        # Processing the call keyword arguments (line 489)
        kwargs_314778 = {}
        # Getting the type of 'isinstance' (line 489)
        isinstance_314775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 489)
        isinstance_call_result_314779 = invoke(stypy.reporting.localization.Localization(__file__, 489, 15), isinstance_314775, *[other_314776, relativedelta_314777], **kwargs_314778)
        
        # Applying the 'not' unary operator (line 489)
        result_not__314780 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 11), 'not', isinstance_call_result_314779)
        
        # Testing the type of an if condition (line 489)
        if_condition_314781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 489, 8), result_not__314780)
        # Assigning a type to the variable 'if_condition_314781' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'if_condition_314781', if_condition_314781)
        # SSA begins for if statement (line 489)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 490)
        NotImplemented_314782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'stypy_return_type', NotImplemented_314782)
        # SSA join for if statement (line 489)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 491)
        self_314783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 11), 'self')
        # Obtaining the member 'weekday' of a type (line 491)
        weekday_314784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 11), self_314783, 'weekday')
        # Getting the type of 'other' (line 491)
        other_314785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 27), 'other')
        # Obtaining the member 'weekday' of a type (line 491)
        weekday_314786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 27), other_314785, 'weekday')
        # Applying the binary operator 'or' (line 491)
        result_or_keyword_314787 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 11), 'or', weekday_314784, weekday_314786)
        
        # Testing the type of an if condition (line 491)
        if_condition_314788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 491, 8), result_or_keyword_314787)
        # Assigning a type to the variable 'if_condition_314788' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'if_condition_314788', if_condition_314788)
        # SSA begins for if statement (line 491)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 492)
        self_314789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 19), 'self')
        # Obtaining the member 'weekday' of a type (line 492)
        weekday_314790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 19), self_314789, 'weekday')
        # Applying the 'not' unary operator (line 492)
        result_not__314791 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 15), 'not', weekday_314790)
        
        
        # Getting the type of 'other' (line 492)
        other_314792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 39), 'other')
        # Obtaining the member 'weekday' of a type (line 492)
        weekday_314793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 39), other_314792, 'weekday')
        # Applying the 'not' unary operator (line 492)
        result_not__314794 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 35), 'not', weekday_314793)
        
        # Applying the binary operator 'or' (line 492)
        result_or_keyword_314795 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 15), 'or', result_not__314791, result_not__314794)
        
        # Testing the type of an if condition (line 492)
        if_condition_314796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 492, 12), result_or_keyword_314795)
        # Assigning a type to the variable 'if_condition_314796' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'if_condition_314796', if_condition_314796)
        # SSA begins for if statement (line 492)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 493)
        False_314797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 16), 'stypy_return_type', False_314797)
        # SSA join for if statement (line 492)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 494)
        self_314798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'self')
        # Obtaining the member 'weekday' of a type (line 494)
        weekday_314799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 15), self_314798, 'weekday')
        # Obtaining the member 'weekday' of a type (line 494)
        weekday_314800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 15), weekday_314799, 'weekday')
        # Getting the type of 'other' (line 494)
        other_314801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 39), 'other')
        # Obtaining the member 'weekday' of a type (line 494)
        weekday_314802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 39), other_314801, 'weekday')
        # Obtaining the member 'weekday' of a type (line 494)
        weekday_314803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 39), weekday_314802, 'weekday')
        # Applying the binary operator '!=' (line 494)
        result_ne_314804 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 15), '!=', weekday_314800, weekday_314803)
        
        # Testing the type of an if condition (line 494)
        if_condition_314805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 12), result_ne_314804)
        # Assigning a type to the variable 'if_condition_314805' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'if_condition_314805', if_condition_314805)
        # SSA begins for if statement (line 494)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 495)
        False_314806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'stypy_return_type', False_314806)
        # SSA join for if statement (line 494)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 496):
        
        # Assigning a Attribute to a Name (line 496):
        # Getting the type of 'self' (line 496)
        self_314807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 21), 'self')
        # Obtaining the member 'weekday' of a type (line 496)
        weekday_314808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 21), self_314807, 'weekday')
        # Obtaining the member 'n' of a type (line 496)
        n_314809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 21), weekday_314808, 'n')
        # Assigning a type to the variable 'tuple_assignment_313104' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'tuple_assignment_313104', n_314809)
        
        # Assigning a Attribute to a Name (line 496):
        # Getting the type of 'other' (line 496)
        other_314810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 37), 'other')
        # Obtaining the member 'weekday' of a type (line 496)
        weekday_314811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 37), other_314810, 'weekday')
        # Obtaining the member 'n' of a type (line 496)
        n_314812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 37), weekday_314811, 'n')
        # Assigning a type to the variable 'tuple_assignment_313105' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'tuple_assignment_313105', n_314812)
        
        # Assigning a Name to a Name (line 496):
        # Getting the type of 'tuple_assignment_313104' (line 496)
        tuple_assignment_313104_314813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'tuple_assignment_313104')
        # Assigning a type to the variable 'n1' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'n1', tuple_assignment_313104_314813)
        
        # Assigning a Name to a Name (line 496):
        # Getting the type of 'tuple_assignment_313105' (line 496)
        tuple_assignment_313105_314814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'tuple_assignment_313105')
        # Assigning a type to the variable 'n2' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 16), 'n2', tuple_assignment_313105_314814)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'n1' (line 497)
        n1_314815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 15), 'n1')
        # Getting the type of 'n2' (line 497)
        n2_314816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 21), 'n2')
        # Applying the binary operator '!=' (line 497)
        result_ne_314817 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 15), '!=', n1_314815, n2_314816)
        
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'n1' (line 497)
        n1_314818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 38), 'n1')
        # Applying the 'not' unary operator (line 497)
        result_not__314819 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 34), 'not', n1_314818)
        
        
        # Getting the type of 'n1' (line 497)
        n1_314820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 44), 'n1')
        int_314821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 50), 'int')
        # Applying the binary operator '==' (line 497)
        result_eq_314822 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 44), '==', n1_314820, int_314821)
        
        # Applying the binary operator 'or' (line 497)
        result_or_keyword_314823 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 34), 'or', result_not__314819, result_eq_314822)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'n2' (line 497)
        n2_314824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 62), 'n2')
        # Applying the 'not' unary operator (line 497)
        result_not__314825 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 58), 'not', n2_314824)
        
        
        # Getting the type of 'n2' (line 497)
        n2_314826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 68), 'n2')
        int_314827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 74), 'int')
        # Applying the binary operator '==' (line 497)
        result_eq_314828 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 68), '==', n2_314826, int_314827)
        
        # Applying the binary operator 'or' (line 497)
        result_or_keyword_314829 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 58), 'or', result_not__314825, result_eq_314828)
        
        # Applying the binary operator 'and' (line 497)
        result_and_keyword_314830 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 33), 'and', result_or_keyword_314823, result_or_keyword_314829)
        
        # Applying the 'not' unary operator (line 497)
        result_not__314831 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 28), 'not', result_and_keyword_314830)
        
        # Applying the binary operator 'and' (line 497)
        result_and_keyword_314832 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 15), 'and', result_ne_314817, result_not__314831)
        
        # Testing the type of an if condition (line 497)
        if_condition_314833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 12), result_and_keyword_314832)
        # Assigning a type to the variable 'if_condition_314833' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'if_condition_314833', if_condition_314833)
        # SSA begins for if statement (line 497)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 498)
        False_314834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 'stypy_return_type', False_314834)
        # SSA join for if statement (line 497)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 491)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 499)
        self_314835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'self')
        # Obtaining the member 'years' of a type (line 499)
        years_314836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 16), self_314835, 'years')
        # Getting the type of 'other' (line 499)
        other_314837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 30), 'other')
        # Obtaining the member 'years' of a type (line 499)
        years_314838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 30), other_314837, 'years')
        # Applying the binary operator '==' (line 499)
        result_eq_314839 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), '==', years_314836, years_314838)
        
        
        # Getting the type of 'self' (line 500)
        self_314840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 16), 'self')
        # Obtaining the member 'months' of a type (line 500)
        months_314841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 16), self_314840, 'months')
        # Getting the type of 'other' (line 500)
        other_314842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 31), 'other')
        # Obtaining the member 'months' of a type (line 500)
        months_314843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 31), other_314842, 'months')
        # Applying the binary operator '==' (line 500)
        result_eq_314844 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 16), '==', months_314841, months_314843)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314845 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_eq_314839, result_eq_314844)
        
        # Getting the type of 'self' (line 501)
        self_314846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'self')
        # Obtaining the member 'days' of a type (line 501)
        days_314847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 16), self_314846, 'days')
        # Getting the type of 'other' (line 501)
        other_314848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 29), 'other')
        # Obtaining the member 'days' of a type (line 501)
        days_314849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 29), other_314848, 'days')
        # Applying the binary operator '==' (line 501)
        result_eq_314850 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 16), '==', days_314847, days_314849)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314851 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314845, result_eq_314850)
        
        # Getting the type of 'self' (line 502)
        self_314852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'self')
        # Obtaining the member 'hours' of a type (line 502)
        hours_314853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 16), self_314852, 'hours')
        # Getting the type of 'other' (line 502)
        other_314854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 30), 'other')
        # Obtaining the member 'hours' of a type (line 502)
        hours_314855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 30), other_314854, 'hours')
        # Applying the binary operator '==' (line 502)
        result_eq_314856 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 16), '==', hours_314853, hours_314855)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314857 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314851, result_eq_314856)
        
        # Getting the type of 'self' (line 503)
        self_314858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 16), 'self')
        # Obtaining the member 'minutes' of a type (line 503)
        minutes_314859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 16), self_314858, 'minutes')
        # Getting the type of 'other' (line 503)
        other_314860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 32), 'other')
        # Obtaining the member 'minutes' of a type (line 503)
        minutes_314861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 32), other_314860, 'minutes')
        # Applying the binary operator '==' (line 503)
        result_eq_314862 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 16), '==', minutes_314859, minutes_314861)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314863 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314857, result_eq_314862)
        
        # Getting the type of 'self' (line 504)
        self_314864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 16), 'self')
        # Obtaining the member 'seconds' of a type (line 504)
        seconds_314865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 16), self_314864, 'seconds')
        # Getting the type of 'other' (line 504)
        other_314866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 32), 'other')
        # Obtaining the member 'seconds' of a type (line 504)
        seconds_314867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 32), other_314866, 'seconds')
        # Applying the binary operator '==' (line 504)
        result_eq_314868 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 16), '==', seconds_314865, seconds_314867)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314869 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314863, result_eq_314868)
        
        # Getting the type of 'self' (line 505)
        self_314870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 16), 'self')
        # Obtaining the member 'microseconds' of a type (line 505)
        microseconds_314871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 16), self_314870, 'microseconds')
        # Getting the type of 'other' (line 505)
        other_314872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 37), 'other')
        # Obtaining the member 'microseconds' of a type (line 505)
        microseconds_314873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 37), other_314872, 'microseconds')
        # Applying the binary operator '==' (line 505)
        result_eq_314874 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 16), '==', microseconds_314871, microseconds_314873)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314875 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314869, result_eq_314874)
        
        # Getting the type of 'self' (line 506)
        self_314876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 16), 'self')
        # Obtaining the member 'leapdays' of a type (line 506)
        leapdays_314877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 16), self_314876, 'leapdays')
        # Getting the type of 'other' (line 506)
        other_314878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 33), 'other')
        # Obtaining the member 'leapdays' of a type (line 506)
        leapdays_314879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 33), other_314878, 'leapdays')
        # Applying the binary operator '==' (line 506)
        result_eq_314880 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 16), '==', leapdays_314877, leapdays_314879)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314881 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314875, result_eq_314880)
        
        # Getting the type of 'self' (line 507)
        self_314882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 16), 'self')
        # Obtaining the member 'year' of a type (line 507)
        year_314883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 16), self_314882, 'year')
        # Getting the type of 'other' (line 507)
        other_314884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 29), 'other')
        # Obtaining the member 'year' of a type (line 507)
        year_314885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 29), other_314884, 'year')
        # Applying the binary operator '==' (line 507)
        result_eq_314886 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 16), '==', year_314883, year_314885)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314887 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314881, result_eq_314886)
        
        # Getting the type of 'self' (line 508)
        self_314888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 16), 'self')
        # Obtaining the member 'month' of a type (line 508)
        month_314889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 16), self_314888, 'month')
        # Getting the type of 'other' (line 508)
        other_314890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 30), 'other')
        # Obtaining the member 'month' of a type (line 508)
        month_314891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 30), other_314890, 'month')
        # Applying the binary operator '==' (line 508)
        result_eq_314892 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 16), '==', month_314889, month_314891)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314893 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314887, result_eq_314892)
        
        # Getting the type of 'self' (line 509)
        self_314894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'self')
        # Obtaining the member 'day' of a type (line 509)
        day_314895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 16), self_314894, 'day')
        # Getting the type of 'other' (line 509)
        other_314896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 28), 'other')
        # Obtaining the member 'day' of a type (line 509)
        day_314897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 28), other_314896, 'day')
        # Applying the binary operator '==' (line 509)
        result_eq_314898 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 16), '==', day_314895, day_314897)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314899 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314893, result_eq_314898)
        
        # Getting the type of 'self' (line 510)
        self_314900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 16), 'self')
        # Obtaining the member 'hour' of a type (line 510)
        hour_314901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 16), self_314900, 'hour')
        # Getting the type of 'other' (line 510)
        other_314902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 29), 'other')
        # Obtaining the member 'hour' of a type (line 510)
        hour_314903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 29), other_314902, 'hour')
        # Applying the binary operator '==' (line 510)
        result_eq_314904 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 16), '==', hour_314901, hour_314903)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314905 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314899, result_eq_314904)
        
        # Getting the type of 'self' (line 511)
        self_314906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 16), 'self')
        # Obtaining the member 'minute' of a type (line 511)
        minute_314907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 16), self_314906, 'minute')
        # Getting the type of 'other' (line 511)
        other_314908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 31), 'other')
        # Obtaining the member 'minute' of a type (line 511)
        minute_314909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 31), other_314908, 'minute')
        # Applying the binary operator '==' (line 511)
        result_eq_314910 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 16), '==', minute_314907, minute_314909)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314911 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314905, result_eq_314910)
        
        # Getting the type of 'self' (line 512)
        self_314912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 16), 'self')
        # Obtaining the member 'second' of a type (line 512)
        second_314913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 16), self_314912, 'second')
        # Getting the type of 'other' (line 512)
        other_314914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 31), 'other')
        # Obtaining the member 'second' of a type (line 512)
        second_314915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 31), other_314914, 'second')
        # Applying the binary operator '==' (line 512)
        result_eq_314916 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 16), '==', second_314913, second_314915)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314917 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314911, result_eq_314916)
        
        # Getting the type of 'self' (line 513)
        self_314918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 16), 'self')
        # Obtaining the member 'microsecond' of a type (line 513)
        microsecond_314919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 16), self_314918, 'microsecond')
        # Getting the type of 'other' (line 513)
        other_314920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 36), 'other')
        # Obtaining the member 'microsecond' of a type (line 513)
        microsecond_314921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 36), other_314920, 'microsecond')
        # Applying the binary operator '==' (line 513)
        result_eq_314922 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 16), '==', microsecond_314919, microsecond_314921)
        
        # Applying the binary operator 'and' (line 499)
        result_and_keyword_314923 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 16), 'and', result_and_keyword_314917, result_eq_314922)
        
        # Assigning a type to the variable 'stypy_return_type' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'stypy_return_type', result_and_keyword_314923)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 488)
        stypy_return_type_314924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314924)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_314924

    
    # Assigning a Name to a Name (line 515):

    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 517, 4, False)
        # Assigning a type to the variable 'self' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.__ne__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.__ne__.__dict__.__setitem__('stypy_function_name', 'relativedelta.__ne__')
        relativedelta.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        relativedelta.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__ne__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to __eq__(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'other' (line 518)
        other_314927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 31), 'other', False)
        # Processing the call keyword arguments (line 518)
        kwargs_314928 = {}
        # Getting the type of 'self' (line 518)
        self_314925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 19), 'self', False)
        # Obtaining the member '__eq__' of a type (line 518)
        eq___314926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 19), self_314925, '__eq__')
        # Calling __eq__(args, kwargs) (line 518)
        eq___call_result_314929 = invoke(stypy.reporting.localization.Localization(__file__, 518, 19), eq___314926, *[other_314927], **kwargs_314928)
        
        # Applying the 'not' unary operator (line 518)
        result_not__314930 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 15), 'not', eq___call_result_314929)
        
        # Assigning a type to the variable 'stypy_return_type' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'stypy_return_type', result_not__314930)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 517)
        stypy_return_type_314931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314931)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_314931


    @norecursion
    def __div__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__div__'
        module_type_store = module_type_store.open_function_context('__div__', 520, 4, False)
        # Assigning a type to the variable 'self' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.__div__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.__div__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.__div__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.__div__.__dict__.__setitem__('stypy_function_name', 'relativedelta.__div__')
        relativedelta.__div__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        relativedelta.__div__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.__div__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.__div__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.__div__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.__div__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.__div__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.__div__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # SSA begins for try-except statement (line 521)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a BinOp to a Name (line 522):
        
        # Assigning a BinOp to a Name (line 522):
        int_314932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 25), 'int')
        
        # Call to float(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'other' (line 522)
        other_314934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 35), 'other', False)
        # Processing the call keyword arguments (line 522)
        kwargs_314935 = {}
        # Getting the type of 'float' (line 522)
        float_314933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 29), 'float', False)
        # Calling float(args, kwargs) (line 522)
        float_call_result_314936 = invoke(stypy.reporting.localization.Localization(__file__, 522, 29), float_314933, *[other_314934], **kwargs_314935)
        
        # Applying the binary operator 'div' (line 522)
        result_div_314937 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 25), 'div', int_314932, float_call_result_314936)
        
        # Assigning a type to the variable 'reciprocal' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'reciprocal', result_div_314937)
        # SSA branch for the except part of a try statement (line 521)
        # SSA branch for the except 'TypeError' branch of a try statement (line 521)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'NotImplemented' (line 524)
        NotImplemented_314938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'stypy_return_type', NotImplemented_314938)
        # SSA join for try-except statement (line 521)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __mul__(...): (line 526)
        # Processing the call arguments (line 526)
        # Getting the type of 'reciprocal' (line 526)
        reciprocal_314941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 28), 'reciprocal', False)
        # Processing the call keyword arguments (line 526)
        kwargs_314942 = {}
        # Getting the type of 'self' (line 526)
        self_314939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 'self', False)
        # Obtaining the member '__mul__' of a type (line 526)
        mul___314940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 15), self_314939, '__mul__')
        # Calling __mul__(args, kwargs) (line 526)
        mul___call_result_314943 = invoke(stypy.reporting.localization.Localization(__file__, 526, 15), mul___314940, *[reciprocal_314941], **kwargs_314942)
        
        # Assigning a type to the variable 'stypy_return_type' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'stypy_return_type', mul___call_result_314943)
        
        # ################# End of '__div__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__div__' in the type store
        # Getting the type of 'stypy_return_type' (line 520)
        stypy_return_type_314944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314944)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__div__'
        return stypy_return_type_314944

    
    # Assigning a Name to a Name (line 528):

    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 530, 4, False)
        # Assigning a type to the variable 'self' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'relativedelta.stypy__repr__')
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        relativedelta.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'relativedelta.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 531):
        
        # Assigning a List to a Name (line 531):
        
        # Obtaining an instance of the builtin type 'list' (line 531)
        list_314945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 531)
        
        # Assigning a type to the variable 'l' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'l', list_314945)
        
        
        # Obtaining an instance of the builtin type 'list' (line 532)
        list_314946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 532)
        # Adding element type (line 532)
        str_314947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 21), 'str', 'years')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 20), list_314946, str_314947)
        # Adding element type (line 532)
        str_314948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 30), 'str', 'months')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 20), list_314946, str_314948)
        # Adding element type (line 532)
        str_314949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 40), 'str', 'days')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 20), list_314946, str_314949)
        # Adding element type (line 532)
        str_314950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 48), 'str', 'leapdays')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 20), list_314946, str_314950)
        # Adding element type (line 532)
        str_314951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 21), 'str', 'hours')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 20), list_314946, str_314951)
        # Adding element type (line 532)
        str_314952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 30), 'str', 'minutes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 20), list_314946, str_314952)
        # Adding element type (line 532)
        str_314953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 41), 'str', 'seconds')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 20), list_314946, str_314953)
        # Adding element type (line 532)
        str_314954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 52), 'str', 'microseconds')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 20), list_314946, str_314954)
        
        # Testing the type of a for loop iterable (line 532)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 532, 8), list_314946)
        # Getting the type of the for loop variable (line 532)
        for_loop_var_314955 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 532, 8), list_314946)
        # Assigning a type to the variable 'attr' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'attr', for_loop_var_314955)
        # SSA begins for a for statement (line 532)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 534):
        
        # Assigning a Call to a Name (line 534):
        
        # Call to getattr(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'self' (line 534)
        self_314957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 28), 'self', False)
        # Getting the type of 'attr' (line 534)
        attr_314958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 34), 'attr', False)
        # Processing the call keyword arguments (line 534)
        kwargs_314959 = {}
        # Getting the type of 'getattr' (line 534)
        getattr_314956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 534)
        getattr_call_result_314960 = invoke(stypy.reporting.localization.Localization(__file__, 534, 20), getattr_314956, *[self_314957, attr_314958], **kwargs_314959)
        
        # Assigning a type to the variable 'value' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'value', getattr_call_result_314960)
        
        # Getting the type of 'value' (line 535)
        value_314961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'value')
        # Testing the type of an if condition (line 535)
        if_condition_314962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 12), value_314961)
        # Assigning a type to the variable 'if_condition_314962' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'if_condition_314962', if_condition_314962)
        # SSA begins for if statement (line 535)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 536)
        # Processing the call arguments (line 536)
        
        # Call to format(...): (line 536)
        # Processing the call keyword arguments (line 536)
        # Getting the type of 'attr' (line 536)
        attr_314967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 57), 'attr', False)
        keyword_314968 = attr_314967
        # Getting the type of 'value' (line 536)
        value_314969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 69), 'value', False)
        keyword_314970 = value_314969
        kwargs_314971 = {'attr': keyword_314968, 'value': keyword_314970}
        str_314965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 25), 'str', '{attr}={value:+g}')
        # Obtaining the member 'format' of a type (line 536)
        format_314966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 25), str_314965, 'format')
        # Calling format(args, kwargs) (line 536)
        format_call_result_314972 = invoke(stypy.reporting.localization.Localization(__file__, 536, 25), format_314966, *[], **kwargs_314971)
        
        # Processing the call keyword arguments (line 536)
        kwargs_314973 = {}
        # Getting the type of 'l' (line 536)
        l_314963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'l', False)
        # Obtaining the member 'append' of a type (line 536)
        append_314964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 16), l_314963, 'append')
        # Calling append(args, kwargs) (line 536)
        append_call_result_314974 = invoke(stypy.reporting.localization.Localization(__file__, 536, 16), append_314964, *[format_call_result_314972], **kwargs_314973)
        
        # SSA join for if statement (line 535)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 537)
        list_314975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 537)
        # Adding element type (line 537)
        str_314976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 21), 'str', 'year')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 20), list_314975, str_314976)
        # Adding element type (line 537)
        str_314977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 29), 'str', 'month')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 20), list_314975, str_314977)
        # Adding element type (line 537)
        str_314978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 38), 'str', 'day')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 20), list_314975, str_314978)
        # Adding element type (line 537)
        str_314979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 45), 'str', 'weekday')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 20), list_314975, str_314979)
        # Adding element type (line 537)
        str_314980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 21), 'str', 'hour')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 20), list_314975, str_314980)
        # Adding element type (line 537)
        str_314981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 29), 'str', 'minute')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 20), list_314975, str_314981)
        # Adding element type (line 537)
        str_314982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 39), 'str', 'second')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 20), list_314975, str_314982)
        # Adding element type (line 537)
        str_314983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 49), 'str', 'microsecond')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 20), list_314975, str_314983)
        
        # Testing the type of a for loop iterable (line 537)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 537, 8), list_314975)
        # Getting the type of the for loop variable (line 537)
        for_loop_var_314984 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 537, 8), list_314975)
        # Assigning a type to the variable 'attr' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'attr', for_loop_var_314984)
        # SSA begins for a for statement (line 537)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 539):
        
        # Assigning a Call to a Name (line 539):
        
        # Call to getattr(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'self' (line 539)
        self_314986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 28), 'self', False)
        # Getting the type of 'attr' (line 539)
        attr_314987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 34), 'attr', False)
        # Processing the call keyword arguments (line 539)
        kwargs_314988 = {}
        # Getting the type of 'getattr' (line 539)
        getattr_314985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 539)
        getattr_call_result_314989 = invoke(stypy.reporting.localization.Localization(__file__, 539, 20), getattr_314985, *[self_314986, attr_314987], **kwargs_314988)
        
        # Assigning a type to the variable 'value' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'value', getattr_call_result_314989)
        
        # Type idiom detected: calculating its left and rigth part (line 540)
        # Getting the type of 'value' (line 540)
        value_314990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'value')
        # Getting the type of 'None' (line 540)
        None_314991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 28), 'None')
        
        (may_be_314992, more_types_in_union_314993) = may_not_be_none(value_314990, None_314991)

        if may_be_314992:

            if more_types_in_union_314993:
                # Runtime conditional SSA (line 540)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 541)
            # Processing the call arguments (line 541)
            
            # Call to format(...): (line 541)
            # Processing the call keyword arguments (line 541)
            # Getting the type of 'attr' (line 541)
            attr_314998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 54), 'attr', False)
            keyword_314999 = attr_314998
            
            # Call to repr(...): (line 541)
            # Processing the call arguments (line 541)
            # Getting the type of 'value' (line 541)
            value_315001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 71), 'value', False)
            # Processing the call keyword arguments (line 541)
            kwargs_315002 = {}
            # Getting the type of 'repr' (line 541)
            repr_315000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 66), 'repr', False)
            # Calling repr(args, kwargs) (line 541)
            repr_call_result_315003 = invoke(stypy.reporting.localization.Localization(__file__, 541, 66), repr_315000, *[value_315001], **kwargs_315002)
            
            keyword_315004 = repr_call_result_315003
            kwargs_315005 = {'attr': keyword_314999, 'value': keyword_315004}
            str_314996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 25), 'str', '{attr}={value}')
            # Obtaining the member 'format' of a type (line 541)
            format_314997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 25), str_314996, 'format')
            # Calling format(args, kwargs) (line 541)
            format_call_result_315006 = invoke(stypy.reporting.localization.Localization(__file__, 541, 25), format_314997, *[], **kwargs_315005)
            
            # Processing the call keyword arguments (line 541)
            kwargs_315007 = {}
            # Getting the type of 'l' (line 541)
            l_314994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'l', False)
            # Obtaining the member 'append' of a type (line 541)
            append_314995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 16), l_314994, 'append')
            # Calling append(args, kwargs) (line 541)
            append_call_result_315008 = invoke(stypy.reporting.localization.Localization(__file__, 541, 16), append_314995, *[format_call_result_315006], **kwargs_315007)
            

            if more_types_in_union_314993:
                # SSA join for if statement (line 540)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to format(...): (line 542)
        # Processing the call keyword arguments (line 542)
        # Getting the type of 'self' (line 542)
        self_315011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 55), 'self', False)
        # Obtaining the member '__class__' of a type (line 542)
        class___315012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 55), self_315011, '__class__')
        # Obtaining the member '__name__' of a type (line 542)
        name___315013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 55), class___315012, '__name__')
        keyword_315014 = name___315013
        
        # Call to join(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'l' (line 543)
        l_315017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 61), 'l', False)
        # Processing the call keyword arguments (line 543)
        kwargs_315018 = {}
        str_315015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 51), 'str', ', ')
        # Obtaining the member 'join' of a type (line 543)
        join_315016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 51), str_315015, 'join')
        # Calling join(args, kwargs) (line 543)
        join_call_result_315019 = invoke(stypy.reporting.localization.Localization(__file__, 543, 51), join_315016, *[l_315017], **kwargs_315018)
        
        keyword_315020 = join_call_result_315019
        kwargs_315021 = {'classname': keyword_315014, 'attrs': keyword_315020}
        str_315009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 15), 'str', '{classname}({attrs})')
        # Obtaining the member 'format' of a type (line 542)
        format_315010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 15), str_315009, 'format')
        # Calling format(args, kwargs) (line 542)
        format_call_result_315022 = invoke(stypy.reporting.localization.Localization(__file__, 542, 15), format_315010, *[], **kwargs_315021)
        
        # Assigning a type to the variable 'stypy_return_type' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'stypy_return_type', format_call_result_315022)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 530)
        stypy_return_type_315023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315023)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_315023


# Assigning a type to the variable 'relativedelta' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'relativedelta', relativedelta)

# Assigning a Name to a Name (line 461):
# Getting the type of 'relativedelta'
relativedelta_315024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'relativedelta')
# Obtaining the member '__bool__' of a type
bool___315025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), relativedelta_315024, '__bool__')
# Getting the type of 'relativedelta'
relativedelta_315026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'relativedelta')
# Setting the type of the member '__nonzero__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), relativedelta_315026, '__nonzero__', bool___315025)

# Assigning a Name to a Name (line 486):
# Getting the type of 'relativedelta'
relativedelta_315027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'relativedelta')
# Obtaining the member '__mul__' of a type
mul___315028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), relativedelta_315027, '__mul__')
# Getting the type of 'relativedelta'
relativedelta_315029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'relativedelta')
# Setting the type of the member '__rmul__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), relativedelta_315029, '__rmul__', mul___315028)

# Assigning a Name to a Name (line 515):
# Getting the type of 'None' (line 515)
None_315030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 15), 'None')
# Getting the type of 'relativedelta'
relativedelta_315031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'relativedelta')
# Setting the type of the member '__hash__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), relativedelta_315031, '__hash__', None_315030)

# Assigning a Name to a Name (line 528):
# Getting the type of 'relativedelta'
relativedelta_315032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'relativedelta')
# Obtaining the member '__div__' of a type
div___315033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), relativedelta_315032, '__div__')
# Getting the type of 'relativedelta'
relativedelta_315034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'relativedelta')
# Setting the type of the member '__truediv__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), relativedelta_315034, '__truediv__', div___315033)

@norecursion
def _sign(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_sign'
    module_type_store = module_type_store.open_function_context('_sign', 546, 0, False)
    
    # Passed parameters checking function
    _sign.stypy_localization = localization
    _sign.stypy_type_of_self = None
    _sign.stypy_type_store = module_type_store
    _sign.stypy_function_name = '_sign'
    _sign.stypy_param_names_list = ['x']
    _sign.stypy_varargs_param_name = None
    _sign.stypy_kwargs_param_name = None
    _sign.stypy_call_defaults = defaults
    _sign.stypy_call_varargs = varargs
    _sign.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sign', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sign', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sign(...)' code ##################

    
    # Call to int(...): (line 547)
    # Processing the call arguments (line 547)
    
    # Call to copysign(...): (line 547)
    # Processing the call arguments (line 547)
    int_315037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 24), 'int')
    # Getting the type of 'x' (line 547)
    x_315038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 27), 'x', False)
    # Processing the call keyword arguments (line 547)
    kwargs_315039 = {}
    # Getting the type of 'copysign' (line 547)
    copysign_315036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 15), 'copysign', False)
    # Calling copysign(args, kwargs) (line 547)
    copysign_call_result_315040 = invoke(stypy.reporting.localization.Localization(__file__, 547, 15), copysign_315036, *[int_315037, x_315038], **kwargs_315039)
    
    # Processing the call keyword arguments (line 547)
    kwargs_315041 = {}
    # Getting the type of 'int' (line 547)
    int_315035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 11), 'int', False)
    # Calling int(args, kwargs) (line 547)
    int_call_result_315042 = invoke(stypy.reporting.localization.Localization(__file__, 547, 11), int_315035, *[copysign_call_result_315040], **kwargs_315041)
    
    # Assigning a type to the variable 'stypy_return_type' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'stypy_return_type', int_call_result_315042)
    
    # ################# End of '_sign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sign' in the type store
    # Getting the type of 'stypy_return_type' (line 546)
    stypy_return_type_315043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_315043)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sign'
    return stypy_return_type_315043

# Assigning a type to the variable '_sign' (line 546)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 0), '_sign', _sign)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
