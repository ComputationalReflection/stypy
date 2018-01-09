
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Test case implementation'''
2: 
3: import collections
4: import sys
5: import functools
6: import difflib
7: import pprint
8: import re
9: import types
10: import warnings
11: 
12: from . import result
13: from .util import (
14:     strclass, safe_repr, unorderable_list_difference,
15:     _count_diff_all_purpose, _count_diff_hashable
16: )
17: 
18: 
19: __unittest = True
20: 
21: 
22: DIFF_OMITTED = ('\nDiff is %s characters long. '
23:                  'Set self.maxDiff to None to see it.')
24: 
25: class SkipTest(Exception):
26:     '''
27:     Raise this exception in a test to skip it.
28: 
29:     Usually you can use TestCase.skipTest() or one of the skipping decorators
30:     instead of raising this directly.
31:     '''
32:     pass
33: 
34: class _ExpectedFailure(Exception):
35:     '''
36:     Raise this when a test is expected to fail.
37: 
38:     This is an implementation detail.
39:     '''
40: 
41:     def __init__(self, exc_info):
42:         super(_ExpectedFailure, self).__init__()
43:         self.exc_info = exc_info
44: 
45: class _UnexpectedSuccess(Exception):
46:     '''
47:     The test was supposed to fail, but it didn't!
48:     '''
49:     pass
50: 
51: def _id(obj):
52:     return obj
53: 
54: def skip(reason):
55:     '''
56:     Unconditionally skip a test.
57:     '''
58:     def decorator(test_item):
59:         if not isinstance(test_item, (type, types.ClassType)):
60:             @functools.wraps(test_item)
61:             def skip_wrapper(*args, **kwargs):
62:                 raise SkipTest(reason)
63:             test_item = skip_wrapper
64: 
65:         test_item.__unittest_skip__ = True
66:         test_item.__unittest_skip_why__ = reason
67:         return test_item
68:     return decorator
69: 
70: def skipIf(condition, reason):
71:     '''
72:     Skip a test if the condition is true.
73:     '''
74:     if condition:
75:         return skip(reason)
76:     return _id
77: 
78: def skipUnless(condition, reason):
79:     '''
80:     Skip a test unless the condition is true.
81:     '''
82:     if not condition:
83:         return skip(reason)
84:     return _id
85: 
86: 
87: def expectedFailure(func):
88:     @functools.wraps(func)
89:     def wrapper(*args, **kwargs):
90:         try:
91:             func(*args, **kwargs)
92:         except Exception:
93:             raise _ExpectedFailure(sys.exc_info())
94:         raise _UnexpectedSuccess
95:     return wrapper
96: 
97: 
98: class _AssertRaisesContext(object):
99:     '''A context manager used to implement TestCase.assertRaises* methods.'''
100: 
101:     def __init__(self, expected, test_case, expected_regexp=None):
102:         self.expected = expected
103:         self.failureException = test_case.failureException
104:         self.expected_regexp = expected_regexp
105: 
106:     def __enter__(self):
107:         return self
108: 
109:     def __exit__(self, exc_type, exc_value, tb):
110:         if exc_type is None:
111:             try:
112:                 exc_name = self.expected.__name__
113:             except AttributeError:
114:                 exc_name = str(self.expected)
115:             raise self.failureException(
116:                 "{0} not raised".format(exc_name))
117:         if not issubclass(exc_type, self.expected):
118:             # let unexpected exceptions pass through
119:             return False
120:         self.exception = exc_value # store for later retrieval
121:         if self.expected_regexp is None:
122:             return True
123: 
124:         expected_regexp = self.expected_regexp
125:         if not expected_regexp.search(str(exc_value)):
126:             raise self.failureException('"%s" does not match "%s"' %
127:                      (expected_regexp.pattern, str(exc_value)))
128:         return True
129: 
130: 
131: class TestCase(object):
132:     '''A class whose instances are single test cases.
133: 
134:     By default, the test code itself should be placed in a method named
135:     'runTest'.
136: 
137:     If the fixture may be used for many test cases, create as
138:     many test methods as are needed. When instantiating such a TestCase
139:     subclass, specify in the constructor arguments the name of the test method
140:     that the instance is to execute.
141: 
142:     Test authors should subclass TestCase for their own tests. Construction
143:     and deconstruction of the test's environment ('fixture') can be
144:     implemented by overriding the 'setUp' and 'tearDown' methods respectively.
145: 
146:     If it is necessary to override the __init__ method, the base class
147:     __init__ method must always be called. It is important that subclasses
148:     should not change the signature of their __init__ method, since instances
149:     of the classes are instantiated automatically by parts of the framework
150:     in order to be run.
151: 
152:     When subclassing TestCase, you can set these attributes:
153:     * failureException: determines which exception will be raised when
154:         the instance's assertion methods fail; test methods raising this
155:         exception will be deemed to have 'failed' rather than 'errored'.
156:     * longMessage: determines whether long messages (including repr of
157:         objects used in assert methods) will be printed on failure in *addition*
158:         to any explicit message passed.
159:     * maxDiff: sets the maximum length of a diff in failure messages
160:         by assert methods using difflib. It is looked up as an instance
161:         attribute so can be configured by individual tests if required.
162:     '''
163: 
164:     failureException = AssertionError
165: 
166:     longMessage = False
167: 
168:     maxDiff = 80*8
169: 
170:     # If a string is longer than _diffThreshold, use normal comparison instead
171:     # of difflib.  See #11763.
172:     _diffThreshold = 2**16
173: 
174:     # Attribute used by TestSuite for classSetUp
175: 
176:     _classSetupFailed = False
177: 
178:     def __init__(self, methodName='runTest'):
179:         '''Create an instance of the class that will use the named test
180:            method when executed. Raises a ValueError if the instance does
181:            not have a method with the specified name.
182:         '''
183:         self._testMethodName = methodName
184:         self._resultForDoCleanups = None
185:         try:
186:             testMethod = getattr(self, methodName)
187:         except AttributeError:
188:             raise ValueError("no such test method in %s: %s" %
189:                   (self.__class__, methodName))
190:         self._testMethodDoc = testMethod.__doc__
191:         self._cleanups = []
192: 
193:         # Map types to custom assertEqual functions that will compare
194:         # instances of said type in more detail to generate a more useful
195:         # error message.
196:         self._type_equality_funcs = {}
197:         self.addTypeEqualityFunc(dict, 'assertDictEqual')
198:         self.addTypeEqualityFunc(list, 'assertListEqual')
199:         self.addTypeEqualityFunc(tuple, 'assertTupleEqual')
200:         self.addTypeEqualityFunc(set, 'assertSetEqual')
201:         self.addTypeEqualityFunc(frozenset, 'assertSetEqual')
202:         try:
203:             self.addTypeEqualityFunc(unicode, 'assertMultiLineEqual')
204:         except NameError:
205:             # No unicode support in this build
206:             pass
207: 
208:     def addTypeEqualityFunc(self, typeobj, function):
209:         '''Add a type specific assertEqual style function to compare a type.
210: 
211:         This method is for use by TestCase subclasses that need to register
212:         their own type equality functions to provide nicer error messages.
213: 
214:         Args:
215:             typeobj: The data type to call this function on when both values
216:                     are of the same type in assertEqual().
217:             function: The callable taking two arguments and an optional
218:                     msg= argument that raises self.failureException with a
219:                     useful error message when the two arguments are not equal.
220:         '''
221:         self._type_equality_funcs[typeobj] = function
222: 
223:     def addCleanup(self, function, *args, **kwargs):
224:         '''Add a function, with arguments, to be called when the test is
225:         completed. Functions added are called on a LIFO basis and are
226:         called after tearDown on test failure or success.
227: 
228:         Cleanup items are called even if setUp fails (unlike tearDown).'''
229:         self._cleanups.append((function, args, kwargs))
230: 
231:     def setUp(self):
232:         "Hook method for setting up the test fixture before exercising it."
233:         pass
234: 
235:     def tearDown(self):
236:         "Hook method for deconstructing the test fixture after testing it."
237:         pass
238: 
239:     @classmethod
240:     def setUpClass(cls):
241:         "Hook method for setting up class fixture before running tests in the class."
242: 
243:     @classmethod
244:     def tearDownClass(cls):
245:         "Hook method for deconstructing the class fixture after running all tests in the class."
246: 
247:     def countTestCases(self):
248:         return 1
249: 
250:     def defaultTestResult(self):
251:         return result.TestResult()
252: 
253:     def shortDescription(self):
254:         '''Returns a one-line description of the test, or None if no
255:         description has been provided.
256: 
257:         The default implementation of this method returns the first line of
258:         the specified test method's docstring.
259:         '''
260:         doc = self._testMethodDoc
261:         return doc and doc.split("\n")[0].strip() or None
262: 
263: 
264:     def id(self):
265:         return "%s.%s" % (strclass(self.__class__), self._testMethodName)
266: 
267:     def __eq__(self, other):
268:         if type(self) is not type(other):
269:             return NotImplemented
270: 
271:         return self._testMethodName == other._testMethodName
272: 
273:     def __ne__(self, other):
274:         return not self == other
275: 
276:     def __hash__(self):
277:         return hash((type(self), self._testMethodName))
278: 
279:     def __str__(self):
280:         return "%s (%s)" % (self._testMethodName, strclass(self.__class__))
281: 
282:     def __repr__(self):
283:         return "<%s testMethod=%s>" % \
284:                (strclass(self.__class__), self._testMethodName)
285: 
286:     def _addSkip(self, result, reason):
287:         addSkip = getattr(result, 'addSkip', None)
288:         if addSkip is not None:
289:             addSkip(self, reason)
290:         else:
291:             warnings.warn("TestResult has no addSkip method, skips not reported",
292:                           RuntimeWarning, 2)
293:             result.addSuccess(self)
294: 
295:     def run(self, result=None):
296:         orig_result = result
297:         if result is None:
298:             result = self.defaultTestResult()
299:             startTestRun = getattr(result, 'startTestRun', None)
300:             if startTestRun is not None:
301:                 startTestRun()
302: 
303:         self._resultForDoCleanups = result
304:         result.startTest(self)
305: 
306:         testMethod = getattr(self, self._testMethodName)
307:         if (getattr(self.__class__, "__unittest_skip__", False) or
308:             getattr(testMethod, "__unittest_skip__", False)):
309:             # If the class or method was skipped.
310:             try:
311:                 skip_why = (getattr(self.__class__, '__unittest_skip_why__', '')
312:                             or getattr(testMethod, '__unittest_skip_why__', ''))
313:                 self._addSkip(result, skip_why)
314:             finally:
315:                 result.stopTest(self)
316:             return
317:         try:
318:             success = False
319:             try:
320:                 self.setUp()
321:             except SkipTest as e:
322:                 self._addSkip(result, str(e))
323:             except KeyboardInterrupt:
324:                 raise
325:             except:
326:                 result.addError(self, sys.exc_info())
327:             else:
328:                 try:
329:                     testMethod()
330:                 except KeyboardInterrupt:
331:                     raise
332:                 except self.failureException:
333:                     result.addFailure(self, sys.exc_info())
334:                 except _ExpectedFailure as e:
335:                     addExpectedFailure = getattr(result, 'addExpectedFailure', None)
336:                     if addExpectedFailure is not None:
337:                         addExpectedFailure(self, e.exc_info)
338:                     else:
339:                         warnings.warn("TestResult has no addExpectedFailure method, reporting as passes",
340:                                       RuntimeWarning)
341:                         result.addSuccess(self)
342:                 except _UnexpectedSuccess:
343:                     addUnexpectedSuccess = getattr(result, 'addUnexpectedSuccess', None)
344:                     if addUnexpectedSuccess is not None:
345:                         addUnexpectedSuccess(self)
346:                     else:
347:                         warnings.warn("TestResult has no addUnexpectedSuccess method, reporting as failures",
348:                                       RuntimeWarning)
349:                         result.addFailure(self, sys.exc_info())
350:                 except SkipTest as e:
351:                     self._addSkip(result, str(e))
352:                 except:
353:                     result.addError(self, sys.exc_info())
354:                 else:
355:                     success = True
356: 
357:                 try:
358:                     self.tearDown()
359:                 except KeyboardInterrupt:
360:                     raise
361:                 except:
362:                     result.addError(self, sys.exc_info())
363:                     success = False
364: 
365:             cleanUpSuccess = self.doCleanups()
366:             success = success and cleanUpSuccess
367:             if success:
368:                 result.addSuccess(self)
369:         finally:
370:             result.stopTest(self)
371:             if orig_result is None:
372:                 stopTestRun = getattr(result, 'stopTestRun', None)
373:                 if stopTestRun is not None:
374:                     stopTestRun()
375: 
376:     def doCleanups(self):
377:         '''Execute all cleanup functions. Normally called for you after
378:         tearDown.'''
379:         result = self._resultForDoCleanups
380:         ok = True
381:         while self._cleanups:
382:             function, args, kwargs = self._cleanups.pop(-1)
383:             try:
384:                 function(*args, **kwargs)
385:             except KeyboardInterrupt:
386:                 raise
387:             except:
388:                 ok = False
389:                 result.addError(self, sys.exc_info())
390:         return ok
391: 
392:     def __call__(self, *args, **kwds):
393:         return self.run(*args, **kwds)
394: 
395:     def debug(self):
396:         '''Run the test without collecting errors in a TestResult'''
397:         self.setUp()
398:         getattr(self, self._testMethodName)()
399:         self.tearDown()
400:         while self._cleanups:
401:             function, args, kwargs = self._cleanups.pop(-1)
402:             function(*args, **kwargs)
403: 
404:     def skipTest(self, reason):
405:         '''Skip this test.'''
406:         raise SkipTest(reason)
407: 
408:     def fail(self, msg=None):
409:         '''Fail immediately, with the given message.'''
410:         raise self.failureException(msg)
411: 
412:     def assertFalse(self, expr, msg=None):
413:         '''Check that the expression is false.'''
414:         if expr:
415:             msg = self._formatMessage(msg, "%s is not false" % safe_repr(expr))
416:             raise self.failureException(msg)
417: 
418:     def assertTrue(self, expr, msg=None):
419:         '''Check that the expression is true.'''
420:         if not expr:
421:             msg = self._formatMessage(msg, "%s is not true" % safe_repr(expr))
422:             raise self.failureException(msg)
423: 
424:     def _formatMessage(self, msg, standardMsg):
425:         '''Honour the longMessage attribute when generating failure messages.
426:         If longMessage is False this means:
427:         * Use only an explicit message if it is provided
428:         * Otherwise use the standard message for the assert
429: 
430:         If longMessage is True:
431:         * Use the standard message
432:         * If an explicit message is provided, plus ' : ' and the explicit message
433:         '''
434:         if not self.longMessage:
435:             return msg or standardMsg
436:         if msg is None:
437:             return standardMsg
438:         try:
439:             # don't switch to '{}' formatting in Python 2.X
440:             # it changes the way unicode input is handled
441:             return '%s : %s' % (standardMsg, msg)
442:         except UnicodeDecodeError:
443:             return  '%s : %s' % (safe_repr(standardMsg), safe_repr(msg))
444: 
445: 
446:     def assertRaises(self, excClass, callableObj=None, *args, **kwargs):
447:         '''Fail unless an exception of class excClass is raised
448:            by callableObj when invoked with arguments args and keyword
449:            arguments kwargs. If a different type of exception is
450:            raised, it will not be caught, and the test case will be
451:            deemed to have suffered an error, exactly as for an
452:            unexpected exception.
453: 
454:            If called with callableObj omitted or None, will return a
455:            context object used like this::
456: 
457:                 with self.assertRaises(SomeException):
458:                     do_something()
459: 
460:            The context manager keeps a reference to the exception as
461:            the 'exception' attribute. This allows you to inspect the
462:            exception after the assertion::
463: 
464:                with self.assertRaises(SomeException) as cm:
465:                    do_something()
466:                the_exception = cm.exception
467:                self.assertEqual(the_exception.error_code, 3)
468:         '''
469:         context = _AssertRaisesContext(excClass, self)
470:         if callableObj is None:
471:             return context
472:         with context:
473:             callableObj(*args, **kwargs)
474: 
475:     def _getAssertEqualityFunc(self, first, second):
476:         '''Get a detailed comparison function for the types of the two args.
477: 
478:         Returns: A callable accepting (first, second, msg=None) that will
479:         raise a failure exception if first != second with a useful human
480:         readable error message for those types.
481:         '''
482:         #
483:         # NOTE(gregory.p.smith): I considered isinstance(first, type(second))
484:         # and vice versa.  I opted for the conservative approach in case
485:         # subclasses are not intended to be compared in detail to their super
486:         # class instances using a type equality func.  This means testing
487:         # subtypes won't automagically use the detailed comparison.  Callers
488:         # should use their type specific assertSpamEqual method to compare
489:         # subclasses if the detailed comparison is desired and appropriate.
490:         # See the discussion in http://bugs.python.org/issue2578.
491:         #
492:         if type(first) is type(second):
493:             asserter = self._type_equality_funcs.get(type(first))
494:             if asserter is not None:
495:                 if isinstance(asserter, basestring):
496:                     asserter = getattr(self, asserter)
497:                 return asserter
498: 
499:         return self._baseAssertEqual
500: 
501:     def _baseAssertEqual(self, first, second, msg=None):
502:         '''The default assertEqual implementation, not type specific.'''
503:         if not first == second:
504:             standardMsg = '%s != %s' % (safe_repr(first), safe_repr(second))
505:             msg = self._formatMessage(msg, standardMsg)
506:             raise self.failureException(msg)
507: 
508:     def assertEqual(self, first, second, msg=None):
509:         '''Fail if the two objects are unequal as determined by the '=='
510:            operator.
511:         '''
512:         assertion_func = self._getAssertEqualityFunc(first, second)
513:         assertion_func(first, second, msg=msg)
514: 
515:     def assertNotEqual(self, first, second, msg=None):
516:         '''Fail if the two objects are equal as determined by the '!='
517:            operator.
518:         '''
519:         if not first != second:
520:             msg = self._formatMessage(msg, '%s == %s' % (safe_repr(first),
521:                                                           safe_repr(second)))
522:             raise self.failureException(msg)
523: 
524: 
525:     def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
526:         '''Fail if the two objects are unequal as determined by their
527:            difference rounded to the given number of decimal places
528:            (default 7) and comparing to zero, or by comparing that the
529:            between the two objects is more than the given delta.
530: 
531:            Note that decimal places (from zero) are usually not the same
532:            as significant digits (measured from the most signficant digit).
533: 
534:            If the two objects compare equal then they will automatically
535:            compare almost equal.
536:         '''
537:         if first == second:
538:             # shortcut
539:             return
540:         if delta is not None and places is not None:
541:             raise TypeError("specify delta or places not both")
542: 
543:         if delta is not None:
544:             if abs(first - second) <= delta:
545:                 return
546: 
547:             standardMsg = '%s != %s within %s delta' % (safe_repr(first),
548:                                                         safe_repr(second),
549:                                                         safe_repr(delta))
550:         else:
551:             if places is None:
552:                 places = 7
553: 
554:             if round(abs(second-first), places) == 0:
555:                 return
556: 
557:             standardMsg = '%s != %s within %r places' % (safe_repr(first),
558:                                                           safe_repr(second),
559:                                                           places)
560:         msg = self._formatMessage(msg, standardMsg)
561:         raise self.failureException(msg)
562: 
563:     def assertNotAlmostEqual(self, first, second, places=None, msg=None, delta=None):
564:         '''Fail if the two objects are equal as determined by their
565:            difference rounded to the given number of decimal places
566:            (default 7) and comparing to zero, or by comparing that the
567:            between the two objects is less than the given delta.
568: 
569:            Note that decimal places (from zero) are usually not the same
570:            as significant digits (measured from the most signficant digit).
571: 
572:            Objects that are equal automatically fail.
573:         '''
574:         if delta is not None and places is not None:
575:             raise TypeError("specify delta or places not both")
576:         if delta is not None:
577:             if not (first == second) and abs(first - second) > delta:
578:                 return
579:             standardMsg = '%s == %s within %s delta' % (safe_repr(first),
580:                                                         safe_repr(second),
581:                                                         safe_repr(delta))
582:         else:
583:             if places is None:
584:                 places = 7
585:             if not (first == second) and round(abs(second-first), places) != 0:
586:                 return
587:             standardMsg = '%s == %s within %r places' % (safe_repr(first),
588:                                                          safe_repr(second),
589:                                                          places)
590: 
591:         msg = self._formatMessage(msg, standardMsg)
592:         raise self.failureException(msg)
593: 
594:     # Synonyms for assertion methods
595: 
596:     # The plurals are undocumented.  Keep them that way to discourage use.
597:     # Do not add more.  Do not remove.
598:     # Going through a deprecation cycle on these would annoy many people.
599:     assertEquals = assertEqual
600:     assertNotEquals = assertNotEqual
601:     assertAlmostEquals = assertAlmostEqual
602:     assertNotAlmostEquals = assertNotAlmostEqual
603:     assert_ = assertTrue
604: 
605:     # These fail* assertion method names are pending deprecation and will
606:     # be a DeprecationWarning in 3.2; http://bugs.python.org/issue2578
607:     def _deprecate(original_func):
608:         def deprecated_func(*args, **kwargs):
609:             warnings.warn(
610:                 'Please use {0} instead.'.format(original_func.__name__),
611:                 PendingDeprecationWarning, 2)
612:             return original_func(*args, **kwargs)
613:         return deprecated_func
614: 
615:     failUnlessEqual = _deprecate(assertEqual)
616:     failIfEqual = _deprecate(assertNotEqual)
617:     failUnlessAlmostEqual = _deprecate(assertAlmostEqual)
618:     failIfAlmostEqual = _deprecate(assertNotAlmostEqual)
619:     failUnless = _deprecate(assertTrue)
620:     failUnlessRaises = _deprecate(assertRaises)
621:     failIf = _deprecate(assertFalse)
622: 
623:     def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
624:         '''An equality assertion for ordered sequences (like lists and tuples).
625: 
626:         For the purposes of this function, a valid ordered sequence type is one
627:         which can be indexed, has a length, and has an equality operator.
628: 
629:         Args:
630:             seq1: The first sequence to compare.
631:             seq2: The second sequence to compare.
632:             seq_type: The expected datatype of the sequences, or None if no
633:                     datatype should be enforced.
634:             msg: Optional message to use on failure instead of a list of
635:                     differences.
636:         '''
637:         if seq_type is not None:
638:             seq_type_name = seq_type.__name__
639:             if not isinstance(seq1, seq_type):
640:                 raise self.failureException('First sequence is not a %s: %s'
641:                                         % (seq_type_name, safe_repr(seq1)))
642:             if not isinstance(seq2, seq_type):
643:                 raise self.failureException('Second sequence is not a %s: %s'
644:                                         % (seq_type_name, safe_repr(seq2)))
645:         else:
646:             seq_type_name = "sequence"
647: 
648:         differing = None
649:         try:
650:             len1 = len(seq1)
651:         except (TypeError, NotImplementedError):
652:             differing = 'First %s has no length.    Non-sequence?' % (
653:                     seq_type_name)
654: 
655:         if differing is None:
656:             try:
657:                 len2 = len(seq2)
658:             except (TypeError, NotImplementedError):
659:                 differing = 'Second %s has no length.    Non-sequence?' % (
660:                         seq_type_name)
661: 
662:         if differing is None:
663:             if seq1 == seq2:
664:                 return
665: 
666:             seq1_repr = safe_repr(seq1)
667:             seq2_repr = safe_repr(seq2)
668:             if len(seq1_repr) > 30:
669:                 seq1_repr = seq1_repr[:30] + '...'
670:             if len(seq2_repr) > 30:
671:                 seq2_repr = seq2_repr[:30] + '...'
672:             elements = (seq_type_name.capitalize(), seq1_repr, seq2_repr)
673:             differing = '%ss differ: %s != %s\n' % elements
674: 
675:             for i in xrange(min(len1, len2)):
676:                 try:
677:                     item1 = seq1[i]
678:                 except (TypeError, IndexError, NotImplementedError):
679:                     differing += ('\nUnable to index element %d of first %s\n' %
680:                                  (i, seq_type_name))
681:                     break
682: 
683:                 try:
684:                     item2 = seq2[i]
685:                 except (TypeError, IndexError, NotImplementedError):
686:                     differing += ('\nUnable to index element %d of second %s\n' %
687:                                  (i, seq_type_name))
688:                     break
689: 
690:                 if item1 != item2:
691:                     differing += ('\nFirst differing element %d:\n%s\n%s\n' %
692:                                  (i, item1, item2))
693:                     break
694:             else:
695:                 if (len1 == len2 and seq_type is None and
696:                     type(seq1) != type(seq2)):
697:                     # The sequences are the same, but have differing types.
698:                     return
699: 
700:             if len1 > len2:
701:                 differing += ('\nFirst %s contains %d additional '
702:                              'elements.\n' % (seq_type_name, len1 - len2))
703:                 try:
704:                     differing += ('First extra element %d:\n%s\n' %
705:                                   (len2, seq1[len2]))
706:                 except (TypeError, IndexError, NotImplementedError):
707:                     differing += ('Unable to index element %d '
708:                                   'of first %s\n' % (len2, seq_type_name))
709:             elif len1 < len2:
710:                 differing += ('\nSecond %s contains %d additional '
711:                              'elements.\n' % (seq_type_name, len2 - len1))
712:                 try:
713:                     differing += ('First extra element %d:\n%s\n' %
714:                                   (len1, seq2[len1]))
715:                 except (TypeError, IndexError, NotImplementedError):
716:                     differing += ('Unable to index element %d '
717:                                   'of second %s\n' % (len1, seq_type_name))
718:         standardMsg = differing
719:         diffMsg = '\n' + '\n'.join(
720:             difflib.ndiff(pprint.pformat(seq1).splitlines(),
721:                           pprint.pformat(seq2).splitlines()))
722:         standardMsg = self._truncateMessage(standardMsg, diffMsg)
723:         msg = self._formatMessage(msg, standardMsg)
724:         self.fail(msg)
725: 
726:     def _truncateMessage(self, message, diff):
727:         max_diff = self.maxDiff
728:         if max_diff is None or len(diff) <= max_diff:
729:             return message + diff
730:         return message + (DIFF_OMITTED % len(diff))
731: 
732:     def assertListEqual(self, list1, list2, msg=None):
733:         '''A list-specific equality assertion.
734: 
735:         Args:
736:             list1: The first list to compare.
737:             list2: The second list to compare.
738:             msg: Optional message to use on failure instead of a list of
739:                     differences.
740: 
741:         '''
742:         self.assertSequenceEqual(list1, list2, msg, seq_type=list)
743: 
744:     def assertTupleEqual(self, tuple1, tuple2, msg=None):
745:         '''A tuple-specific equality assertion.
746: 
747:         Args:
748:             tuple1: The first tuple to compare.
749:             tuple2: The second tuple to compare.
750:             msg: Optional message to use on failure instead of a list of
751:                     differences.
752:         '''
753:         self.assertSequenceEqual(tuple1, tuple2, msg, seq_type=tuple)
754: 
755:     def assertSetEqual(self, set1, set2, msg=None):
756:         '''A set-specific equality assertion.
757: 
758:         Args:
759:             set1: The first set to compare.
760:             set2: The second set to compare.
761:             msg: Optional message to use on failure instead of a list of
762:                     differences.
763: 
764:         assertSetEqual uses ducktyping to support different types of sets, and
765:         is optimized for sets specifically (parameters must support a
766:         difference method).
767:         '''
768:         try:
769:             difference1 = set1.difference(set2)
770:         except TypeError, e:
771:             self.fail('invalid type when attempting set difference: %s' % e)
772:         except AttributeError, e:
773:             self.fail('first argument does not support set difference: %s' % e)
774: 
775:         try:
776:             difference2 = set2.difference(set1)
777:         except TypeError, e:
778:             self.fail('invalid type when attempting set difference: %s' % e)
779:         except AttributeError, e:
780:             self.fail('second argument does not support set difference: %s' % e)
781: 
782:         if not (difference1 or difference2):
783:             return
784: 
785:         lines = []
786:         if difference1:
787:             lines.append('Items in the first set but not the second:')
788:             for item in difference1:
789:                 lines.append(repr(item))
790:         if difference2:
791:             lines.append('Items in the second set but not the first:')
792:             for item in difference2:
793:                 lines.append(repr(item))
794: 
795:         standardMsg = '\n'.join(lines)
796:         self.fail(self._formatMessage(msg, standardMsg))
797: 
798:     def assertIn(self, member, container, msg=None):
799:         '''Just like self.assertTrue(a in b), but with a nicer default message.'''
800:         if member not in container:
801:             standardMsg = '%s not found in %s' % (safe_repr(member),
802:                                                   safe_repr(container))
803:             self.fail(self._formatMessage(msg, standardMsg))
804: 
805:     def assertNotIn(self, member, container, msg=None):
806:         '''Just like self.assertTrue(a not in b), but with a nicer default message.'''
807:         if member in container:
808:             standardMsg = '%s unexpectedly found in %s' % (safe_repr(member),
809:                                                         safe_repr(container))
810:             self.fail(self._formatMessage(msg, standardMsg))
811: 
812:     def assertIs(self, expr1, expr2, msg=None):
813:         '''Just like self.assertTrue(a is b), but with a nicer default message.'''
814:         if expr1 is not expr2:
815:             standardMsg = '%s is not %s' % (safe_repr(expr1),
816:                                              safe_repr(expr2))
817:             self.fail(self._formatMessage(msg, standardMsg))
818: 
819:     def assertIsNot(self, expr1, expr2, msg=None):
820:         '''Just like self.assertTrue(a is not b), but with a nicer default message.'''
821:         if expr1 is expr2:
822:             standardMsg = 'unexpectedly identical: %s' % (safe_repr(expr1),)
823:             self.fail(self._formatMessage(msg, standardMsg))
824: 
825:     def assertDictEqual(self, d1, d2, msg=None):
826:         self.assertIsInstance(d1, dict, 'First argument is not a dictionary')
827:         self.assertIsInstance(d2, dict, 'Second argument is not a dictionary')
828: 
829:         if d1 != d2:
830:             standardMsg = '%s != %s' % (safe_repr(d1, True), safe_repr(d2, True))
831:             diff = ('\n' + '\n'.join(difflib.ndiff(
832:                            pprint.pformat(d1).splitlines(),
833:                            pprint.pformat(d2).splitlines())))
834:             standardMsg = self._truncateMessage(standardMsg, diff)
835:             self.fail(self._formatMessage(msg, standardMsg))
836: 
837:     def assertDictContainsSubset(self, expected, actual, msg=None):
838:         '''Checks whether actual is a superset of expected.'''
839:         missing = []
840:         mismatched = []
841:         for key, value in expected.iteritems():
842:             if key not in actual:
843:                 missing.append(key)
844:             elif value != actual[key]:
845:                 mismatched.append('%s, expected: %s, actual: %s' %
846:                                   (safe_repr(key), safe_repr(value),
847:                                    safe_repr(actual[key])))
848: 
849:         if not (missing or mismatched):
850:             return
851: 
852:         standardMsg = ''
853:         if missing:
854:             standardMsg = 'Missing: %s' % ','.join(safe_repr(m) for m in
855:                                                     missing)
856:         if mismatched:
857:             if standardMsg:
858:                 standardMsg += '; '
859:             standardMsg += 'Mismatched values: %s' % ','.join(mismatched)
860: 
861:         self.fail(self._formatMessage(msg, standardMsg))
862: 
863:     def assertItemsEqual(self, expected_seq, actual_seq, msg=None):
864:         '''An unordered sequence specific comparison. It asserts that
865:         actual_seq and expected_seq have the same element counts.
866:         Equivalent to::
867: 
868:             self.assertEqual(Counter(iter(actual_seq)),
869:                              Counter(iter(expected_seq)))
870: 
871:         Asserts that each element has the same count in both sequences.
872:         Example:
873:             - [0, 1, 1] and [1, 0, 1] compare equal.
874:             - [0, 0, 1] and [0, 1] compare unequal.
875:         '''
876:         first_seq, second_seq = list(expected_seq), list(actual_seq)
877:         with warnings.catch_warnings():
878:             if sys.py3kwarning:
879:                 # Silence Py3k warning raised during the sorting
880:                 for _msg in ["(code|dict|type) inequality comparisons",
881:                              "builtin_function_or_method order comparisons",
882:                              "comparing unequal types"]:
883:                     warnings.filterwarnings("ignore", _msg, DeprecationWarning)
884:             try:
885:                 first = collections.Counter(first_seq)
886:                 second = collections.Counter(second_seq)
887:             except TypeError:
888:                 # Handle case with unhashable elements
889:                 differences = _count_diff_all_purpose(first_seq, second_seq)
890:             else:
891:                 if first == second:
892:                     return
893:                 differences = _count_diff_hashable(first_seq, second_seq)
894: 
895:         if differences:
896:             standardMsg = 'Element counts were not equal:\n'
897:             lines = ['First has %d, Second has %d:  %r' % diff for diff in differences]
898:             diffMsg = '\n'.join(lines)
899:             standardMsg = self._truncateMessage(standardMsg, diffMsg)
900:             msg = self._formatMessage(msg, standardMsg)
901:             self.fail(msg)
902: 
903:     def assertMultiLineEqual(self, first, second, msg=None):
904:         '''Assert that two multi-line strings are equal.'''
905:         self.assertIsInstance(first, basestring,
906:                 'First argument is not a string')
907:         self.assertIsInstance(second, basestring,
908:                 'Second argument is not a string')
909: 
910:         if first != second:
911:             # don't use difflib if the strings are too long
912:             if (len(first) > self._diffThreshold or
913:                 len(second) > self._diffThreshold):
914:                 self._baseAssertEqual(first, second, msg)
915:             firstlines = first.splitlines(True)
916:             secondlines = second.splitlines(True)
917:             if len(firstlines) == 1 and first.strip('\r\n') == first:
918:                 firstlines = [first + '\n']
919:                 secondlines = [second + '\n']
920:             standardMsg = '%s != %s' % (safe_repr(first, True),
921:                                         safe_repr(second, True))
922:             diff = '\n' + ''.join(difflib.ndiff(firstlines, secondlines))
923:             standardMsg = self._truncateMessage(standardMsg, diff)
924:             self.fail(self._formatMessage(msg, standardMsg))
925: 
926:     def assertLess(self, a, b, msg=None):
927:         '''Just like self.assertTrue(a < b), but with a nicer default message.'''
928:         if not a < b:
929:             standardMsg = '%s not less than %s' % (safe_repr(a), safe_repr(b))
930:             self.fail(self._formatMessage(msg, standardMsg))
931: 
932:     def assertLessEqual(self, a, b, msg=None):
933:         '''Just like self.assertTrue(a <= b), but with a nicer default message.'''
934:         if not a <= b:
935:             standardMsg = '%s not less than or equal to %s' % (safe_repr(a), safe_repr(b))
936:             self.fail(self._formatMessage(msg, standardMsg))
937: 
938:     def assertGreater(self, a, b, msg=None):
939:         '''Just like self.assertTrue(a > b), but with a nicer default message.'''
940:         if not a > b:
941:             standardMsg = '%s not greater than %s' % (safe_repr(a), safe_repr(b))
942:             self.fail(self._formatMessage(msg, standardMsg))
943: 
944:     def assertGreaterEqual(self, a, b, msg=None):
945:         '''Just like self.assertTrue(a >= b), but with a nicer default message.'''
946:         if not a >= b:
947:             standardMsg = '%s not greater than or equal to %s' % (safe_repr(a), safe_repr(b))
948:             self.fail(self._formatMessage(msg, standardMsg))
949: 
950:     def assertIsNone(self, obj, msg=None):
951:         '''Same as self.assertTrue(obj is None), with a nicer default message.'''
952:         if obj is not None:
953:             standardMsg = '%s is not None' % (safe_repr(obj),)
954:             self.fail(self._formatMessage(msg, standardMsg))
955: 
956:     def assertIsNotNone(self, obj, msg=None):
957:         '''Included for symmetry with assertIsNone.'''
958:         if obj is None:
959:             standardMsg = 'unexpectedly None'
960:             self.fail(self._formatMessage(msg, standardMsg))
961: 
962:     def assertIsInstance(self, obj, cls, msg=None):
963:         '''Same as self.assertTrue(isinstance(obj, cls)), with a nicer
964:         default message.'''
965:         if not isinstance(obj, cls):
966:             standardMsg = '%s is not an instance of %r' % (safe_repr(obj), cls)
967:             self.fail(self._formatMessage(msg, standardMsg))
968: 
969:     def assertNotIsInstance(self, obj, cls, msg=None):
970:         '''Included for symmetry with assertIsInstance.'''
971:         if isinstance(obj, cls):
972:             standardMsg = '%s is an instance of %r' % (safe_repr(obj), cls)
973:             self.fail(self._formatMessage(msg, standardMsg))
974: 
975:     def assertRaisesRegexp(self, expected_exception, expected_regexp,
976:                            callable_obj=None, *args, **kwargs):
977:         '''Asserts that the message in a raised exception matches a regexp.
978: 
979:         Args:
980:             expected_exception: Exception class expected to be raised.
981:             expected_regexp: Regexp (re pattern object or string) expected
982:                     to be found in error message.
983:             callable_obj: Function to be called.
984:             args: Extra args.
985:             kwargs: Extra kwargs.
986:         '''
987:         if expected_regexp is not None:
988:             expected_regexp = re.compile(expected_regexp)
989:         context = _AssertRaisesContext(expected_exception, self, expected_regexp)
990:         if callable_obj is None:
991:             return context
992:         with context:
993:             callable_obj(*args, **kwargs)
994: 
995:     def assertRegexpMatches(self, text, expected_regexp, msg=None):
996:         '''Fail the test unless the text matches the regular expression.'''
997:         if isinstance(expected_regexp, basestring):
998:             expected_regexp = re.compile(expected_regexp)
999:         if not expected_regexp.search(text):
1000:             msg = msg or "Regexp didn't match"
1001:             msg = '%s: %r not found in %r' % (msg, expected_regexp.pattern, text)
1002:             raise self.failureException(msg)
1003: 
1004:     def assertNotRegexpMatches(self, text, unexpected_regexp, msg=None):
1005:         '''Fail the test if the text matches the regular expression.'''
1006:         if isinstance(unexpected_regexp, basestring):
1007:             unexpected_regexp = re.compile(unexpected_regexp)
1008:         match = unexpected_regexp.search(text)
1009:         if match:
1010:             msg = msg or "Regexp matched"
1011:             msg = '%s: %r matches %r in %r' % (msg,
1012:                                                text[match.start():match.end()],
1013:                                                unexpected_regexp.pattern,
1014:                                                text)
1015:             raise self.failureException(msg)
1016: 
1017: 
1018: class FunctionTestCase(TestCase):
1019:     '''A test case that wraps a test function.
1020: 
1021:     This is useful for slipping pre-existing test functions into the
1022:     unittest framework. Optionally, set-up and tidy-up functions can be
1023:     supplied. As with TestCase, the tidy-up ('tearDown') function will
1024:     always be called if the set-up ('setUp') function ran successfully.
1025:     '''
1026: 
1027:     def __init__(self, testFunc, setUp=None, tearDown=None, description=None):
1028:         super(FunctionTestCase, self).__init__()
1029:         self._setUpFunc = setUp
1030:         self._tearDownFunc = tearDown
1031:         self._testFunc = testFunc
1032:         self._description = description
1033: 
1034:     def setUp(self):
1035:         if self._setUpFunc is not None:
1036:             self._setUpFunc()
1037: 
1038:     def tearDown(self):
1039:         if self._tearDownFunc is not None:
1040:             self._tearDownFunc()
1041: 
1042:     def runTest(self):
1043:         self._testFunc()
1044: 
1045:     def id(self):
1046:         return self._testFunc.__name__
1047: 
1048:     def __eq__(self, other):
1049:         if not isinstance(other, self.__class__):
1050:             return NotImplemented
1051: 
1052:         return self._setUpFunc == other._setUpFunc and \
1053:                self._tearDownFunc == other._tearDownFunc and \
1054:                self._testFunc == other._testFunc and \
1055:                self._description == other._description
1056: 
1057:     def __ne__(self, other):
1058:         return not self == other
1059: 
1060:     def __hash__(self):
1061:         return hash((type(self), self._setUpFunc, self._tearDownFunc,
1062:                      self._testFunc, self._description))
1063: 
1064:     def __str__(self):
1065:         return "%s (%s)" % (strclass(self.__class__),
1066:                             self._testFunc.__name__)
1067: 
1068:     def __repr__(self):
1069:         return "<%s tec=%s>" % (strclass(self.__class__),
1070:                                      self._testFunc)
1071: 
1072:     def shortDescription(self):
1073:         if self._description is not None:
1074:             return self._description
1075:         doc = self._testFunc.__doc__
1076:         return doc and doc.split("\n")[0].strip() or None
1077: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_186534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Test case implementation')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import collections' statement (line 3)
import collections

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'collections', collections, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import functools' statement (line 5)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import difflib' statement (line 6)
import difflib

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'difflib', difflib, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import pprint' statement (line 7)
import pprint

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pprint', pprint, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import re' statement (line 8)
import re

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import types' statement (line 9)
import types

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import warnings' statement (line 10)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from unittest import result' statement (line 12)
from unittest import result

import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'unittest', None, module_type_store, ['result'], [result])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from unittest.util import strclass, safe_repr, unorderable_list_difference, _count_diff_all_purpose, _count_diff_hashable' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_186535 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'unittest.util')

if (type(import_186535) is not StypyTypeError):

    if (import_186535 != 'pyd_module'):
        __import__(import_186535)
        sys_modules_186536 = sys.modules[import_186535]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'unittest.util', sys_modules_186536.module_type_store, module_type_store, ['strclass', 'safe_repr', 'unorderable_list_difference', '_count_diff_all_purpose', '_count_diff_hashable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_186536, sys_modules_186536.module_type_store, module_type_store)
    else:
        from unittest.util import strclass, safe_repr, unorderable_list_difference, _count_diff_all_purpose, _count_diff_hashable

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'unittest.util', None, module_type_store, ['strclass', 'safe_repr', 'unorderable_list_difference', '_count_diff_all_purpose', '_count_diff_hashable'], [strclass, safe_repr, unorderable_list_difference, _count_diff_all_purpose, _count_diff_hashable])

else:
    # Assigning a type to the variable 'unittest.util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'unittest.util', import_186535)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')


# Assigning a Name to a Name (line 19):

# Assigning a Name to a Name (line 19):
# Getting the type of 'True' (line 19)
True_186537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'True')
# Assigning a type to the variable '__unittest' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '__unittest', True_186537)

# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_186538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'str', '\nDiff is %s characters long. Set self.maxDiff to None to see it.')
# Assigning a type to the variable 'DIFF_OMITTED' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'DIFF_OMITTED', str_186538)
# Declaration of the 'SkipTest' class
# Getting the type of 'Exception' (line 25)
Exception_186539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'Exception')

class SkipTest(Exception_186539, ):
    str_186540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', '\n    Raise this exception in a test to skip it.\n\n    Usually you can use TestCase.skipTest() or one of the skipping decorators\n    instead of raising this directly.\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SkipTest.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SkipTest' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'SkipTest', SkipTest)
# Declaration of the '_ExpectedFailure' class
# Getting the type of 'Exception' (line 34)
Exception_186541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'Exception')

class _ExpectedFailure(Exception_186541, ):
    str_186542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', '\n    Raise this when a test is expected to fail.\n\n    This is an implementation detail.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ExpectedFailure.__init__', ['exc_info'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['exc_info'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_186549 = {}
        
        # Call to super(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of '_ExpectedFailure' (line 42)
        _ExpectedFailure_186544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), '_ExpectedFailure', False)
        # Getting the type of 'self' (line 42)
        self_186545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 'self', False)
        # Processing the call keyword arguments (line 42)
        kwargs_186546 = {}
        # Getting the type of 'super' (line 42)
        super_186543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'super', False)
        # Calling super(args, kwargs) (line 42)
        super_call_result_186547 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), super_186543, *[_ExpectedFailure_186544, self_186545], **kwargs_186546)
        
        # Obtaining the member '__init__' of a type (line 42)
        init___186548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), super_call_result_186547, '__init__')
        # Calling __init__(args, kwargs) (line 42)
        init___call_result_186550 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), init___186548, *[], **kwargs_186549)
        
        
        # Assigning a Name to a Attribute (line 43):
        
        # Assigning a Name to a Attribute (line 43):
        # Getting the type of 'exc_info' (line 43)
        exc_info_186551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'exc_info')
        # Getting the type of 'self' (line 43)
        self_186552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'exc_info' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_186552, 'exc_info', exc_info_186551)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable '_ExpectedFailure' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), '_ExpectedFailure', _ExpectedFailure)
# Declaration of the '_UnexpectedSuccess' class
# Getting the type of 'Exception' (line 45)
Exception_186553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'Exception')

class _UnexpectedSuccess(Exception_186553, ):
    str_186554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'str', "\n    The test was supposed to fail, but it didn't!\n    ")
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 0, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_UnexpectedSuccess.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_UnexpectedSuccess' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), '_UnexpectedSuccess', _UnexpectedSuccess)

@norecursion
def _id(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_id'
    module_type_store = module_type_store.open_function_context('_id', 51, 0, False)
    
    # Passed parameters checking function
    _id.stypy_localization = localization
    _id.stypy_type_of_self = None
    _id.stypy_type_store = module_type_store
    _id.stypy_function_name = '_id'
    _id.stypy_param_names_list = ['obj']
    _id.stypy_varargs_param_name = None
    _id.stypy_kwargs_param_name = None
    _id.stypy_call_defaults = defaults
    _id.stypy_call_varargs = varargs
    _id.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_id', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_id', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_id(...)' code ##################

    # Getting the type of 'obj' (line 52)
    obj_186555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'obj')
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type', obj_186555)
    
    # ################# End of '_id(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_id' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_186556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186556)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_id'
    return stypy_return_type_186556

# Assigning a type to the variable '_id' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '_id', _id)

@norecursion
def skip(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'skip'
    module_type_store = module_type_store.open_function_context('skip', 54, 0, False)
    
    # Passed parameters checking function
    skip.stypy_localization = localization
    skip.stypy_type_of_self = None
    skip.stypy_type_store = module_type_store
    skip.stypy_function_name = 'skip'
    skip.stypy_param_names_list = ['reason']
    skip.stypy_varargs_param_name = None
    skip.stypy_kwargs_param_name = None
    skip.stypy_call_defaults = defaults
    skip.stypy_call_varargs = varargs
    skip.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'skip', ['reason'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'skip', localization, ['reason'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'skip(...)' code ##################

    str_186557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'str', '\n    Unconditionally skip a test.\n    ')

    @norecursion
    def decorator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'decorator'
        module_type_store = module_type_store.open_function_context('decorator', 58, 4, False)
        
        # Passed parameters checking function
        decorator.stypy_localization = localization
        decorator.stypy_type_of_self = None
        decorator.stypy_type_store = module_type_store
        decorator.stypy_function_name = 'decorator'
        decorator.stypy_param_names_list = ['test_item']
        decorator.stypy_varargs_param_name = None
        decorator.stypy_kwargs_param_name = None
        decorator.stypy_call_defaults = defaults
        decorator.stypy_call_varargs = varargs
        decorator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'decorator', ['test_item'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'decorator', localization, ['test_item'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'decorator(...)' code ##################

        
        
        
        # Call to isinstance(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'test_item' (line 59)
        test_item_186559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'test_item', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 59)
        tuple_186560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 59)
        # Adding element type (line 59)
        # Getting the type of 'type' (line 59)
        type_186561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 38), 'type', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 38), tuple_186560, type_186561)
        # Adding element type (line 59)
        # Getting the type of 'types' (line 59)
        types_186562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 44), 'types', False)
        # Obtaining the member 'ClassType' of a type (line 59)
        ClassType_186563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 44), types_186562, 'ClassType')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 38), tuple_186560, ClassType_186563)
        
        # Processing the call keyword arguments (line 59)
        kwargs_186564 = {}
        # Getting the type of 'isinstance' (line 59)
        isinstance_186558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 59)
        isinstance_call_result_186565 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), isinstance_186558, *[test_item_186559, tuple_186560], **kwargs_186564)
        
        # Applying the 'not' unary operator (line 59)
        result_not__186566 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), 'not', isinstance_call_result_186565)
        
        # Testing the type of an if condition (line 59)
        if_condition_186567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_not__186566)
        # Assigning a type to the variable 'if_condition_186567' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_186567', if_condition_186567)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def skip_wrapper(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'skip_wrapper'
            module_type_store = module_type_store.open_function_context('skip_wrapper', 60, 12, False)
            
            # Passed parameters checking function
            skip_wrapper.stypy_localization = localization
            skip_wrapper.stypy_type_of_self = None
            skip_wrapper.stypy_type_store = module_type_store
            skip_wrapper.stypy_function_name = 'skip_wrapper'
            skip_wrapper.stypy_param_names_list = []
            skip_wrapper.stypy_varargs_param_name = 'args'
            skip_wrapper.stypy_kwargs_param_name = 'kwargs'
            skip_wrapper.stypy_call_defaults = defaults
            skip_wrapper.stypy_call_varargs = varargs
            skip_wrapper.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'skip_wrapper', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'skip_wrapper', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'skip_wrapper(...)' code ##################

            
            # Call to SkipTest(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'reason' (line 62)
            reason_186569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'reason', False)
            # Processing the call keyword arguments (line 62)
            kwargs_186570 = {}
            # Getting the type of 'SkipTest' (line 62)
            SkipTest_186568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 22), 'SkipTest', False)
            # Calling SkipTest(args, kwargs) (line 62)
            SkipTest_call_result_186571 = invoke(stypy.reporting.localization.Localization(__file__, 62, 22), SkipTest_186568, *[reason_186569], **kwargs_186570)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 62, 16), SkipTest_call_result_186571, 'raise parameter', BaseException)
            
            # ################# End of 'skip_wrapper(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'skip_wrapper' in the type store
            # Getting the type of 'stypy_return_type' (line 60)
            stypy_return_type_186572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_186572)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'skip_wrapper'
            return stypy_return_type_186572

        # Assigning a type to the variable 'skip_wrapper' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'skip_wrapper', skip_wrapper)
        
        # Assigning a Name to a Name (line 63):
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'skip_wrapper' (line 63)
        skip_wrapper_186573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'skip_wrapper')
        # Assigning a type to the variable 'test_item' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'test_item', skip_wrapper_186573)
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 65):
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'True' (line 65)
        True_186574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 38), 'True')
        # Getting the type of 'test_item' (line 65)
        test_item_186575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'test_item')
        # Setting the type of the member '__unittest_skip__' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), test_item_186575, '__unittest_skip__', True_186574)
        
        # Assigning a Name to a Attribute (line 66):
        
        # Assigning a Name to a Attribute (line 66):
        # Getting the type of 'reason' (line 66)
        reason_186576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'reason')
        # Getting the type of 'test_item' (line 66)
        test_item_186577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'test_item')
        # Setting the type of the member '__unittest_skip_why__' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), test_item_186577, '__unittest_skip_why__', reason_186576)
        # Getting the type of 'test_item' (line 67)
        test_item_186578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'test_item')
        # Assigning a type to the variable 'stypy_return_type' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', test_item_186578)
        
        # ################# End of 'decorator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'decorator' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_186579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186579)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'decorator'
        return stypy_return_type_186579

    # Assigning a type to the variable 'decorator' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'decorator', decorator)
    # Getting the type of 'decorator' (line 68)
    decorator_186580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'decorator')
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', decorator_186580)
    
    # ################# End of 'skip(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'skip' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_186581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186581)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'skip'
    return stypy_return_type_186581

# Assigning a type to the variable 'skip' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'skip', skip)

@norecursion
def skipIf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'skipIf'
    module_type_store = module_type_store.open_function_context('skipIf', 70, 0, False)
    
    # Passed parameters checking function
    skipIf.stypy_localization = localization
    skipIf.stypy_type_of_self = None
    skipIf.stypy_type_store = module_type_store
    skipIf.stypy_function_name = 'skipIf'
    skipIf.stypy_param_names_list = ['condition', 'reason']
    skipIf.stypy_varargs_param_name = None
    skipIf.stypy_kwargs_param_name = None
    skipIf.stypy_call_defaults = defaults
    skipIf.stypy_call_varargs = varargs
    skipIf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'skipIf', ['condition', 'reason'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'skipIf', localization, ['condition', 'reason'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'skipIf(...)' code ##################

    str_186582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'str', '\n    Skip a test if the condition is true.\n    ')
    
    # Getting the type of 'condition' (line 74)
    condition_186583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'condition')
    # Testing the type of an if condition (line 74)
    if_condition_186584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 4), condition_186583)
    # Assigning a type to the variable 'if_condition_186584' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'if_condition_186584', if_condition_186584)
    # SSA begins for if statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to skip(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'reason' (line 75)
    reason_186586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'reason', False)
    # Processing the call keyword arguments (line 75)
    kwargs_186587 = {}
    # Getting the type of 'skip' (line 75)
    skip_186585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'skip', False)
    # Calling skip(args, kwargs) (line 75)
    skip_call_result_186588 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), skip_186585, *[reason_186586], **kwargs_186587)
    
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type', skip_call_result_186588)
    # SSA join for if statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of '_id' (line 76)
    _id_186589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), '_id')
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type', _id_186589)
    
    # ################# End of 'skipIf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'skipIf' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_186590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'skipIf'
    return stypy_return_type_186590

# Assigning a type to the variable 'skipIf' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'skipIf', skipIf)

@norecursion
def skipUnless(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'skipUnless'
    module_type_store = module_type_store.open_function_context('skipUnless', 78, 0, False)
    
    # Passed parameters checking function
    skipUnless.stypy_localization = localization
    skipUnless.stypy_type_of_self = None
    skipUnless.stypy_type_store = module_type_store
    skipUnless.stypy_function_name = 'skipUnless'
    skipUnless.stypy_param_names_list = ['condition', 'reason']
    skipUnless.stypy_varargs_param_name = None
    skipUnless.stypy_kwargs_param_name = None
    skipUnless.stypy_call_defaults = defaults
    skipUnless.stypy_call_varargs = varargs
    skipUnless.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'skipUnless', ['condition', 'reason'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'skipUnless', localization, ['condition', 'reason'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'skipUnless(...)' code ##################

    str_186591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', '\n    Skip a test unless the condition is true.\n    ')
    
    
    # Getting the type of 'condition' (line 82)
    condition_186592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'condition')
    # Applying the 'not' unary operator (line 82)
    result_not__186593 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 7), 'not', condition_186592)
    
    # Testing the type of an if condition (line 82)
    if_condition_186594 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 4), result_not__186593)
    # Assigning a type to the variable 'if_condition_186594' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'if_condition_186594', if_condition_186594)
    # SSA begins for if statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to skip(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'reason' (line 83)
    reason_186596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'reason', False)
    # Processing the call keyword arguments (line 83)
    kwargs_186597 = {}
    # Getting the type of 'skip' (line 83)
    skip_186595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'skip', False)
    # Calling skip(args, kwargs) (line 83)
    skip_call_result_186598 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), skip_186595, *[reason_186596], **kwargs_186597)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', skip_call_result_186598)
    # SSA join for if statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of '_id' (line 84)
    _id_186599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), '_id')
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type', _id_186599)
    
    # ################# End of 'skipUnless(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'skipUnless' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_186600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186600)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'skipUnless'
    return stypy_return_type_186600

# Assigning a type to the variable 'skipUnless' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'skipUnless', skipUnless)

@norecursion
def expectedFailure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'expectedFailure'
    module_type_store = module_type_store.open_function_context('expectedFailure', 87, 0, False)
    
    # Passed parameters checking function
    expectedFailure.stypy_localization = localization
    expectedFailure.stypy_type_of_self = None
    expectedFailure.stypy_type_store = module_type_store
    expectedFailure.stypy_function_name = 'expectedFailure'
    expectedFailure.stypy_param_names_list = ['func']
    expectedFailure.stypy_varargs_param_name = None
    expectedFailure.stypy_kwargs_param_name = None
    expectedFailure.stypy_call_defaults = defaults
    expectedFailure.stypy_call_varargs = varargs
    expectedFailure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expectedFailure', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expectedFailure', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expectedFailure(...)' code ##################


    @norecursion
    def wrapper(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrapper'
        module_type_store = module_type_store.open_function_context('wrapper', 88, 4, False)
        
        # Passed parameters checking function
        wrapper.stypy_localization = localization
        wrapper.stypy_type_of_self = None
        wrapper.stypy_type_store = module_type_store
        wrapper.stypy_function_name = 'wrapper'
        wrapper.stypy_param_names_list = []
        wrapper.stypy_varargs_param_name = 'args'
        wrapper.stypy_kwargs_param_name = 'kwargs'
        wrapper.stypy_call_defaults = defaults
        wrapper.stypy_call_varargs = varargs
        wrapper.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrapper', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrapper', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrapper(...)' code ##################

        
        
        # SSA begins for try-except statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to func(...): (line 91)
        # Getting the type of 'args' (line 91)
        args_186602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'args', False)
        # Processing the call keyword arguments (line 91)
        # Getting the type of 'kwargs' (line 91)
        kwargs_186603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'kwargs', False)
        kwargs_186604 = {'kwargs_186603': kwargs_186603}
        # Getting the type of 'func' (line 91)
        func_186601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'func', False)
        # Calling func(args, kwargs) (line 91)
        func_call_result_186605 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), func_186601, *[args_186602], **kwargs_186604)
        
        # SSA branch for the except part of a try statement (line 90)
        # SSA branch for the except 'Exception' branch of a try statement (line 90)
        module_type_store.open_ssa_branch('except')
        
        # Call to _ExpectedFailure(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Call to exc_info(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_186609 = {}
        # Getting the type of 'sys' (line 93)
        sys_186607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 93)
        exc_info_186608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 35), sys_186607, 'exc_info')
        # Calling exc_info(args, kwargs) (line 93)
        exc_info_call_result_186610 = invoke(stypy.reporting.localization.Localization(__file__, 93, 35), exc_info_186608, *[], **kwargs_186609)
        
        # Processing the call keyword arguments (line 93)
        kwargs_186611 = {}
        # Getting the type of '_ExpectedFailure' (line 93)
        _ExpectedFailure_186606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), '_ExpectedFailure', False)
        # Calling _ExpectedFailure(args, kwargs) (line 93)
        _ExpectedFailure_call_result_186612 = invoke(stypy.reporting.localization.Localization(__file__, 93, 18), _ExpectedFailure_186606, *[exc_info_call_result_186610], **kwargs_186611)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 93, 12), _ExpectedFailure_call_result_186612, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of '_UnexpectedSuccess' (line 94)
        _UnexpectedSuccess_186613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), '_UnexpectedSuccess')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 94, 8), _UnexpectedSuccess_186613, 'raise parameter', BaseException)
        
        # ################# End of 'wrapper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrapper' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_186614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrapper'
        return stypy_return_type_186614

    # Assigning a type to the variable 'wrapper' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'wrapper', wrapper)
    # Getting the type of 'wrapper' (line 95)
    wrapper_186615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'wrapper')
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', wrapper_186615)
    
    # ################# End of 'expectedFailure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expectedFailure' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_186616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186616)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expectedFailure'
    return stypy_return_type_186616

# Assigning a type to the variable 'expectedFailure' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'expectedFailure', expectedFailure)
# Declaration of the '_AssertRaisesContext' class

class _AssertRaisesContext(object, ):
    str_186617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'str', 'A context manager used to implement TestCase.assertRaises* methods.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 101)
        None_186618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'None')
        defaults = [None_186618]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_AssertRaisesContext.__init__', ['expected', 'test_case', 'expected_regexp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['expected', 'test_case', 'expected_regexp'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 102):
        
        # Assigning a Name to a Attribute (line 102):
        # Getting the type of 'expected' (line 102)
        expected_186619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'expected')
        # Getting the type of 'self' (line 102)
        self_186620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self')
        # Setting the type of the member 'expected' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_186620, 'expected', expected_186619)
        
        # Assigning a Attribute to a Attribute (line 103):
        
        # Assigning a Attribute to a Attribute (line 103):
        # Getting the type of 'test_case' (line 103)
        test_case_186621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 32), 'test_case')
        # Obtaining the member 'failureException' of a type (line 103)
        failureException_186622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 32), test_case_186621, 'failureException')
        # Getting the type of 'self' (line 103)
        self_186623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self')
        # Setting the type of the member 'failureException' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_186623, 'failureException', failureException_186622)
        
        # Assigning a Name to a Attribute (line 104):
        
        # Assigning a Name to a Attribute (line 104):
        # Getting the type of 'expected_regexp' (line 104)
        expected_regexp_186624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'expected_regexp')
        # Getting the type of 'self' (line 104)
        self_186625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self')
        # Setting the type of the member 'expected_regexp' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_186625, 'expected_regexp', expected_regexp_186624)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __enter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__enter__'
        module_type_store = module_type_store.open_function_context('__enter__', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_localization', localization)
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_function_name', '_AssertRaisesContext.__enter__')
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_param_names_list', [])
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _AssertRaisesContext.__enter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_AssertRaisesContext.__enter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__enter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__enter__(...)' code ##################

        # Getting the type of 'self' (line 107)
        self_186626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', self_186626)
        
        # ################# End of '__enter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__enter__' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_186627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186627)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__enter__'
        return stypy_return_type_186627


    @norecursion
    def __exit__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__exit__'
        module_type_store = module_type_store.open_function_context('__exit__', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_localization', localization)
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_function_name', '_AssertRaisesContext.__exit__')
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_param_names_list', ['exc_type', 'exc_value', 'tb'])
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _AssertRaisesContext.__exit__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_AssertRaisesContext.__exit__', ['exc_type', 'exc_value', 'tb'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__exit__', localization, ['exc_type', 'exc_value', 'tb'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__exit__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 110)
        # Getting the type of 'exc_type' (line 110)
        exc_type_186628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'exc_type')
        # Getting the type of 'None' (line 110)
        None_186629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'None')
        
        (may_be_186630, more_types_in_union_186631) = may_be_none(exc_type_186628, None_186629)

        if may_be_186630:

            if more_types_in_union_186631:
                # Runtime conditional SSA (line 110)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Attribute to a Name (line 112):
            
            # Assigning a Attribute to a Name (line 112):
            # Getting the type of 'self' (line 112)
            self_186632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'self')
            # Obtaining the member 'expected' of a type (line 112)
            expected_186633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 27), self_186632, 'expected')
            # Obtaining the member '__name__' of a type (line 112)
            name___186634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 27), expected_186633, '__name__')
            # Assigning a type to the variable 'exc_name' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'exc_name', name___186634)
            # SSA branch for the except part of a try statement (line 111)
            # SSA branch for the except 'AttributeError' branch of a try statement (line 111)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Call to a Name (line 114):
            
            # Assigning a Call to a Name (line 114):
            
            # Call to str(...): (line 114)
            # Processing the call arguments (line 114)
            # Getting the type of 'self' (line 114)
            self_186636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 31), 'self', False)
            # Obtaining the member 'expected' of a type (line 114)
            expected_186637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 31), self_186636, 'expected')
            # Processing the call keyword arguments (line 114)
            kwargs_186638 = {}
            # Getting the type of 'str' (line 114)
            str_186635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 27), 'str', False)
            # Calling str(args, kwargs) (line 114)
            str_call_result_186639 = invoke(stypy.reporting.localization.Localization(__file__, 114, 27), str_186635, *[expected_186637], **kwargs_186638)
            
            # Assigning a type to the variable 'exc_name' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'exc_name', str_call_result_186639)
            # SSA join for try-except statement (line 111)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to failureException(...): (line 115)
            # Processing the call arguments (line 115)
            
            # Call to format(...): (line 116)
            # Processing the call arguments (line 116)
            # Getting the type of 'exc_name' (line 116)
            exc_name_186644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 40), 'exc_name', False)
            # Processing the call keyword arguments (line 116)
            kwargs_186645 = {}
            str_186642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 16), 'str', '{0} not raised')
            # Obtaining the member 'format' of a type (line 116)
            format_186643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), str_186642, 'format')
            # Calling format(args, kwargs) (line 116)
            format_call_result_186646 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), format_186643, *[exc_name_186644], **kwargs_186645)
            
            # Processing the call keyword arguments (line 115)
            kwargs_186647 = {}
            # Getting the type of 'self' (line 115)
            self_186640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'self', False)
            # Obtaining the member 'failureException' of a type (line 115)
            failureException_186641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 18), self_186640, 'failureException')
            # Calling failureException(args, kwargs) (line 115)
            failureException_call_result_186648 = invoke(stypy.reporting.localization.Localization(__file__, 115, 18), failureException_186641, *[format_call_result_186646], **kwargs_186647)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 115, 12), failureException_call_result_186648, 'raise parameter', BaseException)

            if more_types_in_union_186631:
                # SSA join for if statement (line 110)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to issubclass(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'exc_type' (line 117)
        exc_type_186650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 26), 'exc_type', False)
        # Getting the type of 'self' (line 117)
        self_186651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'self', False)
        # Obtaining the member 'expected' of a type (line 117)
        expected_186652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 36), self_186651, 'expected')
        # Processing the call keyword arguments (line 117)
        kwargs_186653 = {}
        # Getting the type of 'issubclass' (line 117)
        issubclass_186649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 117)
        issubclass_call_result_186654 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), issubclass_186649, *[exc_type_186650, expected_186652], **kwargs_186653)
        
        # Applying the 'not' unary operator (line 117)
        result_not__186655 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 11), 'not', issubclass_call_result_186654)
        
        # Testing the type of an if condition (line 117)
        if_condition_186656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), result_not__186655)
        # Assigning a type to the variable 'if_condition_186656' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'if_condition_186656', if_condition_186656)
        # SSA begins for if statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 119)
        False_186657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'stypy_return_type', False_186657)
        # SSA join for if statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 120):
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'exc_value' (line 120)
        exc_value_186658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'exc_value')
        # Getting the type of 'self' (line 120)
        self_186659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member 'exception' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_186659, 'exception', exc_value_186658)
        
        # Type idiom detected: calculating its left and rigth part (line 121)
        # Getting the type of 'self' (line 121)
        self_186660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'self')
        # Obtaining the member 'expected_regexp' of a type (line 121)
        expected_regexp_186661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), self_186660, 'expected_regexp')
        # Getting the type of 'None' (line 121)
        None_186662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'None')
        
        (may_be_186663, more_types_in_union_186664) = may_be_none(expected_regexp_186661, None_186662)

        if may_be_186663:

            if more_types_in_union_186664:
                # Runtime conditional SSA (line 121)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'True' (line 122)
            True_186665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'stypy_return_type', True_186665)

            if more_types_in_union_186664:
                # SSA join for if statement (line 121)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Name (line 124):
        
        # Assigning a Attribute to a Name (line 124):
        # Getting the type of 'self' (line 124)
        self_186666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 26), 'self')
        # Obtaining the member 'expected_regexp' of a type (line 124)
        expected_regexp_186667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 26), self_186666, 'expected_regexp')
        # Assigning a type to the variable 'expected_regexp' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'expected_regexp', expected_regexp_186667)
        
        
        
        # Call to search(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to str(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'exc_value' (line 125)
        exc_value_186671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 42), 'exc_value', False)
        # Processing the call keyword arguments (line 125)
        kwargs_186672 = {}
        # Getting the type of 'str' (line 125)
        str_186670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 38), 'str', False)
        # Calling str(args, kwargs) (line 125)
        str_call_result_186673 = invoke(stypy.reporting.localization.Localization(__file__, 125, 38), str_186670, *[exc_value_186671], **kwargs_186672)
        
        # Processing the call keyword arguments (line 125)
        kwargs_186674 = {}
        # Getting the type of 'expected_regexp' (line 125)
        expected_regexp_186668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'expected_regexp', False)
        # Obtaining the member 'search' of a type (line 125)
        search_186669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 15), expected_regexp_186668, 'search')
        # Calling search(args, kwargs) (line 125)
        search_call_result_186675 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), search_186669, *[str_call_result_186673], **kwargs_186674)
        
        # Applying the 'not' unary operator (line 125)
        result_not__186676 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), 'not', search_call_result_186675)
        
        # Testing the type of an if condition (line 125)
        if_condition_186677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_not__186676)
        # Assigning a type to the variable 'if_condition_186677' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_186677', if_condition_186677)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to failureException(...): (line 126)
        # Processing the call arguments (line 126)
        str_186680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 40), 'str', '"%s" does not match "%s"')
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_186681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        # Getting the type of 'expected_regexp' (line 127)
        expected_regexp_186682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'expected_regexp', False)
        # Obtaining the member 'pattern' of a type (line 127)
        pattern_186683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 22), expected_regexp_186682, 'pattern')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 22), tuple_186681, pattern_186683)
        # Adding element type (line 127)
        
        # Call to str(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'exc_value' (line 127)
        exc_value_186685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 51), 'exc_value', False)
        # Processing the call keyword arguments (line 127)
        kwargs_186686 = {}
        # Getting the type of 'str' (line 127)
        str_186684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'str', False)
        # Calling str(args, kwargs) (line 127)
        str_call_result_186687 = invoke(stypy.reporting.localization.Localization(__file__, 127, 47), str_186684, *[exc_value_186685], **kwargs_186686)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 22), tuple_186681, str_call_result_186687)
        
        # Applying the binary operator '%' (line 126)
        result_mod_186688 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 40), '%', str_186680, tuple_186681)
        
        # Processing the call keyword arguments (line 126)
        kwargs_186689 = {}
        # Getting the type of 'self' (line 126)
        self_186678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'self', False)
        # Obtaining the member 'failureException' of a type (line 126)
        failureException_186679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 18), self_186678, 'failureException')
        # Calling failureException(args, kwargs) (line 126)
        failureException_call_result_186690 = invoke(stypy.reporting.localization.Localization(__file__, 126, 18), failureException_186679, *[result_mod_186688], **kwargs_186689)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 126, 12), failureException_call_result_186690, 'raise parameter', BaseException)
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'True' (line 128)
        True_186691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'stypy_return_type', True_186691)
        
        # ################# End of '__exit__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__exit__' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_186692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__exit__'
        return stypy_return_type_186692


# Assigning a type to the variable '_AssertRaisesContext' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), '_AssertRaisesContext', _AssertRaisesContext)
# Declaration of the 'TestCase' class

class TestCase(object, ):
    str_186693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, (-1)), 'str', "A class whose instances are single test cases.\n\n    By default, the test code itself should be placed in a method named\n    'runTest'.\n\n    If the fixture may be used for many test cases, create as\n    many test methods as are needed. When instantiating such a TestCase\n    subclass, specify in the constructor arguments the name of the test method\n    that the instance is to execute.\n\n    Test authors should subclass TestCase for their own tests. Construction\n    and deconstruction of the test's environment ('fixture') can be\n    implemented by overriding the 'setUp' and 'tearDown' methods respectively.\n\n    If it is necessary to override the __init__ method, the base class\n    __init__ method must always be called. It is important that subclasses\n    should not change the signature of their __init__ method, since instances\n    of the classes are instantiated automatically by parts of the framework\n    in order to be run.\n\n    When subclassing TestCase, you can set these attributes:\n    * failureException: determines which exception will be raised when\n        the instance's assertion methods fail; test methods raising this\n        exception will be deemed to have 'failed' rather than 'errored'.\n    * longMessage: determines whether long messages (including repr of\n        objects used in assert methods) will be printed on failure in *addition*\n        to any explicit message passed.\n    * maxDiff: sets the maximum length of a diff in failure messages\n        by assert methods using difflib. It is looked up as an instance\n        attribute so can be configured by individual tests if required.\n    ")
    
    # Assigning a Name to a Name (line 164):
    
    # Assigning a Name to a Name (line 166):
    
    # Assigning a BinOp to a Name (line 168):
    
    # Assigning a BinOp to a Name (line 172):
    
    # Assigning a Name to a Name (line 176):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_186694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 34), 'str', 'runTest')
        defaults = [str_186694]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.__init__', ['methodName'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['methodName'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_186695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, (-1)), 'str', 'Create an instance of the class that will use the named test\n           method when executed. Raises a ValueError if the instance does\n           not have a method with the specified name.\n        ')
        
        # Assigning a Name to a Attribute (line 183):
        
        # Assigning a Name to a Attribute (line 183):
        # Getting the type of 'methodName' (line 183)
        methodName_186696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'methodName')
        # Getting the type of 'self' (line 183)
        self_186697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'self')
        # Setting the type of the member '_testMethodName' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), self_186697, '_testMethodName', methodName_186696)
        
        # Assigning a Name to a Attribute (line 184):
        
        # Assigning a Name to a Attribute (line 184):
        # Getting the type of 'None' (line 184)
        None_186698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'None')
        # Getting the type of 'self' (line 184)
        self_186699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self')
        # Setting the type of the member '_resultForDoCleanups' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_186699, '_resultForDoCleanups', None_186698)
        
        
        # SSA begins for try-except statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 186):
        
        # Assigning a Call to a Name (line 186):
        
        # Call to getattr(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'self' (line 186)
        self_186701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 33), 'self', False)
        # Getting the type of 'methodName' (line 186)
        methodName_186702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 39), 'methodName', False)
        # Processing the call keyword arguments (line 186)
        kwargs_186703 = {}
        # Getting the type of 'getattr' (line 186)
        getattr_186700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'getattr', False)
        # Calling getattr(args, kwargs) (line 186)
        getattr_call_result_186704 = invoke(stypy.reporting.localization.Localization(__file__, 186, 25), getattr_186700, *[self_186701, methodName_186702], **kwargs_186703)
        
        # Assigning a type to the variable 'testMethod' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'testMethod', getattr_call_result_186704)
        # SSA branch for the except part of a try statement (line 185)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 185)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 188)
        # Processing the call arguments (line 188)
        str_186706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 29), 'str', 'no such test method in %s: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 189)
        tuple_186707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 189)
        # Adding element type (line 189)
        # Getting the type of 'self' (line 189)
        self_186708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 19), 'self', False)
        # Obtaining the member '__class__' of a type (line 189)
        class___186709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 19), self_186708, '__class__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 19), tuple_186707, class___186709)
        # Adding element type (line 189)
        # Getting the type of 'methodName' (line 189)
        methodName_186710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 35), 'methodName', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 19), tuple_186707, methodName_186710)
        
        # Applying the binary operator '%' (line 188)
        result_mod_186711 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 29), '%', str_186706, tuple_186707)
        
        # Processing the call keyword arguments (line 188)
        kwargs_186712 = {}
        # Getting the type of 'ValueError' (line 188)
        ValueError_186705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 188)
        ValueError_call_result_186713 = invoke(stypy.reporting.localization.Localization(__file__, 188, 18), ValueError_186705, *[result_mod_186711], **kwargs_186712)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 188, 12), ValueError_call_result_186713, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 190):
        
        # Assigning a Attribute to a Attribute (line 190):
        # Getting the type of 'testMethod' (line 190)
        testMethod_186714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 30), 'testMethod')
        # Obtaining the member '__doc__' of a type (line 190)
        doc___186715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 30), testMethod_186714, '__doc__')
        # Getting the type of 'self' (line 190)
        self_186716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'self')
        # Setting the type of the member '_testMethodDoc' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), self_186716, '_testMethodDoc', doc___186715)
        
        # Assigning a List to a Attribute (line 191):
        
        # Assigning a List to a Attribute (line 191):
        
        # Obtaining an instance of the builtin type 'list' (line 191)
        list_186717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 191)
        
        # Getting the type of 'self' (line 191)
        self_186718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self')
        # Setting the type of the member '_cleanups' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_186718, '_cleanups', list_186717)
        
        # Assigning a Dict to a Attribute (line 196):
        
        # Assigning a Dict to a Attribute (line 196):
        
        # Obtaining an instance of the builtin type 'dict' (line 196)
        dict_186719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 36), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 196)
        
        # Getting the type of 'self' (line 196)
        self_186720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Setting the type of the member '_type_equality_funcs' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_186720, '_type_equality_funcs', dict_186719)
        
        # Call to addTypeEqualityFunc(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'dict' (line 197)
        dict_186723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 33), 'dict', False)
        str_186724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 39), 'str', 'assertDictEqual')
        # Processing the call keyword arguments (line 197)
        kwargs_186725 = {}
        # Getting the type of 'self' (line 197)
        self_186721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self', False)
        # Obtaining the member 'addTypeEqualityFunc' of a type (line 197)
        addTypeEqualityFunc_186722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_186721, 'addTypeEqualityFunc')
        # Calling addTypeEqualityFunc(args, kwargs) (line 197)
        addTypeEqualityFunc_call_result_186726 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), addTypeEqualityFunc_186722, *[dict_186723, str_186724], **kwargs_186725)
        
        
        # Call to addTypeEqualityFunc(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'list' (line 198)
        list_186729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 33), 'list', False)
        str_186730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 39), 'str', 'assertListEqual')
        # Processing the call keyword arguments (line 198)
        kwargs_186731 = {}
        # Getting the type of 'self' (line 198)
        self_186727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self', False)
        # Obtaining the member 'addTypeEqualityFunc' of a type (line 198)
        addTypeEqualityFunc_186728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_186727, 'addTypeEqualityFunc')
        # Calling addTypeEqualityFunc(args, kwargs) (line 198)
        addTypeEqualityFunc_call_result_186732 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), addTypeEqualityFunc_186728, *[list_186729, str_186730], **kwargs_186731)
        
        
        # Call to addTypeEqualityFunc(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'tuple' (line 199)
        tuple_186735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'tuple', False)
        str_186736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 40), 'str', 'assertTupleEqual')
        # Processing the call keyword arguments (line 199)
        kwargs_186737 = {}
        # Getting the type of 'self' (line 199)
        self_186733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self', False)
        # Obtaining the member 'addTypeEqualityFunc' of a type (line 199)
        addTypeEqualityFunc_186734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_186733, 'addTypeEqualityFunc')
        # Calling addTypeEqualityFunc(args, kwargs) (line 199)
        addTypeEqualityFunc_call_result_186738 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), addTypeEqualityFunc_186734, *[tuple_186735, str_186736], **kwargs_186737)
        
        
        # Call to addTypeEqualityFunc(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'set' (line 200)
        set_186741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 33), 'set', False)
        str_186742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 38), 'str', 'assertSetEqual')
        # Processing the call keyword arguments (line 200)
        kwargs_186743 = {}
        # Getting the type of 'self' (line 200)
        self_186739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self', False)
        # Obtaining the member 'addTypeEqualityFunc' of a type (line 200)
        addTypeEqualityFunc_186740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_186739, 'addTypeEqualityFunc')
        # Calling addTypeEqualityFunc(args, kwargs) (line 200)
        addTypeEqualityFunc_call_result_186744 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), addTypeEqualityFunc_186740, *[set_186741, str_186742], **kwargs_186743)
        
        
        # Call to addTypeEqualityFunc(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'frozenset' (line 201)
        frozenset_186747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'frozenset', False)
        str_186748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 44), 'str', 'assertSetEqual')
        # Processing the call keyword arguments (line 201)
        kwargs_186749 = {}
        # Getting the type of 'self' (line 201)
        self_186745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'self', False)
        # Obtaining the member 'addTypeEqualityFunc' of a type (line 201)
        addTypeEqualityFunc_186746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), self_186745, 'addTypeEqualityFunc')
        # Calling addTypeEqualityFunc(args, kwargs) (line 201)
        addTypeEqualityFunc_call_result_186750 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), addTypeEqualityFunc_186746, *[frozenset_186747, str_186748], **kwargs_186749)
        
        
        
        # SSA begins for try-except statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to addTypeEqualityFunc(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'unicode' (line 203)
        unicode_186753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 37), 'unicode', False)
        str_186754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 46), 'str', 'assertMultiLineEqual')
        # Processing the call keyword arguments (line 203)
        kwargs_186755 = {}
        # Getting the type of 'self' (line 203)
        self_186751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'self', False)
        # Obtaining the member 'addTypeEqualityFunc' of a type (line 203)
        addTypeEqualityFunc_186752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 12), self_186751, 'addTypeEqualityFunc')
        # Calling addTypeEqualityFunc(args, kwargs) (line 203)
        addTypeEqualityFunc_call_result_186756 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), addTypeEqualityFunc_186752, *[unicode_186753, str_186754], **kwargs_186755)
        
        # SSA branch for the except part of a try statement (line 202)
        # SSA branch for the except 'NameError' branch of a try statement (line 202)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def addTypeEqualityFunc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addTypeEqualityFunc'
        module_type_store = module_type_store.open_function_context('addTypeEqualityFunc', 208, 4, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_localization', localization)
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_function_name', 'TestCase.addTypeEqualityFunc')
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_param_names_list', ['typeobj', 'function'])
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.addTypeEqualityFunc.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.addTypeEqualityFunc', ['typeobj', 'function'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addTypeEqualityFunc', localization, ['typeobj', 'function'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addTypeEqualityFunc(...)' code ##################

        str_186757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'str', 'Add a type specific assertEqual style function to compare a type.\n\n        This method is for use by TestCase subclasses that need to register\n        their own type equality functions to provide nicer error messages.\n\n        Args:\n            typeobj: The data type to call this function on when both values\n                    are of the same type in assertEqual().\n            function: The callable taking two arguments and an optional\n                    msg= argument that raises self.failureException with a\n                    useful error message when the two arguments are not equal.\n        ')
        
        # Assigning a Name to a Subscript (line 221):
        
        # Assigning a Name to a Subscript (line 221):
        # Getting the type of 'function' (line 221)
        function_186758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 45), 'function')
        # Getting the type of 'self' (line 221)
        self_186759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self')
        # Obtaining the member '_type_equality_funcs' of a type (line 221)
        _type_equality_funcs_186760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_186759, '_type_equality_funcs')
        # Getting the type of 'typeobj' (line 221)
        typeobj_186761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 34), 'typeobj')
        # Storing an element on a container (line 221)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 8), _type_equality_funcs_186760, (typeobj_186761, function_186758))
        
        # ################# End of 'addTypeEqualityFunc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addTypeEqualityFunc' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_186762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186762)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addTypeEqualityFunc'
        return stypy_return_type_186762


    @norecursion
    def addCleanup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addCleanup'
        module_type_store = module_type_store.open_function_context('addCleanup', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.addCleanup.__dict__.__setitem__('stypy_localization', localization)
        TestCase.addCleanup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.addCleanup.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.addCleanup.__dict__.__setitem__('stypy_function_name', 'TestCase.addCleanup')
        TestCase.addCleanup.__dict__.__setitem__('stypy_param_names_list', ['function'])
        TestCase.addCleanup.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        TestCase.addCleanup.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        TestCase.addCleanup.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.addCleanup.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.addCleanup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.addCleanup.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.addCleanup', ['function'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addCleanup', localization, ['function'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addCleanup(...)' code ##################

        str_186763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, (-1)), 'str', 'Add a function, with arguments, to be called when the test is\n        completed. Functions added are called on a LIFO basis and are\n        called after tearDown on test failure or success.\n\n        Cleanup items are called even if setUp fails (unlike tearDown).')
        
        # Call to append(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Obtaining an instance of the builtin type 'tuple' (line 229)
        tuple_186767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 229)
        # Adding element type (line 229)
        # Getting the type of 'function' (line 229)
        function_186768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 31), 'function', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 31), tuple_186767, function_186768)
        # Adding element type (line 229)
        # Getting the type of 'args' (line 229)
        args_186769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 41), 'args', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 31), tuple_186767, args_186769)
        # Adding element type (line 229)
        # Getting the type of 'kwargs' (line 229)
        kwargs_186770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 47), 'kwargs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 31), tuple_186767, kwargs_186770)
        
        # Processing the call keyword arguments (line 229)
        kwargs_186771 = {}
        # Getting the type of 'self' (line 229)
        self_186764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self', False)
        # Obtaining the member '_cleanups' of a type (line 229)
        _cleanups_186765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_186764, '_cleanups')
        # Obtaining the member 'append' of a type (line 229)
        append_186766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), _cleanups_186765, 'append')
        # Calling append(args, kwargs) (line 229)
        append_call_result_186772 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), append_186766, *[tuple_186767], **kwargs_186771)
        
        
        # ################# End of 'addCleanup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addCleanup' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_186773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186773)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addCleanup'
        return stypy_return_type_186773


    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        TestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.setUp.__dict__.__setitem__('stypy_function_name', 'TestCase.setUp')
        TestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.setUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setUp(...)' code ##################

        str_186774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 8), 'str', 'Hook method for setting up the test fixture before exercising it.')
        pass
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_186775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186775)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_186775


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        TestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'TestCase.tearDown')
        TestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.tearDown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tearDown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tearDown(...)' code ##################

        str_186776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 8), 'str', 'Hook method for deconstructing the test fixture after testing it.')
        pass
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_186777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186777)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_186777


    @norecursion
    def setUpClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUpClass'
        module_type_store = module_type_store.open_function_context('setUpClass', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.setUpClass.__dict__.__setitem__('stypy_localization', localization)
        TestCase.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.setUpClass.__dict__.__setitem__('stypy_function_name', 'TestCase.setUpClass')
        TestCase.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.setUpClass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setUpClass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setUpClass(...)' code ##################

        str_186778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 8), 'str', 'Hook method for setting up class fixture before running tests in the class.')
        
        # ################# End of 'setUpClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUpClass' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_186779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186779)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUpClass'
        return stypy_return_type_186779


    @norecursion
    def tearDownClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDownClass'
        module_type_store = module_type_store.open_function_context('tearDownClass', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
        TestCase.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.tearDownClass.__dict__.__setitem__('stypy_function_name', 'TestCase.tearDownClass')
        TestCase.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.tearDownClass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tearDownClass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tearDownClass(...)' code ##################

        str_186780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 8), 'str', 'Hook method for deconstructing the class fixture after running all tests in the class.')
        
        # ################# End of 'tearDownClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDownClass' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_186781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDownClass'
        return stypy_return_type_186781


    @norecursion
    def countTestCases(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'countTestCases'
        module_type_store = module_type_store.open_function_context('countTestCases', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.countTestCases.__dict__.__setitem__('stypy_localization', localization)
        TestCase.countTestCases.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.countTestCases.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.countTestCases.__dict__.__setitem__('stypy_function_name', 'TestCase.countTestCases')
        TestCase.countTestCases.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.countTestCases.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.countTestCases.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.countTestCases.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.countTestCases.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.countTestCases.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.countTestCases.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.countTestCases', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'countTestCases', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'countTestCases(...)' code ##################

        int_186782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'stypy_return_type', int_186782)
        
        # ################# End of 'countTestCases(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'countTestCases' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_186783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186783)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'countTestCases'
        return stypy_return_type_186783


    @norecursion
    def defaultTestResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'defaultTestResult'
        module_type_store = module_type_store.open_function_context('defaultTestResult', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_localization', localization)
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_function_name', 'TestCase.defaultTestResult')
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.defaultTestResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.defaultTestResult', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'defaultTestResult', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'defaultTestResult(...)' code ##################

        
        # Call to TestResult(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_186786 = {}
        # Getting the type of 'result' (line 251)
        result_186784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'result', False)
        # Obtaining the member 'TestResult' of a type (line 251)
        TestResult_186785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), result_186784, 'TestResult')
        # Calling TestResult(args, kwargs) (line 251)
        TestResult_call_result_186787 = invoke(stypy.reporting.localization.Localization(__file__, 251, 15), TestResult_186785, *[], **kwargs_186786)
        
        # Assigning a type to the variable 'stypy_return_type' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'stypy_return_type', TestResult_call_result_186787)
        
        # ################# End of 'defaultTestResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'defaultTestResult' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_186788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186788)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'defaultTestResult'
        return stypy_return_type_186788


    @norecursion
    def shortDescription(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shortDescription'
        module_type_store = module_type_store.open_function_context('shortDescription', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.shortDescription.__dict__.__setitem__('stypy_localization', localization)
        TestCase.shortDescription.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.shortDescription.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.shortDescription.__dict__.__setitem__('stypy_function_name', 'TestCase.shortDescription')
        TestCase.shortDescription.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.shortDescription.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.shortDescription.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.shortDescription.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.shortDescription.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.shortDescription.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.shortDescription.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.shortDescription', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shortDescription', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shortDescription(...)' code ##################

        str_186789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, (-1)), 'str', "Returns a one-line description of the test, or None if no\n        description has been provided.\n\n        The default implementation of this method returns the first line of\n        the specified test method's docstring.\n        ")
        
        # Assigning a Attribute to a Name (line 260):
        
        # Assigning a Attribute to a Name (line 260):
        # Getting the type of 'self' (line 260)
        self_186790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 14), 'self')
        # Obtaining the member '_testMethodDoc' of a type (line 260)
        _testMethodDoc_186791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 14), self_186790, '_testMethodDoc')
        # Assigning a type to the variable 'doc' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'doc', _testMethodDoc_186791)
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'doc' (line 261)
        doc_186792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'doc')
        
        # Call to strip(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_186802 = {}
        
        # Obtaining the type of the subscript
        int_186793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 39), 'int')
        
        # Call to split(...): (line 261)
        # Processing the call arguments (line 261)
        str_186796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 33), 'str', '\n')
        # Processing the call keyword arguments (line 261)
        kwargs_186797 = {}
        # Getting the type of 'doc' (line 261)
        doc_186794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'doc', False)
        # Obtaining the member 'split' of a type (line 261)
        split_186795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 23), doc_186794, 'split')
        # Calling split(args, kwargs) (line 261)
        split_call_result_186798 = invoke(stypy.reporting.localization.Localization(__file__, 261, 23), split_186795, *[str_186796], **kwargs_186797)
        
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___186799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 23), split_call_result_186798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_186800 = invoke(stypy.reporting.localization.Localization(__file__, 261, 23), getitem___186799, int_186793)
        
        # Obtaining the member 'strip' of a type (line 261)
        strip_186801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 23), subscript_call_result_186800, 'strip')
        # Calling strip(args, kwargs) (line 261)
        strip_call_result_186803 = invoke(stypy.reporting.localization.Localization(__file__, 261, 23), strip_186801, *[], **kwargs_186802)
        
        # Applying the binary operator 'and' (line 261)
        result_and_keyword_186804 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 15), 'and', doc_186792, strip_call_result_186803)
        
        # Getting the type of 'None' (line 261)
        None_186805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 53), 'None')
        # Applying the binary operator 'or' (line 261)
        result_or_keyword_186806 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 15), 'or', result_and_keyword_186804, None_186805)
        
        # Assigning a type to the variable 'stypy_return_type' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'stypy_return_type', result_or_keyword_186806)
        
        # ################# End of 'shortDescription(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shortDescription' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_186807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186807)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shortDescription'
        return stypy_return_type_186807


    @norecursion
    def id(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'id'
        module_type_store = module_type_store.open_function_context('id', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.id.__dict__.__setitem__('stypy_localization', localization)
        TestCase.id.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.id.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.id.__dict__.__setitem__('stypy_function_name', 'TestCase.id')
        TestCase.id.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.id.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.id.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.id.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.id.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.id.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.id.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.id', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'id', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'id(...)' code ##################

        str_186808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 15), 'str', '%s.%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 265)
        tuple_186809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 265)
        # Adding element type (line 265)
        
        # Call to strclass(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'self' (line 265)
        self_186811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'self', False)
        # Obtaining the member '__class__' of a type (line 265)
        class___186812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 35), self_186811, '__class__')
        # Processing the call keyword arguments (line 265)
        kwargs_186813 = {}
        # Getting the type of 'strclass' (line 265)
        strclass_186810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 26), 'strclass', False)
        # Calling strclass(args, kwargs) (line 265)
        strclass_call_result_186814 = invoke(stypy.reporting.localization.Localization(__file__, 265, 26), strclass_186810, *[class___186812], **kwargs_186813)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 26), tuple_186809, strclass_call_result_186814)
        # Adding element type (line 265)
        # Getting the type of 'self' (line 265)
        self_186815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 52), 'self')
        # Obtaining the member '_testMethodName' of a type (line 265)
        _testMethodName_186816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 52), self_186815, '_testMethodName')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 26), tuple_186809, _testMethodName_186816)
        
        # Applying the binary operator '%' (line 265)
        result_mod_186817 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), '%', str_186808, tuple_186809)
        
        # Assigning a type to the variable 'stypy_return_type' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'stypy_return_type', result_mod_186817)
        
        # ################# End of 'id(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'id' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_186818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186818)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'id'
        return stypy_return_type_186818


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 267, 4, False)
        # Assigning a type to the variable 'self' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'TestCase.__eq__')
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 268)
        # Getting the type of 'self' (line 268)
        self_186819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'self')
        
        # Call to type(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'other' (line 268)
        other_186821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 34), 'other', False)
        # Processing the call keyword arguments (line 268)
        kwargs_186822 = {}
        # Getting the type of 'type' (line 268)
        type_186820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'type', False)
        # Calling type(args, kwargs) (line 268)
        type_call_result_186823 = invoke(stypy.reporting.localization.Localization(__file__, 268, 29), type_186820, *[other_186821], **kwargs_186822)
        
        
        (may_be_186824, more_types_in_union_186825) = may_not_be_type(self_186819, type_call_result_186823)

        if may_be_186824:

            if more_types_in_union_186825:
                # Runtime conditional SSA (line 268)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 268)
            self_186826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self')
            # Assigning a type to the variable 'self' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self', remove_type_from_union(self_186826, type_call_result_186823))
            # Getting the type of 'NotImplemented' (line 269)
            NotImplemented_186827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'NotImplemented')
            # Assigning a type to the variable 'stypy_return_type' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'stypy_return_type', NotImplemented_186827)

            if more_types_in_union_186825:
                # SSA join for if statement (line 268)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 271)
        self_186828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'self')
        # Obtaining the member '_testMethodName' of a type (line 271)
        _testMethodName_186829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 15), self_186828, '_testMethodName')
        # Getting the type of 'other' (line 271)
        other_186830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 39), 'other')
        # Obtaining the member '_testMethodName' of a type (line 271)
        _testMethodName_186831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 39), other_186830, '_testMethodName')
        # Applying the binary operator '==' (line 271)
        result_eq_186832 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 15), '==', _testMethodName_186829, _testMethodName_186831)
        
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'stypy_return_type', result_eq_186832)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 267)
        stypy_return_type_186833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186833)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_186833


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.__ne__.__dict__.__setitem__('stypy_localization', localization)
        TestCase.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.__ne__.__dict__.__setitem__('stypy_function_name', 'TestCase.__ne__')
        TestCase.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        TestCase.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.__ne__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 274)
        self_186834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'self')
        # Getting the type of 'other' (line 274)
        other_186835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 27), 'other')
        # Applying the binary operator '==' (line 274)
        result_eq_186836 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 19), '==', self_186834, other_186835)
        
        # Applying the 'not' unary operator (line 274)
        result_not__186837 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 15), 'not', result_eq_186836)
        
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'stypy_return_type', result_not__186837)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_186838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186838)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_186838


    @norecursion
    def stypy__hash__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__hash__'
        module_type_store = module_type_store.open_function_context('__hash__', 276, 4, False)
        # Assigning a type to the variable 'self' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_localization', localization)
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_function_name', 'TestCase.__hash__')
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.stypy__hash__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.__hash__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__hash__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__hash__(...)' code ##################

        
        # Call to hash(...): (line 277)
        # Processing the call arguments (line 277)
        
        # Obtaining an instance of the builtin type 'tuple' (line 277)
        tuple_186840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 277)
        # Adding element type (line 277)
        
        # Call to type(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'self' (line 277)
        self_186842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 26), 'self', False)
        # Processing the call keyword arguments (line 277)
        kwargs_186843 = {}
        # Getting the type of 'type' (line 277)
        type_186841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'type', False)
        # Calling type(args, kwargs) (line 277)
        type_call_result_186844 = invoke(stypy.reporting.localization.Localization(__file__, 277, 21), type_186841, *[self_186842], **kwargs_186843)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 21), tuple_186840, type_call_result_186844)
        # Adding element type (line 277)
        # Getting the type of 'self' (line 277)
        self_186845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 33), 'self', False)
        # Obtaining the member '_testMethodName' of a type (line 277)
        _testMethodName_186846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 33), self_186845, '_testMethodName')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 21), tuple_186840, _testMethodName_186846)
        
        # Processing the call keyword arguments (line 277)
        kwargs_186847 = {}
        # Getting the type of 'hash' (line 277)
        hash_186839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'hash', False)
        # Calling hash(args, kwargs) (line 277)
        hash_call_result_186848 = invoke(stypy.reporting.localization.Localization(__file__, 277, 15), hash_186839, *[tuple_186840], **kwargs_186847)
        
        # Assigning a type to the variable 'stypy_return_type' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'stypy_return_type', hash_call_result_186848)
        
        # ################# End of '__hash__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__hash__' in the type store
        # Getting the type of 'stypy_return_type' (line 276)
        stypy_return_type_186849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__hash__'
        return stypy_return_type_186849


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        TestCase.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.stypy__str__.__dict__.__setitem__('stypy_function_name', 'TestCase.__str__')
        TestCase.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.__str__', [], None, None, defaults, varargs, kwargs)

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

        str_186850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 15), 'str', '%s (%s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 280)
        tuple_186851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 280)
        # Adding element type (line 280)
        # Getting the type of 'self' (line 280)
        self_186852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 28), 'self')
        # Obtaining the member '_testMethodName' of a type (line 280)
        _testMethodName_186853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 28), self_186852, '_testMethodName')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 28), tuple_186851, _testMethodName_186853)
        # Adding element type (line 280)
        
        # Call to strclass(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'self' (line 280)
        self_186855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 59), 'self', False)
        # Obtaining the member '__class__' of a type (line 280)
        class___186856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 59), self_186855, '__class__')
        # Processing the call keyword arguments (line 280)
        kwargs_186857 = {}
        # Getting the type of 'strclass' (line 280)
        strclass_186854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 50), 'strclass', False)
        # Calling strclass(args, kwargs) (line 280)
        strclass_call_result_186858 = invoke(stypy.reporting.localization.Localization(__file__, 280, 50), strclass_186854, *[class___186856], **kwargs_186857)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 28), tuple_186851, strclass_call_result_186858)
        
        # Applying the binary operator '%' (line 280)
        result_mod_186859 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), '%', str_186850, tuple_186851)
        
        # Assigning a type to the variable 'stypy_return_type' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type', result_mod_186859)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_186860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186860)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_186860


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 282, 4, False)
        # Assigning a type to the variable 'self' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'TestCase.__repr__')
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_186861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 15), 'str', '<%s testMethod=%s>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_186862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        # Adding element type (line 284)
        
        # Call to strclass(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'self' (line 284)
        self_186864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'self', False)
        # Obtaining the member '__class__' of a type (line 284)
        class___186865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 25), self_186864, '__class__')
        # Processing the call keyword arguments (line 284)
        kwargs_186866 = {}
        # Getting the type of 'strclass' (line 284)
        strclass_186863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'strclass', False)
        # Calling strclass(args, kwargs) (line 284)
        strclass_call_result_186867 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), strclass_186863, *[class___186865], **kwargs_186866)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 16), tuple_186862, strclass_call_result_186867)
        # Adding element type (line 284)
        # Getting the type of 'self' (line 284)
        self_186868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 42), 'self')
        # Obtaining the member '_testMethodName' of a type (line 284)
        _testMethodName_186869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 42), self_186868, '_testMethodName')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 16), tuple_186862, _testMethodName_186869)
        
        # Applying the binary operator '%' (line 283)
        result_mod_186870 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 15), '%', str_186861, tuple_186862)
        
        # Assigning a type to the variable 'stypy_return_type' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'stypy_return_type', result_mod_186870)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 282)
        stypy_return_type_186871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186871)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_186871


    @norecursion
    def _addSkip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_addSkip'
        module_type_store = module_type_store.open_function_context('_addSkip', 286, 4, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase._addSkip.__dict__.__setitem__('stypy_localization', localization)
        TestCase._addSkip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase._addSkip.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase._addSkip.__dict__.__setitem__('stypy_function_name', 'TestCase._addSkip')
        TestCase._addSkip.__dict__.__setitem__('stypy_param_names_list', ['result', 'reason'])
        TestCase._addSkip.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase._addSkip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase._addSkip.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase._addSkip.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase._addSkip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase._addSkip.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase._addSkip', ['result', 'reason'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_addSkip', localization, ['result', 'reason'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_addSkip(...)' code ##################

        
        # Assigning a Call to a Name (line 287):
        
        # Assigning a Call to a Name (line 287):
        
        # Call to getattr(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'result' (line 287)
        result_186873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'result', False)
        str_186874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 34), 'str', 'addSkip')
        # Getting the type of 'None' (line 287)
        None_186875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 45), 'None', False)
        # Processing the call keyword arguments (line 287)
        kwargs_186876 = {}
        # Getting the type of 'getattr' (line 287)
        getattr_186872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 287)
        getattr_call_result_186877 = invoke(stypy.reporting.localization.Localization(__file__, 287, 18), getattr_186872, *[result_186873, str_186874, None_186875], **kwargs_186876)
        
        # Assigning a type to the variable 'addSkip' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'addSkip', getattr_call_result_186877)
        
        # Type idiom detected: calculating its left and rigth part (line 288)
        # Getting the type of 'addSkip' (line 288)
        addSkip_186878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'addSkip')
        # Getting the type of 'None' (line 288)
        None_186879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 26), 'None')
        
        (may_be_186880, more_types_in_union_186881) = may_not_be_none(addSkip_186878, None_186879)

        if may_be_186880:

            if more_types_in_union_186881:
                # Runtime conditional SSA (line 288)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to addSkip(...): (line 289)
            # Processing the call arguments (line 289)
            # Getting the type of 'self' (line 289)
            self_186883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'self', False)
            # Getting the type of 'reason' (line 289)
            reason_186884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'reason', False)
            # Processing the call keyword arguments (line 289)
            kwargs_186885 = {}
            # Getting the type of 'addSkip' (line 289)
            addSkip_186882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'addSkip', False)
            # Calling addSkip(args, kwargs) (line 289)
            addSkip_call_result_186886 = invoke(stypy.reporting.localization.Localization(__file__, 289, 12), addSkip_186882, *[self_186883, reason_186884], **kwargs_186885)
            

            if more_types_in_union_186881:
                # Runtime conditional SSA for else branch (line 288)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_186880) or more_types_in_union_186881):
            
            # Call to warn(...): (line 291)
            # Processing the call arguments (line 291)
            str_186889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 26), 'str', 'TestResult has no addSkip method, skips not reported')
            # Getting the type of 'RuntimeWarning' (line 292)
            RuntimeWarning_186890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'RuntimeWarning', False)
            int_186891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 42), 'int')
            # Processing the call keyword arguments (line 291)
            kwargs_186892 = {}
            # Getting the type of 'warnings' (line 291)
            warnings_186887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 291)
            warn_186888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), warnings_186887, 'warn')
            # Calling warn(args, kwargs) (line 291)
            warn_call_result_186893 = invoke(stypy.reporting.localization.Localization(__file__, 291, 12), warn_186888, *[str_186889, RuntimeWarning_186890, int_186891], **kwargs_186892)
            
            
            # Call to addSuccess(...): (line 293)
            # Processing the call arguments (line 293)
            # Getting the type of 'self' (line 293)
            self_186896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'self', False)
            # Processing the call keyword arguments (line 293)
            kwargs_186897 = {}
            # Getting the type of 'result' (line 293)
            result_186894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'result', False)
            # Obtaining the member 'addSuccess' of a type (line 293)
            addSuccess_186895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), result_186894, 'addSuccess')
            # Calling addSuccess(args, kwargs) (line 293)
            addSuccess_call_result_186898 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), addSuccess_186895, *[self_186896], **kwargs_186897)
            

            if (may_be_186880 and more_types_in_union_186881):
                # SSA join for if statement (line 288)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_addSkip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_addSkip' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_186899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_186899)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_addSkip'
        return stypy_return_type_186899


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 295)
        None_186900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'None')
        defaults = [None_186900]
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 295, 4, False)
        # Assigning a type to the variable 'self' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.run.__dict__.__setitem__('stypy_localization', localization)
        TestCase.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.run.__dict__.__setitem__('stypy_function_name', 'TestCase.run')
        TestCase.run.__dict__.__setitem__('stypy_param_names_list', ['result'])
        TestCase.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.run.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.run', ['result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, ['result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Assigning a Name to a Name (line 296):
        
        # Assigning a Name to a Name (line 296):
        # Getting the type of 'result' (line 296)
        result_186901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'result')
        # Assigning a type to the variable 'orig_result' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'orig_result', result_186901)
        
        # Type idiom detected: calculating its left and rigth part (line 297)
        # Getting the type of 'result' (line 297)
        result_186902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'result')
        # Getting the type of 'None' (line 297)
        None_186903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'None')
        
        (may_be_186904, more_types_in_union_186905) = may_be_none(result_186902, None_186903)

        if may_be_186904:

            if more_types_in_union_186905:
                # Runtime conditional SSA (line 297)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 298):
            
            # Assigning a Call to a Name (line 298):
            
            # Call to defaultTestResult(...): (line 298)
            # Processing the call keyword arguments (line 298)
            kwargs_186908 = {}
            # Getting the type of 'self' (line 298)
            self_186906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'self', False)
            # Obtaining the member 'defaultTestResult' of a type (line 298)
            defaultTestResult_186907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 21), self_186906, 'defaultTestResult')
            # Calling defaultTestResult(args, kwargs) (line 298)
            defaultTestResult_call_result_186909 = invoke(stypy.reporting.localization.Localization(__file__, 298, 21), defaultTestResult_186907, *[], **kwargs_186908)
            
            # Assigning a type to the variable 'result' (line 298)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'result', defaultTestResult_call_result_186909)
            
            # Assigning a Call to a Name (line 299):
            
            # Assigning a Call to a Name (line 299):
            
            # Call to getattr(...): (line 299)
            # Processing the call arguments (line 299)
            # Getting the type of 'result' (line 299)
            result_186911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 35), 'result', False)
            str_186912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 43), 'str', 'startTestRun')
            # Getting the type of 'None' (line 299)
            None_186913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 59), 'None', False)
            # Processing the call keyword arguments (line 299)
            kwargs_186914 = {}
            # Getting the type of 'getattr' (line 299)
            getattr_186910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 27), 'getattr', False)
            # Calling getattr(args, kwargs) (line 299)
            getattr_call_result_186915 = invoke(stypy.reporting.localization.Localization(__file__, 299, 27), getattr_186910, *[result_186911, str_186912, None_186913], **kwargs_186914)
            
            # Assigning a type to the variable 'startTestRun' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'startTestRun', getattr_call_result_186915)
            
            # Type idiom detected: calculating its left and rigth part (line 300)
            # Getting the type of 'startTestRun' (line 300)
            startTestRun_186916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'startTestRun')
            # Getting the type of 'None' (line 300)
            None_186917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 35), 'None')
            
            (may_be_186918, more_types_in_union_186919) = may_not_be_none(startTestRun_186916, None_186917)

            if may_be_186918:

                if more_types_in_union_186919:
                    # Runtime conditional SSA (line 300)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to startTestRun(...): (line 301)
                # Processing the call keyword arguments (line 301)
                kwargs_186921 = {}
                # Getting the type of 'startTestRun' (line 301)
                startTestRun_186920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'startTestRun', False)
                # Calling startTestRun(args, kwargs) (line 301)
                startTestRun_call_result_186922 = invoke(stypy.reporting.localization.Localization(__file__, 301, 16), startTestRun_186920, *[], **kwargs_186921)
                

                if more_types_in_union_186919:
                    # SSA join for if statement (line 300)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_186905:
                # SSA join for if statement (line 297)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 303):
        
        # Assigning a Name to a Attribute (line 303):
        # Getting the type of 'result' (line 303)
        result_186923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 36), 'result')
        # Getting the type of 'self' (line 303)
        self_186924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'self')
        # Setting the type of the member '_resultForDoCleanups' of a type (line 303)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), self_186924, '_resultForDoCleanups', result_186923)
        
        # Call to startTest(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'self' (line 304)
        self_186927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'self', False)
        # Processing the call keyword arguments (line 304)
        kwargs_186928 = {}
        # Getting the type of 'result' (line 304)
        result_186925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'result', False)
        # Obtaining the member 'startTest' of a type (line 304)
        startTest_186926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), result_186925, 'startTest')
        # Calling startTest(args, kwargs) (line 304)
        startTest_call_result_186929 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), startTest_186926, *[self_186927], **kwargs_186928)
        
        
        # Assigning a Call to a Name (line 306):
        
        # Assigning a Call to a Name (line 306):
        
        # Call to getattr(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'self' (line 306)
        self_186931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 29), 'self', False)
        # Getting the type of 'self' (line 306)
        self_186932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 35), 'self', False)
        # Obtaining the member '_testMethodName' of a type (line 306)
        _testMethodName_186933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 35), self_186932, '_testMethodName')
        # Processing the call keyword arguments (line 306)
        kwargs_186934 = {}
        # Getting the type of 'getattr' (line 306)
        getattr_186930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'getattr', False)
        # Calling getattr(args, kwargs) (line 306)
        getattr_call_result_186935 = invoke(stypy.reporting.localization.Localization(__file__, 306, 21), getattr_186930, *[self_186931, _testMethodName_186933], **kwargs_186934)
        
        # Assigning a type to the variable 'testMethod' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'testMethod', getattr_call_result_186935)
        
        
        # Evaluating a boolean operation
        
        # Call to getattr(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'self' (line 307)
        self_186937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'self', False)
        # Obtaining the member '__class__' of a type (line 307)
        class___186938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 20), self_186937, '__class__')
        str_186939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 36), 'str', '__unittest_skip__')
        # Getting the type of 'False' (line 307)
        False_186940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 57), 'False', False)
        # Processing the call keyword arguments (line 307)
        kwargs_186941 = {}
        # Getting the type of 'getattr' (line 307)
        getattr_186936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'getattr', False)
        # Calling getattr(args, kwargs) (line 307)
        getattr_call_result_186942 = invoke(stypy.reporting.localization.Localization(__file__, 307, 12), getattr_186936, *[class___186938, str_186939, False_186940], **kwargs_186941)
        
        
        # Call to getattr(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'testMethod' (line 308)
        testMethod_186944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'testMethod', False)
        str_186945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 32), 'str', '__unittest_skip__')
        # Getting the type of 'False' (line 308)
        False_186946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 53), 'False', False)
        # Processing the call keyword arguments (line 308)
        kwargs_186947 = {}
        # Getting the type of 'getattr' (line 308)
        getattr_186943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'getattr', False)
        # Calling getattr(args, kwargs) (line 308)
        getattr_call_result_186948 = invoke(stypy.reporting.localization.Localization(__file__, 308, 12), getattr_186943, *[testMethod_186944, str_186945, False_186946], **kwargs_186947)
        
        # Applying the binary operator 'or' (line 307)
        result_or_keyword_186949 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 12), 'or', getattr_call_result_186942, getattr_call_result_186948)
        
        # Testing the type of an if condition (line 307)
        if_condition_186950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 8), result_or_keyword_186949)
        # Assigning a type to the variable 'if_condition_186950' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'if_condition_186950', if_condition_186950)
        # SSA begins for if statement (line 307)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Try-finally block (line 310)
        
        # Assigning a BoolOp to a Name (line 311):
        
        # Assigning a BoolOp to a Name (line 311):
        
        # Evaluating a boolean operation
        
        # Call to getattr(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'self' (line 311)
        self_186952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 36), 'self', False)
        # Obtaining the member '__class__' of a type (line 311)
        class___186953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 36), self_186952, '__class__')
        str_186954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 52), 'str', '__unittest_skip_why__')
        str_186955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 77), 'str', '')
        # Processing the call keyword arguments (line 311)
        kwargs_186956 = {}
        # Getting the type of 'getattr' (line 311)
        getattr_186951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 28), 'getattr', False)
        # Calling getattr(args, kwargs) (line 311)
        getattr_call_result_186957 = invoke(stypy.reporting.localization.Localization(__file__, 311, 28), getattr_186951, *[class___186953, str_186954, str_186955], **kwargs_186956)
        
        
        # Call to getattr(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'testMethod' (line 312)
        testMethod_186959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 39), 'testMethod', False)
        str_186960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 51), 'str', '__unittest_skip_why__')
        str_186961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 76), 'str', '')
        # Processing the call keyword arguments (line 312)
        kwargs_186962 = {}
        # Getting the type of 'getattr' (line 312)
        getattr_186958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 31), 'getattr', False)
        # Calling getattr(args, kwargs) (line 312)
        getattr_call_result_186963 = invoke(stypy.reporting.localization.Localization(__file__, 312, 31), getattr_186958, *[testMethod_186959, str_186960, str_186961], **kwargs_186962)
        
        # Applying the binary operator 'or' (line 311)
        result_or_keyword_186964 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 28), 'or', getattr_call_result_186957, getattr_call_result_186963)
        
        # Assigning a type to the variable 'skip_why' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'skip_why', result_or_keyword_186964)
        
        # Call to _addSkip(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'result' (line 313)
        result_186967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), 'result', False)
        # Getting the type of 'skip_why' (line 313)
        skip_why_186968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 38), 'skip_why', False)
        # Processing the call keyword arguments (line 313)
        kwargs_186969 = {}
        # Getting the type of 'self' (line 313)
        self_186965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'self', False)
        # Obtaining the member '_addSkip' of a type (line 313)
        _addSkip_186966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 16), self_186965, '_addSkip')
        # Calling _addSkip(args, kwargs) (line 313)
        _addSkip_call_result_186970 = invoke(stypy.reporting.localization.Localization(__file__, 313, 16), _addSkip_186966, *[result_186967, skip_why_186968], **kwargs_186969)
        
        
        # finally branch of the try-finally block (line 310)
        
        # Call to stopTest(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'self' (line 315)
        self_186973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 32), 'self', False)
        # Processing the call keyword arguments (line 315)
        kwargs_186974 = {}
        # Getting the type of 'result' (line 315)
        result_186971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'result', False)
        # Obtaining the member 'stopTest' of a type (line 315)
        stopTest_186972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 16), result_186971, 'stopTest')
        # Calling stopTest(args, kwargs) (line 315)
        stopTest_call_result_186975 = invoke(stypy.reporting.localization.Localization(__file__, 315, 16), stopTest_186972, *[self_186973], **kwargs_186974)
        
        
        # Assigning a type to the variable 'stypy_return_type' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 307)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Try-finally block (line 317)
        
        # Assigning a Name to a Name (line 318):
        
        # Assigning a Name to a Name (line 318):
        # Getting the type of 'False' (line 318)
        False_186976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 22), 'False')
        # Assigning a type to the variable 'success' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'success', False_186976)
        
        
        # SSA begins for try-except statement (line 319)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to setUp(...): (line 320)
        # Processing the call keyword arguments (line 320)
        kwargs_186979 = {}
        # Getting the type of 'self' (line 320)
        self_186977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'self', False)
        # Obtaining the member 'setUp' of a type (line 320)
        setUp_186978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 16), self_186977, 'setUp')
        # Calling setUp(args, kwargs) (line 320)
        setUp_call_result_186980 = invoke(stypy.reporting.localization.Localization(__file__, 320, 16), setUp_186978, *[], **kwargs_186979)
        
        # SSA branch for the except part of a try statement (line 319)
        # SSA branch for the except 'SkipTest' branch of a try statement (line 319)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'SkipTest' (line 321)
        SkipTest_186981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 'SkipTest')
        # Assigning a type to the variable 'e' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'e', SkipTest_186981)
        
        # Call to _addSkip(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'result' (line 322)
        result_186984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 30), 'result', False)
        
        # Call to str(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'e' (line 322)
        e_186986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 42), 'e', False)
        # Processing the call keyword arguments (line 322)
        kwargs_186987 = {}
        # Getting the type of 'str' (line 322)
        str_186985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 38), 'str', False)
        # Calling str(args, kwargs) (line 322)
        str_call_result_186988 = invoke(stypy.reporting.localization.Localization(__file__, 322, 38), str_186985, *[e_186986], **kwargs_186987)
        
        # Processing the call keyword arguments (line 322)
        kwargs_186989 = {}
        # Getting the type of 'self' (line 322)
        self_186982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 16), 'self', False)
        # Obtaining the member '_addSkip' of a type (line 322)
        _addSkip_186983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 16), self_186982, '_addSkip')
        # Calling _addSkip(args, kwargs) (line 322)
        _addSkip_call_result_186990 = invoke(stypy.reporting.localization.Localization(__file__, 322, 16), _addSkip_186983, *[result_186984, str_call_result_186988], **kwargs_186989)
        
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 319)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the except '<any exception>' branch of a try statement (line 319)
        module_type_store.open_ssa_branch('except')
        
        # Call to addError(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'self' (line 326)
        self_186993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 'self', False)
        
        # Call to exc_info(...): (line 326)
        # Processing the call keyword arguments (line 326)
        kwargs_186996 = {}
        # Getting the type of 'sys' (line 326)
        sys_186994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 38), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 326)
        exc_info_186995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 38), sys_186994, 'exc_info')
        # Calling exc_info(args, kwargs) (line 326)
        exc_info_call_result_186997 = invoke(stypy.reporting.localization.Localization(__file__, 326, 38), exc_info_186995, *[], **kwargs_186996)
        
        # Processing the call keyword arguments (line 326)
        kwargs_186998 = {}
        # Getting the type of 'result' (line 326)
        result_186991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'result', False)
        # Obtaining the member 'addError' of a type (line 326)
        addError_186992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 16), result_186991, 'addError')
        # Calling addError(args, kwargs) (line 326)
        addError_call_result_186999 = invoke(stypy.reporting.localization.Localization(__file__, 326, 16), addError_186992, *[self_186993, exc_info_call_result_186997], **kwargs_186998)
        
        # SSA branch for the else branch of a try statement (line 319)
        module_type_store.open_ssa_branch('except else')
        
        
        # SSA begins for try-except statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to testMethod(...): (line 329)
        # Processing the call keyword arguments (line 329)
        kwargs_187001 = {}
        # Getting the type of 'testMethod' (line 329)
        testMethod_187000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'testMethod', False)
        # Calling testMethod(args, kwargs) (line 329)
        testMethod_call_result_187002 = invoke(stypy.reporting.localization.Localization(__file__, 329, 20), testMethod_187000, *[], **kwargs_187001)
        
        # SSA branch for the except part of a try statement (line 328)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 328)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the except 'Attribute' branch of a try statement (line 328)
        module_type_store.open_ssa_branch('except')
        
        # Call to addFailure(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'self' (line 333)
        self_187005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 38), 'self', False)
        
        # Call to exc_info(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_187008 = {}
        # Getting the type of 'sys' (line 333)
        sys_187006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 44), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 333)
        exc_info_187007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 44), sys_187006, 'exc_info')
        # Calling exc_info(args, kwargs) (line 333)
        exc_info_call_result_187009 = invoke(stypy.reporting.localization.Localization(__file__, 333, 44), exc_info_187007, *[], **kwargs_187008)
        
        # Processing the call keyword arguments (line 333)
        kwargs_187010 = {}
        # Getting the type of 'result' (line 333)
        result_187003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'result', False)
        # Obtaining the member 'addFailure' of a type (line 333)
        addFailure_187004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 20), result_187003, 'addFailure')
        # Calling addFailure(args, kwargs) (line 333)
        addFailure_call_result_187011 = invoke(stypy.reporting.localization.Localization(__file__, 333, 20), addFailure_187004, *[self_187005, exc_info_call_result_187009], **kwargs_187010)
        
        # SSA branch for the except '_ExpectedFailure' branch of a try statement (line 328)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of '_ExpectedFailure' (line 334)
        _ExpectedFailure_187012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 23), '_ExpectedFailure')
        # Assigning a type to the variable 'e' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'e', _ExpectedFailure_187012)
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to getattr(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'result' (line 335)
        result_187014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 49), 'result', False)
        str_187015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 57), 'str', 'addExpectedFailure')
        # Getting the type of 'None' (line 335)
        None_187016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 79), 'None', False)
        # Processing the call keyword arguments (line 335)
        kwargs_187017 = {}
        # Getting the type of 'getattr' (line 335)
        getattr_187013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 41), 'getattr', False)
        # Calling getattr(args, kwargs) (line 335)
        getattr_call_result_187018 = invoke(stypy.reporting.localization.Localization(__file__, 335, 41), getattr_187013, *[result_187014, str_187015, None_187016], **kwargs_187017)
        
        # Assigning a type to the variable 'addExpectedFailure' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'addExpectedFailure', getattr_call_result_187018)
        
        # Type idiom detected: calculating its left and rigth part (line 336)
        # Getting the type of 'addExpectedFailure' (line 336)
        addExpectedFailure_187019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'addExpectedFailure')
        # Getting the type of 'None' (line 336)
        None_187020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 49), 'None')
        
        (may_be_187021, more_types_in_union_187022) = may_not_be_none(addExpectedFailure_187019, None_187020)

        if may_be_187021:

            if more_types_in_union_187022:
                # Runtime conditional SSA (line 336)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to addExpectedFailure(...): (line 337)
            # Processing the call arguments (line 337)
            # Getting the type of 'self' (line 337)
            self_187024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 43), 'self', False)
            # Getting the type of 'e' (line 337)
            e_187025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 49), 'e', False)
            # Obtaining the member 'exc_info' of a type (line 337)
            exc_info_187026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 49), e_187025, 'exc_info')
            # Processing the call keyword arguments (line 337)
            kwargs_187027 = {}
            # Getting the type of 'addExpectedFailure' (line 337)
            addExpectedFailure_187023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'addExpectedFailure', False)
            # Calling addExpectedFailure(args, kwargs) (line 337)
            addExpectedFailure_call_result_187028 = invoke(stypy.reporting.localization.Localization(__file__, 337, 24), addExpectedFailure_187023, *[self_187024, exc_info_187026], **kwargs_187027)
            

            if more_types_in_union_187022:
                # Runtime conditional SSA for else branch (line 336)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_187021) or more_types_in_union_187022):
            
            # Call to warn(...): (line 339)
            # Processing the call arguments (line 339)
            str_187031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 38), 'str', 'TestResult has no addExpectedFailure method, reporting as passes')
            # Getting the type of 'RuntimeWarning' (line 340)
            RuntimeWarning_187032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 38), 'RuntimeWarning', False)
            # Processing the call keyword arguments (line 339)
            kwargs_187033 = {}
            # Getting the type of 'warnings' (line 339)
            warnings_187029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 24), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 339)
            warn_187030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 24), warnings_187029, 'warn')
            # Calling warn(args, kwargs) (line 339)
            warn_call_result_187034 = invoke(stypy.reporting.localization.Localization(__file__, 339, 24), warn_187030, *[str_187031, RuntimeWarning_187032], **kwargs_187033)
            
            
            # Call to addSuccess(...): (line 341)
            # Processing the call arguments (line 341)
            # Getting the type of 'self' (line 341)
            self_187037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 42), 'self', False)
            # Processing the call keyword arguments (line 341)
            kwargs_187038 = {}
            # Getting the type of 'result' (line 341)
            result_187035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 24), 'result', False)
            # Obtaining the member 'addSuccess' of a type (line 341)
            addSuccess_187036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 24), result_187035, 'addSuccess')
            # Calling addSuccess(args, kwargs) (line 341)
            addSuccess_call_result_187039 = invoke(stypy.reporting.localization.Localization(__file__, 341, 24), addSuccess_187036, *[self_187037], **kwargs_187038)
            

            if (may_be_187021 and more_types_in_union_187022):
                # SSA join for if statement (line 336)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the except '_UnexpectedSuccess' branch of a try statement (line 328)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to getattr(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'result' (line 343)
        result_187041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 51), 'result', False)
        str_187042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 59), 'str', 'addUnexpectedSuccess')
        # Getting the type of 'None' (line 343)
        None_187043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 83), 'None', False)
        # Processing the call keyword arguments (line 343)
        kwargs_187044 = {}
        # Getting the type of 'getattr' (line 343)
        getattr_187040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 43), 'getattr', False)
        # Calling getattr(args, kwargs) (line 343)
        getattr_call_result_187045 = invoke(stypy.reporting.localization.Localization(__file__, 343, 43), getattr_187040, *[result_187041, str_187042, None_187043], **kwargs_187044)
        
        # Assigning a type to the variable 'addUnexpectedSuccess' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'addUnexpectedSuccess', getattr_call_result_187045)
        
        # Type idiom detected: calculating its left and rigth part (line 344)
        # Getting the type of 'addUnexpectedSuccess' (line 344)
        addUnexpectedSuccess_187046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'addUnexpectedSuccess')
        # Getting the type of 'None' (line 344)
        None_187047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 51), 'None')
        
        (may_be_187048, more_types_in_union_187049) = may_not_be_none(addUnexpectedSuccess_187046, None_187047)

        if may_be_187048:

            if more_types_in_union_187049:
                # Runtime conditional SSA (line 344)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to addUnexpectedSuccess(...): (line 345)
            # Processing the call arguments (line 345)
            # Getting the type of 'self' (line 345)
            self_187051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 45), 'self', False)
            # Processing the call keyword arguments (line 345)
            kwargs_187052 = {}
            # Getting the type of 'addUnexpectedSuccess' (line 345)
            addUnexpectedSuccess_187050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'addUnexpectedSuccess', False)
            # Calling addUnexpectedSuccess(args, kwargs) (line 345)
            addUnexpectedSuccess_call_result_187053 = invoke(stypy.reporting.localization.Localization(__file__, 345, 24), addUnexpectedSuccess_187050, *[self_187051], **kwargs_187052)
            

            if more_types_in_union_187049:
                # Runtime conditional SSA for else branch (line 344)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_187048) or more_types_in_union_187049):
            
            # Call to warn(...): (line 347)
            # Processing the call arguments (line 347)
            str_187056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 38), 'str', 'TestResult has no addUnexpectedSuccess method, reporting as failures')
            # Getting the type of 'RuntimeWarning' (line 348)
            RuntimeWarning_187057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 38), 'RuntimeWarning', False)
            # Processing the call keyword arguments (line 347)
            kwargs_187058 = {}
            # Getting the type of 'warnings' (line 347)
            warnings_187054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 347)
            warn_187055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 24), warnings_187054, 'warn')
            # Calling warn(args, kwargs) (line 347)
            warn_call_result_187059 = invoke(stypy.reporting.localization.Localization(__file__, 347, 24), warn_187055, *[str_187056, RuntimeWarning_187057], **kwargs_187058)
            
            
            # Call to addFailure(...): (line 349)
            # Processing the call arguments (line 349)
            # Getting the type of 'self' (line 349)
            self_187062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 42), 'self', False)
            
            # Call to exc_info(...): (line 349)
            # Processing the call keyword arguments (line 349)
            kwargs_187065 = {}
            # Getting the type of 'sys' (line 349)
            sys_187063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 48), 'sys', False)
            # Obtaining the member 'exc_info' of a type (line 349)
            exc_info_187064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 48), sys_187063, 'exc_info')
            # Calling exc_info(args, kwargs) (line 349)
            exc_info_call_result_187066 = invoke(stypy.reporting.localization.Localization(__file__, 349, 48), exc_info_187064, *[], **kwargs_187065)
            
            # Processing the call keyword arguments (line 349)
            kwargs_187067 = {}
            # Getting the type of 'result' (line 349)
            result_187060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'result', False)
            # Obtaining the member 'addFailure' of a type (line 349)
            addFailure_187061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 24), result_187060, 'addFailure')
            # Calling addFailure(args, kwargs) (line 349)
            addFailure_call_result_187068 = invoke(stypy.reporting.localization.Localization(__file__, 349, 24), addFailure_187061, *[self_187062, exc_info_call_result_187066], **kwargs_187067)
            

            if (may_be_187048 and more_types_in_union_187049):
                # SSA join for if statement (line 344)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the except 'SkipTest' branch of a try statement (line 328)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'SkipTest' (line 350)
        SkipTest_187069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 23), 'SkipTest')
        # Assigning a type to the variable 'e' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'e', SkipTest_187069)
        
        # Call to _addSkip(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'result' (line 351)
        result_187072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 34), 'result', False)
        
        # Call to str(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'e' (line 351)
        e_187074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 46), 'e', False)
        # Processing the call keyword arguments (line 351)
        kwargs_187075 = {}
        # Getting the type of 'str' (line 351)
        str_187073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 42), 'str', False)
        # Calling str(args, kwargs) (line 351)
        str_call_result_187076 = invoke(stypy.reporting.localization.Localization(__file__, 351, 42), str_187073, *[e_187074], **kwargs_187075)
        
        # Processing the call keyword arguments (line 351)
        kwargs_187077 = {}
        # Getting the type of 'self' (line 351)
        self_187070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 20), 'self', False)
        # Obtaining the member '_addSkip' of a type (line 351)
        _addSkip_187071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 20), self_187070, '_addSkip')
        # Calling _addSkip(args, kwargs) (line 351)
        _addSkip_call_result_187078 = invoke(stypy.reporting.localization.Localization(__file__, 351, 20), _addSkip_187071, *[result_187072, str_call_result_187076], **kwargs_187077)
        
        # SSA branch for the except '<any exception>' branch of a try statement (line 328)
        module_type_store.open_ssa_branch('except')
        
        # Call to addError(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'self' (line 353)
        self_187081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 36), 'self', False)
        
        # Call to exc_info(...): (line 353)
        # Processing the call keyword arguments (line 353)
        kwargs_187084 = {}
        # Getting the type of 'sys' (line 353)
        sys_187082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 42), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 353)
        exc_info_187083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 42), sys_187082, 'exc_info')
        # Calling exc_info(args, kwargs) (line 353)
        exc_info_call_result_187085 = invoke(stypy.reporting.localization.Localization(__file__, 353, 42), exc_info_187083, *[], **kwargs_187084)
        
        # Processing the call keyword arguments (line 353)
        kwargs_187086 = {}
        # Getting the type of 'result' (line 353)
        result_187079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'result', False)
        # Obtaining the member 'addError' of a type (line 353)
        addError_187080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 20), result_187079, 'addError')
        # Calling addError(args, kwargs) (line 353)
        addError_call_result_187087 = invoke(stypy.reporting.localization.Localization(__file__, 353, 20), addError_187080, *[self_187081, exc_info_call_result_187085], **kwargs_187086)
        
        # SSA branch for the else branch of a try statement (line 328)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Name to a Name (line 355):
        
        # Assigning a Name to a Name (line 355):
        # Getting the type of 'True' (line 355)
        True_187088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 30), 'True')
        # Assigning a type to the variable 'success' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'success', True_187088)
        # SSA join for try-except statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to tearDown(...): (line 358)
        # Processing the call keyword arguments (line 358)
        kwargs_187091 = {}
        # Getting the type of 'self' (line 358)
        self_187089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'self', False)
        # Obtaining the member 'tearDown' of a type (line 358)
        tearDown_187090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 20), self_187089, 'tearDown')
        # Calling tearDown(args, kwargs) (line 358)
        tearDown_call_result_187092 = invoke(stypy.reporting.localization.Localization(__file__, 358, 20), tearDown_187090, *[], **kwargs_187091)
        
        # SSA branch for the except part of a try statement (line 357)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 357)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the except '<any exception>' branch of a try statement (line 357)
        module_type_store.open_ssa_branch('except')
        
        # Call to addError(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'self' (line 362)
        self_187095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 36), 'self', False)
        
        # Call to exc_info(...): (line 362)
        # Processing the call keyword arguments (line 362)
        kwargs_187098 = {}
        # Getting the type of 'sys' (line 362)
        sys_187096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 42), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 362)
        exc_info_187097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 42), sys_187096, 'exc_info')
        # Calling exc_info(args, kwargs) (line 362)
        exc_info_call_result_187099 = invoke(stypy.reporting.localization.Localization(__file__, 362, 42), exc_info_187097, *[], **kwargs_187098)
        
        # Processing the call keyword arguments (line 362)
        kwargs_187100 = {}
        # Getting the type of 'result' (line 362)
        result_187093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 20), 'result', False)
        # Obtaining the member 'addError' of a type (line 362)
        addError_187094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 20), result_187093, 'addError')
        # Calling addError(args, kwargs) (line 362)
        addError_call_result_187101 = invoke(stypy.reporting.localization.Localization(__file__, 362, 20), addError_187094, *[self_187095, exc_info_call_result_187099], **kwargs_187100)
        
        
        # Assigning a Name to a Name (line 363):
        
        # Assigning a Name to a Name (line 363):
        # Getting the type of 'False' (line 363)
        False_187102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 30), 'False')
        # Assigning a type to the variable 'success' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 20), 'success', False_187102)
        # SSA join for try-except statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 319)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 365):
        
        # Assigning a Call to a Name (line 365):
        
        # Call to doCleanups(...): (line 365)
        # Processing the call keyword arguments (line 365)
        kwargs_187105 = {}
        # Getting the type of 'self' (line 365)
        self_187103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 29), 'self', False)
        # Obtaining the member 'doCleanups' of a type (line 365)
        doCleanups_187104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 29), self_187103, 'doCleanups')
        # Calling doCleanups(args, kwargs) (line 365)
        doCleanups_call_result_187106 = invoke(stypy.reporting.localization.Localization(__file__, 365, 29), doCleanups_187104, *[], **kwargs_187105)
        
        # Assigning a type to the variable 'cleanUpSuccess' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'cleanUpSuccess', doCleanups_call_result_187106)
        
        # Assigning a BoolOp to a Name (line 366):
        
        # Assigning a BoolOp to a Name (line 366):
        
        # Evaluating a boolean operation
        # Getting the type of 'success' (line 366)
        success_187107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'success')
        # Getting the type of 'cleanUpSuccess' (line 366)
        cleanUpSuccess_187108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 34), 'cleanUpSuccess')
        # Applying the binary operator 'and' (line 366)
        result_and_keyword_187109 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 22), 'and', success_187107, cleanUpSuccess_187108)
        
        # Assigning a type to the variable 'success' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'success', result_and_keyword_187109)
        
        # Getting the type of 'success' (line 367)
        success_187110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 15), 'success')
        # Testing the type of an if condition (line 367)
        if_condition_187111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 12), success_187110)
        # Assigning a type to the variable 'if_condition_187111' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'if_condition_187111', if_condition_187111)
        # SSA begins for if statement (line 367)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to addSuccess(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'self' (line 368)
        self_187114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 34), 'self', False)
        # Processing the call keyword arguments (line 368)
        kwargs_187115 = {}
        # Getting the type of 'result' (line 368)
        result_187112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'result', False)
        # Obtaining the member 'addSuccess' of a type (line 368)
        addSuccess_187113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 16), result_187112, 'addSuccess')
        # Calling addSuccess(args, kwargs) (line 368)
        addSuccess_call_result_187116 = invoke(stypy.reporting.localization.Localization(__file__, 368, 16), addSuccess_187113, *[self_187114], **kwargs_187115)
        
        # SSA join for if statement (line 367)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 317)
        
        # Call to stopTest(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'self' (line 370)
        self_187119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 28), 'self', False)
        # Processing the call keyword arguments (line 370)
        kwargs_187120 = {}
        # Getting the type of 'result' (line 370)
        result_187117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'result', False)
        # Obtaining the member 'stopTest' of a type (line 370)
        stopTest_187118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), result_187117, 'stopTest')
        # Calling stopTest(args, kwargs) (line 370)
        stopTest_call_result_187121 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), stopTest_187118, *[self_187119], **kwargs_187120)
        
        
        # Type idiom detected: calculating its left and rigth part (line 371)
        # Getting the type of 'orig_result' (line 371)
        orig_result_187122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'orig_result')
        # Getting the type of 'None' (line 371)
        None_187123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 30), 'None')
        
        (may_be_187124, more_types_in_union_187125) = may_be_none(orig_result_187122, None_187123)

        if may_be_187124:

            if more_types_in_union_187125:
                # Runtime conditional SSA (line 371)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 372):
            
            # Assigning a Call to a Name (line 372):
            
            # Call to getattr(...): (line 372)
            # Processing the call arguments (line 372)
            # Getting the type of 'result' (line 372)
            result_187127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 38), 'result', False)
            str_187128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 46), 'str', 'stopTestRun')
            # Getting the type of 'None' (line 372)
            None_187129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 61), 'None', False)
            # Processing the call keyword arguments (line 372)
            kwargs_187130 = {}
            # Getting the type of 'getattr' (line 372)
            getattr_187126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 30), 'getattr', False)
            # Calling getattr(args, kwargs) (line 372)
            getattr_call_result_187131 = invoke(stypy.reporting.localization.Localization(__file__, 372, 30), getattr_187126, *[result_187127, str_187128, None_187129], **kwargs_187130)
            
            # Assigning a type to the variable 'stopTestRun' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'stopTestRun', getattr_call_result_187131)
            
            # Type idiom detected: calculating its left and rigth part (line 373)
            # Getting the type of 'stopTestRun' (line 373)
            stopTestRun_187132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'stopTestRun')
            # Getting the type of 'None' (line 373)
            None_187133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 38), 'None')
            
            (may_be_187134, more_types_in_union_187135) = may_not_be_none(stopTestRun_187132, None_187133)

            if may_be_187134:

                if more_types_in_union_187135:
                    # Runtime conditional SSA (line 373)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to stopTestRun(...): (line 374)
                # Processing the call keyword arguments (line 374)
                kwargs_187137 = {}
                # Getting the type of 'stopTestRun' (line 374)
                stopTestRun_187136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'stopTestRun', False)
                # Calling stopTestRun(args, kwargs) (line 374)
                stopTestRun_call_result_187138 = invoke(stypy.reporting.localization.Localization(__file__, 374, 20), stopTestRun_187136, *[], **kwargs_187137)
                

                if more_types_in_union_187135:
                    # SSA join for if statement (line 373)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_187125:
                # SSA join for if statement (line 371)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 295)
        stypy_return_type_187139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187139)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_187139


    @norecursion
    def doCleanups(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'doCleanups'
        module_type_store = module_type_store.open_function_context('doCleanups', 376, 4, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.doCleanups.__dict__.__setitem__('stypy_localization', localization)
        TestCase.doCleanups.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.doCleanups.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.doCleanups.__dict__.__setitem__('stypy_function_name', 'TestCase.doCleanups')
        TestCase.doCleanups.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.doCleanups.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.doCleanups.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.doCleanups.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.doCleanups.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.doCleanups.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.doCleanups.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.doCleanups', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'doCleanups', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'doCleanups(...)' code ##################

        str_187140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, (-1)), 'str', 'Execute all cleanup functions. Normally called for you after\n        tearDown.')
        
        # Assigning a Attribute to a Name (line 379):
        
        # Assigning a Attribute to a Name (line 379):
        # Getting the type of 'self' (line 379)
        self_187141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 17), 'self')
        # Obtaining the member '_resultForDoCleanups' of a type (line 379)
        _resultForDoCleanups_187142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 17), self_187141, '_resultForDoCleanups')
        # Assigning a type to the variable 'result' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'result', _resultForDoCleanups_187142)
        
        # Assigning a Name to a Name (line 380):
        
        # Assigning a Name to a Name (line 380):
        # Getting the type of 'True' (line 380)
        True_187143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 13), 'True')
        # Assigning a type to the variable 'ok' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'ok', True_187143)
        
        # Getting the type of 'self' (line 381)
        self_187144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 14), 'self')
        # Obtaining the member '_cleanups' of a type (line 381)
        _cleanups_187145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 14), self_187144, '_cleanups')
        # Testing the type of an if condition (line 381)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 8), _cleanups_187145)
        # SSA begins for while statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 382):
        
        # Assigning a Call to a Name:
        
        # Call to pop(...): (line 382)
        # Processing the call arguments (line 382)
        int_187149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 56), 'int')
        # Processing the call keyword arguments (line 382)
        kwargs_187150 = {}
        # Getting the type of 'self' (line 382)
        self_187146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 37), 'self', False)
        # Obtaining the member '_cleanups' of a type (line 382)
        _cleanups_187147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 37), self_187146, '_cleanups')
        # Obtaining the member 'pop' of a type (line 382)
        pop_187148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 37), _cleanups_187147, 'pop')
        # Calling pop(args, kwargs) (line 382)
        pop_call_result_187151 = invoke(stypy.reporting.localization.Localization(__file__, 382, 37), pop_187148, *[int_187149], **kwargs_187150)
        
        # Assigning a type to the variable 'call_assignment_186524' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186524', pop_call_result_187151)
        
        # Assigning a Call to a Name (line 382):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_187154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 12), 'int')
        # Processing the call keyword arguments
        kwargs_187155 = {}
        # Getting the type of 'call_assignment_186524' (line 382)
        call_assignment_186524_187152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186524', False)
        # Obtaining the member '__getitem__' of a type (line 382)
        getitem___187153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 12), call_assignment_186524_187152, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_187156 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___187153, *[int_187154], **kwargs_187155)
        
        # Assigning a type to the variable 'call_assignment_186525' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186525', getitem___call_result_187156)
        
        # Assigning a Name to a Name (line 382):
        # Getting the type of 'call_assignment_186525' (line 382)
        call_assignment_186525_187157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186525')
        # Assigning a type to the variable 'function' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'function', call_assignment_186525_187157)
        
        # Assigning a Call to a Name (line 382):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_187160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 12), 'int')
        # Processing the call keyword arguments
        kwargs_187161 = {}
        # Getting the type of 'call_assignment_186524' (line 382)
        call_assignment_186524_187158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186524', False)
        # Obtaining the member '__getitem__' of a type (line 382)
        getitem___187159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 12), call_assignment_186524_187158, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_187162 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___187159, *[int_187160], **kwargs_187161)
        
        # Assigning a type to the variable 'call_assignment_186526' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186526', getitem___call_result_187162)
        
        # Assigning a Name to a Name (line 382):
        # Getting the type of 'call_assignment_186526' (line 382)
        call_assignment_186526_187163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186526')
        # Assigning a type to the variable 'args' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 22), 'args', call_assignment_186526_187163)
        
        # Assigning a Call to a Name (line 382):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_187166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 12), 'int')
        # Processing the call keyword arguments
        kwargs_187167 = {}
        # Getting the type of 'call_assignment_186524' (line 382)
        call_assignment_186524_187164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186524', False)
        # Obtaining the member '__getitem__' of a type (line 382)
        getitem___187165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 12), call_assignment_186524_187164, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_187168 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___187165, *[int_187166], **kwargs_187167)
        
        # Assigning a type to the variable 'call_assignment_186527' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186527', getitem___call_result_187168)
        
        # Assigning a Name to a Name (line 382):
        # Getting the type of 'call_assignment_186527' (line 382)
        call_assignment_186527_187169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'call_assignment_186527')
        # Assigning a type to the variable 'kwargs' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 28), 'kwargs', call_assignment_186527_187169)
        
        
        # SSA begins for try-except statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to function(...): (line 384)
        # Getting the type of 'args' (line 384)
        args_187171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 26), 'args', False)
        # Processing the call keyword arguments (line 384)
        # Getting the type of 'kwargs' (line 384)
        kwargs_187172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 34), 'kwargs', False)
        kwargs_187173 = {'kwargs_187172': kwargs_187172}
        # Getting the type of 'function' (line 384)
        function_187170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'function', False)
        # Calling function(args, kwargs) (line 384)
        function_call_result_187174 = invoke(stypy.reporting.localization.Localization(__file__, 384, 16), function_187170, *[args_187171], **kwargs_187173)
        
        # SSA branch for the except part of a try statement (line 383)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 383)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the except '<any exception>' branch of a try statement (line 383)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 388):
        
        # Assigning a Name to a Name (line 388):
        # Getting the type of 'False' (line 388)
        False_187175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 21), 'False')
        # Assigning a type to the variable 'ok' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'ok', False_187175)
        
        # Call to addError(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'self' (line 389)
        self_187178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'self', False)
        
        # Call to exc_info(...): (line 389)
        # Processing the call keyword arguments (line 389)
        kwargs_187181 = {}
        # Getting the type of 'sys' (line 389)
        sys_187179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 38), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 389)
        exc_info_187180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 38), sys_187179, 'exc_info')
        # Calling exc_info(args, kwargs) (line 389)
        exc_info_call_result_187182 = invoke(stypy.reporting.localization.Localization(__file__, 389, 38), exc_info_187180, *[], **kwargs_187181)
        
        # Processing the call keyword arguments (line 389)
        kwargs_187183 = {}
        # Getting the type of 'result' (line 389)
        result_187176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'result', False)
        # Obtaining the member 'addError' of a type (line 389)
        addError_187177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 16), result_187176, 'addError')
        # Calling addError(args, kwargs) (line 389)
        addError_call_result_187184 = invoke(stypy.reporting.localization.Localization(__file__, 389, 16), addError_187177, *[self_187178, exc_info_call_result_187182], **kwargs_187183)
        
        # SSA join for try-except statement (line 383)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 381)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'ok' (line 390)
        ok_187185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'ok')
        # Assigning a type to the variable 'stypy_return_type' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'stypy_return_type', ok_187185)
        
        # ################# End of 'doCleanups(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'doCleanups' in the type store
        # Getting the type of 'stypy_return_type' (line 376)
        stypy_return_type_187186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187186)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'doCleanups'
        return stypy_return_type_187186


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 392, 4, False)
        # Assigning a type to the variable 'self' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.__call__.__dict__.__setitem__('stypy_localization', localization)
        TestCase.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.__call__.__dict__.__setitem__('stypy_function_name', 'TestCase.__call__')
        TestCase.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        TestCase.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwds')
        TestCase.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.__call__', [], 'args', 'kwds', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to run(...): (line 393)
        # Getting the type of 'args' (line 393)
        args_187189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 25), 'args', False)
        # Processing the call keyword arguments (line 393)
        # Getting the type of 'kwds' (line 393)
        kwds_187190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 33), 'kwds', False)
        kwargs_187191 = {'kwds_187190': kwds_187190}
        # Getting the type of 'self' (line 393)
        self_187187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'self', False)
        # Obtaining the member 'run' of a type (line 393)
        run_187188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 15), self_187187, 'run')
        # Calling run(args, kwargs) (line 393)
        run_call_result_187192 = invoke(stypy.reporting.localization.Localization(__file__, 393, 15), run_187188, *[args_187189], **kwargs_187191)
        
        # Assigning a type to the variable 'stypy_return_type' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'stypy_return_type', run_call_result_187192)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 392)
        stypy_return_type_187193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187193)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_187193


    @norecursion
    def debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'debug'
        module_type_store = module_type_store.open_function_context('debug', 395, 4, False)
        # Assigning a type to the variable 'self' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.debug.__dict__.__setitem__('stypy_localization', localization)
        TestCase.debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.debug.__dict__.__setitem__('stypy_function_name', 'TestCase.debug')
        TestCase.debug.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.debug', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'debug', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'debug(...)' code ##################

        str_187194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 8), 'str', 'Run the test without collecting errors in a TestResult')
        
        # Call to setUp(...): (line 397)
        # Processing the call keyword arguments (line 397)
        kwargs_187197 = {}
        # Getting the type of 'self' (line 397)
        self_187195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self', False)
        # Obtaining the member 'setUp' of a type (line 397)
        setUp_187196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_187195, 'setUp')
        # Calling setUp(args, kwargs) (line 397)
        setUp_call_result_187198 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), setUp_187196, *[], **kwargs_187197)
        
        
        # Call to (...): (line 398)
        # Processing the call keyword arguments (line 398)
        kwargs_187205 = {}
        
        # Call to getattr(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'self' (line 398)
        self_187200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 16), 'self', False)
        # Getting the type of 'self' (line 398)
        self_187201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 22), 'self', False)
        # Obtaining the member '_testMethodName' of a type (line 398)
        _testMethodName_187202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 22), self_187201, '_testMethodName')
        # Processing the call keyword arguments (line 398)
        kwargs_187203 = {}
        # Getting the type of 'getattr' (line 398)
        getattr_187199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'getattr', False)
        # Calling getattr(args, kwargs) (line 398)
        getattr_call_result_187204 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), getattr_187199, *[self_187200, _testMethodName_187202], **kwargs_187203)
        
        # Calling (args, kwargs) (line 398)
        _call_result_187206 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), getattr_call_result_187204, *[], **kwargs_187205)
        
        
        # Call to tearDown(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_187209 = {}
        # Getting the type of 'self' (line 399)
        self_187207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self', False)
        # Obtaining the member 'tearDown' of a type (line 399)
        tearDown_187208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_187207, 'tearDown')
        # Calling tearDown(args, kwargs) (line 399)
        tearDown_call_result_187210 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), tearDown_187208, *[], **kwargs_187209)
        
        
        # Getting the type of 'self' (line 400)
        self_187211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 14), 'self')
        # Obtaining the member '_cleanups' of a type (line 400)
        _cleanups_187212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 14), self_187211, '_cleanups')
        # Testing the type of an if condition (line 400)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 8), _cleanups_187212)
        # SSA begins for while statement (line 400)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 401):
        
        # Assigning a Call to a Name:
        
        # Call to pop(...): (line 401)
        # Processing the call arguments (line 401)
        int_187216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 56), 'int')
        # Processing the call keyword arguments (line 401)
        kwargs_187217 = {}
        # Getting the type of 'self' (line 401)
        self_187213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 37), 'self', False)
        # Obtaining the member '_cleanups' of a type (line 401)
        _cleanups_187214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 37), self_187213, '_cleanups')
        # Obtaining the member 'pop' of a type (line 401)
        pop_187215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 37), _cleanups_187214, 'pop')
        # Calling pop(args, kwargs) (line 401)
        pop_call_result_187218 = invoke(stypy.reporting.localization.Localization(__file__, 401, 37), pop_187215, *[int_187216], **kwargs_187217)
        
        # Assigning a type to the variable 'call_assignment_186528' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186528', pop_call_result_187218)
        
        # Assigning a Call to a Name (line 401):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_187221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 12), 'int')
        # Processing the call keyword arguments
        kwargs_187222 = {}
        # Getting the type of 'call_assignment_186528' (line 401)
        call_assignment_186528_187219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186528', False)
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___187220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 12), call_assignment_186528_187219, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_187223 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___187220, *[int_187221], **kwargs_187222)
        
        # Assigning a type to the variable 'call_assignment_186529' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186529', getitem___call_result_187223)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'call_assignment_186529' (line 401)
        call_assignment_186529_187224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186529')
        # Assigning a type to the variable 'function' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'function', call_assignment_186529_187224)
        
        # Assigning a Call to a Name (line 401):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_187227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 12), 'int')
        # Processing the call keyword arguments
        kwargs_187228 = {}
        # Getting the type of 'call_assignment_186528' (line 401)
        call_assignment_186528_187225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186528', False)
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___187226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 12), call_assignment_186528_187225, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_187229 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___187226, *[int_187227], **kwargs_187228)
        
        # Assigning a type to the variable 'call_assignment_186530' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186530', getitem___call_result_187229)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'call_assignment_186530' (line 401)
        call_assignment_186530_187230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186530')
        # Assigning a type to the variable 'args' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 22), 'args', call_assignment_186530_187230)
        
        # Assigning a Call to a Name (line 401):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_187233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 12), 'int')
        # Processing the call keyword arguments
        kwargs_187234 = {}
        # Getting the type of 'call_assignment_186528' (line 401)
        call_assignment_186528_187231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186528', False)
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___187232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 12), call_assignment_186528_187231, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_187235 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___187232, *[int_187233], **kwargs_187234)
        
        # Assigning a type to the variable 'call_assignment_186531' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186531', getitem___call_result_187235)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'call_assignment_186531' (line 401)
        call_assignment_186531_187236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'call_assignment_186531')
        # Assigning a type to the variable 'kwargs' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 28), 'kwargs', call_assignment_186531_187236)
        
        # Call to function(...): (line 402)
        # Getting the type of 'args' (line 402)
        args_187238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 22), 'args', False)
        # Processing the call keyword arguments (line 402)
        # Getting the type of 'kwargs' (line 402)
        kwargs_187239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 30), 'kwargs', False)
        kwargs_187240 = {'kwargs_187239': kwargs_187239}
        # Getting the type of 'function' (line 402)
        function_187237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'function', False)
        # Calling function(args, kwargs) (line 402)
        function_call_result_187241 = invoke(stypy.reporting.localization.Localization(__file__, 402, 12), function_187237, *[args_187238], **kwargs_187240)
        
        # SSA join for while statement (line 400)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'debug' in the type store
        # Getting the type of 'stypy_return_type' (line 395)
        stypy_return_type_187242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187242)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'debug'
        return stypy_return_type_187242


    @norecursion
    def skipTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'skipTest'
        module_type_store = module_type_store.open_function_context('skipTest', 404, 4, False)
        # Assigning a type to the variable 'self' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.skipTest.__dict__.__setitem__('stypy_localization', localization)
        TestCase.skipTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.skipTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.skipTest.__dict__.__setitem__('stypy_function_name', 'TestCase.skipTest')
        TestCase.skipTest.__dict__.__setitem__('stypy_param_names_list', ['reason'])
        TestCase.skipTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.skipTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.skipTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.skipTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.skipTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.skipTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.skipTest', ['reason'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'skipTest', localization, ['reason'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'skipTest(...)' code ##################

        str_187243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'str', 'Skip this test.')
        
        # Call to SkipTest(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'reason' (line 406)
        reason_187245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 23), 'reason', False)
        # Processing the call keyword arguments (line 406)
        kwargs_187246 = {}
        # Getting the type of 'SkipTest' (line 406)
        SkipTest_187244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 14), 'SkipTest', False)
        # Calling SkipTest(args, kwargs) (line 406)
        SkipTest_call_result_187247 = invoke(stypy.reporting.localization.Localization(__file__, 406, 14), SkipTest_187244, *[reason_187245], **kwargs_187246)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 406, 8), SkipTest_call_result_187247, 'raise parameter', BaseException)
        
        # ################# End of 'skipTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'skipTest' in the type store
        # Getting the type of 'stypy_return_type' (line 404)
        stypy_return_type_187248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187248)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'skipTest'
        return stypy_return_type_187248


    @norecursion
    def fail(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 408)
        None_187249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 23), 'None')
        defaults = [None_187249]
        # Create a new context for function 'fail'
        module_type_store = module_type_store.open_function_context('fail', 408, 4, False)
        # Assigning a type to the variable 'self' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.fail.__dict__.__setitem__('stypy_localization', localization)
        TestCase.fail.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.fail.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.fail.__dict__.__setitem__('stypy_function_name', 'TestCase.fail')
        TestCase.fail.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        TestCase.fail.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.fail.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.fail.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.fail.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.fail.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.fail.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.fail', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fail', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fail(...)' code ##################

        str_187250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 8), 'str', 'Fail immediately, with the given message.')
        
        # Call to failureException(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'msg' (line 410)
        msg_187253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 36), 'msg', False)
        # Processing the call keyword arguments (line 410)
        kwargs_187254 = {}
        # Getting the type of 'self' (line 410)
        self_187251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 14), 'self', False)
        # Obtaining the member 'failureException' of a type (line 410)
        failureException_187252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 14), self_187251, 'failureException')
        # Calling failureException(args, kwargs) (line 410)
        failureException_call_result_187255 = invoke(stypy.reporting.localization.Localization(__file__, 410, 14), failureException_187252, *[msg_187253], **kwargs_187254)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 410, 8), failureException_call_result_187255, 'raise parameter', BaseException)
        
        # ################# End of 'fail(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fail' in the type store
        # Getting the type of 'stypy_return_type' (line 408)
        stypy_return_type_187256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fail'
        return stypy_return_type_187256


    @norecursion
    def assertFalse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 412)
        None_187257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 36), 'None')
        defaults = [None_187257]
        # Create a new context for function 'assertFalse'
        module_type_store = module_type_store.open_function_context('assertFalse', 412, 4, False)
        # Assigning a type to the variable 'self' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertFalse.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertFalse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertFalse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertFalse.__dict__.__setitem__('stypy_function_name', 'TestCase.assertFalse')
        TestCase.assertFalse.__dict__.__setitem__('stypy_param_names_list', ['expr', 'msg'])
        TestCase.assertFalse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertFalse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertFalse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertFalse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertFalse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertFalse.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertFalse', ['expr', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertFalse', localization, ['expr', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertFalse(...)' code ##################

        str_187258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 8), 'str', 'Check that the expression is false.')
        
        # Getting the type of 'expr' (line 414)
        expr_187259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 11), 'expr')
        # Testing the type of an if condition (line 414)
        if_condition_187260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 8), expr_187259)
        # Assigning a type to the variable 'if_condition_187260' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'if_condition_187260', if_condition_187260)
        # SSA begins for if statement (line 414)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 415):
        
        # Assigning a Call to a Name (line 415):
        
        # Call to _formatMessage(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'msg' (line 415)
        msg_187263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 38), 'msg', False)
        str_187264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 43), 'str', '%s is not false')
        
        # Call to safe_repr(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'expr' (line 415)
        expr_187266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 73), 'expr', False)
        # Processing the call keyword arguments (line 415)
        kwargs_187267 = {}
        # Getting the type of 'safe_repr' (line 415)
        safe_repr_187265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 63), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 415)
        safe_repr_call_result_187268 = invoke(stypy.reporting.localization.Localization(__file__, 415, 63), safe_repr_187265, *[expr_187266], **kwargs_187267)
        
        # Applying the binary operator '%' (line 415)
        result_mod_187269 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 43), '%', str_187264, safe_repr_call_result_187268)
        
        # Processing the call keyword arguments (line 415)
        kwargs_187270 = {}
        # Getting the type of 'self' (line 415)
        self_187261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 18), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 415)
        _formatMessage_187262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 18), self_187261, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 415)
        _formatMessage_call_result_187271 = invoke(stypy.reporting.localization.Localization(__file__, 415, 18), _formatMessage_187262, *[msg_187263, result_mod_187269], **kwargs_187270)
        
        # Assigning a type to the variable 'msg' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'msg', _formatMessage_call_result_187271)
        
        # Call to failureException(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'msg' (line 416)
        msg_187274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 40), 'msg', False)
        # Processing the call keyword arguments (line 416)
        kwargs_187275 = {}
        # Getting the type of 'self' (line 416)
        self_187272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 18), 'self', False)
        # Obtaining the member 'failureException' of a type (line 416)
        failureException_187273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 18), self_187272, 'failureException')
        # Calling failureException(args, kwargs) (line 416)
        failureException_call_result_187276 = invoke(stypy.reporting.localization.Localization(__file__, 416, 18), failureException_187273, *[msg_187274], **kwargs_187275)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 416, 12), failureException_call_result_187276, 'raise parameter', BaseException)
        # SSA join for if statement (line 414)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertFalse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertFalse' in the type store
        # Getting the type of 'stypy_return_type' (line 412)
        stypy_return_type_187277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187277)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertFalse'
        return stypy_return_type_187277


    @norecursion
    def assertTrue(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 418)
        None_187278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 35), 'None')
        defaults = [None_187278]
        # Create a new context for function 'assertTrue'
        module_type_store = module_type_store.open_function_context('assertTrue', 418, 4, False)
        # Assigning a type to the variable 'self' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertTrue.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertTrue.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertTrue.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertTrue.__dict__.__setitem__('stypy_function_name', 'TestCase.assertTrue')
        TestCase.assertTrue.__dict__.__setitem__('stypy_param_names_list', ['expr', 'msg'])
        TestCase.assertTrue.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertTrue.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertTrue.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertTrue.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertTrue.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertTrue.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertTrue', ['expr', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertTrue', localization, ['expr', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertTrue(...)' code ##################

        str_187279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 8), 'str', 'Check that the expression is true.')
        
        
        # Getting the type of 'expr' (line 420)
        expr_187280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'expr')
        # Applying the 'not' unary operator (line 420)
        result_not__187281 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 11), 'not', expr_187280)
        
        # Testing the type of an if condition (line 420)
        if_condition_187282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 8), result_not__187281)
        # Assigning a type to the variable 'if_condition_187282' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'if_condition_187282', if_condition_187282)
        # SSA begins for if statement (line 420)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 421):
        
        # Assigning a Call to a Name (line 421):
        
        # Call to _formatMessage(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'msg' (line 421)
        msg_187285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 38), 'msg', False)
        str_187286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 43), 'str', '%s is not true')
        
        # Call to safe_repr(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'expr' (line 421)
        expr_187288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 72), 'expr', False)
        # Processing the call keyword arguments (line 421)
        kwargs_187289 = {}
        # Getting the type of 'safe_repr' (line 421)
        safe_repr_187287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 62), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 421)
        safe_repr_call_result_187290 = invoke(stypy.reporting.localization.Localization(__file__, 421, 62), safe_repr_187287, *[expr_187288], **kwargs_187289)
        
        # Applying the binary operator '%' (line 421)
        result_mod_187291 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 43), '%', str_187286, safe_repr_call_result_187290)
        
        # Processing the call keyword arguments (line 421)
        kwargs_187292 = {}
        # Getting the type of 'self' (line 421)
        self_187283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 18), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 421)
        _formatMessage_187284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 18), self_187283, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 421)
        _formatMessage_call_result_187293 = invoke(stypy.reporting.localization.Localization(__file__, 421, 18), _formatMessage_187284, *[msg_187285, result_mod_187291], **kwargs_187292)
        
        # Assigning a type to the variable 'msg' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'msg', _formatMessage_call_result_187293)
        
        # Call to failureException(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'msg' (line 422)
        msg_187296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 40), 'msg', False)
        # Processing the call keyword arguments (line 422)
        kwargs_187297 = {}
        # Getting the type of 'self' (line 422)
        self_187294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 18), 'self', False)
        # Obtaining the member 'failureException' of a type (line 422)
        failureException_187295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 18), self_187294, 'failureException')
        # Calling failureException(args, kwargs) (line 422)
        failureException_call_result_187298 = invoke(stypy.reporting.localization.Localization(__file__, 422, 18), failureException_187295, *[msg_187296], **kwargs_187297)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 422, 12), failureException_call_result_187298, 'raise parameter', BaseException)
        # SSA join for if statement (line 420)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertTrue(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertTrue' in the type store
        # Getting the type of 'stypy_return_type' (line 418)
        stypy_return_type_187299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187299)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertTrue'
        return stypy_return_type_187299


    @norecursion
    def _formatMessage(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_formatMessage'
        module_type_store = module_type_store.open_function_context('_formatMessage', 424, 4, False)
        # Assigning a type to the variable 'self' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase._formatMessage.__dict__.__setitem__('stypy_localization', localization)
        TestCase._formatMessage.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase._formatMessage.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase._formatMessage.__dict__.__setitem__('stypy_function_name', 'TestCase._formatMessage')
        TestCase._formatMessage.__dict__.__setitem__('stypy_param_names_list', ['msg', 'standardMsg'])
        TestCase._formatMessage.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase._formatMessage.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase._formatMessage.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase._formatMessage.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase._formatMessage.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase._formatMessage.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase._formatMessage', ['msg', 'standardMsg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_formatMessage', localization, ['msg', 'standardMsg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_formatMessage(...)' code ##################

        str_187300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, (-1)), 'str', "Honour the longMessage attribute when generating failure messages.\n        If longMessage is False this means:\n        * Use only an explicit message if it is provided\n        * Otherwise use the standard message for the assert\n\n        If longMessage is True:\n        * Use the standard message\n        * If an explicit message is provided, plus ' : ' and the explicit message\n        ")
        
        
        # Getting the type of 'self' (line 434)
        self_187301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'self')
        # Obtaining the member 'longMessage' of a type (line 434)
        longMessage_187302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 15), self_187301, 'longMessage')
        # Applying the 'not' unary operator (line 434)
        result_not__187303 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 11), 'not', longMessage_187302)
        
        # Testing the type of an if condition (line 434)
        if_condition_187304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 8), result_not__187303)
        # Assigning a type to the variable 'if_condition_187304' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'if_condition_187304', if_condition_187304)
        # SSA begins for if statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Evaluating a boolean operation
        # Getting the type of 'msg' (line 435)
        msg_187305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 19), 'msg')
        # Getting the type of 'standardMsg' (line 435)
        standardMsg_187306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 26), 'standardMsg')
        # Applying the binary operator 'or' (line 435)
        result_or_keyword_187307 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 19), 'or', msg_187305, standardMsg_187306)
        
        # Assigning a type to the variable 'stypy_return_type' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'stypy_return_type', result_or_keyword_187307)
        # SSA join for if statement (line 434)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 436)
        # Getting the type of 'msg' (line 436)
        msg_187308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'msg')
        # Getting the type of 'None' (line 436)
        None_187309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 18), 'None')
        
        (may_be_187310, more_types_in_union_187311) = may_be_none(msg_187308, None_187309)

        if may_be_187310:

            if more_types_in_union_187311:
                # Runtime conditional SSA (line 436)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'standardMsg' (line 437)
            standardMsg_187312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'standardMsg')
            # Assigning a type to the variable 'stypy_return_type' (line 437)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'stypy_return_type', standardMsg_187312)

            if more_types_in_union_187311:
                # SSA join for if statement (line 436)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # SSA begins for try-except statement (line 438)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        str_187313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 19), 'str', '%s : %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 441)
        tuple_187314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 441)
        # Adding element type (line 441)
        # Getting the type of 'standardMsg' (line 441)
        standardMsg_187315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 32), 'standardMsg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 32), tuple_187314, standardMsg_187315)
        # Adding element type (line 441)
        # Getting the type of 'msg' (line 441)
        msg_187316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 45), 'msg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 32), tuple_187314, msg_187316)
        
        # Applying the binary operator '%' (line 441)
        result_mod_187317 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 19), '%', str_187313, tuple_187314)
        
        # Assigning a type to the variable 'stypy_return_type' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'stypy_return_type', result_mod_187317)
        # SSA branch for the except part of a try statement (line 438)
        # SSA branch for the except 'UnicodeDecodeError' branch of a try statement (line 438)
        module_type_store.open_ssa_branch('except')
        str_187318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 20), 'str', '%s : %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 443)
        tuple_187319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 443)
        # Adding element type (line 443)
        
        # Call to safe_repr(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'standardMsg' (line 443)
        standardMsg_187321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 43), 'standardMsg', False)
        # Processing the call keyword arguments (line 443)
        kwargs_187322 = {}
        # Getting the type of 'safe_repr' (line 443)
        safe_repr_187320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 33), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 443)
        safe_repr_call_result_187323 = invoke(stypy.reporting.localization.Localization(__file__, 443, 33), safe_repr_187320, *[standardMsg_187321], **kwargs_187322)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 33), tuple_187319, safe_repr_call_result_187323)
        # Adding element type (line 443)
        
        # Call to safe_repr(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'msg' (line 443)
        msg_187325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 67), 'msg', False)
        # Processing the call keyword arguments (line 443)
        kwargs_187326 = {}
        # Getting the type of 'safe_repr' (line 443)
        safe_repr_187324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 57), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 443)
        safe_repr_call_result_187327 = invoke(stypy.reporting.localization.Localization(__file__, 443, 57), safe_repr_187324, *[msg_187325], **kwargs_187326)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 33), tuple_187319, safe_repr_call_result_187327)
        
        # Applying the binary operator '%' (line 443)
        result_mod_187328 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 20), '%', str_187318, tuple_187319)
        
        # Assigning a type to the variable 'stypy_return_type' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'stypy_return_type', result_mod_187328)
        # SSA join for try-except statement (line 438)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_formatMessage(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_formatMessage' in the type store
        # Getting the type of 'stypy_return_type' (line 424)
        stypy_return_type_187329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187329)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_formatMessage'
        return stypy_return_type_187329


    @norecursion
    def assertRaises(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 446)
        None_187330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 49), 'None')
        defaults = [None_187330]
        # Create a new context for function 'assertRaises'
        module_type_store = module_type_store.open_function_context('assertRaises', 446, 4, False)
        # Assigning a type to the variable 'self' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertRaises.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertRaises.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertRaises.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertRaises.__dict__.__setitem__('stypy_function_name', 'TestCase.assertRaises')
        TestCase.assertRaises.__dict__.__setitem__('stypy_param_names_list', ['excClass', 'callableObj'])
        TestCase.assertRaises.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        TestCase.assertRaises.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        TestCase.assertRaises.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertRaises.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertRaises.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertRaises.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertRaises', ['excClass', 'callableObj'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertRaises', localization, ['excClass', 'callableObj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertRaises(...)' code ##################

        str_187331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, (-1)), 'str', "Fail unless an exception of class excClass is raised\n           by callableObj when invoked with arguments args and keyword\n           arguments kwargs. If a different type of exception is\n           raised, it will not be caught, and the test case will be\n           deemed to have suffered an error, exactly as for an\n           unexpected exception.\n\n           If called with callableObj omitted or None, will return a\n           context object used like this::\n\n                with self.assertRaises(SomeException):\n                    do_something()\n\n           The context manager keeps a reference to the exception as\n           the 'exception' attribute. This allows you to inspect the\n           exception after the assertion::\n\n               with self.assertRaises(SomeException) as cm:\n                   do_something()\n               the_exception = cm.exception\n               self.assertEqual(the_exception.error_code, 3)\n        ")
        
        # Assigning a Call to a Name (line 469):
        
        # Assigning a Call to a Name (line 469):
        
        # Call to _AssertRaisesContext(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'excClass' (line 469)
        excClass_187333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 39), 'excClass', False)
        # Getting the type of 'self' (line 469)
        self_187334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 49), 'self', False)
        # Processing the call keyword arguments (line 469)
        kwargs_187335 = {}
        # Getting the type of '_AssertRaisesContext' (line 469)
        _AssertRaisesContext_187332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 18), '_AssertRaisesContext', False)
        # Calling _AssertRaisesContext(args, kwargs) (line 469)
        _AssertRaisesContext_call_result_187336 = invoke(stypy.reporting.localization.Localization(__file__, 469, 18), _AssertRaisesContext_187332, *[excClass_187333, self_187334], **kwargs_187335)
        
        # Assigning a type to the variable 'context' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'context', _AssertRaisesContext_call_result_187336)
        
        # Type idiom detected: calculating its left and rigth part (line 470)
        # Getting the type of 'callableObj' (line 470)
        callableObj_187337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 11), 'callableObj')
        # Getting the type of 'None' (line 470)
        None_187338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'None')
        
        (may_be_187339, more_types_in_union_187340) = may_be_none(callableObj_187337, None_187338)

        if may_be_187339:

            if more_types_in_union_187340:
                # Runtime conditional SSA (line 470)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'context' (line 471)
            context_187341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 19), 'context')
            # Assigning a type to the variable 'stypy_return_type' (line 471)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'stypy_return_type', context_187341)

            if more_types_in_union_187340:
                # SSA join for if statement (line 470)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'context' (line 472)
        context_187342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 13), 'context')
        with_187343 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 472, 13), context_187342, 'with parameter', '__enter__', '__exit__')

        if with_187343:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 472)
            enter___187344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 13), context_187342, '__enter__')
            with_enter_187345 = invoke(stypy.reporting.localization.Localization(__file__, 472, 13), enter___187344)
            
            # Call to callableObj(...): (line 473)
            # Getting the type of 'args' (line 473)
            args_187347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 25), 'args', False)
            # Processing the call keyword arguments (line 473)
            # Getting the type of 'kwargs' (line 473)
            kwargs_187348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 33), 'kwargs', False)
            kwargs_187349 = {'kwargs_187348': kwargs_187348}
            # Getting the type of 'callableObj' (line 473)
            callableObj_187346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'callableObj', False)
            # Calling callableObj(args, kwargs) (line 473)
            callableObj_call_result_187350 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), callableObj_187346, *[args_187347], **kwargs_187349)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 472)
            exit___187351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 13), context_187342, '__exit__')
            with_exit_187352 = invoke(stypy.reporting.localization.Localization(__file__, 472, 13), exit___187351, None, None, None)

        
        # ################# End of 'assertRaises(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertRaises' in the type store
        # Getting the type of 'stypy_return_type' (line 446)
        stypy_return_type_187353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertRaises'
        return stypy_return_type_187353


    @norecursion
    def _getAssertEqualityFunc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_getAssertEqualityFunc'
        module_type_store = module_type_store.open_function_context('_getAssertEqualityFunc', 475, 4, False)
        # Assigning a type to the variable 'self' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_localization', localization)
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_function_name', 'TestCase._getAssertEqualityFunc')
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_param_names_list', ['first', 'second'])
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase._getAssertEqualityFunc.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase._getAssertEqualityFunc', ['first', 'second'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_getAssertEqualityFunc', localization, ['first', 'second'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_getAssertEqualityFunc(...)' code ##################

        str_187354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, (-1)), 'str', 'Get a detailed comparison function for the types of the two args.\n\n        Returns: A callable accepting (first, second, msg=None) that will\n        raise a failure exception if first != second with a useful human\n        readable error message for those types.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 492)
        # Getting the type of 'first' (line 492)
        first_187355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 16), 'first')
        
        # Call to type(...): (line 492)
        # Processing the call arguments (line 492)
        # Getting the type of 'second' (line 492)
        second_187357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 31), 'second', False)
        # Processing the call keyword arguments (line 492)
        kwargs_187358 = {}
        # Getting the type of 'type' (line 492)
        type_187356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 26), 'type', False)
        # Calling type(args, kwargs) (line 492)
        type_call_result_187359 = invoke(stypy.reporting.localization.Localization(__file__, 492, 26), type_187356, *[second_187357], **kwargs_187358)
        
        
        (may_be_187360, more_types_in_union_187361) = may_be_type(first_187355, type_call_result_187359)

        if may_be_187360:

            if more_types_in_union_187361:
                # Runtime conditional SSA (line 492)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'first' (line 492)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'first', type_call_result_187359())
            
            # Assigning a Call to a Name (line 493):
            
            # Assigning a Call to a Name (line 493):
            
            # Call to get(...): (line 493)
            # Processing the call arguments (line 493)
            
            # Call to type(...): (line 493)
            # Processing the call arguments (line 493)
            # Getting the type of 'first' (line 493)
            first_187366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 58), 'first', False)
            # Processing the call keyword arguments (line 493)
            kwargs_187367 = {}
            # Getting the type of 'type' (line 493)
            type_187365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 53), 'type', False)
            # Calling type(args, kwargs) (line 493)
            type_call_result_187368 = invoke(stypy.reporting.localization.Localization(__file__, 493, 53), type_187365, *[first_187366], **kwargs_187367)
            
            # Processing the call keyword arguments (line 493)
            kwargs_187369 = {}
            # Getting the type of 'self' (line 493)
            self_187362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 23), 'self', False)
            # Obtaining the member '_type_equality_funcs' of a type (line 493)
            _type_equality_funcs_187363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 23), self_187362, '_type_equality_funcs')
            # Obtaining the member 'get' of a type (line 493)
            get_187364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 23), _type_equality_funcs_187363, 'get')
            # Calling get(args, kwargs) (line 493)
            get_call_result_187370 = invoke(stypy.reporting.localization.Localization(__file__, 493, 23), get_187364, *[type_call_result_187368], **kwargs_187369)
            
            # Assigning a type to the variable 'asserter' (line 493)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'asserter', get_call_result_187370)
            
            # Type idiom detected: calculating its left and rigth part (line 494)
            # Getting the type of 'asserter' (line 494)
            asserter_187371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'asserter')
            # Getting the type of 'None' (line 494)
            None_187372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 31), 'None')
            
            (may_be_187373, more_types_in_union_187374) = may_not_be_none(asserter_187371, None_187372)

            if may_be_187373:

                if more_types_in_union_187374:
                    # Runtime conditional SSA (line 494)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Type idiom detected: calculating its left and rigth part (line 495)
                # Getting the type of 'basestring' (line 495)
                basestring_187375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 40), 'basestring')
                # Getting the type of 'asserter' (line 495)
                asserter_187376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 30), 'asserter')
                
                (may_be_187377, more_types_in_union_187378) = may_be_subtype(basestring_187375, asserter_187376)

                if may_be_187377:

                    if more_types_in_union_187378:
                        # Runtime conditional SSA (line 495)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'asserter' (line 495)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'asserter', remove_not_subtype_from_union(asserter_187376, basestring))
                    
                    # Assigning a Call to a Name (line 496):
                    
                    # Assigning a Call to a Name (line 496):
                    
                    # Call to getattr(...): (line 496)
                    # Processing the call arguments (line 496)
                    # Getting the type of 'self' (line 496)
                    self_187380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 39), 'self', False)
                    # Getting the type of 'asserter' (line 496)
                    asserter_187381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 45), 'asserter', False)
                    # Processing the call keyword arguments (line 496)
                    kwargs_187382 = {}
                    # Getting the type of 'getattr' (line 496)
                    getattr_187379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 31), 'getattr', False)
                    # Calling getattr(args, kwargs) (line 496)
                    getattr_call_result_187383 = invoke(stypy.reporting.localization.Localization(__file__, 496, 31), getattr_187379, *[self_187380, asserter_187381], **kwargs_187382)
                    
                    # Assigning a type to the variable 'asserter' (line 496)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 20), 'asserter', getattr_call_result_187383)

                    if more_types_in_union_187378:
                        # SSA join for if statement (line 495)
                        module_type_store = module_type_store.join_ssa_context()


                
                # Getting the type of 'asserter' (line 497)
                asserter_187384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 23), 'asserter')
                # Assigning a type to the variable 'stypy_return_type' (line 497)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'stypy_return_type', asserter_187384)

                if more_types_in_union_187374:
                    # SSA join for if statement (line 494)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_187361:
                # SSA join for if statement (line 492)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 499)
        self_187385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), 'self')
        # Obtaining the member '_baseAssertEqual' of a type (line 499)
        _baseAssertEqual_187386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 15), self_187385, '_baseAssertEqual')
        # Assigning a type to the variable 'stypy_return_type' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'stypy_return_type', _baseAssertEqual_187386)
        
        # ################# End of '_getAssertEqualityFunc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_getAssertEqualityFunc' in the type store
        # Getting the type of 'stypy_return_type' (line 475)
        stypy_return_type_187387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_getAssertEqualityFunc'
        return stypy_return_type_187387


    @norecursion
    def _baseAssertEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 501)
        None_187388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 50), 'None')
        defaults = [None_187388]
        # Create a new context for function '_baseAssertEqual'
        module_type_store = module_type_store.open_function_context('_baseAssertEqual', 501, 4, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_function_name', 'TestCase._baseAssertEqual')
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_param_names_list', ['first', 'second', 'msg'])
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase._baseAssertEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase._baseAssertEqual', ['first', 'second', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_baseAssertEqual', localization, ['first', 'second', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_baseAssertEqual(...)' code ##################

        str_187389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 8), 'str', 'The default assertEqual implementation, not type specific.')
        
        
        
        # Getting the type of 'first' (line 503)
        first_187390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 15), 'first')
        # Getting the type of 'second' (line 503)
        second_187391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 24), 'second')
        # Applying the binary operator '==' (line 503)
        result_eq_187392 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 15), '==', first_187390, second_187391)
        
        # Applying the 'not' unary operator (line 503)
        result_not__187393 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 11), 'not', result_eq_187392)
        
        # Testing the type of an if condition (line 503)
        if_condition_187394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 503, 8), result_not__187393)
        # Assigning a type to the variable 'if_condition_187394' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'if_condition_187394', if_condition_187394)
        # SSA begins for if statement (line 503)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 504):
        
        # Assigning a BinOp to a Name (line 504):
        str_187395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 26), 'str', '%s != %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 504)
        tuple_187396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 504)
        # Adding element type (line 504)
        
        # Call to safe_repr(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'first' (line 504)
        first_187398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 50), 'first', False)
        # Processing the call keyword arguments (line 504)
        kwargs_187399 = {}
        # Getting the type of 'safe_repr' (line 504)
        safe_repr_187397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 40), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 504)
        safe_repr_call_result_187400 = invoke(stypy.reporting.localization.Localization(__file__, 504, 40), safe_repr_187397, *[first_187398], **kwargs_187399)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 40), tuple_187396, safe_repr_call_result_187400)
        # Adding element type (line 504)
        
        # Call to safe_repr(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'second' (line 504)
        second_187402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 68), 'second', False)
        # Processing the call keyword arguments (line 504)
        kwargs_187403 = {}
        # Getting the type of 'safe_repr' (line 504)
        safe_repr_187401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 58), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 504)
        safe_repr_call_result_187404 = invoke(stypy.reporting.localization.Localization(__file__, 504, 58), safe_repr_187401, *[second_187402], **kwargs_187403)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 40), tuple_187396, safe_repr_call_result_187404)
        
        # Applying the binary operator '%' (line 504)
        result_mod_187405 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 26), '%', str_187395, tuple_187396)
        
        # Assigning a type to the variable 'standardMsg' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'standardMsg', result_mod_187405)
        
        # Assigning a Call to a Name (line 505):
        
        # Assigning a Call to a Name (line 505):
        
        # Call to _formatMessage(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'msg' (line 505)
        msg_187408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 38), 'msg', False)
        # Getting the type of 'standardMsg' (line 505)
        standardMsg_187409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 43), 'standardMsg', False)
        # Processing the call keyword arguments (line 505)
        kwargs_187410 = {}
        # Getting the type of 'self' (line 505)
        self_187406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 18), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 505)
        _formatMessage_187407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 18), self_187406, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 505)
        _formatMessage_call_result_187411 = invoke(stypy.reporting.localization.Localization(__file__, 505, 18), _formatMessage_187407, *[msg_187408, standardMsg_187409], **kwargs_187410)
        
        # Assigning a type to the variable 'msg' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'msg', _formatMessage_call_result_187411)
        
        # Call to failureException(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'msg' (line 506)
        msg_187414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 40), 'msg', False)
        # Processing the call keyword arguments (line 506)
        kwargs_187415 = {}
        # Getting the type of 'self' (line 506)
        self_187412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 18), 'self', False)
        # Obtaining the member 'failureException' of a type (line 506)
        failureException_187413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 18), self_187412, 'failureException')
        # Calling failureException(args, kwargs) (line 506)
        failureException_call_result_187416 = invoke(stypy.reporting.localization.Localization(__file__, 506, 18), failureException_187413, *[msg_187414], **kwargs_187415)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 506, 12), failureException_call_result_187416, 'raise parameter', BaseException)
        # SSA join for if statement (line 503)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_baseAssertEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_baseAssertEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 501)
        stypy_return_type_187417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187417)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_baseAssertEqual'
        return stypy_return_type_187417


    @norecursion
    def assertEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 508)
        None_187418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 45), 'None')
        defaults = [None_187418]
        # Create a new context for function 'assertEqual'
        module_type_store = module_type_store.open_function_context('assertEqual', 508, 4, False)
        # Assigning a type to the variable 'self' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertEqual')
        TestCase.assertEqual.__dict__.__setitem__('stypy_param_names_list', ['first', 'second', 'msg'])
        TestCase.assertEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertEqual', ['first', 'second', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertEqual', localization, ['first', 'second', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertEqual(...)' code ##################

        str_187419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, (-1)), 'str', "Fail if the two objects are unequal as determined by the '=='\n           operator.\n        ")
        
        # Assigning a Call to a Name (line 512):
        
        # Assigning a Call to a Name (line 512):
        
        # Call to _getAssertEqualityFunc(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'first' (line 512)
        first_187422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 53), 'first', False)
        # Getting the type of 'second' (line 512)
        second_187423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 60), 'second', False)
        # Processing the call keyword arguments (line 512)
        kwargs_187424 = {}
        # Getting the type of 'self' (line 512)
        self_187420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 25), 'self', False)
        # Obtaining the member '_getAssertEqualityFunc' of a type (line 512)
        _getAssertEqualityFunc_187421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 25), self_187420, '_getAssertEqualityFunc')
        # Calling _getAssertEqualityFunc(args, kwargs) (line 512)
        _getAssertEqualityFunc_call_result_187425 = invoke(stypy.reporting.localization.Localization(__file__, 512, 25), _getAssertEqualityFunc_187421, *[first_187422, second_187423], **kwargs_187424)
        
        # Assigning a type to the variable 'assertion_func' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'assertion_func', _getAssertEqualityFunc_call_result_187425)
        
        # Call to assertion_func(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'first' (line 513)
        first_187427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 23), 'first', False)
        # Getting the type of 'second' (line 513)
        second_187428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 30), 'second', False)
        # Processing the call keyword arguments (line 513)
        # Getting the type of 'msg' (line 513)
        msg_187429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 42), 'msg', False)
        keyword_187430 = msg_187429
        kwargs_187431 = {'msg': keyword_187430}
        # Getting the type of 'assertion_func' (line 513)
        assertion_func_187426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'assertion_func', False)
        # Calling assertion_func(args, kwargs) (line 513)
        assertion_func_call_result_187432 = invoke(stypy.reporting.localization.Localization(__file__, 513, 8), assertion_func_187426, *[first_187427, second_187428], **kwargs_187431)
        
        
        # ################# End of 'assertEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 508)
        stypy_return_type_187433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187433)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertEqual'
        return stypy_return_type_187433


    @norecursion
    def assertNotEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 515)
        None_187434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 48), 'None')
        defaults = [None_187434]
        # Create a new context for function 'assertNotEqual'
        module_type_store = module_type_store.open_function_context('assertNotEqual', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertNotEqual')
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_param_names_list', ['first', 'second', 'msg'])
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertNotEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertNotEqual', ['first', 'second', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertNotEqual', localization, ['first', 'second', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertNotEqual(...)' code ##################

        str_187435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, (-1)), 'str', "Fail if the two objects are equal as determined by the '!='\n           operator.\n        ")
        
        
        
        # Getting the type of 'first' (line 519)
        first_187436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'first')
        # Getting the type of 'second' (line 519)
        second_187437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 24), 'second')
        # Applying the binary operator '!=' (line 519)
        result_ne_187438 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 15), '!=', first_187436, second_187437)
        
        # Applying the 'not' unary operator (line 519)
        result_not__187439 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 11), 'not', result_ne_187438)
        
        # Testing the type of an if condition (line 519)
        if_condition_187440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 519, 8), result_not__187439)
        # Assigning a type to the variable 'if_condition_187440' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'if_condition_187440', if_condition_187440)
        # SSA begins for if statement (line 519)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 520):
        
        # Assigning a Call to a Name (line 520):
        
        # Call to _formatMessage(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'msg' (line 520)
        msg_187443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 38), 'msg', False)
        str_187444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 43), 'str', '%s == %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 520)
        tuple_187445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 520)
        # Adding element type (line 520)
        
        # Call to safe_repr(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'first' (line 520)
        first_187447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 67), 'first', False)
        # Processing the call keyword arguments (line 520)
        kwargs_187448 = {}
        # Getting the type of 'safe_repr' (line 520)
        safe_repr_187446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 57), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 520)
        safe_repr_call_result_187449 = invoke(stypy.reporting.localization.Localization(__file__, 520, 57), safe_repr_187446, *[first_187447], **kwargs_187448)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 57), tuple_187445, safe_repr_call_result_187449)
        # Adding element type (line 520)
        
        # Call to safe_repr(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'second' (line 521)
        second_187451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 68), 'second', False)
        # Processing the call keyword arguments (line 521)
        kwargs_187452 = {}
        # Getting the type of 'safe_repr' (line 521)
        safe_repr_187450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 58), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 521)
        safe_repr_call_result_187453 = invoke(stypy.reporting.localization.Localization(__file__, 521, 58), safe_repr_187450, *[second_187451], **kwargs_187452)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 57), tuple_187445, safe_repr_call_result_187453)
        
        # Applying the binary operator '%' (line 520)
        result_mod_187454 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 43), '%', str_187444, tuple_187445)
        
        # Processing the call keyword arguments (line 520)
        kwargs_187455 = {}
        # Getting the type of 'self' (line 520)
        self_187441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 18), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 520)
        _formatMessage_187442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 18), self_187441, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 520)
        _formatMessage_call_result_187456 = invoke(stypy.reporting.localization.Localization(__file__, 520, 18), _formatMessage_187442, *[msg_187443, result_mod_187454], **kwargs_187455)
        
        # Assigning a type to the variable 'msg' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'msg', _formatMessage_call_result_187456)
        
        # Call to failureException(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'msg' (line 522)
        msg_187459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 40), 'msg', False)
        # Processing the call keyword arguments (line 522)
        kwargs_187460 = {}
        # Getting the type of 'self' (line 522)
        self_187457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 18), 'self', False)
        # Obtaining the member 'failureException' of a type (line 522)
        failureException_187458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 18), self_187457, 'failureException')
        # Calling failureException(args, kwargs) (line 522)
        failureException_call_result_187461 = invoke(stypy.reporting.localization.Localization(__file__, 522, 18), failureException_187458, *[msg_187459], **kwargs_187460)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 522, 12), failureException_call_result_187461, 'raise parameter', BaseException)
        # SSA join for if statement (line 519)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertNotEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertNotEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_187462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187462)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertNotEqual'
        return stypy_return_type_187462


    @norecursion
    def assertAlmostEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 525)
        None_187463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 54), 'None')
        # Getting the type of 'None' (line 525)
        None_187464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 64), 'None')
        # Getting the type of 'None' (line 525)
        None_187465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 76), 'None')
        defaults = [None_187463, None_187464, None_187465]
        # Create a new context for function 'assertAlmostEqual'
        module_type_store = module_type_store.open_function_context('assertAlmostEqual', 525, 4, False)
        # Assigning a type to the variable 'self' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertAlmostEqual')
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_param_names_list', ['first', 'second', 'places', 'msg', 'delta'])
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertAlmostEqual.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertAlmostEqual', ['first', 'second', 'places', 'msg', 'delta'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertAlmostEqual', localization, ['first', 'second', 'places', 'msg', 'delta'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertAlmostEqual(...)' code ##################

        str_187466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, (-1)), 'str', 'Fail if the two objects are unequal as determined by their\n           difference rounded to the given number of decimal places\n           (default 7) and comparing to zero, or by comparing that the\n           between the two objects is more than the given delta.\n\n           Note that decimal places (from zero) are usually not the same\n           as significant digits (measured from the most signficant digit).\n\n           If the two objects compare equal then they will automatically\n           compare almost equal.\n        ')
        
        
        # Getting the type of 'first' (line 537)
        first_187467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 11), 'first')
        # Getting the type of 'second' (line 537)
        second_187468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 20), 'second')
        # Applying the binary operator '==' (line 537)
        result_eq_187469 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 11), '==', first_187467, second_187468)
        
        # Testing the type of an if condition (line 537)
        if_condition_187470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 537, 8), result_eq_187469)
        # Assigning a type to the variable 'if_condition_187470' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'if_condition_187470', if_condition_187470)
        # SSA begins for if statement (line 537)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 537)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'delta' (line 540)
        delta_187471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 11), 'delta')
        # Getting the type of 'None' (line 540)
        None_187472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 24), 'None')
        # Applying the binary operator 'isnot' (line 540)
        result_is_not_187473 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 11), 'isnot', delta_187471, None_187472)
        
        
        # Getting the type of 'places' (line 540)
        places_187474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 33), 'places')
        # Getting the type of 'None' (line 540)
        None_187475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 47), 'None')
        # Applying the binary operator 'isnot' (line 540)
        result_is_not_187476 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 33), 'isnot', places_187474, None_187475)
        
        # Applying the binary operator 'and' (line 540)
        result_and_keyword_187477 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 11), 'and', result_is_not_187473, result_is_not_187476)
        
        # Testing the type of an if condition (line 540)
        if_condition_187478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 8), result_and_keyword_187477)
        # Assigning a type to the variable 'if_condition_187478' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'if_condition_187478', if_condition_187478)
        # SSA begins for if statement (line 540)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 541)
        # Processing the call arguments (line 541)
        str_187480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 28), 'str', 'specify delta or places not both')
        # Processing the call keyword arguments (line 541)
        kwargs_187481 = {}
        # Getting the type of 'TypeError' (line 541)
        TypeError_187479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 541)
        TypeError_call_result_187482 = invoke(stypy.reporting.localization.Localization(__file__, 541, 18), TypeError_187479, *[str_187480], **kwargs_187481)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 541, 12), TypeError_call_result_187482, 'raise parameter', BaseException)
        # SSA join for if statement (line 540)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 543)
        # Getting the type of 'delta' (line 543)
        delta_187483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'delta')
        # Getting the type of 'None' (line 543)
        None_187484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 24), 'None')
        
        (may_be_187485, more_types_in_union_187486) = may_not_be_none(delta_187483, None_187484)

        if may_be_187485:

            if more_types_in_union_187486:
                # Runtime conditional SSA (line 543)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            
            # Call to abs(...): (line 544)
            # Processing the call arguments (line 544)
            # Getting the type of 'first' (line 544)
            first_187488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 19), 'first', False)
            # Getting the type of 'second' (line 544)
            second_187489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 27), 'second', False)
            # Applying the binary operator '-' (line 544)
            result_sub_187490 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 19), '-', first_187488, second_187489)
            
            # Processing the call keyword arguments (line 544)
            kwargs_187491 = {}
            # Getting the type of 'abs' (line 544)
            abs_187487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 15), 'abs', False)
            # Calling abs(args, kwargs) (line 544)
            abs_call_result_187492 = invoke(stypy.reporting.localization.Localization(__file__, 544, 15), abs_187487, *[result_sub_187490], **kwargs_187491)
            
            # Getting the type of 'delta' (line 544)
            delta_187493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 38), 'delta')
            # Applying the binary operator '<=' (line 544)
            result_le_187494 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 15), '<=', abs_call_result_187492, delta_187493)
            
            # Testing the type of an if condition (line 544)
            if_condition_187495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 544, 12), result_le_187494)
            # Assigning a type to the variable 'if_condition_187495' (line 544)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'if_condition_187495', if_condition_187495)
            # SSA begins for if statement (line 544)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 545)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 544)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 547):
            
            # Assigning a BinOp to a Name (line 547):
            str_187496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 26), 'str', '%s != %s within %s delta')
            
            # Obtaining an instance of the builtin type 'tuple' (line 547)
            tuple_187497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 56), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 547)
            # Adding element type (line 547)
            
            # Call to safe_repr(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'first' (line 547)
            first_187499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 66), 'first', False)
            # Processing the call keyword arguments (line 547)
            kwargs_187500 = {}
            # Getting the type of 'safe_repr' (line 547)
            safe_repr_187498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 56), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 547)
            safe_repr_call_result_187501 = invoke(stypy.reporting.localization.Localization(__file__, 547, 56), safe_repr_187498, *[first_187499], **kwargs_187500)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 56), tuple_187497, safe_repr_call_result_187501)
            # Adding element type (line 547)
            
            # Call to safe_repr(...): (line 548)
            # Processing the call arguments (line 548)
            # Getting the type of 'second' (line 548)
            second_187503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 66), 'second', False)
            # Processing the call keyword arguments (line 548)
            kwargs_187504 = {}
            # Getting the type of 'safe_repr' (line 548)
            safe_repr_187502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 56), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 548)
            safe_repr_call_result_187505 = invoke(stypy.reporting.localization.Localization(__file__, 548, 56), safe_repr_187502, *[second_187503], **kwargs_187504)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 56), tuple_187497, safe_repr_call_result_187505)
            # Adding element type (line 547)
            
            # Call to safe_repr(...): (line 549)
            # Processing the call arguments (line 549)
            # Getting the type of 'delta' (line 549)
            delta_187507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 66), 'delta', False)
            # Processing the call keyword arguments (line 549)
            kwargs_187508 = {}
            # Getting the type of 'safe_repr' (line 549)
            safe_repr_187506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 56), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 549)
            safe_repr_call_result_187509 = invoke(stypy.reporting.localization.Localization(__file__, 549, 56), safe_repr_187506, *[delta_187507], **kwargs_187508)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 547, 56), tuple_187497, safe_repr_call_result_187509)
            
            # Applying the binary operator '%' (line 547)
            result_mod_187510 = python_operator(stypy.reporting.localization.Localization(__file__, 547, 26), '%', str_187496, tuple_187497)
            
            # Assigning a type to the variable 'standardMsg' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'standardMsg', result_mod_187510)

            if more_types_in_union_187486:
                # Runtime conditional SSA for else branch (line 543)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_187485) or more_types_in_union_187486):
            
            # Type idiom detected: calculating its left and rigth part (line 551)
            # Getting the type of 'places' (line 551)
            places_187511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 15), 'places')
            # Getting the type of 'None' (line 551)
            None_187512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 25), 'None')
            
            (may_be_187513, more_types_in_union_187514) = may_be_none(places_187511, None_187512)

            if may_be_187513:

                if more_types_in_union_187514:
                    # Runtime conditional SSA (line 551)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Num to a Name (line 552):
                
                # Assigning a Num to a Name (line 552):
                int_187515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 25), 'int')
                # Assigning a type to the variable 'places' (line 552)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'places', int_187515)

                if more_types_in_union_187514:
                    # SSA join for if statement (line 551)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            
            # Call to round(...): (line 554)
            # Processing the call arguments (line 554)
            
            # Call to abs(...): (line 554)
            # Processing the call arguments (line 554)
            # Getting the type of 'second' (line 554)
            second_187518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 25), 'second', False)
            # Getting the type of 'first' (line 554)
            first_187519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 32), 'first', False)
            # Applying the binary operator '-' (line 554)
            result_sub_187520 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 25), '-', second_187518, first_187519)
            
            # Processing the call keyword arguments (line 554)
            kwargs_187521 = {}
            # Getting the type of 'abs' (line 554)
            abs_187517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'abs', False)
            # Calling abs(args, kwargs) (line 554)
            abs_call_result_187522 = invoke(stypy.reporting.localization.Localization(__file__, 554, 21), abs_187517, *[result_sub_187520], **kwargs_187521)
            
            # Getting the type of 'places' (line 554)
            places_187523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 40), 'places', False)
            # Processing the call keyword arguments (line 554)
            kwargs_187524 = {}
            # Getting the type of 'round' (line 554)
            round_187516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 15), 'round', False)
            # Calling round(args, kwargs) (line 554)
            round_call_result_187525 = invoke(stypy.reporting.localization.Localization(__file__, 554, 15), round_187516, *[abs_call_result_187522, places_187523], **kwargs_187524)
            
            int_187526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 51), 'int')
            # Applying the binary operator '==' (line 554)
            result_eq_187527 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 15), '==', round_call_result_187525, int_187526)
            
            # Testing the type of an if condition (line 554)
            if_condition_187528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 12), result_eq_187527)
            # Assigning a type to the variable 'if_condition_187528' (line 554)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'if_condition_187528', if_condition_187528)
            # SSA begins for if statement (line 554)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 555)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 16), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 554)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 557):
            
            # Assigning a BinOp to a Name (line 557):
            str_187529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 26), 'str', '%s != %s within %r places')
            
            # Obtaining an instance of the builtin type 'tuple' (line 557)
            tuple_187530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 57), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 557)
            # Adding element type (line 557)
            
            # Call to safe_repr(...): (line 557)
            # Processing the call arguments (line 557)
            # Getting the type of 'first' (line 557)
            first_187532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 67), 'first', False)
            # Processing the call keyword arguments (line 557)
            kwargs_187533 = {}
            # Getting the type of 'safe_repr' (line 557)
            safe_repr_187531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 57), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 557)
            safe_repr_call_result_187534 = invoke(stypy.reporting.localization.Localization(__file__, 557, 57), safe_repr_187531, *[first_187532], **kwargs_187533)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 57), tuple_187530, safe_repr_call_result_187534)
            # Adding element type (line 557)
            
            # Call to safe_repr(...): (line 558)
            # Processing the call arguments (line 558)
            # Getting the type of 'second' (line 558)
            second_187536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 68), 'second', False)
            # Processing the call keyword arguments (line 558)
            kwargs_187537 = {}
            # Getting the type of 'safe_repr' (line 558)
            safe_repr_187535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 58), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 558)
            safe_repr_call_result_187538 = invoke(stypy.reporting.localization.Localization(__file__, 558, 58), safe_repr_187535, *[second_187536], **kwargs_187537)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 57), tuple_187530, safe_repr_call_result_187538)
            # Adding element type (line 557)
            # Getting the type of 'places' (line 559)
            places_187539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 58), 'places')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 57), tuple_187530, places_187539)
            
            # Applying the binary operator '%' (line 557)
            result_mod_187540 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 26), '%', str_187529, tuple_187530)
            
            # Assigning a type to the variable 'standardMsg' (line 557)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'standardMsg', result_mod_187540)

            if (may_be_187485 and more_types_in_union_187486):
                # SSA join for if statement (line 543)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 560):
        
        # Assigning a Call to a Name (line 560):
        
        # Call to _formatMessage(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'msg' (line 560)
        msg_187543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 34), 'msg', False)
        # Getting the type of 'standardMsg' (line 560)
        standardMsg_187544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 39), 'standardMsg', False)
        # Processing the call keyword arguments (line 560)
        kwargs_187545 = {}
        # Getting the type of 'self' (line 560)
        self_187541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 14), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 560)
        _formatMessage_187542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 14), self_187541, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 560)
        _formatMessage_call_result_187546 = invoke(stypy.reporting.localization.Localization(__file__, 560, 14), _formatMessage_187542, *[msg_187543, standardMsg_187544], **kwargs_187545)
        
        # Assigning a type to the variable 'msg' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'msg', _formatMessage_call_result_187546)
        
        # Call to failureException(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'msg' (line 561)
        msg_187549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 36), 'msg', False)
        # Processing the call keyword arguments (line 561)
        kwargs_187550 = {}
        # Getting the type of 'self' (line 561)
        self_187547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 14), 'self', False)
        # Obtaining the member 'failureException' of a type (line 561)
        failureException_187548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 14), self_187547, 'failureException')
        # Calling failureException(args, kwargs) (line 561)
        failureException_call_result_187551 = invoke(stypy.reporting.localization.Localization(__file__, 561, 14), failureException_187548, *[msg_187549], **kwargs_187550)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 561, 8), failureException_call_result_187551, 'raise parameter', BaseException)
        
        # ################# End of 'assertAlmostEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertAlmostEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 525)
        stypy_return_type_187552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertAlmostEqual'
        return stypy_return_type_187552


    @norecursion
    def assertNotAlmostEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 563)
        None_187553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 57), 'None')
        # Getting the type of 'None' (line 563)
        None_187554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 67), 'None')
        # Getting the type of 'None' (line 563)
        None_187555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 79), 'None')
        defaults = [None_187553, None_187554, None_187555]
        # Create a new context for function 'assertNotAlmostEqual'
        module_type_store = module_type_store.open_function_context('assertNotAlmostEqual', 563, 4, False)
        # Assigning a type to the variable 'self' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertNotAlmostEqual')
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_param_names_list', ['first', 'second', 'places', 'msg', 'delta'])
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertNotAlmostEqual.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertNotAlmostEqual', ['first', 'second', 'places', 'msg', 'delta'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertNotAlmostEqual', localization, ['first', 'second', 'places', 'msg', 'delta'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertNotAlmostEqual(...)' code ##################

        str_187556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, (-1)), 'str', 'Fail if the two objects are equal as determined by their\n           difference rounded to the given number of decimal places\n           (default 7) and comparing to zero, or by comparing that the\n           between the two objects is less than the given delta.\n\n           Note that decimal places (from zero) are usually not the same\n           as significant digits (measured from the most signficant digit).\n\n           Objects that are equal automatically fail.\n        ')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'delta' (line 574)
        delta_187557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 11), 'delta')
        # Getting the type of 'None' (line 574)
        None_187558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 24), 'None')
        # Applying the binary operator 'isnot' (line 574)
        result_is_not_187559 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 11), 'isnot', delta_187557, None_187558)
        
        
        # Getting the type of 'places' (line 574)
        places_187560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 33), 'places')
        # Getting the type of 'None' (line 574)
        None_187561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 47), 'None')
        # Applying the binary operator 'isnot' (line 574)
        result_is_not_187562 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 33), 'isnot', places_187560, None_187561)
        
        # Applying the binary operator 'and' (line 574)
        result_and_keyword_187563 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 11), 'and', result_is_not_187559, result_is_not_187562)
        
        # Testing the type of an if condition (line 574)
        if_condition_187564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 8), result_and_keyword_187563)
        # Assigning a type to the variable 'if_condition_187564' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'if_condition_187564', if_condition_187564)
        # SSA begins for if statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 575)
        # Processing the call arguments (line 575)
        str_187566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 28), 'str', 'specify delta or places not both')
        # Processing the call keyword arguments (line 575)
        kwargs_187567 = {}
        # Getting the type of 'TypeError' (line 575)
        TypeError_187565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 575)
        TypeError_call_result_187568 = invoke(stypy.reporting.localization.Localization(__file__, 575, 18), TypeError_187565, *[str_187566], **kwargs_187567)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 575, 12), TypeError_call_result_187568, 'raise parameter', BaseException)
        # SSA join for if statement (line 574)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 576)
        # Getting the type of 'delta' (line 576)
        delta_187569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'delta')
        # Getting the type of 'None' (line 576)
        None_187570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 24), 'None')
        
        (may_be_187571, more_types_in_union_187572) = may_not_be_none(delta_187569, None_187570)

        if may_be_187571:

            if more_types_in_union_187572:
                # Runtime conditional SSA (line 576)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Evaluating a boolean operation
            
            
            # Getting the type of 'first' (line 577)
            first_187573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 20), 'first')
            # Getting the type of 'second' (line 577)
            second_187574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 29), 'second')
            # Applying the binary operator '==' (line 577)
            result_eq_187575 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 20), '==', first_187573, second_187574)
            
            # Applying the 'not' unary operator (line 577)
            result_not__187576 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 15), 'not', result_eq_187575)
            
            
            
            # Call to abs(...): (line 577)
            # Processing the call arguments (line 577)
            # Getting the type of 'first' (line 577)
            first_187578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 45), 'first', False)
            # Getting the type of 'second' (line 577)
            second_187579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 53), 'second', False)
            # Applying the binary operator '-' (line 577)
            result_sub_187580 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 45), '-', first_187578, second_187579)
            
            # Processing the call keyword arguments (line 577)
            kwargs_187581 = {}
            # Getting the type of 'abs' (line 577)
            abs_187577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 41), 'abs', False)
            # Calling abs(args, kwargs) (line 577)
            abs_call_result_187582 = invoke(stypy.reporting.localization.Localization(__file__, 577, 41), abs_187577, *[result_sub_187580], **kwargs_187581)
            
            # Getting the type of 'delta' (line 577)
            delta_187583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 63), 'delta')
            # Applying the binary operator '>' (line 577)
            result_gt_187584 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 41), '>', abs_call_result_187582, delta_187583)
            
            # Applying the binary operator 'and' (line 577)
            result_and_keyword_187585 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 15), 'and', result_not__187576, result_gt_187584)
            
            # Testing the type of an if condition (line 577)
            if_condition_187586 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 577, 12), result_and_keyword_187585)
            # Assigning a type to the variable 'if_condition_187586' (line 577)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'if_condition_187586', if_condition_187586)
            # SSA begins for if statement (line 577)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 578)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 16), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 577)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 579):
            
            # Assigning a BinOp to a Name (line 579):
            str_187587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 26), 'str', '%s == %s within %s delta')
            
            # Obtaining an instance of the builtin type 'tuple' (line 579)
            tuple_187588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 56), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 579)
            # Adding element type (line 579)
            
            # Call to safe_repr(...): (line 579)
            # Processing the call arguments (line 579)
            # Getting the type of 'first' (line 579)
            first_187590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 66), 'first', False)
            # Processing the call keyword arguments (line 579)
            kwargs_187591 = {}
            # Getting the type of 'safe_repr' (line 579)
            safe_repr_187589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 56), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 579)
            safe_repr_call_result_187592 = invoke(stypy.reporting.localization.Localization(__file__, 579, 56), safe_repr_187589, *[first_187590], **kwargs_187591)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 56), tuple_187588, safe_repr_call_result_187592)
            # Adding element type (line 579)
            
            # Call to safe_repr(...): (line 580)
            # Processing the call arguments (line 580)
            # Getting the type of 'second' (line 580)
            second_187594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 66), 'second', False)
            # Processing the call keyword arguments (line 580)
            kwargs_187595 = {}
            # Getting the type of 'safe_repr' (line 580)
            safe_repr_187593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 56), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 580)
            safe_repr_call_result_187596 = invoke(stypy.reporting.localization.Localization(__file__, 580, 56), safe_repr_187593, *[second_187594], **kwargs_187595)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 56), tuple_187588, safe_repr_call_result_187596)
            # Adding element type (line 579)
            
            # Call to safe_repr(...): (line 581)
            # Processing the call arguments (line 581)
            # Getting the type of 'delta' (line 581)
            delta_187598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 66), 'delta', False)
            # Processing the call keyword arguments (line 581)
            kwargs_187599 = {}
            # Getting the type of 'safe_repr' (line 581)
            safe_repr_187597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 56), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 581)
            safe_repr_call_result_187600 = invoke(stypy.reporting.localization.Localization(__file__, 581, 56), safe_repr_187597, *[delta_187598], **kwargs_187599)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 56), tuple_187588, safe_repr_call_result_187600)
            
            # Applying the binary operator '%' (line 579)
            result_mod_187601 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 26), '%', str_187587, tuple_187588)
            
            # Assigning a type to the variable 'standardMsg' (line 579)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'standardMsg', result_mod_187601)

            if more_types_in_union_187572:
                # Runtime conditional SSA for else branch (line 576)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_187571) or more_types_in_union_187572):
            
            # Type idiom detected: calculating its left and rigth part (line 583)
            # Getting the type of 'places' (line 583)
            places_187602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 15), 'places')
            # Getting the type of 'None' (line 583)
            None_187603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 25), 'None')
            
            (may_be_187604, more_types_in_union_187605) = may_be_none(places_187602, None_187603)

            if may_be_187604:

                if more_types_in_union_187605:
                    # Runtime conditional SSA (line 583)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Num to a Name (line 584):
                
                # Assigning a Num to a Name (line 584):
                int_187606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 25), 'int')
                # Assigning a type to the variable 'places' (line 584)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'places', int_187606)

                if more_types_in_union_187605:
                    # SSA join for if statement (line 583)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Evaluating a boolean operation
            
            
            # Getting the type of 'first' (line 585)
            first_187607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'first')
            # Getting the type of 'second' (line 585)
            second_187608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 29), 'second')
            # Applying the binary operator '==' (line 585)
            result_eq_187609 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 20), '==', first_187607, second_187608)
            
            # Applying the 'not' unary operator (line 585)
            result_not__187610 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 15), 'not', result_eq_187609)
            
            
            
            # Call to round(...): (line 585)
            # Processing the call arguments (line 585)
            
            # Call to abs(...): (line 585)
            # Processing the call arguments (line 585)
            # Getting the type of 'second' (line 585)
            second_187613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 51), 'second', False)
            # Getting the type of 'first' (line 585)
            first_187614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 58), 'first', False)
            # Applying the binary operator '-' (line 585)
            result_sub_187615 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 51), '-', second_187613, first_187614)
            
            # Processing the call keyword arguments (line 585)
            kwargs_187616 = {}
            # Getting the type of 'abs' (line 585)
            abs_187612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 47), 'abs', False)
            # Calling abs(args, kwargs) (line 585)
            abs_call_result_187617 = invoke(stypy.reporting.localization.Localization(__file__, 585, 47), abs_187612, *[result_sub_187615], **kwargs_187616)
            
            # Getting the type of 'places' (line 585)
            places_187618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 66), 'places', False)
            # Processing the call keyword arguments (line 585)
            kwargs_187619 = {}
            # Getting the type of 'round' (line 585)
            round_187611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 41), 'round', False)
            # Calling round(args, kwargs) (line 585)
            round_call_result_187620 = invoke(stypy.reporting.localization.Localization(__file__, 585, 41), round_187611, *[abs_call_result_187617, places_187618], **kwargs_187619)
            
            int_187621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 77), 'int')
            # Applying the binary operator '!=' (line 585)
            result_ne_187622 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 41), '!=', round_call_result_187620, int_187621)
            
            # Applying the binary operator 'and' (line 585)
            result_and_keyword_187623 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 15), 'and', result_not__187610, result_ne_187622)
            
            # Testing the type of an if condition (line 585)
            if_condition_187624 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 12), result_and_keyword_187623)
            # Assigning a type to the variable 'if_condition_187624' (line 585)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'if_condition_187624', if_condition_187624)
            # SSA begins for if statement (line 585)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 586)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 585)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 587):
            
            # Assigning a BinOp to a Name (line 587):
            str_187625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 26), 'str', '%s == %s within %r places')
            
            # Obtaining an instance of the builtin type 'tuple' (line 587)
            tuple_187626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 57), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 587)
            # Adding element type (line 587)
            
            # Call to safe_repr(...): (line 587)
            # Processing the call arguments (line 587)
            # Getting the type of 'first' (line 587)
            first_187628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 67), 'first', False)
            # Processing the call keyword arguments (line 587)
            kwargs_187629 = {}
            # Getting the type of 'safe_repr' (line 587)
            safe_repr_187627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 57), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 587)
            safe_repr_call_result_187630 = invoke(stypy.reporting.localization.Localization(__file__, 587, 57), safe_repr_187627, *[first_187628], **kwargs_187629)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 57), tuple_187626, safe_repr_call_result_187630)
            # Adding element type (line 587)
            
            # Call to safe_repr(...): (line 588)
            # Processing the call arguments (line 588)
            # Getting the type of 'second' (line 588)
            second_187632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 67), 'second', False)
            # Processing the call keyword arguments (line 588)
            kwargs_187633 = {}
            # Getting the type of 'safe_repr' (line 588)
            safe_repr_187631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 57), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 588)
            safe_repr_call_result_187634 = invoke(stypy.reporting.localization.Localization(__file__, 588, 57), safe_repr_187631, *[second_187632], **kwargs_187633)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 57), tuple_187626, safe_repr_call_result_187634)
            # Adding element type (line 587)
            # Getting the type of 'places' (line 589)
            places_187635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 57), 'places')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 57), tuple_187626, places_187635)
            
            # Applying the binary operator '%' (line 587)
            result_mod_187636 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 26), '%', str_187625, tuple_187626)
            
            # Assigning a type to the variable 'standardMsg' (line 587)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'standardMsg', result_mod_187636)

            if (may_be_187571 and more_types_in_union_187572):
                # SSA join for if statement (line 576)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 591):
        
        # Assigning a Call to a Name (line 591):
        
        # Call to _formatMessage(...): (line 591)
        # Processing the call arguments (line 591)
        # Getting the type of 'msg' (line 591)
        msg_187639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 34), 'msg', False)
        # Getting the type of 'standardMsg' (line 591)
        standardMsg_187640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 39), 'standardMsg', False)
        # Processing the call keyword arguments (line 591)
        kwargs_187641 = {}
        # Getting the type of 'self' (line 591)
        self_187637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 14), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 591)
        _formatMessage_187638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 14), self_187637, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 591)
        _formatMessage_call_result_187642 = invoke(stypy.reporting.localization.Localization(__file__, 591, 14), _formatMessage_187638, *[msg_187639, standardMsg_187640], **kwargs_187641)
        
        # Assigning a type to the variable 'msg' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'msg', _formatMessage_call_result_187642)
        
        # Call to failureException(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'msg' (line 592)
        msg_187645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 36), 'msg', False)
        # Processing the call keyword arguments (line 592)
        kwargs_187646 = {}
        # Getting the type of 'self' (line 592)
        self_187643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 14), 'self', False)
        # Obtaining the member 'failureException' of a type (line 592)
        failureException_187644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 14), self_187643, 'failureException')
        # Calling failureException(args, kwargs) (line 592)
        failureException_call_result_187647 = invoke(stypy.reporting.localization.Localization(__file__, 592, 14), failureException_187644, *[msg_187645], **kwargs_187646)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 592, 8), failureException_call_result_187647, 'raise parameter', BaseException)
        
        # ################# End of 'assertNotAlmostEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertNotAlmostEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 563)
        stypy_return_type_187648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187648)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertNotAlmostEqual'
        return stypy_return_type_187648

    
    # Assigning a Name to a Name (line 599):
    
    # Assigning a Name to a Name (line 600):
    
    # Assigning a Name to a Name (line 601):
    
    # Assigning a Name to a Name (line 602):
    
    # Assigning a Name to a Name (line 603):

    @staticmethod
    @norecursion
    def _deprecate(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_deprecate'
        module_type_store = module_type_store.open_function_context('_deprecate', 607, 4, False)
        
        # Passed parameters checking function
        TestCase._deprecate.__dict__.__setitem__('stypy_localization', localization)
        TestCase._deprecate.__dict__.__setitem__('stypy_type_of_self', None)
        TestCase._deprecate.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase._deprecate.__dict__.__setitem__('stypy_function_name', '_deprecate')
        TestCase._deprecate.__dict__.__setitem__('stypy_param_names_list', ['original_func'])
        TestCase._deprecate.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase._deprecate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase._deprecate.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase._deprecate.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase._deprecate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase._deprecate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '_deprecate', ['original_func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_deprecate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_deprecate(...)' code ##################


        @norecursion
        def deprecated_func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'deprecated_func'
            module_type_store = module_type_store.open_function_context('deprecated_func', 608, 8, False)
            
            # Passed parameters checking function
            deprecated_func.stypy_localization = localization
            deprecated_func.stypy_type_of_self = None
            deprecated_func.stypy_type_store = module_type_store
            deprecated_func.stypy_function_name = 'deprecated_func'
            deprecated_func.stypy_param_names_list = []
            deprecated_func.stypy_varargs_param_name = 'args'
            deprecated_func.stypy_kwargs_param_name = 'kwargs'
            deprecated_func.stypy_call_defaults = defaults
            deprecated_func.stypy_call_varargs = varargs
            deprecated_func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'deprecated_func', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'deprecated_func', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'deprecated_func(...)' code ##################

            
            # Call to warn(...): (line 609)
            # Processing the call arguments (line 609)
            
            # Call to format(...): (line 610)
            # Processing the call arguments (line 610)
            # Getting the type of 'original_func' (line 610)
            original_func_187653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 49), 'original_func', False)
            # Obtaining the member '__name__' of a type (line 610)
            name___187654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 49), original_func_187653, '__name__')
            # Processing the call keyword arguments (line 610)
            kwargs_187655 = {}
            str_187651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 16), 'str', 'Please use {0} instead.')
            # Obtaining the member 'format' of a type (line 610)
            format_187652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 16), str_187651, 'format')
            # Calling format(args, kwargs) (line 610)
            format_call_result_187656 = invoke(stypy.reporting.localization.Localization(__file__, 610, 16), format_187652, *[name___187654], **kwargs_187655)
            
            # Getting the type of 'PendingDeprecationWarning' (line 611)
            PendingDeprecationWarning_187657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'PendingDeprecationWarning', False)
            int_187658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 43), 'int')
            # Processing the call keyword arguments (line 609)
            kwargs_187659 = {}
            # Getting the type of 'warnings' (line 609)
            warnings_187649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 609)
            warn_187650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 12), warnings_187649, 'warn')
            # Calling warn(args, kwargs) (line 609)
            warn_call_result_187660 = invoke(stypy.reporting.localization.Localization(__file__, 609, 12), warn_187650, *[format_call_result_187656, PendingDeprecationWarning_187657, int_187658], **kwargs_187659)
            
            
            # Call to original_func(...): (line 612)
            # Getting the type of 'args' (line 612)
            args_187662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 34), 'args', False)
            # Processing the call keyword arguments (line 612)
            # Getting the type of 'kwargs' (line 612)
            kwargs_187663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 42), 'kwargs', False)
            kwargs_187664 = {'kwargs_187663': kwargs_187663}
            # Getting the type of 'original_func' (line 612)
            original_func_187661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 19), 'original_func', False)
            # Calling original_func(args, kwargs) (line 612)
            original_func_call_result_187665 = invoke(stypy.reporting.localization.Localization(__file__, 612, 19), original_func_187661, *[args_187662], **kwargs_187664)
            
            # Assigning a type to the variable 'stypy_return_type' (line 612)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'stypy_return_type', original_func_call_result_187665)
            
            # ################# End of 'deprecated_func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'deprecated_func' in the type store
            # Getting the type of 'stypy_return_type' (line 608)
            stypy_return_type_187666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_187666)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'deprecated_func'
            return stypy_return_type_187666

        # Assigning a type to the variable 'deprecated_func' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'deprecated_func', deprecated_func)
        # Getting the type of 'deprecated_func' (line 613)
        deprecated_func_187667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 15), 'deprecated_func')
        # Assigning a type to the variable 'stypy_return_type' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'stypy_return_type', deprecated_func_187667)
        
        # ################# End of '_deprecate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_deprecate' in the type store
        # Getting the type of 'stypy_return_type' (line 607)
        stypy_return_type_187668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187668)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_deprecate'
        return stypy_return_type_187668

    
    # Assigning a Call to a Name (line 615):
    
    # Assigning a Call to a Name (line 616):
    
    # Assigning a Call to a Name (line 617):
    
    # Assigning a Call to a Name (line 618):
    
    # Assigning a Call to a Name (line 619):
    
    # Assigning a Call to a Name (line 620):
    
    # Assigning a Call to a Name (line 621):

    @norecursion
    def assertSequenceEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 623)
        None_187669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 50), 'None')
        # Getting the type of 'None' (line 623)
        None_187670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 65), 'None')
        defaults = [None_187669, None_187670]
        # Create a new context for function 'assertSequenceEqual'
        module_type_store = module_type_store.open_function_context('assertSequenceEqual', 623, 4, False)
        # Assigning a type to the variable 'self' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertSequenceEqual')
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_param_names_list', ['seq1', 'seq2', 'msg', 'seq_type'])
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertSequenceEqual.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertSequenceEqual', ['seq1', 'seq2', 'msg', 'seq_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertSequenceEqual', localization, ['seq1', 'seq2', 'msg', 'seq_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertSequenceEqual(...)' code ##################

        str_187671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, (-1)), 'str', 'An equality assertion for ordered sequences (like lists and tuples).\n\n        For the purposes of this function, a valid ordered sequence type is one\n        which can be indexed, has a length, and has an equality operator.\n\n        Args:\n            seq1: The first sequence to compare.\n            seq2: The second sequence to compare.\n            seq_type: The expected datatype of the sequences, or None if no\n                    datatype should be enforced.\n            msg: Optional message to use on failure instead of a list of\n                    differences.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 637)
        # Getting the type of 'seq_type' (line 637)
        seq_type_187672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'seq_type')
        # Getting the type of 'None' (line 637)
        None_187673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 27), 'None')
        
        (may_be_187674, more_types_in_union_187675) = may_not_be_none(seq_type_187672, None_187673)

        if may_be_187674:

            if more_types_in_union_187675:
                # Runtime conditional SSA (line 637)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 638):
            
            # Assigning a Attribute to a Name (line 638):
            # Getting the type of 'seq_type' (line 638)
            seq_type_187676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 28), 'seq_type')
            # Obtaining the member '__name__' of a type (line 638)
            name___187677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 28), seq_type_187676, '__name__')
            # Assigning a type to the variable 'seq_type_name' (line 638)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 12), 'seq_type_name', name___187677)
            
            
            
            # Call to isinstance(...): (line 639)
            # Processing the call arguments (line 639)
            # Getting the type of 'seq1' (line 639)
            seq1_187679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 30), 'seq1', False)
            # Getting the type of 'seq_type' (line 639)
            seq_type_187680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 36), 'seq_type', False)
            # Processing the call keyword arguments (line 639)
            kwargs_187681 = {}
            # Getting the type of 'isinstance' (line 639)
            isinstance_187678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 639)
            isinstance_call_result_187682 = invoke(stypy.reporting.localization.Localization(__file__, 639, 19), isinstance_187678, *[seq1_187679, seq_type_187680], **kwargs_187681)
            
            # Applying the 'not' unary operator (line 639)
            result_not__187683 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 15), 'not', isinstance_call_result_187682)
            
            # Testing the type of an if condition (line 639)
            if_condition_187684 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 639, 12), result_not__187683)
            # Assigning a type to the variable 'if_condition_187684' (line 639)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'if_condition_187684', if_condition_187684)
            # SSA begins for if statement (line 639)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to failureException(...): (line 640)
            # Processing the call arguments (line 640)
            str_187687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 44), 'str', 'First sequence is not a %s: %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 641)
            tuple_187688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 43), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 641)
            # Adding element type (line 641)
            # Getting the type of 'seq_type_name' (line 641)
            seq_type_name_187689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 43), 'seq_type_name', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 43), tuple_187688, seq_type_name_187689)
            # Adding element type (line 641)
            
            # Call to safe_repr(...): (line 641)
            # Processing the call arguments (line 641)
            # Getting the type of 'seq1' (line 641)
            seq1_187691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 68), 'seq1', False)
            # Processing the call keyword arguments (line 641)
            kwargs_187692 = {}
            # Getting the type of 'safe_repr' (line 641)
            safe_repr_187690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 58), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 641)
            safe_repr_call_result_187693 = invoke(stypy.reporting.localization.Localization(__file__, 641, 58), safe_repr_187690, *[seq1_187691], **kwargs_187692)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 43), tuple_187688, safe_repr_call_result_187693)
            
            # Applying the binary operator '%' (line 640)
            result_mod_187694 = python_operator(stypy.reporting.localization.Localization(__file__, 640, 44), '%', str_187687, tuple_187688)
            
            # Processing the call keyword arguments (line 640)
            kwargs_187695 = {}
            # Getting the type of 'self' (line 640)
            self_187685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 22), 'self', False)
            # Obtaining the member 'failureException' of a type (line 640)
            failureException_187686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 22), self_187685, 'failureException')
            # Calling failureException(args, kwargs) (line 640)
            failureException_call_result_187696 = invoke(stypy.reporting.localization.Localization(__file__, 640, 22), failureException_187686, *[result_mod_187694], **kwargs_187695)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 640, 16), failureException_call_result_187696, 'raise parameter', BaseException)
            # SSA join for if statement (line 639)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Call to isinstance(...): (line 642)
            # Processing the call arguments (line 642)
            # Getting the type of 'seq2' (line 642)
            seq2_187698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 30), 'seq2', False)
            # Getting the type of 'seq_type' (line 642)
            seq_type_187699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 36), 'seq_type', False)
            # Processing the call keyword arguments (line 642)
            kwargs_187700 = {}
            # Getting the type of 'isinstance' (line 642)
            isinstance_187697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 642)
            isinstance_call_result_187701 = invoke(stypy.reporting.localization.Localization(__file__, 642, 19), isinstance_187697, *[seq2_187698, seq_type_187699], **kwargs_187700)
            
            # Applying the 'not' unary operator (line 642)
            result_not__187702 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 15), 'not', isinstance_call_result_187701)
            
            # Testing the type of an if condition (line 642)
            if_condition_187703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 642, 12), result_not__187702)
            # Assigning a type to the variable 'if_condition_187703' (line 642)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'if_condition_187703', if_condition_187703)
            # SSA begins for if statement (line 642)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to failureException(...): (line 643)
            # Processing the call arguments (line 643)
            str_187706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 44), 'str', 'Second sequence is not a %s: %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 644)
            tuple_187707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 43), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 644)
            # Adding element type (line 644)
            # Getting the type of 'seq_type_name' (line 644)
            seq_type_name_187708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 43), 'seq_type_name', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 43), tuple_187707, seq_type_name_187708)
            # Adding element type (line 644)
            
            # Call to safe_repr(...): (line 644)
            # Processing the call arguments (line 644)
            # Getting the type of 'seq2' (line 644)
            seq2_187710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 68), 'seq2', False)
            # Processing the call keyword arguments (line 644)
            kwargs_187711 = {}
            # Getting the type of 'safe_repr' (line 644)
            safe_repr_187709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 58), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 644)
            safe_repr_call_result_187712 = invoke(stypy.reporting.localization.Localization(__file__, 644, 58), safe_repr_187709, *[seq2_187710], **kwargs_187711)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 43), tuple_187707, safe_repr_call_result_187712)
            
            # Applying the binary operator '%' (line 643)
            result_mod_187713 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 44), '%', str_187706, tuple_187707)
            
            # Processing the call keyword arguments (line 643)
            kwargs_187714 = {}
            # Getting the type of 'self' (line 643)
            self_187704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 22), 'self', False)
            # Obtaining the member 'failureException' of a type (line 643)
            failureException_187705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 22), self_187704, 'failureException')
            # Calling failureException(args, kwargs) (line 643)
            failureException_call_result_187715 = invoke(stypy.reporting.localization.Localization(__file__, 643, 22), failureException_187705, *[result_mod_187713], **kwargs_187714)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 643, 16), failureException_call_result_187715, 'raise parameter', BaseException)
            # SSA join for if statement (line 642)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_187675:
                # Runtime conditional SSA for else branch (line 637)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_187674) or more_types_in_union_187675):
            
            # Assigning a Str to a Name (line 646):
            
            # Assigning a Str to a Name (line 646):
            str_187716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 28), 'str', 'sequence')
            # Assigning a type to the variable 'seq_type_name' (line 646)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 12), 'seq_type_name', str_187716)

            if (may_be_187674 and more_types_in_union_187675):
                # SSA join for if statement (line 637)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 648):
        
        # Assigning a Name to a Name (line 648):
        # Getting the type of 'None' (line 648)
        None_187717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 20), 'None')
        # Assigning a type to the variable 'differing' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'differing', None_187717)
        
        
        # SSA begins for try-except statement (line 649)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 650):
        
        # Assigning a Call to a Name (line 650):
        
        # Call to len(...): (line 650)
        # Processing the call arguments (line 650)
        # Getting the type of 'seq1' (line 650)
        seq1_187719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 23), 'seq1', False)
        # Processing the call keyword arguments (line 650)
        kwargs_187720 = {}
        # Getting the type of 'len' (line 650)
        len_187718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 19), 'len', False)
        # Calling len(args, kwargs) (line 650)
        len_call_result_187721 = invoke(stypy.reporting.localization.Localization(__file__, 650, 19), len_187718, *[seq1_187719], **kwargs_187720)
        
        # Assigning a type to the variable 'len1' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'len1', len_call_result_187721)
        # SSA branch for the except part of a try statement (line 649)
        # SSA branch for the except 'Tuple' branch of a try statement (line 649)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a BinOp to a Name (line 652):
        
        # Assigning a BinOp to a Name (line 652):
        str_187722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 24), 'str', 'First %s has no length.    Non-sequence?')
        # Getting the type of 'seq_type_name' (line 653)
        seq_type_name_187723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 20), 'seq_type_name')
        # Applying the binary operator '%' (line 652)
        result_mod_187724 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 24), '%', str_187722, seq_type_name_187723)
        
        # Assigning a type to the variable 'differing' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'differing', result_mod_187724)
        # SSA join for try-except statement (line 649)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 655)
        # Getting the type of 'differing' (line 655)
        differing_187725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 11), 'differing')
        # Getting the type of 'None' (line 655)
        None_187726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 24), 'None')
        
        (may_be_187727, more_types_in_union_187728) = may_be_none(differing_187725, None_187726)

        if may_be_187727:

            if more_types_in_union_187728:
                # Runtime conditional SSA (line 655)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 656)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 657):
            
            # Assigning a Call to a Name (line 657):
            
            # Call to len(...): (line 657)
            # Processing the call arguments (line 657)
            # Getting the type of 'seq2' (line 657)
            seq2_187730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 27), 'seq2', False)
            # Processing the call keyword arguments (line 657)
            kwargs_187731 = {}
            # Getting the type of 'len' (line 657)
            len_187729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 23), 'len', False)
            # Calling len(args, kwargs) (line 657)
            len_call_result_187732 = invoke(stypy.reporting.localization.Localization(__file__, 657, 23), len_187729, *[seq2_187730], **kwargs_187731)
            
            # Assigning a type to the variable 'len2' (line 657)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 16), 'len2', len_call_result_187732)
            # SSA branch for the except part of a try statement (line 656)
            # SSA branch for the except 'Tuple' branch of a try statement (line 656)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a BinOp to a Name (line 659):
            
            # Assigning a BinOp to a Name (line 659):
            str_187733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 28), 'str', 'Second %s has no length.    Non-sequence?')
            # Getting the type of 'seq_type_name' (line 660)
            seq_type_name_187734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 24), 'seq_type_name')
            # Applying the binary operator '%' (line 659)
            result_mod_187735 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 28), '%', str_187733, seq_type_name_187734)
            
            # Assigning a type to the variable 'differing' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'differing', result_mod_187735)
            # SSA join for try-except statement (line 656)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_187728:
                # SSA join for if statement (line 655)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 662)
        # Getting the type of 'differing' (line 662)
        differing_187736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 11), 'differing')
        # Getting the type of 'None' (line 662)
        None_187737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 24), 'None')
        
        (may_be_187738, more_types_in_union_187739) = may_be_none(differing_187736, None_187737)

        if may_be_187738:

            if more_types_in_union_187739:
                # Runtime conditional SSA (line 662)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'seq1' (line 663)
            seq1_187740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 15), 'seq1')
            # Getting the type of 'seq2' (line 663)
            seq2_187741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 23), 'seq2')
            # Applying the binary operator '==' (line 663)
            result_eq_187742 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 15), '==', seq1_187740, seq2_187741)
            
            # Testing the type of an if condition (line 663)
            if_condition_187743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 663, 12), result_eq_187742)
            # Assigning a type to the variable 'if_condition_187743' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'if_condition_187743', if_condition_187743)
            # SSA begins for if statement (line 663)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 664)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 16), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 663)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 666):
            
            # Assigning a Call to a Name (line 666):
            
            # Call to safe_repr(...): (line 666)
            # Processing the call arguments (line 666)
            # Getting the type of 'seq1' (line 666)
            seq1_187745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 34), 'seq1', False)
            # Processing the call keyword arguments (line 666)
            kwargs_187746 = {}
            # Getting the type of 'safe_repr' (line 666)
            safe_repr_187744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 24), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 666)
            safe_repr_call_result_187747 = invoke(stypy.reporting.localization.Localization(__file__, 666, 24), safe_repr_187744, *[seq1_187745], **kwargs_187746)
            
            # Assigning a type to the variable 'seq1_repr' (line 666)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'seq1_repr', safe_repr_call_result_187747)
            
            # Assigning a Call to a Name (line 667):
            
            # Assigning a Call to a Name (line 667):
            
            # Call to safe_repr(...): (line 667)
            # Processing the call arguments (line 667)
            # Getting the type of 'seq2' (line 667)
            seq2_187749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 34), 'seq2', False)
            # Processing the call keyword arguments (line 667)
            kwargs_187750 = {}
            # Getting the type of 'safe_repr' (line 667)
            safe_repr_187748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 24), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 667)
            safe_repr_call_result_187751 = invoke(stypy.reporting.localization.Localization(__file__, 667, 24), safe_repr_187748, *[seq2_187749], **kwargs_187750)
            
            # Assigning a type to the variable 'seq2_repr' (line 667)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'seq2_repr', safe_repr_call_result_187751)
            
            
            
            # Call to len(...): (line 668)
            # Processing the call arguments (line 668)
            # Getting the type of 'seq1_repr' (line 668)
            seq1_repr_187753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 19), 'seq1_repr', False)
            # Processing the call keyword arguments (line 668)
            kwargs_187754 = {}
            # Getting the type of 'len' (line 668)
            len_187752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'len', False)
            # Calling len(args, kwargs) (line 668)
            len_call_result_187755 = invoke(stypy.reporting.localization.Localization(__file__, 668, 15), len_187752, *[seq1_repr_187753], **kwargs_187754)
            
            int_187756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 32), 'int')
            # Applying the binary operator '>' (line 668)
            result_gt_187757 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 15), '>', len_call_result_187755, int_187756)
            
            # Testing the type of an if condition (line 668)
            if_condition_187758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 12), result_gt_187757)
            # Assigning a type to the variable 'if_condition_187758' (line 668)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'if_condition_187758', if_condition_187758)
            # SSA begins for if statement (line 668)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 669):
            
            # Assigning a BinOp to a Name (line 669):
            
            # Obtaining the type of the subscript
            int_187759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 39), 'int')
            slice_187760 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 669, 28), None, int_187759, None)
            # Getting the type of 'seq1_repr' (line 669)
            seq1_repr_187761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 28), 'seq1_repr')
            # Obtaining the member '__getitem__' of a type (line 669)
            getitem___187762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 28), seq1_repr_187761, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 669)
            subscript_call_result_187763 = invoke(stypy.reporting.localization.Localization(__file__, 669, 28), getitem___187762, slice_187760)
            
            str_187764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 45), 'str', '...')
            # Applying the binary operator '+' (line 669)
            result_add_187765 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 28), '+', subscript_call_result_187763, str_187764)
            
            # Assigning a type to the variable 'seq1_repr' (line 669)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 16), 'seq1_repr', result_add_187765)
            # SSA join for if statement (line 668)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Call to len(...): (line 670)
            # Processing the call arguments (line 670)
            # Getting the type of 'seq2_repr' (line 670)
            seq2_repr_187767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 19), 'seq2_repr', False)
            # Processing the call keyword arguments (line 670)
            kwargs_187768 = {}
            # Getting the type of 'len' (line 670)
            len_187766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 15), 'len', False)
            # Calling len(args, kwargs) (line 670)
            len_call_result_187769 = invoke(stypy.reporting.localization.Localization(__file__, 670, 15), len_187766, *[seq2_repr_187767], **kwargs_187768)
            
            int_187770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 32), 'int')
            # Applying the binary operator '>' (line 670)
            result_gt_187771 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 15), '>', len_call_result_187769, int_187770)
            
            # Testing the type of an if condition (line 670)
            if_condition_187772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 12), result_gt_187771)
            # Assigning a type to the variable 'if_condition_187772' (line 670)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'if_condition_187772', if_condition_187772)
            # SSA begins for if statement (line 670)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 671):
            
            # Assigning a BinOp to a Name (line 671):
            
            # Obtaining the type of the subscript
            int_187773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 39), 'int')
            slice_187774 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 671, 28), None, int_187773, None)
            # Getting the type of 'seq2_repr' (line 671)
            seq2_repr_187775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 28), 'seq2_repr')
            # Obtaining the member '__getitem__' of a type (line 671)
            getitem___187776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 28), seq2_repr_187775, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 671)
            subscript_call_result_187777 = invoke(stypy.reporting.localization.Localization(__file__, 671, 28), getitem___187776, slice_187774)
            
            str_187778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 45), 'str', '...')
            # Applying the binary operator '+' (line 671)
            result_add_187779 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 28), '+', subscript_call_result_187777, str_187778)
            
            # Assigning a type to the variable 'seq2_repr' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'seq2_repr', result_add_187779)
            # SSA join for if statement (line 670)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Tuple to a Name (line 672):
            
            # Assigning a Tuple to a Name (line 672):
            
            # Obtaining an instance of the builtin type 'tuple' (line 672)
            tuple_187780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 24), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 672)
            # Adding element type (line 672)
            
            # Call to capitalize(...): (line 672)
            # Processing the call keyword arguments (line 672)
            kwargs_187783 = {}
            # Getting the type of 'seq_type_name' (line 672)
            seq_type_name_187781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 24), 'seq_type_name', False)
            # Obtaining the member 'capitalize' of a type (line 672)
            capitalize_187782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 24), seq_type_name_187781, 'capitalize')
            # Calling capitalize(args, kwargs) (line 672)
            capitalize_call_result_187784 = invoke(stypy.reporting.localization.Localization(__file__, 672, 24), capitalize_187782, *[], **kwargs_187783)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 24), tuple_187780, capitalize_call_result_187784)
            # Adding element type (line 672)
            # Getting the type of 'seq1_repr' (line 672)
            seq1_repr_187785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 52), 'seq1_repr')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 24), tuple_187780, seq1_repr_187785)
            # Adding element type (line 672)
            # Getting the type of 'seq2_repr' (line 672)
            seq2_repr_187786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 63), 'seq2_repr')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 24), tuple_187780, seq2_repr_187786)
            
            # Assigning a type to the variable 'elements' (line 672)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'elements', tuple_187780)
            
            # Assigning a BinOp to a Name (line 673):
            
            # Assigning a BinOp to a Name (line 673):
            str_187787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 24), 'str', '%ss differ: %s != %s\n')
            # Getting the type of 'elements' (line 673)
            elements_187788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 51), 'elements')
            # Applying the binary operator '%' (line 673)
            result_mod_187789 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 24), '%', str_187787, elements_187788)
            
            # Assigning a type to the variable 'differing' (line 673)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'differing', result_mod_187789)
            
            
            # Call to xrange(...): (line 675)
            # Processing the call arguments (line 675)
            
            # Call to min(...): (line 675)
            # Processing the call arguments (line 675)
            # Getting the type of 'len1' (line 675)
            len1_187792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 32), 'len1', False)
            # Getting the type of 'len2' (line 675)
            len2_187793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 38), 'len2', False)
            # Processing the call keyword arguments (line 675)
            kwargs_187794 = {}
            # Getting the type of 'min' (line 675)
            min_187791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 28), 'min', False)
            # Calling min(args, kwargs) (line 675)
            min_call_result_187795 = invoke(stypy.reporting.localization.Localization(__file__, 675, 28), min_187791, *[len1_187792, len2_187793], **kwargs_187794)
            
            # Processing the call keyword arguments (line 675)
            kwargs_187796 = {}
            # Getting the type of 'xrange' (line 675)
            xrange_187790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 675)
            xrange_call_result_187797 = invoke(stypy.reporting.localization.Localization(__file__, 675, 21), xrange_187790, *[min_call_result_187795], **kwargs_187796)
            
            # Testing the type of a for loop iterable (line 675)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 675, 12), xrange_call_result_187797)
            # Getting the type of the for loop variable (line 675)
            for_loop_var_187798 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 675, 12), xrange_call_result_187797)
            # Assigning a type to the variable 'i' (line 675)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'i', for_loop_var_187798)
            # SSA begins for a for statement (line 675)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # SSA begins for try-except statement (line 676)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Subscript to a Name (line 677):
            
            # Assigning a Subscript to a Name (line 677):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 677)
            i_187799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 33), 'i')
            # Getting the type of 'seq1' (line 677)
            seq1_187800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 28), 'seq1')
            # Obtaining the member '__getitem__' of a type (line 677)
            getitem___187801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 28), seq1_187800, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 677)
            subscript_call_result_187802 = invoke(stypy.reporting.localization.Localization(__file__, 677, 28), getitem___187801, i_187799)
            
            # Assigning a type to the variable 'item1' (line 677)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 20), 'item1', subscript_call_result_187802)
            # SSA branch for the except part of a try statement (line 676)
            # SSA branch for the except 'Tuple' branch of a try statement (line 676)
            module_type_store.open_ssa_branch('except')
            
            # Getting the type of 'differing' (line 679)
            differing_187803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 20), 'differing')
            str_187804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 34), 'str', '\nUnable to index element %d of first %s\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 680)
            tuple_187805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 680)
            # Adding element type (line 680)
            # Getting the type of 'i' (line 680)
            i_187806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 34), 'i')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 34), tuple_187805, i_187806)
            # Adding element type (line 680)
            # Getting the type of 'seq_type_name' (line 680)
            seq_type_name_187807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 37), 'seq_type_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 34), tuple_187805, seq_type_name_187807)
            
            # Applying the binary operator '%' (line 679)
            result_mod_187808 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 34), '%', str_187804, tuple_187805)
            
            # Applying the binary operator '+=' (line 679)
            result_iadd_187809 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 20), '+=', differing_187803, result_mod_187808)
            # Assigning a type to the variable 'differing' (line 679)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 20), 'differing', result_iadd_187809)
            
            # SSA join for try-except statement (line 676)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # SSA begins for try-except statement (line 683)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Subscript to a Name (line 684):
            
            # Assigning a Subscript to a Name (line 684):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 684)
            i_187810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 33), 'i')
            # Getting the type of 'seq2' (line 684)
            seq2_187811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 28), 'seq2')
            # Obtaining the member '__getitem__' of a type (line 684)
            getitem___187812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 28), seq2_187811, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 684)
            subscript_call_result_187813 = invoke(stypy.reporting.localization.Localization(__file__, 684, 28), getitem___187812, i_187810)
            
            # Assigning a type to the variable 'item2' (line 684)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 20), 'item2', subscript_call_result_187813)
            # SSA branch for the except part of a try statement (line 683)
            # SSA branch for the except 'Tuple' branch of a try statement (line 683)
            module_type_store.open_ssa_branch('except')
            
            # Getting the type of 'differing' (line 686)
            differing_187814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 20), 'differing')
            str_187815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 34), 'str', '\nUnable to index element %d of second %s\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 687)
            tuple_187816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 687)
            # Adding element type (line 687)
            # Getting the type of 'i' (line 687)
            i_187817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 34), 'i')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 34), tuple_187816, i_187817)
            # Adding element type (line 687)
            # Getting the type of 'seq_type_name' (line 687)
            seq_type_name_187818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 37), 'seq_type_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 687, 34), tuple_187816, seq_type_name_187818)
            
            # Applying the binary operator '%' (line 686)
            result_mod_187819 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 34), '%', str_187815, tuple_187816)
            
            # Applying the binary operator '+=' (line 686)
            result_iadd_187820 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 20), '+=', differing_187814, result_mod_187819)
            # Assigning a type to the variable 'differing' (line 686)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 20), 'differing', result_iadd_187820)
            
            # SSA join for try-except statement (line 683)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'item1' (line 690)
            item1_187821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 19), 'item1')
            # Getting the type of 'item2' (line 690)
            item2_187822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 28), 'item2')
            # Applying the binary operator '!=' (line 690)
            result_ne_187823 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 19), '!=', item1_187821, item2_187822)
            
            # Testing the type of an if condition (line 690)
            if_condition_187824 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 690, 16), result_ne_187823)
            # Assigning a type to the variable 'if_condition_187824' (line 690)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'if_condition_187824', if_condition_187824)
            # SSA begins for if statement (line 690)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'differing' (line 691)
            differing_187825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 20), 'differing')
            str_187826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 34), 'str', '\nFirst differing element %d:\n%s\n%s\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 692)
            tuple_187827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 692)
            # Adding element type (line 692)
            # Getting the type of 'i' (line 692)
            i_187828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 34), 'i')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 34), tuple_187827, i_187828)
            # Adding element type (line 692)
            # Getting the type of 'item1' (line 692)
            item1_187829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 37), 'item1')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 34), tuple_187827, item1_187829)
            # Adding element type (line 692)
            # Getting the type of 'item2' (line 692)
            item2_187830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 44), 'item2')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 34), tuple_187827, item2_187830)
            
            # Applying the binary operator '%' (line 691)
            result_mod_187831 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 34), '%', str_187826, tuple_187827)
            
            # Applying the binary operator '+=' (line 691)
            result_iadd_187832 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 20), '+=', differing_187825, result_mod_187831)
            # Assigning a type to the variable 'differing' (line 691)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 20), 'differing', result_iadd_187832)
            
            # SSA join for if statement (line 690)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of a for statement (line 675)
            module_type_store.open_ssa_branch('for loop else')
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'len1' (line 695)
            len1_187833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 20), 'len1')
            # Getting the type of 'len2' (line 695)
            len2_187834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 28), 'len2')
            # Applying the binary operator '==' (line 695)
            result_eq_187835 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 20), '==', len1_187833, len2_187834)
            
            
            # Getting the type of 'seq_type' (line 695)
            seq_type_187836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 37), 'seq_type')
            # Getting the type of 'None' (line 695)
            None_187837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 49), 'None')
            # Applying the binary operator 'is' (line 695)
            result_is__187838 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 37), 'is', seq_type_187836, None_187837)
            
            # Applying the binary operator 'and' (line 695)
            result_and_keyword_187839 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 20), 'and', result_eq_187835, result_is__187838)
            
            
            # Call to type(...): (line 696)
            # Processing the call arguments (line 696)
            # Getting the type of 'seq1' (line 696)
            seq1_187841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 25), 'seq1', False)
            # Processing the call keyword arguments (line 696)
            kwargs_187842 = {}
            # Getting the type of 'type' (line 696)
            type_187840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 20), 'type', False)
            # Calling type(args, kwargs) (line 696)
            type_call_result_187843 = invoke(stypy.reporting.localization.Localization(__file__, 696, 20), type_187840, *[seq1_187841], **kwargs_187842)
            
            
            # Call to type(...): (line 696)
            # Processing the call arguments (line 696)
            # Getting the type of 'seq2' (line 696)
            seq2_187845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 39), 'seq2', False)
            # Processing the call keyword arguments (line 696)
            kwargs_187846 = {}
            # Getting the type of 'type' (line 696)
            type_187844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 34), 'type', False)
            # Calling type(args, kwargs) (line 696)
            type_call_result_187847 = invoke(stypy.reporting.localization.Localization(__file__, 696, 34), type_187844, *[seq2_187845], **kwargs_187846)
            
            # Applying the binary operator '!=' (line 696)
            result_ne_187848 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 20), '!=', type_call_result_187843, type_call_result_187847)
            
            # Applying the binary operator 'and' (line 695)
            result_and_keyword_187849 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 20), 'and', result_and_keyword_187839, result_ne_187848)
            
            # Testing the type of an if condition (line 695)
            if_condition_187850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 695, 16), result_and_keyword_187849)
            # Assigning a type to the variable 'if_condition_187850' (line 695)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'if_condition_187850', if_condition_187850)
            # SSA begins for if statement (line 695)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 698)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 20), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 695)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'len1' (line 700)
            len1_187851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 15), 'len1')
            # Getting the type of 'len2' (line 700)
            len2_187852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 22), 'len2')
            # Applying the binary operator '>' (line 700)
            result_gt_187853 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 15), '>', len1_187851, len2_187852)
            
            # Testing the type of an if condition (line 700)
            if_condition_187854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 700, 12), result_gt_187853)
            # Assigning a type to the variable 'if_condition_187854' (line 700)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'if_condition_187854', if_condition_187854)
            # SSA begins for if statement (line 700)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'differing' (line 701)
            differing_187855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'differing')
            str_187856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 30), 'str', '\nFirst %s contains %d additional elements.\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 702)
            tuple_187857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 702)
            # Adding element type (line 702)
            # Getting the type of 'seq_type_name' (line 702)
            seq_type_name_187858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 46), 'seq_type_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 46), tuple_187857, seq_type_name_187858)
            # Adding element type (line 702)
            # Getting the type of 'len1' (line 702)
            len1_187859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 61), 'len1')
            # Getting the type of 'len2' (line 702)
            len2_187860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 68), 'len2')
            # Applying the binary operator '-' (line 702)
            result_sub_187861 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 61), '-', len1_187859, len2_187860)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 46), tuple_187857, result_sub_187861)
            
            # Applying the binary operator '%' (line 701)
            result_mod_187862 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 30), '%', str_187856, tuple_187857)
            
            # Applying the binary operator '+=' (line 701)
            result_iadd_187863 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 16), '+=', differing_187855, result_mod_187862)
            # Assigning a type to the variable 'differing' (line 701)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'differing', result_iadd_187863)
            
            
            
            # SSA begins for try-except statement (line 703)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Getting the type of 'differing' (line 704)
            differing_187864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 20), 'differing')
            str_187865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 34), 'str', 'First extra element %d:\n%s\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 705)
            tuple_187866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 705)
            # Adding element type (line 705)
            # Getting the type of 'len2' (line 705)
            len2_187867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 35), 'len2')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 35), tuple_187866, len2_187867)
            # Adding element type (line 705)
            
            # Obtaining the type of the subscript
            # Getting the type of 'len2' (line 705)
            len2_187868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 46), 'len2')
            # Getting the type of 'seq1' (line 705)
            seq1_187869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 41), 'seq1')
            # Obtaining the member '__getitem__' of a type (line 705)
            getitem___187870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 41), seq1_187869, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 705)
            subscript_call_result_187871 = invoke(stypy.reporting.localization.Localization(__file__, 705, 41), getitem___187870, len2_187868)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 35), tuple_187866, subscript_call_result_187871)
            
            # Applying the binary operator '%' (line 704)
            result_mod_187872 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 34), '%', str_187865, tuple_187866)
            
            # Applying the binary operator '+=' (line 704)
            result_iadd_187873 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 20), '+=', differing_187864, result_mod_187872)
            # Assigning a type to the variable 'differing' (line 704)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 20), 'differing', result_iadd_187873)
            
            # SSA branch for the except part of a try statement (line 703)
            # SSA branch for the except 'Tuple' branch of a try statement (line 703)
            module_type_store.open_ssa_branch('except')
            
            # Getting the type of 'differing' (line 707)
            differing_187874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 20), 'differing')
            str_187875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 34), 'str', 'Unable to index element %d of first %s\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 708)
            tuple_187876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 53), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 708)
            # Adding element type (line 708)
            # Getting the type of 'len2' (line 708)
            len2_187877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 53), 'len2')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 53), tuple_187876, len2_187877)
            # Adding element type (line 708)
            # Getting the type of 'seq_type_name' (line 708)
            seq_type_name_187878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 59), 'seq_type_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 53), tuple_187876, seq_type_name_187878)
            
            # Applying the binary operator '%' (line 707)
            result_mod_187879 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 34), '%', str_187875, tuple_187876)
            
            # Applying the binary operator '+=' (line 707)
            result_iadd_187880 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 20), '+=', differing_187874, result_mod_187879)
            # Assigning a type to the variable 'differing' (line 707)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 20), 'differing', result_iadd_187880)
            
            # SSA join for try-except statement (line 703)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 700)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'len1' (line 709)
            len1_187881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 17), 'len1')
            # Getting the type of 'len2' (line 709)
            len2_187882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 24), 'len2')
            # Applying the binary operator '<' (line 709)
            result_lt_187883 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 17), '<', len1_187881, len2_187882)
            
            # Testing the type of an if condition (line 709)
            if_condition_187884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 709, 17), result_lt_187883)
            # Assigning a type to the variable 'if_condition_187884' (line 709)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 17), 'if_condition_187884', if_condition_187884)
            # SSA begins for if statement (line 709)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'differing' (line 710)
            differing_187885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 16), 'differing')
            str_187886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 30), 'str', '\nSecond %s contains %d additional elements.\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 711)
            tuple_187887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 711)
            # Adding element type (line 711)
            # Getting the type of 'seq_type_name' (line 711)
            seq_type_name_187888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 46), 'seq_type_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 46), tuple_187887, seq_type_name_187888)
            # Adding element type (line 711)
            # Getting the type of 'len2' (line 711)
            len2_187889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 61), 'len2')
            # Getting the type of 'len1' (line 711)
            len1_187890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 68), 'len1')
            # Applying the binary operator '-' (line 711)
            result_sub_187891 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 61), '-', len2_187889, len1_187890)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 46), tuple_187887, result_sub_187891)
            
            # Applying the binary operator '%' (line 710)
            result_mod_187892 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 30), '%', str_187886, tuple_187887)
            
            # Applying the binary operator '+=' (line 710)
            result_iadd_187893 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 16), '+=', differing_187885, result_mod_187892)
            # Assigning a type to the variable 'differing' (line 710)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 16), 'differing', result_iadd_187893)
            
            
            
            # SSA begins for try-except statement (line 712)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Getting the type of 'differing' (line 713)
            differing_187894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 20), 'differing')
            str_187895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 34), 'str', 'First extra element %d:\n%s\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 714)
            tuple_187896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 714)
            # Adding element type (line 714)
            # Getting the type of 'len1' (line 714)
            len1_187897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 35), 'len1')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 35), tuple_187896, len1_187897)
            # Adding element type (line 714)
            
            # Obtaining the type of the subscript
            # Getting the type of 'len1' (line 714)
            len1_187898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 46), 'len1')
            # Getting the type of 'seq2' (line 714)
            seq2_187899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 41), 'seq2')
            # Obtaining the member '__getitem__' of a type (line 714)
            getitem___187900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 41), seq2_187899, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 714)
            subscript_call_result_187901 = invoke(stypy.reporting.localization.Localization(__file__, 714, 41), getitem___187900, len1_187898)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 35), tuple_187896, subscript_call_result_187901)
            
            # Applying the binary operator '%' (line 713)
            result_mod_187902 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 34), '%', str_187895, tuple_187896)
            
            # Applying the binary operator '+=' (line 713)
            result_iadd_187903 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 20), '+=', differing_187894, result_mod_187902)
            # Assigning a type to the variable 'differing' (line 713)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 20), 'differing', result_iadd_187903)
            
            # SSA branch for the except part of a try statement (line 712)
            # SSA branch for the except 'Tuple' branch of a try statement (line 712)
            module_type_store.open_ssa_branch('except')
            
            # Getting the type of 'differing' (line 716)
            differing_187904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 20), 'differing')
            str_187905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 34), 'str', 'Unable to index element %d of second %s\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 717)
            tuple_187906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 54), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 717)
            # Adding element type (line 717)
            # Getting the type of 'len1' (line 717)
            len1_187907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 54), 'len1')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 717, 54), tuple_187906, len1_187907)
            # Adding element type (line 717)
            # Getting the type of 'seq_type_name' (line 717)
            seq_type_name_187908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 60), 'seq_type_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 717, 54), tuple_187906, seq_type_name_187908)
            
            # Applying the binary operator '%' (line 716)
            result_mod_187909 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 34), '%', str_187905, tuple_187906)
            
            # Applying the binary operator '+=' (line 716)
            result_iadd_187910 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 20), '+=', differing_187904, result_mod_187909)
            # Assigning a type to the variable 'differing' (line 716)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 20), 'differing', result_iadd_187910)
            
            # SSA join for try-except statement (line 712)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 709)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 700)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_187739:
                # SSA join for if statement (line 662)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 718):
        
        # Assigning a Name to a Name (line 718):
        # Getting the type of 'differing' (line 718)
        differing_187911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 22), 'differing')
        # Assigning a type to the variable 'standardMsg' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'standardMsg', differing_187911)
        
        # Assigning a BinOp to a Name (line 719):
        
        # Assigning a BinOp to a Name (line 719):
        str_187912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 18), 'str', '\n')
        
        # Call to join(...): (line 719)
        # Processing the call arguments (line 719)
        
        # Call to ndiff(...): (line 720)
        # Processing the call arguments (line 720)
        
        # Call to splitlines(...): (line 720)
        # Processing the call keyword arguments (line 720)
        kwargs_187923 = {}
        
        # Call to pformat(...): (line 720)
        # Processing the call arguments (line 720)
        # Getting the type of 'seq1' (line 720)
        seq1_187919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 41), 'seq1', False)
        # Processing the call keyword arguments (line 720)
        kwargs_187920 = {}
        # Getting the type of 'pprint' (line 720)
        pprint_187917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 26), 'pprint', False)
        # Obtaining the member 'pformat' of a type (line 720)
        pformat_187918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 26), pprint_187917, 'pformat')
        # Calling pformat(args, kwargs) (line 720)
        pformat_call_result_187921 = invoke(stypy.reporting.localization.Localization(__file__, 720, 26), pformat_187918, *[seq1_187919], **kwargs_187920)
        
        # Obtaining the member 'splitlines' of a type (line 720)
        splitlines_187922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 26), pformat_call_result_187921, 'splitlines')
        # Calling splitlines(args, kwargs) (line 720)
        splitlines_call_result_187924 = invoke(stypy.reporting.localization.Localization(__file__, 720, 26), splitlines_187922, *[], **kwargs_187923)
        
        
        # Call to splitlines(...): (line 721)
        # Processing the call keyword arguments (line 721)
        kwargs_187931 = {}
        
        # Call to pformat(...): (line 721)
        # Processing the call arguments (line 721)
        # Getting the type of 'seq2' (line 721)
        seq2_187927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 41), 'seq2', False)
        # Processing the call keyword arguments (line 721)
        kwargs_187928 = {}
        # Getting the type of 'pprint' (line 721)
        pprint_187925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 26), 'pprint', False)
        # Obtaining the member 'pformat' of a type (line 721)
        pformat_187926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 26), pprint_187925, 'pformat')
        # Calling pformat(args, kwargs) (line 721)
        pformat_call_result_187929 = invoke(stypy.reporting.localization.Localization(__file__, 721, 26), pformat_187926, *[seq2_187927], **kwargs_187928)
        
        # Obtaining the member 'splitlines' of a type (line 721)
        splitlines_187930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 26), pformat_call_result_187929, 'splitlines')
        # Calling splitlines(args, kwargs) (line 721)
        splitlines_call_result_187932 = invoke(stypy.reporting.localization.Localization(__file__, 721, 26), splitlines_187930, *[], **kwargs_187931)
        
        # Processing the call keyword arguments (line 720)
        kwargs_187933 = {}
        # Getting the type of 'difflib' (line 720)
        difflib_187915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 12), 'difflib', False)
        # Obtaining the member 'ndiff' of a type (line 720)
        ndiff_187916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 12), difflib_187915, 'ndiff')
        # Calling ndiff(args, kwargs) (line 720)
        ndiff_call_result_187934 = invoke(stypy.reporting.localization.Localization(__file__, 720, 12), ndiff_187916, *[splitlines_call_result_187924, splitlines_call_result_187932], **kwargs_187933)
        
        # Processing the call keyword arguments (line 719)
        kwargs_187935 = {}
        str_187913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 25), 'str', '\n')
        # Obtaining the member 'join' of a type (line 719)
        join_187914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 25), str_187913, 'join')
        # Calling join(args, kwargs) (line 719)
        join_call_result_187936 = invoke(stypy.reporting.localization.Localization(__file__, 719, 25), join_187914, *[ndiff_call_result_187934], **kwargs_187935)
        
        # Applying the binary operator '+' (line 719)
        result_add_187937 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 18), '+', str_187912, join_call_result_187936)
        
        # Assigning a type to the variable 'diffMsg' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'diffMsg', result_add_187937)
        
        # Assigning a Call to a Name (line 722):
        
        # Assigning a Call to a Name (line 722):
        
        # Call to _truncateMessage(...): (line 722)
        # Processing the call arguments (line 722)
        # Getting the type of 'standardMsg' (line 722)
        standardMsg_187940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 44), 'standardMsg', False)
        # Getting the type of 'diffMsg' (line 722)
        diffMsg_187941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 57), 'diffMsg', False)
        # Processing the call keyword arguments (line 722)
        kwargs_187942 = {}
        # Getting the type of 'self' (line 722)
        self_187938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 22), 'self', False)
        # Obtaining the member '_truncateMessage' of a type (line 722)
        _truncateMessage_187939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 22), self_187938, '_truncateMessage')
        # Calling _truncateMessage(args, kwargs) (line 722)
        _truncateMessage_call_result_187943 = invoke(stypy.reporting.localization.Localization(__file__, 722, 22), _truncateMessage_187939, *[standardMsg_187940, diffMsg_187941], **kwargs_187942)
        
        # Assigning a type to the variable 'standardMsg' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'standardMsg', _truncateMessage_call_result_187943)
        
        # Assigning a Call to a Name (line 723):
        
        # Assigning a Call to a Name (line 723):
        
        # Call to _formatMessage(...): (line 723)
        # Processing the call arguments (line 723)
        # Getting the type of 'msg' (line 723)
        msg_187946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 34), 'msg', False)
        # Getting the type of 'standardMsg' (line 723)
        standardMsg_187947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 39), 'standardMsg', False)
        # Processing the call keyword arguments (line 723)
        kwargs_187948 = {}
        # Getting the type of 'self' (line 723)
        self_187944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 14), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 723)
        _formatMessage_187945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 14), self_187944, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 723)
        _formatMessage_call_result_187949 = invoke(stypy.reporting.localization.Localization(__file__, 723, 14), _formatMessage_187945, *[msg_187946, standardMsg_187947], **kwargs_187948)
        
        # Assigning a type to the variable 'msg' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'msg', _formatMessage_call_result_187949)
        
        # Call to fail(...): (line 724)
        # Processing the call arguments (line 724)
        # Getting the type of 'msg' (line 724)
        msg_187952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 18), 'msg', False)
        # Processing the call keyword arguments (line 724)
        kwargs_187953 = {}
        # Getting the type of 'self' (line 724)
        self_187950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'self', False)
        # Obtaining the member 'fail' of a type (line 724)
        fail_187951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 8), self_187950, 'fail')
        # Calling fail(args, kwargs) (line 724)
        fail_call_result_187954 = invoke(stypy.reporting.localization.Localization(__file__, 724, 8), fail_187951, *[msg_187952], **kwargs_187953)
        
        
        # ################# End of 'assertSequenceEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertSequenceEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 623)
        stypy_return_type_187955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187955)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertSequenceEqual'
        return stypy_return_type_187955


    @norecursion
    def _truncateMessage(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_truncateMessage'
        module_type_store = module_type_store.open_function_context('_truncateMessage', 726, 4, False)
        # Assigning a type to the variable 'self' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase._truncateMessage.__dict__.__setitem__('stypy_localization', localization)
        TestCase._truncateMessage.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase._truncateMessage.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase._truncateMessage.__dict__.__setitem__('stypy_function_name', 'TestCase._truncateMessage')
        TestCase._truncateMessage.__dict__.__setitem__('stypy_param_names_list', ['message', 'diff'])
        TestCase._truncateMessage.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase._truncateMessage.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase._truncateMessage.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase._truncateMessage.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase._truncateMessage.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase._truncateMessage.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase._truncateMessage', ['message', 'diff'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_truncateMessage', localization, ['message', 'diff'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_truncateMessage(...)' code ##################

        
        # Assigning a Attribute to a Name (line 727):
        
        # Assigning a Attribute to a Name (line 727):
        # Getting the type of 'self' (line 727)
        self_187956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 19), 'self')
        # Obtaining the member 'maxDiff' of a type (line 727)
        maxDiff_187957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 19), self_187956, 'maxDiff')
        # Assigning a type to the variable 'max_diff' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'max_diff', maxDiff_187957)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'max_diff' (line 728)
        max_diff_187958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 11), 'max_diff')
        # Getting the type of 'None' (line 728)
        None_187959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 23), 'None')
        # Applying the binary operator 'is' (line 728)
        result_is__187960 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 11), 'is', max_diff_187958, None_187959)
        
        
        
        # Call to len(...): (line 728)
        # Processing the call arguments (line 728)
        # Getting the type of 'diff' (line 728)
        diff_187962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 35), 'diff', False)
        # Processing the call keyword arguments (line 728)
        kwargs_187963 = {}
        # Getting the type of 'len' (line 728)
        len_187961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 31), 'len', False)
        # Calling len(args, kwargs) (line 728)
        len_call_result_187964 = invoke(stypy.reporting.localization.Localization(__file__, 728, 31), len_187961, *[diff_187962], **kwargs_187963)
        
        # Getting the type of 'max_diff' (line 728)
        max_diff_187965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 44), 'max_diff')
        # Applying the binary operator '<=' (line 728)
        result_le_187966 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 31), '<=', len_call_result_187964, max_diff_187965)
        
        # Applying the binary operator 'or' (line 728)
        result_or_keyword_187967 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 11), 'or', result_is__187960, result_le_187966)
        
        # Testing the type of an if condition (line 728)
        if_condition_187968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 728, 8), result_or_keyword_187967)
        # Assigning a type to the variable 'if_condition_187968' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'if_condition_187968', if_condition_187968)
        # SSA begins for if statement (line 728)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'message' (line 729)
        message_187969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 19), 'message')
        # Getting the type of 'diff' (line 729)
        diff_187970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 29), 'diff')
        # Applying the binary operator '+' (line 729)
        result_add_187971 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 19), '+', message_187969, diff_187970)
        
        # Assigning a type to the variable 'stypy_return_type' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 12), 'stypy_return_type', result_add_187971)
        # SSA join for if statement (line 728)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'message' (line 730)
        message_187972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 15), 'message')
        # Getting the type of 'DIFF_OMITTED' (line 730)
        DIFF_OMITTED_187973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 26), 'DIFF_OMITTED')
        
        # Call to len(...): (line 730)
        # Processing the call arguments (line 730)
        # Getting the type of 'diff' (line 730)
        diff_187975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 45), 'diff', False)
        # Processing the call keyword arguments (line 730)
        kwargs_187976 = {}
        # Getting the type of 'len' (line 730)
        len_187974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 41), 'len', False)
        # Calling len(args, kwargs) (line 730)
        len_call_result_187977 = invoke(stypy.reporting.localization.Localization(__file__, 730, 41), len_187974, *[diff_187975], **kwargs_187976)
        
        # Applying the binary operator '%' (line 730)
        result_mod_187978 = python_operator(stypy.reporting.localization.Localization(__file__, 730, 26), '%', DIFF_OMITTED_187973, len_call_result_187977)
        
        # Applying the binary operator '+' (line 730)
        result_add_187979 = python_operator(stypy.reporting.localization.Localization(__file__, 730, 15), '+', message_187972, result_mod_187978)
        
        # Assigning a type to the variable 'stypy_return_type' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'stypy_return_type', result_add_187979)
        
        # ################# End of '_truncateMessage(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_truncateMessage' in the type store
        # Getting the type of 'stypy_return_type' (line 726)
        stypy_return_type_187980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187980)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_truncateMessage'
        return stypy_return_type_187980


    @norecursion
    def assertListEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 732)
        None_187981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 48), 'None')
        defaults = [None_187981]
        # Create a new context for function 'assertListEqual'
        module_type_store = module_type_store.open_function_context('assertListEqual', 732, 4, False)
        # Assigning a type to the variable 'self' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertListEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertListEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertListEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertListEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertListEqual')
        TestCase.assertListEqual.__dict__.__setitem__('stypy_param_names_list', ['list1', 'list2', 'msg'])
        TestCase.assertListEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertListEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertListEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertListEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertListEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertListEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertListEqual', ['list1', 'list2', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertListEqual', localization, ['list1', 'list2', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertListEqual(...)' code ##################

        str_187982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, (-1)), 'str', 'A list-specific equality assertion.\n\n        Args:\n            list1: The first list to compare.\n            list2: The second list to compare.\n            msg: Optional message to use on failure instead of a list of\n                    differences.\n\n        ')
        
        # Call to assertSequenceEqual(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'list1' (line 742)
        list1_187985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 33), 'list1', False)
        # Getting the type of 'list2' (line 742)
        list2_187986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 40), 'list2', False)
        # Getting the type of 'msg' (line 742)
        msg_187987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 47), 'msg', False)
        # Processing the call keyword arguments (line 742)
        # Getting the type of 'list' (line 742)
        list_187988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 61), 'list', False)
        keyword_187989 = list_187988
        kwargs_187990 = {'seq_type': keyword_187989}
        # Getting the type of 'self' (line 742)
        self_187983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'self', False)
        # Obtaining the member 'assertSequenceEqual' of a type (line 742)
        assertSequenceEqual_187984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 8), self_187983, 'assertSequenceEqual')
        # Calling assertSequenceEqual(args, kwargs) (line 742)
        assertSequenceEqual_call_result_187991 = invoke(stypy.reporting.localization.Localization(__file__, 742, 8), assertSequenceEqual_187984, *[list1_187985, list2_187986, msg_187987], **kwargs_187990)
        
        
        # ################# End of 'assertListEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertListEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 732)
        stypy_return_type_187992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_187992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertListEqual'
        return stypy_return_type_187992


    @norecursion
    def assertTupleEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 744)
        None_187993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 51), 'None')
        defaults = [None_187993]
        # Create a new context for function 'assertTupleEqual'
        module_type_store = module_type_store.open_function_context('assertTupleEqual', 744, 4, False)
        # Assigning a type to the variable 'self' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertTupleEqual')
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_param_names_list', ['tuple1', 'tuple2', 'msg'])
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertTupleEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertTupleEqual', ['tuple1', 'tuple2', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertTupleEqual', localization, ['tuple1', 'tuple2', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertTupleEqual(...)' code ##################

        str_187994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, (-1)), 'str', 'A tuple-specific equality assertion.\n\n        Args:\n            tuple1: The first tuple to compare.\n            tuple2: The second tuple to compare.\n            msg: Optional message to use on failure instead of a list of\n                    differences.\n        ')
        
        # Call to assertSequenceEqual(...): (line 753)
        # Processing the call arguments (line 753)
        # Getting the type of 'tuple1' (line 753)
        tuple1_187997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 33), 'tuple1', False)
        # Getting the type of 'tuple2' (line 753)
        tuple2_187998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 41), 'tuple2', False)
        # Getting the type of 'msg' (line 753)
        msg_187999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 49), 'msg', False)
        # Processing the call keyword arguments (line 753)
        # Getting the type of 'tuple' (line 753)
        tuple_188000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 63), 'tuple', False)
        keyword_188001 = tuple_188000
        kwargs_188002 = {'seq_type': keyword_188001}
        # Getting the type of 'self' (line 753)
        self_187995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'self', False)
        # Obtaining the member 'assertSequenceEqual' of a type (line 753)
        assertSequenceEqual_187996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 8), self_187995, 'assertSequenceEqual')
        # Calling assertSequenceEqual(args, kwargs) (line 753)
        assertSequenceEqual_call_result_188003 = invoke(stypy.reporting.localization.Localization(__file__, 753, 8), assertSequenceEqual_187996, *[tuple1_187997, tuple2_187998, msg_187999], **kwargs_188002)
        
        
        # ################# End of 'assertTupleEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertTupleEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 744)
        stypy_return_type_188004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188004)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertTupleEqual'
        return stypy_return_type_188004


    @norecursion
    def assertSetEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 755)
        None_188005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 45), 'None')
        defaults = [None_188005]
        # Create a new context for function 'assertSetEqual'
        module_type_store = module_type_store.open_function_context('assertSetEqual', 755, 4, False)
        # Assigning a type to the variable 'self' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertSetEqual')
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_param_names_list', ['set1', 'set2', 'msg'])
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertSetEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertSetEqual', ['set1', 'set2', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertSetEqual', localization, ['set1', 'set2', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertSetEqual(...)' code ##################

        str_188006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, (-1)), 'str', 'A set-specific equality assertion.\n\n        Args:\n            set1: The first set to compare.\n            set2: The second set to compare.\n            msg: Optional message to use on failure instead of a list of\n                    differences.\n\n        assertSetEqual uses ducktyping to support different types of sets, and\n        is optimized for sets specifically (parameters must support a\n        difference method).\n        ')
        
        
        # SSA begins for try-except statement (line 768)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 769):
        
        # Assigning a Call to a Name (line 769):
        
        # Call to difference(...): (line 769)
        # Processing the call arguments (line 769)
        # Getting the type of 'set2' (line 769)
        set2_188009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 42), 'set2', False)
        # Processing the call keyword arguments (line 769)
        kwargs_188010 = {}
        # Getting the type of 'set1' (line 769)
        set1_188007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 26), 'set1', False)
        # Obtaining the member 'difference' of a type (line 769)
        difference_188008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 26), set1_188007, 'difference')
        # Calling difference(args, kwargs) (line 769)
        difference_call_result_188011 = invoke(stypy.reporting.localization.Localization(__file__, 769, 26), difference_188008, *[set2_188009], **kwargs_188010)
        
        # Assigning a type to the variable 'difference1' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'difference1', difference_call_result_188011)
        # SSA branch for the except part of a try statement (line 768)
        # SSA branch for the except 'TypeError' branch of a try statement (line 768)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'TypeError' (line 770)
        TypeError_188012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 15), 'TypeError')
        # Assigning a type to the variable 'e' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'e', TypeError_188012)
        
        # Call to fail(...): (line 771)
        # Processing the call arguments (line 771)
        str_188015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 22), 'str', 'invalid type when attempting set difference: %s')
        # Getting the type of 'e' (line 771)
        e_188016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 74), 'e', False)
        # Applying the binary operator '%' (line 771)
        result_mod_188017 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 22), '%', str_188015, e_188016)
        
        # Processing the call keyword arguments (line 771)
        kwargs_188018 = {}
        # Getting the type of 'self' (line 771)
        self_188013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 771)
        fail_188014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 12), self_188013, 'fail')
        # Calling fail(args, kwargs) (line 771)
        fail_call_result_188019 = invoke(stypy.reporting.localization.Localization(__file__, 771, 12), fail_188014, *[result_mod_188017], **kwargs_188018)
        
        # SSA branch for the except 'AttributeError' branch of a try statement (line 768)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'AttributeError' (line 772)
        AttributeError_188020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 15), 'AttributeError')
        # Assigning a type to the variable 'e' (line 772)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 8), 'e', AttributeError_188020)
        
        # Call to fail(...): (line 773)
        # Processing the call arguments (line 773)
        str_188023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 22), 'str', 'first argument does not support set difference: %s')
        # Getting the type of 'e' (line 773)
        e_188024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 77), 'e', False)
        # Applying the binary operator '%' (line 773)
        result_mod_188025 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 22), '%', str_188023, e_188024)
        
        # Processing the call keyword arguments (line 773)
        kwargs_188026 = {}
        # Getting the type of 'self' (line 773)
        self_188021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 773)
        fail_188022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 12), self_188021, 'fail')
        # Calling fail(args, kwargs) (line 773)
        fail_call_result_188027 = invoke(stypy.reporting.localization.Localization(__file__, 773, 12), fail_188022, *[result_mod_188025], **kwargs_188026)
        
        # SSA join for try-except statement (line 768)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 775)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 776):
        
        # Assigning a Call to a Name (line 776):
        
        # Call to difference(...): (line 776)
        # Processing the call arguments (line 776)
        # Getting the type of 'set1' (line 776)
        set1_188030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 42), 'set1', False)
        # Processing the call keyword arguments (line 776)
        kwargs_188031 = {}
        # Getting the type of 'set2' (line 776)
        set2_188028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 26), 'set2', False)
        # Obtaining the member 'difference' of a type (line 776)
        difference_188029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 26), set2_188028, 'difference')
        # Calling difference(args, kwargs) (line 776)
        difference_call_result_188032 = invoke(stypy.reporting.localization.Localization(__file__, 776, 26), difference_188029, *[set1_188030], **kwargs_188031)
        
        # Assigning a type to the variable 'difference2' (line 776)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 12), 'difference2', difference_call_result_188032)
        # SSA branch for the except part of a try statement (line 775)
        # SSA branch for the except 'TypeError' branch of a try statement (line 775)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'TypeError' (line 777)
        TypeError_188033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 15), 'TypeError')
        # Assigning a type to the variable 'e' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'e', TypeError_188033)
        
        # Call to fail(...): (line 778)
        # Processing the call arguments (line 778)
        str_188036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 22), 'str', 'invalid type when attempting set difference: %s')
        # Getting the type of 'e' (line 778)
        e_188037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 74), 'e', False)
        # Applying the binary operator '%' (line 778)
        result_mod_188038 = python_operator(stypy.reporting.localization.Localization(__file__, 778, 22), '%', str_188036, e_188037)
        
        # Processing the call keyword arguments (line 778)
        kwargs_188039 = {}
        # Getting the type of 'self' (line 778)
        self_188034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 778)
        fail_188035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 12), self_188034, 'fail')
        # Calling fail(args, kwargs) (line 778)
        fail_call_result_188040 = invoke(stypy.reporting.localization.Localization(__file__, 778, 12), fail_188035, *[result_mod_188038], **kwargs_188039)
        
        # SSA branch for the except 'AttributeError' branch of a try statement (line 775)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'AttributeError' (line 779)
        AttributeError_188041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 15), 'AttributeError')
        # Assigning a type to the variable 'e' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'e', AttributeError_188041)
        
        # Call to fail(...): (line 780)
        # Processing the call arguments (line 780)
        str_188044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 22), 'str', 'second argument does not support set difference: %s')
        # Getting the type of 'e' (line 780)
        e_188045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 78), 'e', False)
        # Applying the binary operator '%' (line 780)
        result_mod_188046 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 22), '%', str_188044, e_188045)
        
        # Processing the call keyword arguments (line 780)
        kwargs_188047 = {}
        # Getting the type of 'self' (line 780)
        self_188042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 780)
        fail_188043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 12), self_188042, 'fail')
        # Calling fail(args, kwargs) (line 780)
        fail_call_result_188048 = invoke(stypy.reporting.localization.Localization(__file__, 780, 12), fail_188043, *[result_mod_188046], **kwargs_188047)
        
        # SSA join for try-except statement (line 775)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'difference1' (line 782)
        difference1_188049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 16), 'difference1')
        # Getting the type of 'difference2' (line 782)
        difference2_188050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 31), 'difference2')
        # Applying the binary operator 'or' (line 782)
        result_or_keyword_188051 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 16), 'or', difference1_188049, difference2_188050)
        
        # Applying the 'not' unary operator (line 782)
        result_not__188052 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 11), 'not', result_or_keyword_188051)
        
        # Testing the type of an if condition (line 782)
        if_condition_188053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 782, 8), result_not__188052)
        # Assigning a type to the variable 'if_condition_188053' (line 782)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'if_condition_188053', if_condition_188053)
        # SSA begins for if statement (line 782)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 782)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 785):
        
        # Assigning a List to a Name (line 785):
        
        # Obtaining an instance of the builtin type 'list' (line 785)
        list_188054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 785)
        
        # Assigning a type to the variable 'lines' (line 785)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'lines', list_188054)
        
        # Getting the type of 'difference1' (line 786)
        difference1_188055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 11), 'difference1')
        # Testing the type of an if condition (line 786)
        if_condition_188056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 786, 8), difference1_188055)
        # Assigning a type to the variable 'if_condition_188056' (line 786)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'if_condition_188056', if_condition_188056)
        # SSA begins for if statement (line 786)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 787)
        # Processing the call arguments (line 787)
        str_188059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 25), 'str', 'Items in the first set but not the second:')
        # Processing the call keyword arguments (line 787)
        kwargs_188060 = {}
        # Getting the type of 'lines' (line 787)
        lines_188057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 12), 'lines', False)
        # Obtaining the member 'append' of a type (line 787)
        append_188058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 12), lines_188057, 'append')
        # Calling append(args, kwargs) (line 787)
        append_call_result_188061 = invoke(stypy.reporting.localization.Localization(__file__, 787, 12), append_188058, *[str_188059], **kwargs_188060)
        
        
        # Getting the type of 'difference1' (line 788)
        difference1_188062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 24), 'difference1')
        # Testing the type of a for loop iterable (line 788)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 788, 12), difference1_188062)
        # Getting the type of the for loop variable (line 788)
        for_loop_var_188063 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 788, 12), difference1_188062)
        # Assigning a type to the variable 'item' (line 788)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 12), 'item', for_loop_var_188063)
        # SSA begins for a for statement (line 788)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 789)
        # Processing the call arguments (line 789)
        
        # Call to repr(...): (line 789)
        # Processing the call arguments (line 789)
        # Getting the type of 'item' (line 789)
        item_188067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 34), 'item', False)
        # Processing the call keyword arguments (line 789)
        kwargs_188068 = {}
        # Getting the type of 'repr' (line 789)
        repr_188066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 29), 'repr', False)
        # Calling repr(args, kwargs) (line 789)
        repr_call_result_188069 = invoke(stypy.reporting.localization.Localization(__file__, 789, 29), repr_188066, *[item_188067], **kwargs_188068)
        
        # Processing the call keyword arguments (line 789)
        kwargs_188070 = {}
        # Getting the type of 'lines' (line 789)
        lines_188064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 16), 'lines', False)
        # Obtaining the member 'append' of a type (line 789)
        append_188065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 16), lines_188064, 'append')
        # Calling append(args, kwargs) (line 789)
        append_call_result_188071 = invoke(stypy.reporting.localization.Localization(__file__, 789, 16), append_188065, *[repr_call_result_188069], **kwargs_188070)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 786)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'difference2' (line 790)
        difference2_188072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 11), 'difference2')
        # Testing the type of an if condition (line 790)
        if_condition_188073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 790, 8), difference2_188072)
        # Assigning a type to the variable 'if_condition_188073' (line 790)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'if_condition_188073', if_condition_188073)
        # SSA begins for if statement (line 790)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 791)
        # Processing the call arguments (line 791)
        str_188076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 25), 'str', 'Items in the second set but not the first:')
        # Processing the call keyword arguments (line 791)
        kwargs_188077 = {}
        # Getting the type of 'lines' (line 791)
        lines_188074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'lines', False)
        # Obtaining the member 'append' of a type (line 791)
        append_188075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 12), lines_188074, 'append')
        # Calling append(args, kwargs) (line 791)
        append_call_result_188078 = invoke(stypy.reporting.localization.Localization(__file__, 791, 12), append_188075, *[str_188076], **kwargs_188077)
        
        
        # Getting the type of 'difference2' (line 792)
        difference2_188079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 24), 'difference2')
        # Testing the type of a for loop iterable (line 792)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 792, 12), difference2_188079)
        # Getting the type of the for loop variable (line 792)
        for_loop_var_188080 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 792, 12), difference2_188079)
        # Assigning a type to the variable 'item' (line 792)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 12), 'item', for_loop_var_188080)
        # SSA begins for a for statement (line 792)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 793)
        # Processing the call arguments (line 793)
        
        # Call to repr(...): (line 793)
        # Processing the call arguments (line 793)
        # Getting the type of 'item' (line 793)
        item_188084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 34), 'item', False)
        # Processing the call keyword arguments (line 793)
        kwargs_188085 = {}
        # Getting the type of 'repr' (line 793)
        repr_188083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 29), 'repr', False)
        # Calling repr(args, kwargs) (line 793)
        repr_call_result_188086 = invoke(stypy.reporting.localization.Localization(__file__, 793, 29), repr_188083, *[item_188084], **kwargs_188085)
        
        # Processing the call keyword arguments (line 793)
        kwargs_188087 = {}
        # Getting the type of 'lines' (line 793)
        lines_188081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 16), 'lines', False)
        # Obtaining the member 'append' of a type (line 793)
        append_188082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 16), lines_188081, 'append')
        # Calling append(args, kwargs) (line 793)
        append_call_result_188088 = invoke(stypy.reporting.localization.Localization(__file__, 793, 16), append_188082, *[repr_call_result_188086], **kwargs_188087)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 790)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 795):
        
        # Assigning a Call to a Name (line 795):
        
        # Call to join(...): (line 795)
        # Processing the call arguments (line 795)
        # Getting the type of 'lines' (line 795)
        lines_188091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 32), 'lines', False)
        # Processing the call keyword arguments (line 795)
        kwargs_188092 = {}
        str_188089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 22), 'str', '\n')
        # Obtaining the member 'join' of a type (line 795)
        join_188090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 22), str_188089, 'join')
        # Calling join(args, kwargs) (line 795)
        join_call_result_188093 = invoke(stypy.reporting.localization.Localization(__file__, 795, 22), join_188090, *[lines_188091], **kwargs_188092)
        
        # Assigning a type to the variable 'standardMsg' (line 795)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'standardMsg', join_call_result_188093)
        
        # Call to fail(...): (line 796)
        # Processing the call arguments (line 796)
        
        # Call to _formatMessage(...): (line 796)
        # Processing the call arguments (line 796)
        # Getting the type of 'msg' (line 796)
        msg_188098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 38), 'msg', False)
        # Getting the type of 'standardMsg' (line 796)
        standardMsg_188099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 43), 'standardMsg', False)
        # Processing the call keyword arguments (line 796)
        kwargs_188100 = {}
        # Getting the type of 'self' (line 796)
        self_188096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 18), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 796)
        _formatMessage_188097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 18), self_188096, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 796)
        _formatMessage_call_result_188101 = invoke(stypy.reporting.localization.Localization(__file__, 796, 18), _formatMessage_188097, *[msg_188098, standardMsg_188099], **kwargs_188100)
        
        # Processing the call keyword arguments (line 796)
        kwargs_188102 = {}
        # Getting the type of 'self' (line 796)
        self_188094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 8), 'self', False)
        # Obtaining the member 'fail' of a type (line 796)
        fail_188095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 8), self_188094, 'fail')
        # Calling fail(args, kwargs) (line 796)
        fail_call_result_188103 = invoke(stypy.reporting.localization.Localization(__file__, 796, 8), fail_188095, *[_formatMessage_call_result_188101], **kwargs_188102)
        
        
        # ################# End of 'assertSetEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertSetEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 755)
        stypy_return_type_188104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188104)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertSetEqual'
        return stypy_return_type_188104


    @norecursion
    def assertIn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 798)
        None_188105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 46), 'None')
        defaults = [None_188105]
        # Create a new context for function 'assertIn'
        module_type_store = module_type_store.open_function_context('assertIn', 798, 4, False)
        # Assigning a type to the variable 'self' (line 799)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertIn.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertIn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertIn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertIn.__dict__.__setitem__('stypy_function_name', 'TestCase.assertIn')
        TestCase.assertIn.__dict__.__setitem__('stypy_param_names_list', ['member', 'container', 'msg'])
        TestCase.assertIn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertIn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertIn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertIn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertIn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertIn.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertIn', ['member', 'container', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertIn', localization, ['member', 'container', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertIn(...)' code ##################

        str_188106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 8), 'str', 'Just like self.assertTrue(a in b), but with a nicer default message.')
        
        
        # Getting the type of 'member' (line 800)
        member_188107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 11), 'member')
        # Getting the type of 'container' (line 800)
        container_188108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 25), 'container')
        # Applying the binary operator 'notin' (line 800)
        result_contains_188109 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 11), 'notin', member_188107, container_188108)
        
        # Testing the type of an if condition (line 800)
        if_condition_188110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 800, 8), result_contains_188109)
        # Assigning a type to the variable 'if_condition_188110' (line 800)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 8), 'if_condition_188110', if_condition_188110)
        # SSA begins for if statement (line 800)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 801):
        
        # Assigning a BinOp to a Name (line 801):
        str_188111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 26), 'str', '%s not found in %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 801)
        tuple_188112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 801)
        # Adding element type (line 801)
        
        # Call to safe_repr(...): (line 801)
        # Processing the call arguments (line 801)
        # Getting the type of 'member' (line 801)
        member_188114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 60), 'member', False)
        # Processing the call keyword arguments (line 801)
        kwargs_188115 = {}
        # Getting the type of 'safe_repr' (line 801)
        safe_repr_188113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 50), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 801)
        safe_repr_call_result_188116 = invoke(stypy.reporting.localization.Localization(__file__, 801, 50), safe_repr_188113, *[member_188114], **kwargs_188115)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 801, 50), tuple_188112, safe_repr_call_result_188116)
        # Adding element type (line 801)
        
        # Call to safe_repr(...): (line 802)
        # Processing the call arguments (line 802)
        # Getting the type of 'container' (line 802)
        container_188118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 60), 'container', False)
        # Processing the call keyword arguments (line 802)
        kwargs_188119 = {}
        # Getting the type of 'safe_repr' (line 802)
        safe_repr_188117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 50), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 802)
        safe_repr_call_result_188120 = invoke(stypy.reporting.localization.Localization(__file__, 802, 50), safe_repr_188117, *[container_188118], **kwargs_188119)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 801, 50), tuple_188112, safe_repr_call_result_188120)
        
        # Applying the binary operator '%' (line 801)
        result_mod_188121 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 26), '%', str_188111, tuple_188112)
        
        # Assigning a type to the variable 'standardMsg' (line 801)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 12), 'standardMsg', result_mod_188121)
        
        # Call to fail(...): (line 803)
        # Processing the call arguments (line 803)
        
        # Call to _formatMessage(...): (line 803)
        # Processing the call arguments (line 803)
        # Getting the type of 'msg' (line 803)
        msg_188126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 803)
        standardMsg_188127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 803)
        kwargs_188128 = {}
        # Getting the type of 'self' (line 803)
        self_188124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 803)
        _formatMessage_188125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 22), self_188124, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 803)
        _formatMessage_call_result_188129 = invoke(stypy.reporting.localization.Localization(__file__, 803, 22), _formatMessage_188125, *[msg_188126, standardMsg_188127], **kwargs_188128)
        
        # Processing the call keyword arguments (line 803)
        kwargs_188130 = {}
        # Getting the type of 'self' (line 803)
        self_188122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 803)
        fail_188123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 12), self_188122, 'fail')
        # Calling fail(args, kwargs) (line 803)
        fail_call_result_188131 = invoke(stypy.reporting.localization.Localization(__file__, 803, 12), fail_188123, *[_formatMessage_call_result_188129], **kwargs_188130)
        
        # SSA join for if statement (line 800)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertIn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertIn' in the type store
        # Getting the type of 'stypy_return_type' (line 798)
        stypy_return_type_188132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188132)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertIn'
        return stypy_return_type_188132


    @norecursion
    def assertNotIn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 805)
        None_188133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 49), 'None')
        defaults = [None_188133]
        # Create a new context for function 'assertNotIn'
        module_type_store = module_type_store.open_function_context('assertNotIn', 805, 4, False)
        # Assigning a type to the variable 'self' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertNotIn.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertNotIn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertNotIn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertNotIn.__dict__.__setitem__('stypy_function_name', 'TestCase.assertNotIn')
        TestCase.assertNotIn.__dict__.__setitem__('stypy_param_names_list', ['member', 'container', 'msg'])
        TestCase.assertNotIn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertNotIn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertNotIn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertNotIn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertNotIn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertNotIn.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertNotIn', ['member', 'container', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertNotIn', localization, ['member', 'container', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertNotIn(...)' code ##################

        str_188134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 8), 'str', 'Just like self.assertTrue(a not in b), but with a nicer default message.')
        
        
        # Getting the type of 'member' (line 807)
        member_188135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 11), 'member')
        # Getting the type of 'container' (line 807)
        container_188136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 21), 'container')
        # Applying the binary operator 'in' (line 807)
        result_contains_188137 = python_operator(stypy.reporting.localization.Localization(__file__, 807, 11), 'in', member_188135, container_188136)
        
        # Testing the type of an if condition (line 807)
        if_condition_188138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 807, 8), result_contains_188137)
        # Assigning a type to the variable 'if_condition_188138' (line 807)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'if_condition_188138', if_condition_188138)
        # SSA begins for if statement (line 807)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 808):
        
        # Assigning a BinOp to a Name (line 808):
        str_188139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 26), 'str', '%s unexpectedly found in %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 808)
        tuple_188140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 808)
        # Adding element type (line 808)
        
        # Call to safe_repr(...): (line 808)
        # Processing the call arguments (line 808)
        # Getting the type of 'member' (line 808)
        member_188142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 69), 'member', False)
        # Processing the call keyword arguments (line 808)
        kwargs_188143 = {}
        # Getting the type of 'safe_repr' (line 808)
        safe_repr_188141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 59), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 808)
        safe_repr_call_result_188144 = invoke(stypy.reporting.localization.Localization(__file__, 808, 59), safe_repr_188141, *[member_188142], **kwargs_188143)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 59), tuple_188140, safe_repr_call_result_188144)
        # Adding element type (line 808)
        
        # Call to safe_repr(...): (line 809)
        # Processing the call arguments (line 809)
        # Getting the type of 'container' (line 809)
        container_188146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 66), 'container', False)
        # Processing the call keyword arguments (line 809)
        kwargs_188147 = {}
        # Getting the type of 'safe_repr' (line 809)
        safe_repr_188145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 56), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 809)
        safe_repr_call_result_188148 = invoke(stypy.reporting.localization.Localization(__file__, 809, 56), safe_repr_188145, *[container_188146], **kwargs_188147)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 59), tuple_188140, safe_repr_call_result_188148)
        
        # Applying the binary operator '%' (line 808)
        result_mod_188149 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 26), '%', str_188139, tuple_188140)
        
        # Assigning a type to the variable 'standardMsg' (line 808)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 12), 'standardMsg', result_mod_188149)
        
        # Call to fail(...): (line 810)
        # Processing the call arguments (line 810)
        
        # Call to _formatMessage(...): (line 810)
        # Processing the call arguments (line 810)
        # Getting the type of 'msg' (line 810)
        msg_188154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 810)
        standardMsg_188155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 810)
        kwargs_188156 = {}
        # Getting the type of 'self' (line 810)
        self_188152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 810)
        _formatMessage_188153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 22), self_188152, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 810)
        _formatMessage_call_result_188157 = invoke(stypy.reporting.localization.Localization(__file__, 810, 22), _formatMessage_188153, *[msg_188154, standardMsg_188155], **kwargs_188156)
        
        # Processing the call keyword arguments (line 810)
        kwargs_188158 = {}
        # Getting the type of 'self' (line 810)
        self_188150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 810)
        fail_188151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 12), self_188150, 'fail')
        # Calling fail(args, kwargs) (line 810)
        fail_call_result_188159 = invoke(stypy.reporting.localization.Localization(__file__, 810, 12), fail_188151, *[_formatMessage_call_result_188157], **kwargs_188158)
        
        # SSA join for if statement (line 807)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertNotIn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertNotIn' in the type store
        # Getting the type of 'stypy_return_type' (line 805)
        stypy_return_type_188160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188160)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertNotIn'
        return stypy_return_type_188160


    @norecursion
    def assertIs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 812)
        None_188161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 41), 'None')
        defaults = [None_188161]
        # Create a new context for function 'assertIs'
        module_type_store = module_type_store.open_function_context('assertIs', 812, 4, False)
        # Assigning a type to the variable 'self' (line 813)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertIs.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertIs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertIs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertIs.__dict__.__setitem__('stypy_function_name', 'TestCase.assertIs')
        TestCase.assertIs.__dict__.__setitem__('stypy_param_names_list', ['expr1', 'expr2', 'msg'])
        TestCase.assertIs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertIs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertIs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertIs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertIs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertIs.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertIs', ['expr1', 'expr2', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertIs', localization, ['expr1', 'expr2', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertIs(...)' code ##################

        str_188162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 8), 'str', 'Just like self.assertTrue(a is b), but with a nicer default message.')
        
        
        # Getting the type of 'expr1' (line 814)
        expr1_188163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 11), 'expr1')
        # Getting the type of 'expr2' (line 814)
        expr2_188164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 24), 'expr2')
        # Applying the binary operator 'isnot' (line 814)
        result_is_not_188165 = python_operator(stypy.reporting.localization.Localization(__file__, 814, 11), 'isnot', expr1_188163, expr2_188164)
        
        # Testing the type of an if condition (line 814)
        if_condition_188166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 814, 8), result_is_not_188165)
        # Assigning a type to the variable 'if_condition_188166' (line 814)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 8), 'if_condition_188166', if_condition_188166)
        # SSA begins for if statement (line 814)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 815):
        
        # Assigning a BinOp to a Name (line 815):
        str_188167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 26), 'str', '%s is not %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 815)
        tuple_188168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 815)
        # Adding element type (line 815)
        
        # Call to safe_repr(...): (line 815)
        # Processing the call arguments (line 815)
        # Getting the type of 'expr1' (line 815)
        expr1_188170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 54), 'expr1', False)
        # Processing the call keyword arguments (line 815)
        kwargs_188171 = {}
        # Getting the type of 'safe_repr' (line 815)
        safe_repr_188169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 44), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 815)
        safe_repr_call_result_188172 = invoke(stypy.reporting.localization.Localization(__file__, 815, 44), safe_repr_188169, *[expr1_188170], **kwargs_188171)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 44), tuple_188168, safe_repr_call_result_188172)
        # Adding element type (line 815)
        
        # Call to safe_repr(...): (line 816)
        # Processing the call arguments (line 816)
        # Getting the type of 'expr2' (line 816)
        expr2_188174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 55), 'expr2', False)
        # Processing the call keyword arguments (line 816)
        kwargs_188175 = {}
        # Getting the type of 'safe_repr' (line 816)
        safe_repr_188173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 45), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 816)
        safe_repr_call_result_188176 = invoke(stypy.reporting.localization.Localization(__file__, 816, 45), safe_repr_188173, *[expr2_188174], **kwargs_188175)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 44), tuple_188168, safe_repr_call_result_188176)
        
        # Applying the binary operator '%' (line 815)
        result_mod_188177 = python_operator(stypy.reporting.localization.Localization(__file__, 815, 26), '%', str_188167, tuple_188168)
        
        # Assigning a type to the variable 'standardMsg' (line 815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 12), 'standardMsg', result_mod_188177)
        
        # Call to fail(...): (line 817)
        # Processing the call arguments (line 817)
        
        # Call to _formatMessage(...): (line 817)
        # Processing the call arguments (line 817)
        # Getting the type of 'msg' (line 817)
        msg_188182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 817)
        standardMsg_188183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 817)
        kwargs_188184 = {}
        # Getting the type of 'self' (line 817)
        self_188180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 817)
        _formatMessage_188181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 22), self_188180, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 817)
        _formatMessage_call_result_188185 = invoke(stypy.reporting.localization.Localization(__file__, 817, 22), _formatMessage_188181, *[msg_188182, standardMsg_188183], **kwargs_188184)
        
        # Processing the call keyword arguments (line 817)
        kwargs_188186 = {}
        # Getting the type of 'self' (line 817)
        self_188178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 817)
        fail_188179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 12), self_188178, 'fail')
        # Calling fail(args, kwargs) (line 817)
        fail_call_result_188187 = invoke(stypy.reporting.localization.Localization(__file__, 817, 12), fail_188179, *[_formatMessage_call_result_188185], **kwargs_188186)
        
        # SSA join for if statement (line 814)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertIs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertIs' in the type store
        # Getting the type of 'stypy_return_type' (line 812)
        stypy_return_type_188188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188188)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertIs'
        return stypy_return_type_188188


    @norecursion
    def assertIsNot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 819)
        None_188189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 44), 'None')
        defaults = [None_188189]
        # Create a new context for function 'assertIsNot'
        module_type_store = module_type_store.open_function_context('assertIsNot', 819, 4, False)
        # Assigning a type to the variable 'self' (line 820)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertIsNot.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertIsNot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertIsNot.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertIsNot.__dict__.__setitem__('stypy_function_name', 'TestCase.assertIsNot')
        TestCase.assertIsNot.__dict__.__setitem__('stypy_param_names_list', ['expr1', 'expr2', 'msg'])
        TestCase.assertIsNot.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertIsNot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertIsNot.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertIsNot.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertIsNot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertIsNot.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertIsNot', ['expr1', 'expr2', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertIsNot', localization, ['expr1', 'expr2', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertIsNot(...)' code ##################

        str_188190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 8), 'str', 'Just like self.assertTrue(a is not b), but with a nicer default message.')
        
        
        # Getting the type of 'expr1' (line 821)
        expr1_188191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 11), 'expr1')
        # Getting the type of 'expr2' (line 821)
        expr2_188192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 20), 'expr2')
        # Applying the binary operator 'is' (line 821)
        result_is__188193 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 11), 'is', expr1_188191, expr2_188192)
        
        # Testing the type of an if condition (line 821)
        if_condition_188194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 8), result_is__188193)
        # Assigning a type to the variable 'if_condition_188194' (line 821)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 8), 'if_condition_188194', if_condition_188194)
        # SSA begins for if statement (line 821)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 822):
        
        # Assigning a BinOp to a Name (line 822):
        str_188195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 26), 'str', 'unexpectedly identical: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 822)
        tuple_188196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 822)
        # Adding element type (line 822)
        
        # Call to safe_repr(...): (line 822)
        # Processing the call arguments (line 822)
        # Getting the type of 'expr1' (line 822)
        expr1_188198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 68), 'expr1', False)
        # Processing the call keyword arguments (line 822)
        kwargs_188199 = {}
        # Getting the type of 'safe_repr' (line 822)
        safe_repr_188197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 58), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 822)
        safe_repr_call_result_188200 = invoke(stypy.reporting.localization.Localization(__file__, 822, 58), safe_repr_188197, *[expr1_188198], **kwargs_188199)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 822, 58), tuple_188196, safe_repr_call_result_188200)
        
        # Applying the binary operator '%' (line 822)
        result_mod_188201 = python_operator(stypy.reporting.localization.Localization(__file__, 822, 26), '%', str_188195, tuple_188196)
        
        # Assigning a type to the variable 'standardMsg' (line 822)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 12), 'standardMsg', result_mod_188201)
        
        # Call to fail(...): (line 823)
        # Processing the call arguments (line 823)
        
        # Call to _formatMessage(...): (line 823)
        # Processing the call arguments (line 823)
        # Getting the type of 'msg' (line 823)
        msg_188206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 823)
        standardMsg_188207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 823)
        kwargs_188208 = {}
        # Getting the type of 'self' (line 823)
        self_188204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 823)
        _formatMessage_188205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 823, 22), self_188204, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 823)
        _formatMessage_call_result_188209 = invoke(stypy.reporting.localization.Localization(__file__, 823, 22), _formatMessage_188205, *[msg_188206, standardMsg_188207], **kwargs_188208)
        
        # Processing the call keyword arguments (line 823)
        kwargs_188210 = {}
        # Getting the type of 'self' (line 823)
        self_188202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 823)
        fail_188203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 823, 12), self_188202, 'fail')
        # Calling fail(args, kwargs) (line 823)
        fail_call_result_188211 = invoke(stypy.reporting.localization.Localization(__file__, 823, 12), fail_188203, *[_formatMessage_call_result_188209], **kwargs_188210)
        
        # SSA join for if statement (line 821)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertIsNot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertIsNot' in the type store
        # Getting the type of 'stypy_return_type' (line 819)
        stypy_return_type_188212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188212)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertIsNot'
        return stypy_return_type_188212


    @norecursion
    def assertDictEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 825)
        None_188213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 42), 'None')
        defaults = [None_188213]
        # Create a new context for function 'assertDictEqual'
        module_type_store = module_type_store.open_function_context('assertDictEqual', 825, 4, False)
        # Assigning a type to the variable 'self' (line 826)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertDictEqual')
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_param_names_list', ['d1', 'd2', 'msg'])
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertDictEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertDictEqual', ['d1', 'd2', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertDictEqual', localization, ['d1', 'd2', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertDictEqual(...)' code ##################

        
        # Call to assertIsInstance(...): (line 826)
        # Processing the call arguments (line 826)
        # Getting the type of 'd1' (line 826)
        d1_188216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 30), 'd1', False)
        # Getting the type of 'dict' (line 826)
        dict_188217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 34), 'dict', False)
        str_188218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 40), 'str', 'First argument is not a dictionary')
        # Processing the call keyword arguments (line 826)
        kwargs_188219 = {}
        # Getting the type of 'self' (line 826)
        self_188214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 826)
        assertIsInstance_188215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 8), self_188214, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 826)
        assertIsInstance_call_result_188220 = invoke(stypy.reporting.localization.Localization(__file__, 826, 8), assertIsInstance_188215, *[d1_188216, dict_188217, str_188218], **kwargs_188219)
        
        
        # Call to assertIsInstance(...): (line 827)
        # Processing the call arguments (line 827)
        # Getting the type of 'd2' (line 827)
        d2_188223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 30), 'd2', False)
        # Getting the type of 'dict' (line 827)
        dict_188224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 34), 'dict', False)
        str_188225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 40), 'str', 'Second argument is not a dictionary')
        # Processing the call keyword arguments (line 827)
        kwargs_188226 = {}
        # Getting the type of 'self' (line 827)
        self_188221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 827)
        assertIsInstance_188222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 8), self_188221, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 827)
        assertIsInstance_call_result_188227 = invoke(stypy.reporting.localization.Localization(__file__, 827, 8), assertIsInstance_188222, *[d2_188223, dict_188224, str_188225], **kwargs_188226)
        
        
        
        # Getting the type of 'd1' (line 829)
        d1_188228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 11), 'd1')
        # Getting the type of 'd2' (line 829)
        d2_188229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 17), 'd2')
        # Applying the binary operator '!=' (line 829)
        result_ne_188230 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 11), '!=', d1_188228, d2_188229)
        
        # Testing the type of an if condition (line 829)
        if_condition_188231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 829, 8), result_ne_188230)
        # Assigning a type to the variable 'if_condition_188231' (line 829)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'if_condition_188231', if_condition_188231)
        # SSA begins for if statement (line 829)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 830):
        
        # Assigning a BinOp to a Name (line 830):
        str_188232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 26), 'str', '%s != %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 830)
        tuple_188233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 830)
        # Adding element type (line 830)
        
        # Call to safe_repr(...): (line 830)
        # Processing the call arguments (line 830)
        # Getting the type of 'd1' (line 830)
        d1_188235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 50), 'd1', False)
        # Getting the type of 'True' (line 830)
        True_188236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 54), 'True', False)
        # Processing the call keyword arguments (line 830)
        kwargs_188237 = {}
        # Getting the type of 'safe_repr' (line 830)
        safe_repr_188234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 40), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 830)
        safe_repr_call_result_188238 = invoke(stypy.reporting.localization.Localization(__file__, 830, 40), safe_repr_188234, *[d1_188235, True_188236], **kwargs_188237)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 830, 40), tuple_188233, safe_repr_call_result_188238)
        # Adding element type (line 830)
        
        # Call to safe_repr(...): (line 830)
        # Processing the call arguments (line 830)
        # Getting the type of 'd2' (line 830)
        d2_188240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 71), 'd2', False)
        # Getting the type of 'True' (line 830)
        True_188241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 75), 'True', False)
        # Processing the call keyword arguments (line 830)
        kwargs_188242 = {}
        # Getting the type of 'safe_repr' (line 830)
        safe_repr_188239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 61), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 830)
        safe_repr_call_result_188243 = invoke(stypy.reporting.localization.Localization(__file__, 830, 61), safe_repr_188239, *[d2_188240, True_188241], **kwargs_188242)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 830, 40), tuple_188233, safe_repr_call_result_188243)
        
        # Applying the binary operator '%' (line 830)
        result_mod_188244 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 26), '%', str_188232, tuple_188233)
        
        # Assigning a type to the variable 'standardMsg' (line 830)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 12), 'standardMsg', result_mod_188244)
        
        # Assigning a BinOp to a Name (line 831):
        
        # Assigning a BinOp to a Name (line 831):
        str_188245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 20), 'str', '\n')
        
        # Call to join(...): (line 831)
        # Processing the call arguments (line 831)
        
        # Call to ndiff(...): (line 831)
        # Processing the call arguments (line 831)
        
        # Call to splitlines(...): (line 832)
        # Processing the call keyword arguments (line 832)
        kwargs_188256 = {}
        
        # Call to pformat(...): (line 832)
        # Processing the call arguments (line 832)
        # Getting the type of 'd1' (line 832)
        d1_188252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 42), 'd1', False)
        # Processing the call keyword arguments (line 832)
        kwargs_188253 = {}
        # Getting the type of 'pprint' (line 832)
        pprint_188250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 27), 'pprint', False)
        # Obtaining the member 'pformat' of a type (line 832)
        pformat_188251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 27), pprint_188250, 'pformat')
        # Calling pformat(args, kwargs) (line 832)
        pformat_call_result_188254 = invoke(stypy.reporting.localization.Localization(__file__, 832, 27), pformat_188251, *[d1_188252], **kwargs_188253)
        
        # Obtaining the member 'splitlines' of a type (line 832)
        splitlines_188255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 27), pformat_call_result_188254, 'splitlines')
        # Calling splitlines(args, kwargs) (line 832)
        splitlines_call_result_188257 = invoke(stypy.reporting.localization.Localization(__file__, 832, 27), splitlines_188255, *[], **kwargs_188256)
        
        
        # Call to splitlines(...): (line 833)
        # Processing the call keyword arguments (line 833)
        kwargs_188264 = {}
        
        # Call to pformat(...): (line 833)
        # Processing the call arguments (line 833)
        # Getting the type of 'd2' (line 833)
        d2_188260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 42), 'd2', False)
        # Processing the call keyword arguments (line 833)
        kwargs_188261 = {}
        # Getting the type of 'pprint' (line 833)
        pprint_188258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 27), 'pprint', False)
        # Obtaining the member 'pformat' of a type (line 833)
        pformat_188259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 27), pprint_188258, 'pformat')
        # Calling pformat(args, kwargs) (line 833)
        pformat_call_result_188262 = invoke(stypy.reporting.localization.Localization(__file__, 833, 27), pformat_188259, *[d2_188260], **kwargs_188261)
        
        # Obtaining the member 'splitlines' of a type (line 833)
        splitlines_188263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 27), pformat_call_result_188262, 'splitlines')
        # Calling splitlines(args, kwargs) (line 833)
        splitlines_call_result_188265 = invoke(stypy.reporting.localization.Localization(__file__, 833, 27), splitlines_188263, *[], **kwargs_188264)
        
        # Processing the call keyword arguments (line 831)
        kwargs_188266 = {}
        # Getting the type of 'difflib' (line 831)
        difflib_188248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 37), 'difflib', False)
        # Obtaining the member 'ndiff' of a type (line 831)
        ndiff_188249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 37), difflib_188248, 'ndiff')
        # Calling ndiff(args, kwargs) (line 831)
        ndiff_call_result_188267 = invoke(stypy.reporting.localization.Localization(__file__, 831, 37), ndiff_188249, *[splitlines_call_result_188257, splitlines_call_result_188265], **kwargs_188266)
        
        # Processing the call keyword arguments (line 831)
        kwargs_188268 = {}
        str_188246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 27), 'str', '\n')
        # Obtaining the member 'join' of a type (line 831)
        join_188247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 27), str_188246, 'join')
        # Calling join(args, kwargs) (line 831)
        join_call_result_188269 = invoke(stypy.reporting.localization.Localization(__file__, 831, 27), join_188247, *[ndiff_call_result_188267], **kwargs_188268)
        
        # Applying the binary operator '+' (line 831)
        result_add_188270 = python_operator(stypy.reporting.localization.Localization(__file__, 831, 20), '+', str_188245, join_call_result_188269)
        
        # Assigning a type to the variable 'diff' (line 831)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 12), 'diff', result_add_188270)
        
        # Assigning a Call to a Name (line 834):
        
        # Assigning a Call to a Name (line 834):
        
        # Call to _truncateMessage(...): (line 834)
        # Processing the call arguments (line 834)
        # Getting the type of 'standardMsg' (line 834)
        standardMsg_188273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 48), 'standardMsg', False)
        # Getting the type of 'diff' (line 834)
        diff_188274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 61), 'diff', False)
        # Processing the call keyword arguments (line 834)
        kwargs_188275 = {}
        # Getting the type of 'self' (line 834)
        self_188271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 26), 'self', False)
        # Obtaining the member '_truncateMessage' of a type (line 834)
        _truncateMessage_188272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 26), self_188271, '_truncateMessage')
        # Calling _truncateMessage(args, kwargs) (line 834)
        _truncateMessage_call_result_188276 = invoke(stypy.reporting.localization.Localization(__file__, 834, 26), _truncateMessage_188272, *[standardMsg_188273, diff_188274], **kwargs_188275)
        
        # Assigning a type to the variable 'standardMsg' (line 834)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 12), 'standardMsg', _truncateMessage_call_result_188276)
        
        # Call to fail(...): (line 835)
        # Processing the call arguments (line 835)
        
        # Call to _formatMessage(...): (line 835)
        # Processing the call arguments (line 835)
        # Getting the type of 'msg' (line 835)
        msg_188281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 835)
        standardMsg_188282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 835)
        kwargs_188283 = {}
        # Getting the type of 'self' (line 835)
        self_188279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 835)
        _formatMessage_188280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 22), self_188279, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 835)
        _formatMessage_call_result_188284 = invoke(stypy.reporting.localization.Localization(__file__, 835, 22), _formatMessage_188280, *[msg_188281, standardMsg_188282], **kwargs_188283)
        
        # Processing the call keyword arguments (line 835)
        kwargs_188285 = {}
        # Getting the type of 'self' (line 835)
        self_188277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 835)
        fail_188278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 12), self_188277, 'fail')
        # Calling fail(args, kwargs) (line 835)
        fail_call_result_188286 = invoke(stypy.reporting.localization.Localization(__file__, 835, 12), fail_188278, *[_formatMessage_call_result_188284], **kwargs_188285)
        
        # SSA join for if statement (line 829)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertDictEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertDictEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 825)
        stypy_return_type_188287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188287)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertDictEqual'
        return stypy_return_type_188287


    @norecursion
    def assertDictContainsSubset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 837)
        None_188288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 61), 'None')
        defaults = [None_188288]
        # Create a new context for function 'assertDictContainsSubset'
        module_type_store = module_type_store.open_function_context('assertDictContainsSubset', 837, 4, False)
        # Assigning a type to the variable 'self' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_function_name', 'TestCase.assertDictContainsSubset')
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_param_names_list', ['expected', 'actual', 'msg'])
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertDictContainsSubset.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertDictContainsSubset', ['expected', 'actual', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertDictContainsSubset', localization, ['expected', 'actual', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertDictContainsSubset(...)' code ##################

        str_188289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 8), 'str', 'Checks whether actual is a superset of expected.')
        
        # Assigning a List to a Name (line 839):
        
        # Assigning a List to a Name (line 839):
        
        # Obtaining an instance of the builtin type 'list' (line 839)
        list_188290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 839)
        
        # Assigning a type to the variable 'missing' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'missing', list_188290)
        
        # Assigning a List to a Name (line 840):
        
        # Assigning a List to a Name (line 840):
        
        # Obtaining an instance of the builtin type 'list' (line 840)
        list_188291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 840)
        
        # Assigning a type to the variable 'mismatched' (line 840)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'mismatched', list_188291)
        
        
        # Call to iteritems(...): (line 841)
        # Processing the call keyword arguments (line 841)
        kwargs_188294 = {}
        # Getting the type of 'expected' (line 841)
        expected_188292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 26), 'expected', False)
        # Obtaining the member 'iteritems' of a type (line 841)
        iteritems_188293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 26), expected_188292, 'iteritems')
        # Calling iteritems(args, kwargs) (line 841)
        iteritems_call_result_188295 = invoke(stypy.reporting.localization.Localization(__file__, 841, 26), iteritems_188293, *[], **kwargs_188294)
        
        # Testing the type of a for loop iterable (line 841)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 841, 8), iteritems_call_result_188295)
        # Getting the type of the for loop variable (line 841)
        for_loop_var_188296 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 841, 8), iteritems_call_result_188295)
        # Assigning a type to the variable 'key' (line 841)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 841, 8), for_loop_var_188296))
        # Assigning a type to the variable 'value' (line 841)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 841, 8), for_loop_var_188296))
        # SSA begins for a for statement (line 841)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'key' (line 842)
        key_188297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 15), 'key')
        # Getting the type of 'actual' (line 842)
        actual_188298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 26), 'actual')
        # Applying the binary operator 'notin' (line 842)
        result_contains_188299 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 15), 'notin', key_188297, actual_188298)
        
        # Testing the type of an if condition (line 842)
        if_condition_188300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 842, 12), result_contains_188299)
        # Assigning a type to the variable 'if_condition_188300' (line 842)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'if_condition_188300', if_condition_188300)
        # SSA begins for if statement (line 842)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 843)
        # Processing the call arguments (line 843)
        # Getting the type of 'key' (line 843)
        key_188303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 31), 'key', False)
        # Processing the call keyword arguments (line 843)
        kwargs_188304 = {}
        # Getting the type of 'missing' (line 843)
        missing_188301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 16), 'missing', False)
        # Obtaining the member 'append' of a type (line 843)
        append_188302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 16), missing_188301, 'append')
        # Calling append(args, kwargs) (line 843)
        append_call_result_188305 = invoke(stypy.reporting.localization.Localization(__file__, 843, 16), append_188302, *[key_188303], **kwargs_188304)
        
        # SSA branch for the else part of an if statement (line 842)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'value' (line 844)
        value_188306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 17), 'value')
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 844)
        key_188307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 33), 'key')
        # Getting the type of 'actual' (line 844)
        actual_188308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 26), 'actual')
        # Obtaining the member '__getitem__' of a type (line 844)
        getitem___188309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 26), actual_188308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 844)
        subscript_call_result_188310 = invoke(stypy.reporting.localization.Localization(__file__, 844, 26), getitem___188309, key_188307)
        
        # Applying the binary operator '!=' (line 844)
        result_ne_188311 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 17), '!=', value_188306, subscript_call_result_188310)
        
        # Testing the type of an if condition (line 844)
        if_condition_188312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 844, 17), result_ne_188311)
        # Assigning a type to the variable 'if_condition_188312' (line 844)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 17), 'if_condition_188312', if_condition_188312)
        # SSA begins for if statement (line 844)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 845)
        # Processing the call arguments (line 845)
        str_188315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 34), 'str', '%s, expected: %s, actual: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 846)
        tuple_188316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 846)
        # Adding element type (line 846)
        
        # Call to safe_repr(...): (line 846)
        # Processing the call arguments (line 846)
        # Getting the type of 'key' (line 846)
        key_188318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 45), 'key', False)
        # Processing the call keyword arguments (line 846)
        kwargs_188319 = {}
        # Getting the type of 'safe_repr' (line 846)
        safe_repr_188317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 35), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 846)
        safe_repr_call_result_188320 = invoke(stypy.reporting.localization.Localization(__file__, 846, 35), safe_repr_188317, *[key_188318], **kwargs_188319)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 35), tuple_188316, safe_repr_call_result_188320)
        # Adding element type (line 846)
        
        # Call to safe_repr(...): (line 846)
        # Processing the call arguments (line 846)
        # Getting the type of 'value' (line 846)
        value_188322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 61), 'value', False)
        # Processing the call keyword arguments (line 846)
        kwargs_188323 = {}
        # Getting the type of 'safe_repr' (line 846)
        safe_repr_188321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 51), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 846)
        safe_repr_call_result_188324 = invoke(stypy.reporting.localization.Localization(__file__, 846, 51), safe_repr_188321, *[value_188322], **kwargs_188323)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 35), tuple_188316, safe_repr_call_result_188324)
        # Adding element type (line 846)
        
        # Call to safe_repr(...): (line 847)
        # Processing the call arguments (line 847)
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 847)
        key_188326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 52), 'key', False)
        # Getting the type of 'actual' (line 847)
        actual_188327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 45), 'actual', False)
        # Obtaining the member '__getitem__' of a type (line 847)
        getitem___188328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 45), actual_188327, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 847)
        subscript_call_result_188329 = invoke(stypy.reporting.localization.Localization(__file__, 847, 45), getitem___188328, key_188326)
        
        # Processing the call keyword arguments (line 847)
        kwargs_188330 = {}
        # Getting the type of 'safe_repr' (line 847)
        safe_repr_188325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 35), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 847)
        safe_repr_call_result_188331 = invoke(stypy.reporting.localization.Localization(__file__, 847, 35), safe_repr_188325, *[subscript_call_result_188329], **kwargs_188330)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 35), tuple_188316, safe_repr_call_result_188331)
        
        # Applying the binary operator '%' (line 845)
        result_mod_188332 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 34), '%', str_188315, tuple_188316)
        
        # Processing the call keyword arguments (line 845)
        kwargs_188333 = {}
        # Getting the type of 'mismatched' (line 845)
        mismatched_188313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 16), 'mismatched', False)
        # Obtaining the member 'append' of a type (line 845)
        append_188314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 16), mismatched_188313, 'append')
        # Calling append(args, kwargs) (line 845)
        append_call_result_188334 = invoke(stypy.reporting.localization.Localization(__file__, 845, 16), append_188314, *[result_mod_188332], **kwargs_188333)
        
        # SSA join for if statement (line 844)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 842)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'missing' (line 849)
        missing_188335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 16), 'missing')
        # Getting the type of 'mismatched' (line 849)
        mismatched_188336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 27), 'mismatched')
        # Applying the binary operator 'or' (line 849)
        result_or_keyword_188337 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 16), 'or', missing_188335, mismatched_188336)
        
        # Applying the 'not' unary operator (line 849)
        result_not__188338 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 11), 'not', result_or_keyword_188337)
        
        # Testing the type of an if condition (line 849)
        if_condition_188339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 849, 8), result_not__188338)
        # Assigning a type to the variable 'if_condition_188339' (line 849)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'if_condition_188339', if_condition_188339)
        # SSA begins for if statement (line 849)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 850)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 849)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 852):
        
        # Assigning a Str to a Name (line 852):
        str_188340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 22), 'str', '')
        # Assigning a type to the variable 'standardMsg' (line 852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 8), 'standardMsg', str_188340)
        
        # Getting the type of 'missing' (line 853)
        missing_188341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 11), 'missing')
        # Testing the type of an if condition (line 853)
        if_condition_188342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 853, 8), missing_188341)
        # Assigning a type to the variable 'if_condition_188342' (line 853)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'if_condition_188342', if_condition_188342)
        # SSA begins for if statement (line 853)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 854):
        
        # Assigning a BinOp to a Name (line 854):
        str_188343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 26), 'str', 'Missing: %s')
        
        # Call to join(...): (line 854)
        # Processing the call arguments (line 854)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 854, 51, True)
        # Calculating comprehension expression
        # Getting the type of 'missing' (line 855)
        missing_188350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 52), 'missing', False)
        comprehension_188351 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 854, 51), missing_188350)
        # Assigning a type to the variable 'm' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 51), 'm', comprehension_188351)
        
        # Call to safe_repr(...): (line 854)
        # Processing the call arguments (line 854)
        # Getting the type of 'm' (line 854)
        m_188347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 61), 'm', False)
        # Processing the call keyword arguments (line 854)
        kwargs_188348 = {}
        # Getting the type of 'safe_repr' (line 854)
        safe_repr_188346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 51), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 854)
        safe_repr_call_result_188349 = invoke(stypy.reporting.localization.Localization(__file__, 854, 51), safe_repr_188346, *[m_188347], **kwargs_188348)
        
        list_188352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 51), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 854, 51), list_188352, safe_repr_call_result_188349)
        # Processing the call keyword arguments (line 854)
        kwargs_188353 = {}
        str_188344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 42), 'str', ',')
        # Obtaining the member 'join' of a type (line 854)
        join_188345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 42), str_188344, 'join')
        # Calling join(args, kwargs) (line 854)
        join_call_result_188354 = invoke(stypy.reporting.localization.Localization(__file__, 854, 42), join_188345, *[list_188352], **kwargs_188353)
        
        # Applying the binary operator '%' (line 854)
        result_mod_188355 = python_operator(stypy.reporting.localization.Localization(__file__, 854, 26), '%', str_188343, join_call_result_188354)
        
        # Assigning a type to the variable 'standardMsg' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 12), 'standardMsg', result_mod_188355)
        # SSA join for if statement (line 853)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'mismatched' (line 856)
        mismatched_188356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 11), 'mismatched')
        # Testing the type of an if condition (line 856)
        if_condition_188357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 856, 8), mismatched_188356)
        # Assigning a type to the variable 'if_condition_188357' (line 856)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 8), 'if_condition_188357', if_condition_188357)
        # SSA begins for if statement (line 856)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'standardMsg' (line 857)
        standardMsg_188358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 15), 'standardMsg')
        # Testing the type of an if condition (line 857)
        if_condition_188359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 857, 12), standardMsg_188358)
        # Assigning a type to the variable 'if_condition_188359' (line 857)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 12), 'if_condition_188359', if_condition_188359)
        # SSA begins for if statement (line 857)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'standardMsg' (line 858)
        standardMsg_188360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 16), 'standardMsg')
        str_188361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, 31), 'str', '; ')
        # Applying the binary operator '+=' (line 858)
        result_iadd_188362 = python_operator(stypy.reporting.localization.Localization(__file__, 858, 16), '+=', standardMsg_188360, str_188361)
        # Assigning a type to the variable 'standardMsg' (line 858)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 858, 16), 'standardMsg', result_iadd_188362)
        
        # SSA join for if statement (line 857)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'standardMsg' (line 859)
        standardMsg_188363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 12), 'standardMsg')
        str_188364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 27), 'str', 'Mismatched values: %s')
        
        # Call to join(...): (line 859)
        # Processing the call arguments (line 859)
        # Getting the type of 'mismatched' (line 859)
        mismatched_188367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 62), 'mismatched', False)
        # Processing the call keyword arguments (line 859)
        kwargs_188368 = {}
        str_188365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 53), 'str', ',')
        # Obtaining the member 'join' of a type (line 859)
        join_188366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 53), str_188365, 'join')
        # Calling join(args, kwargs) (line 859)
        join_call_result_188369 = invoke(stypy.reporting.localization.Localization(__file__, 859, 53), join_188366, *[mismatched_188367], **kwargs_188368)
        
        # Applying the binary operator '%' (line 859)
        result_mod_188370 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 27), '%', str_188364, join_call_result_188369)
        
        # Applying the binary operator '+=' (line 859)
        result_iadd_188371 = python_operator(stypy.reporting.localization.Localization(__file__, 859, 12), '+=', standardMsg_188363, result_mod_188370)
        # Assigning a type to the variable 'standardMsg' (line 859)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 12), 'standardMsg', result_iadd_188371)
        
        # SSA join for if statement (line 856)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to fail(...): (line 861)
        # Processing the call arguments (line 861)
        
        # Call to _formatMessage(...): (line 861)
        # Processing the call arguments (line 861)
        # Getting the type of 'msg' (line 861)
        msg_188376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 38), 'msg', False)
        # Getting the type of 'standardMsg' (line 861)
        standardMsg_188377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 43), 'standardMsg', False)
        # Processing the call keyword arguments (line 861)
        kwargs_188378 = {}
        # Getting the type of 'self' (line 861)
        self_188374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 18), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 861)
        _formatMessage_188375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 18), self_188374, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 861)
        _formatMessage_call_result_188379 = invoke(stypy.reporting.localization.Localization(__file__, 861, 18), _formatMessage_188375, *[msg_188376, standardMsg_188377], **kwargs_188378)
        
        # Processing the call keyword arguments (line 861)
        kwargs_188380 = {}
        # Getting the type of 'self' (line 861)
        self_188372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 8), 'self', False)
        # Obtaining the member 'fail' of a type (line 861)
        fail_188373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 8), self_188372, 'fail')
        # Calling fail(args, kwargs) (line 861)
        fail_call_result_188381 = invoke(stypy.reporting.localization.Localization(__file__, 861, 8), fail_188373, *[_formatMessage_call_result_188379], **kwargs_188380)
        
        
        # ################# End of 'assertDictContainsSubset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertDictContainsSubset' in the type store
        # Getting the type of 'stypy_return_type' (line 837)
        stypy_return_type_188382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188382)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertDictContainsSubset'
        return stypy_return_type_188382


    @norecursion
    def assertItemsEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 863)
        None_188383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 61), 'None')
        defaults = [None_188383]
        # Create a new context for function 'assertItemsEqual'
        module_type_store = module_type_store.open_function_context('assertItemsEqual', 863, 4, False)
        # Assigning a type to the variable 'self' (line 864)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertItemsEqual')
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_param_names_list', ['expected_seq', 'actual_seq', 'msg'])
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertItemsEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertItemsEqual', ['expected_seq', 'actual_seq', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertItemsEqual', localization, ['expected_seq', 'actual_seq', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertItemsEqual(...)' code ##################

        str_188384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, (-1)), 'str', 'An unordered sequence specific comparison. It asserts that\n        actual_seq and expected_seq have the same element counts.\n        Equivalent to::\n\n            self.assertEqual(Counter(iter(actual_seq)),\n                             Counter(iter(expected_seq)))\n\n        Asserts that each element has the same count in both sequences.\n        Example:\n            - [0, 1, 1] and [1, 0, 1] compare equal.\n            - [0, 0, 1] and [0, 1] compare unequal.\n        ')
        
        # Assigning a Tuple to a Tuple (line 876):
        
        # Assigning a Call to a Name (line 876):
        
        # Call to list(...): (line 876)
        # Processing the call arguments (line 876)
        # Getting the type of 'expected_seq' (line 876)
        expected_seq_188386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 37), 'expected_seq', False)
        # Processing the call keyword arguments (line 876)
        kwargs_188387 = {}
        # Getting the type of 'list' (line 876)
        list_188385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 32), 'list', False)
        # Calling list(args, kwargs) (line 876)
        list_call_result_188388 = invoke(stypy.reporting.localization.Localization(__file__, 876, 32), list_188385, *[expected_seq_188386], **kwargs_188387)
        
        # Assigning a type to the variable 'tuple_assignment_186532' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'tuple_assignment_186532', list_call_result_188388)
        
        # Assigning a Call to a Name (line 876):
        
        # Call to list(...): (line 876)
        # Processing the call arguments (line 876)
        # Getting the type of 'actual_seq' (line 876)
        actual_seq_188390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 57), 'actual_seq', False)
        # Processing the call keyword arguments (line 876)
        kwargs_188391 = {}
        # Getting the type of 'list' (line 876)
        list_188389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 52), 'list', False)
        # Calling list(args, kwargs) (line 876)
        list_call_result_188392 = invoke(stypy.reporting.localization.Localization(__file__, 876, 52), list_188389, *[actual_seq_188390], **kwargs_188391)
        
        # Assigning a type to the variable 'tuple_assignment_186533' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'tuple_assignment_186533', list_call_result_188392)
        
        # Assigning a Name to a Name (line 876):
        # Getting the type of 'tuple_assignment_186532' (line 876)
        tuple_assignment_186532_188393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'tuple_assignment_186532')
        # Assigning a type to the variable 'first_seq' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'first_seq', tuple_assignment_186532_188393)
        
        # Assigning a Name to a Name (line 876):
        # Getting the type of 'tuple_assignment_186533' (line 876)
        tuple_assignment_186533_188394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'tuple_assignment_186533')
        # Assigning a type to the variable 'second_seq' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 19), 'second_seq', tuple_assignment_186533_188394)
        
        # Call to catch_warnings(...): (line 877)
        # Processing the call keyword arguments (line 877)
        kwargs_188397 = {}
        # Getting the type of 'warnings' (line 877)
        warnings_188395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 13), 'warnings', False)
        # Obtaining the member 'catch_warnings' of a type (line 877)
        catch_warnings_188396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 13), warnings_188395, 'catch_warnings')
        # Calling catch_warnings(args, kwargs) (line 877)
        catch_warnings_call_result_188398 = invoke(stypy.reporting.localization.Localization(__file__, 877, 13), catch_warnings_188396, *[], **kwargs_188397)
        
        with_188399 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 877, 13), catch_warnings_call_result_188398, 'with parameter', '__enter__', '__exit__')

        if with_188399:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 877)
            enter___188400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 13), catch_warnings_call_result_188398, '__enter__')
            with_enter_188401 = invoke(stypy.reporting.localization.Localization(__file__, 877, 13), enter___188400)
            
            # Getting the type of 'sys' (line 878)
            sys_188402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 15), 'sys')
            # Obtaining the member 'py3kwarning' of a type (line 878)
            py3kwarning_188403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 15), sys_188402, 'py3kwarning')
            # Testing the type of an if condition (line 878)
            if_condition_188404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 878, 12), py3kwarning_188403)
            # Assigning a type to the variable 'if_condition_188404' (line 878)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 12), 'if_condition_188404', if_condition_188404)
            # SSA begins for if statement (line 878)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Obtaining an instance of the builtin type 'list' (line 880)
            list_188405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 880)
            # Adding element type (line 880)
            str_188406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 29), 'str', '(code|dict|type) inequality comparisons')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 880, 28), list_188405, str_188406)
            # Adding element type (line 880)
            str_188407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 29), 'str', 'builtin_function_or_method order comparisons')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 880, 28), list_188405, str_188407)
            # Adding element type (line 880)
            str_188408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 29), 'str', 'comparing unequal types')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 880, 28), list_188405, str_188408)
            
            # Testing the type of a for loop iterable (line 880)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 880, 16), list_188405)
            # Getting the type of the for loop variable (line 880)
            for_loop_var_188409 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 880, 16), list_188405)
            # Assigning a type to the variable '_msg' (line 880)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 16), '_msg', for_loop_var_188409)
            # SSA begins for a for statement (line 880)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to filterwarnings(...): (line 883)
            # Processing the call arguments (line 883)
            str_188412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 44), 'str', 'ignore')
            # Getting the type of '_msg' (line 883)
            _msg_188413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 54), '_msg', False)
            # Getting the type of 'DeprecationWarning' (line 883)
            DeprecationWarning_188414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 60), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 883)
            kwargs_188415 = {}
            # Getting the type of 'warnings' (line 883)
            warnings_188410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 20), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 883)
            filterwarnings_188411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 20), warnings_188410, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 883)
            filterwarnings_call_result_188416 = invoke(stypy.reporting.localization.Localization(__file__, 883, 20), filterwarnings_188411, *[str_188412, _msg_188413, DeprecationWarning_188414], **kwargs_188415)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 878)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # SSA begins for try-except statement (line 884)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 885):
            
            # Assigning a Call to a Name (line 885):
            
            # Call to Counter(...): (line 885)
            # Processing the call arguments (line 885)
            # Getting the type of 'first_seq' (line 885)
            first_seq_188419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 44), 'first_seq', False)
            # Processing the call keyword arguments (line 885)
            kwargs_188420 = {}
            # Getting the type of 'collections' (line 885)
            collections_188417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 24), 'collections', False)
            # Obtaining the member 'Counter' of a type (line 885)
            Counter_188418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 24), collections_188417, 'Counter')
            # Calling Counter(args, kwargs) (line 885)
            Counter_call_result_188421 = invoke(stypy.reporting.localization.Localization(__file__, 885, 24), Counter_188418, *[first_seq_188419], **kwargs_188420)
            
            # Assigning a type to the variable 'first' (line 885)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 16), 'first', Counter_call_result_188421)
            
            # Assigning a Call to a Name (line 886):
            
            # Assigning a Call to a Name (line 886):
            
            # Call to Counter(...): (line 886)
            # Processing the call arguments (line 886)
            # Getting the type of 'second_seq' (line 886)
            second_seq_188424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 45), 'second_seq', False)
            # Processing the call keyword arguments (line 886)
            kwargs_188425 = {}
            # Getting the type of 'collections' (line 886)
            collections_188422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 25), 'collections', False)
            # Obtaining the member 'Counter' of a type (line 886)
            Counter_188423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 25), collections_188422, 'Counter')
            # Calling Counter(args, kwargs) (line 886)
            Counter_call_result_188426 = invoke(stypy.reporting.localization.Localization(__file__, 886, 25), Counter_188423, *[second_seq_188424], **kwargs_188425)
            
            # Assigning a type to the variable 'second' (line 886)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 16), 'second', Counter_call_result_188426)
            # SSA branch for the except part of a try statement (line 884)
            # SSA branch for the except 'TypeError' branch of a try statement (line 884)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Call to a Name (line 889):
            
            # Assigning a Call to a Name (line 889):
            
            # Call to _count_diff_all_purpose(...): (line 889)
            # Processing the call arguments (line 889)
            # Getting the type of 'first_seq' (line 889)
            first_seq_188428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 54), 'first_seq', False)
            # Getting the type of 'second_seq' (line 889)
            second_seq_188429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 65), 'second_seq', False)
            # Processing the call keyword arguments (line 889)
            kwargs_188430 = {}
            # Getting the type of '_count_diff_all_purpose' (line 889)
            _count_diff_all_purpose_188427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 30), '_count_diff_all_purpose', False)
            # Calling _count_diff_all_purpose(args, kwargs) (line 889)
            _count_diff_all_purpose_call_result_188431 = invoke(stypy.reporting.localization.Localization(__file__, 889, 30), _count_diff_all_purpose_188427, *[first_seq_188428, second_seq_188429], **kwargs_188430)
            
            # Assigning a type to the variable 'differences' (line 889)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 16), 'differences', _count_diff_all_purpose_call_result_188431)
            # SSA branch for the else branch of a try statement (line 884)
            module_type_store.open_ssa_branch('except else')
            
            
            # Getting the type of 'first' (line 891)
            first_188432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 19), 'first')
            # Getting the type of 'second' (line 891)
            second_188433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 28), 'second')
            # Applying the binary operator '==' (line 891)
            result_eq_188434 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 19), '==', first_188432, second_188433)
            
            # Testing the type of an if condition (line 891)
            if_condition_188435 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 891, 16), result_eq_188434)
            # Assigning a type to the variable 'if_condition_188435' (line 891)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 16), 'if_condition_188435', if_condition_188435)
            # SSA begins for if statement (line 891)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 892)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 20), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 891)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 893):
            
            # Assigning a Call to a Name (line 893):
            
            # Call to _count_diff_hashable(...): (line 893)
            # Processing the call arguments (line 893)
            # Getting the type of 'first_seq' (line 893)
            first_seq_188437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 51), 'first_seq', False)
            # Getting the type of 'second_seq' (line 893)
            second_seq_188438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 62), 'second_seq', False)
            # Processing the call keyword arguments (line 893)
            kwargs_188439 = {}
            # Getting the type of '_count_diff_hashable' (line 893)
            _count_diff_hashable_188436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 30), '_count_diff_hashable', False)
            # Calling _count_diff_hashable(args, kwargs) (line 893)
            _count_diff_hashable_call_result_188440 = invoke(stypy.reporting.localization.Localization(__file__, 893, 30), _count_diff_hashable_188436, *[first_seq_188437, second_seq_188438], **kwargs_188439)
            
            # Assigning a type to the variable 'differences' (line 893)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 16), 'differences', _count_diff_hashable_call_result_188440)
            # SSA join for try-except statement (line 884)
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 877)
            exit___188441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 13), catch_warnings_call_result_188398, '__exit__')
            with_exit_188442 = invoke(stypy.reporting.localization.Localization(__file__, 877, 13), exit___188441, None, None, None)

        
        # Getting the type of 'differences' (line 895)
        differences_188443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 11), 'differences')
        # Testing the type of an if condition (line 895)
        if_condition_188444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 895, 8), differences_188443)
        # Assigning a type to the variable 'if_condition_188444' (line 895)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 8), 'if_condition_188444', if_condition_188444)
        # SSA begins for if statement (line 895)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 896):
        
        # Assigning a Str to a Name (line 896):
        str_188445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 26), 'str', 'Element counts were not equal:\n')
        # Assigning a type to the variable 'standardMsg' (line 896)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 12), 'standardMsg', str_188445)
        
        # Assigning a ListComp to a Name (line 897):
        
        # Assigning a ListComp to a Name (line 897):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'differences' (line 897)
        differences_188449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 75), 'differences')
        comprehension_188450 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 21), differences_188449)
        # Assigning a type to the variable 'diff' (line 897)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 21), 'diff', comprehension_188450)
        str_188446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 21), 'str', 'First has %d, Second has %d:  %r')
        # Getting the type of 'diff' (line 897)
        diff_188447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 58), 'diff')
        # Applying the binary operator '%' (line 897)
        result_mod_188448 = python_operator(stypy.reporting.localization.Localization(__file__, 897, 21), '%', str_188446, diff_188447)
        
        list_188451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 21), list_188451, result_mod_188448)
        # Assigning a type to the variable 'lines' (line 897)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 12), 'lines', list_188451)
        
        # Assigning a Call to a Name (line 898):
        
        # Assigning a Call to a Name (line 898):
        
        # Call to join(...): (line 898)
        # Processing the call arguments (line 898)
        # Getting the type of 'lines' (line 898)
        lines_188454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 32), 'lines', False)
        # Processing the call keyword arguments (line 898)
        kwargs_188455 = {}
        str_188452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 22), 'str', '\n')
        # Obtaining the member 'join' of a type (line 898)
        join_188453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 898, 22), str_188452, 'join')
        # Calling join(args, kwargs) (line 898)
        join_call_result_188456 = invoke(stypy.reporting.localization.Localization(__file__, 898, 22), join_188453, *[lines_188454], **kwargs_188455)
        
        # Assigning a type to the variable 'diffMsg' (line 898)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 12), 'diffMsg', join_call_result_188456)
        
        # Assigning a Call to a Name (line 899):
        
        # Assigning a Call to a Name (line 899):
        
        # Call to _truncateMessage(...): (line 899)
        # Processing the call arguments (line 899)
        # Getting the type of 'standardMsg' (line 899)
        standardMsg_188459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 48), 'standardMsg', False)
        # Getting the type of 'diffMsg' (line 899)
        diffMsg_188460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 61), 'diffMsg', False)
        # Processing the call keyword arguments (line 899)
        kwargs_188461 = {}
        # Getting the type of 'self' (line 899)
        self_188457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 26), 'self', False)
        # Obtaining the member '_truncateMessage' of a type (line 899)
        _truncateMessage_188458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 26), self_188457, '_truncateMessage')
        # Calling _truncateMessage(args, kwargs) (line 899)
        _truncateMessage_call_result_188462 = invoke(stypy.reporting.localization.Localization(__file__, 899, 26), _truncateMessage_188458, *[standardMsg_188459, diffMsg_188460], **kwargs_188461)
        
        # Assigning a type to the variable 'standardMsg' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'standardMsg', _truncateMessage_call_result_188462)
        
        # Assigning a Call to a Name (line 900):
        
        # Assigning a Call to a Name (line 900):
        
        # Call to _formatMessage(...): (line 900)
        # Processing the call arguments (line 900)
        # Getting the type of 'msg' (line 900)
        msg_188465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 38), 'msg', False)
        # Getting the type of 'standardMsg' (line 900)
        standardMsg_188466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 43), 'standardMsg', False)
        # Processing the call keyword arguments (line 900)
        kwargs_188467 = {}
        # Getting the type of 'self' (line 900)
        self_188463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 18), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 900)
        _formatMessage_188464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 18), self_188463, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 900)
        _formatMessage_call_result_188468 = invoke(stypy.reporting.localization.Localization(__file__, 900, 18), _formatMessage_188464, *[msg_188465, standardMsg_188466], **kwargs_188467)
        
        # Assigning a type to the variable 'msg' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'msg', _formatMessage_call_result_188468)
        
        # Call to fail(...): (line 901)
        # Processing the call arguments (line 901)
        # Getting the type of 'msg' (line 901)
        msg_188471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 22), 'msg', False)
        # Processing the call keyword arguments (line 901)
        kwargs_188472 = {}
        # Getting the type of 'self' (line 901)
        self_188469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 901)
        fail_188470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 12), self_188469, 'fail')
        # Calling fail(args, kwargs) (line 901)
        fail_call_result_188473 = invoke(stypy.reporting.localization.Localization(__file__, 901, 12), fail_188470, *[msg_188471], **kwargs_188472)
        
        # SSA join for if statement (line 895)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertItemsEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertItemsEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 863)
        stypy_return_type_188474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188474)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertItemsEqual'
        return stypy_return_type_188474


    @norecursion
    def assertMultiLineEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 903)
        None_188475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 54), 'None')
        defaults = [None_188475]
        # Create a new context for function 'assertMultiLineEqual'
        module_type_store = module_type_store.open_function_context('assertMultiLineEqual', 903, 4, False)
        # Assigning a type to the variable 'self' (line 904)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertMultiLineEqual')
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_param_names_list', ['first', 'second', 'msg'])
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertMultiLineEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertMultiLineEqual', ['first', 'second', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertMultiLineEqual', localization, ['first', 'second', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertMultiLineEqual(...)' code ##################

        str_188476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 8), 'str', 'Assert that two multi-line strings are equal.')
        
        # Call to assertIsInstance(...): (line 905)
        # Processing the call arguments (line 905)
        # Getting the type of 'first' (line 905)
        first_188479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 30), 'first', False)
        # Getting the type of 'basestring' (line 905)
        basestring_188480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 37), 'basestring', False)
        str_188481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 16), 'str', 'First argument is not a string')
        # Processing the call keyword arguments (line 905)
        kwargs_188482 = {}
        # Getting the type of 'self' (line 905)
        self_188477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 905)
        assertIsInstance_188478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 8), self_188477, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 905)
        assertIsInstance_call_result_188483 = invoke(stypy.reporting.localization.Localization(__file__, 905, 8), assertIsInstance_188478, *[first_188479, basestring_188480, str_188481], **kwargs_188482)
        
        
        # Call to assertIsInstance(...): (line 907)
        # Processing the call arguments (line 907)
        # Getting the type of 'second' (line 907)
        second_188486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 30), 'second', False)
        # Getting the type of 'basestring' (line 907)
        basestring_188487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 38), 'basestring', False)
        str_188488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 908, 16), 'str', 'Second argument is not a string')
        # Processing the call keyword arguments (line 907)
        kwargs_188489 = {}
        # Getting the type of 'self' (line 907)
        self_188484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 907)
        assertIsInstance_188485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 8), self_188484, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 907)
        assertIsInstance_call_result_188490 = invoke(stypy.reporting.localization.Localization(__file__, 907, 8), assertIsInstance_188485, *[second_188486, basestring_188487, str_188488], **kwargs_188489)
        
        
        
        # Getting the type of 'first' (line 910)
        first_188491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 11), 'first')
        # Getting the type of 'second' (line 910)
        second_188492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 20), 'second')
        # Applying the binary operator '!=' (line 910)
        result_ne_188493 = python_operator(stypy.reporting.localization.Localization(__file__, 910, 11), '!=', first_188491, second_188492)
        
        # Testing the type of an if condition (line 910)
        if_condition_188494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 910, 8), result_ne_188493)
        # Assigning a type to the variable 'if_condition_188494' (line 910)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 8), 'if_condition_188494', if_condition_188494)
        # SSA begins for if statement (line 910)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 912)
        # Processing the call arguments (line 912)
        # Getting the type of 'first' (line 912)
        first_188496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 20), 'first', False)
        # Processing the call keyword arguments (line 912)
        kwargs_188497 = {}
        # Getting the type of 'len' (line 912)
        len_188495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 16), 'len', False)
        # Calling len(args, kwargs) (line 912)
        len_call_result_188498 = invoke(stypy.reporting.localization.Localization(__file__, 912, 16), len_188495, *[first_188496], **kwargs_188497)
        
        # Getting the type of 'self' (line 912)
        self_188499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 29), 'self')
        # Obtaining the member '_diffThreshold' of a type (line 912)
        _diffThreshold_188500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 29), self_188499, '_diffThreshold')
        # Applying the binary operator '>' (line 912)
        result_gt_188501 = python_operator(stypy.reporting.localization.Localization(__file__, 912, 16), '>', len_call_result_188498, _diffThreshold_188500)
        
        
        
        # Call to len(...): (line 913)
        # Processing the call arguments (line 913)
        # Getting the type of 'second' (line 913)
        second_188503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 20), 'second', False)
        # Processing the call keyword arguments (line 913)
        kwargs_188504 = {}
        # Getting the type of 'len' (line 913)
        len_188502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 16), 'len', False)
        # Calling len(args, kwargs) (line 913)
        len_call_result_188505 = invoke(stypy.reporting.localization.Localization(__file__, 913, 16), len_188502, *[second_188503], **kwargs_188504)
        
        # Getting the type of 'self' (line 913)
        self_188506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 30), 'self')
        # Obtaining the member '_diffThreshold' of a type (line 913)
        _diffThreshold_188507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 30), self_188506, '_diffThreshold')
        # Applying the binary operator '>' (line 913)
        result_gt_188508 = python_operator(stypy.reporting.localization.Localization(__file__, 913, 16), '>', len_call_result_188505, _diffThreshold_188507)
        
        # Applying the binary operator 'or' (line 912)
        result_or_keyword_188509 = python_operator(stypy.reporting.localization.Localization(__file__, 912, 16), 'or', result_gt_188501, result_gt_188508)
        
        # Testing the type of an if condition (line 912)
        if_condition_188510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 912, 12), result_or_keyword_188509)
        # Assigning a type to the variable 'if_condition_188510' (line 912)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 12), 'if_condition_188510', if_condition_188510)
        # SSA begins for if statement (line 912)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _baseAssertEqual(...): (line 914)
        # Processing the call arguments (line 914)
        # Getting the type of 'first' (line 914)
        first_188513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 38), 'first', False)
        # Getting the type of 'second' (line 914)
        second_188514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 45), 'second', False)
        # Getting the type of 'msg' (line 914)
        msg_188515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 53), 'msg', False)
        # Processing the call keyword arguments (line 914)
        kwargs_188516 = {}
        # Getting the type of 'self' (line 914)
        self_188511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 16), 'self', False)
        # Obtaining the member '_baseAssertEqual' of a type (line 914)
        _baseAssertEqual_188512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 16), self_188511, '_baseAssertEqual')
        # Calling _baseAssertEqual(args, kwargs) (line 914)
        _baseAssertEqual_call_result_188517 = invoke(stypy.reporting.localization.Localization(__file__, 914, 16), _baseAssertEqual_188512, *[first_188513, second_188514, msg_188515], **kwargs_188516)
        
        # SSA join for if statement (line 912)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 915):
        
        # Assigning a Call to a Name (line 915):
        
        # Call to splitlines(...): (line 915)
        # Processing the call arguments (line 915)
        # Getting the type of 'True' (line 915)
        True_188520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 42), 'True', False)
        # Processing the call keyword arguments (line 915)
        kwargs_188521 = {}
        # Getting the type of 'first' (line 915)
        first_188518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 25), 'first', False)
        # Obtaining the member 'splitlines' of a type (line 915)
        splitlines_188519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 25), first_188518, 'splitlines')
        # Calling splitlines(args, kwargs) (line 915)
        splitlines_call_result_188522 = invoke(stypy.reporting.localization.Localization(__file__, 915, 25), splitlines_188519, *[True_188520], **kwargs_188521)
        
        # Assigning a type to the variable 'firstlines' (line 915)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 12), 'firstlines', splitlines_call_result_188522)
        
        # Assigning a Call to a Name (line 916):
        
        # Assigning a Call to a Name (line 916):
        
        # Call to splitlines(...): (line 916)
        # Processing the call arguments (line 916)
        # Getting the type of 'True' (line 916)
        True_188525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 44), 'True', False)
        # Processing the call keyword arguments (line 916)
        kwargs_188526 = {}
        # Getting the type of 'second' (line 916)
        second_188523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 26), 'second', False)
        # Obtaining the member 'splitlines' of a type (line 916)
        splitlines_188524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 26), second_188523, 'splitlines')
        # Calling splitlines(args, kwargs) (line 916)
        splitlines_call_result_188527 = invoke(stypy.reporting.localization.Localization(__file__, 916, 26), splitlines_188524, *[True_188525], **kwargs_188526)
        
        # Assigning a type to the variable 'secondlines' (line 916)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 12), 'secondlines', splitlines_call_result_188527)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 917)
        # Processing the call arguments (line 917)
        # Getting the type of 'firstlines' (line 917)
        firstlines_188529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 19), 'firstlines', False)
        # Processing the call keyword arguments (line 917)
        kwargs_188530 = {}
        # Getting the type of 'len' (line 917)
        len_188528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 15), 'len', False)
        # Calling len(args, kwargs) (line 917)
        len_call_result_188531 = invoke(stypy.reporting.localization.Localization(__file__, 917, 15), len_188528, *[firstlines_188529], **kwargs_188530)
        
        int_188532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, 34), 'int')
        # Applying the binary operator '==' (line 917)
        result_eq_188533 = python_operator(stypy.reporting.localization.Localization(__file__, 917, 15), '==', len_call_result_188531, int_188532)
        
        
        
        # Call to strip(...): (line 917)
        # Processing the call arguments (line 917)
        str_188536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, 52), 'str', '\r\n')
        # Processing the call keyword arguments (line 917)
        kwargs_188537 = {}
        # Getting the type of 'first' (line 917)
        first_188534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 40), 'first', False)
        # Obtaining the member 'strip' of a type (line 917)
        strip_188535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 40), first_188534, 'strip')
        # Calling strip(args, kwargs) (line 917)
        strip_call_result_188538 = invoke(stypy.reporting.localization.Localization(__file__, 917, 40), strip_188535, *[str_188536], **kwargs_188537)
        
        # Getting the type of 'first' (line 917)
        first_188539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 63), 'first')
        # Applying the binary operator '==' (line 917)
        result_eq_188540 = python_operator(stypy.reporting.localization.Localization(__file__, 917, 40), '==', strip_call_result_188538, first_188539)
        
        # Applying the binary operator 'and' (line 917)
        result_and_keyword_188541 = python_operator(stypy.reporting.localization.Localization(__file__, 917, 15), 'and', result_eq_188533, result_eq_188540)
        
        # Testing the type of an if condition (line 917)
        if_condition_188542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 917, 12), result_and_keyword_188541)
        # Assigning a type to the variable 'if_condition_188542' (line 917)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 12), 'if_condition_188542', if_condition_188542)
        # SSA begins for if statement (line 917)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 918):
        
        # Assigning a List to a Name (line 918):
        
        # Obtaining an instance of the builtin type 'list' (line 918)
        list_188543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 918)
        # Adding element type (line 918)
        # Getting the type of 'first' (line 918)
        first_188544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 30), 'first')
        str_188545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 38), 'str', '\n')
        # Applying the binary operator '+' (line 918)
        result_add_188546 = python_operator(stypy.reporting.localization.Localization(__file__, 918, 30), '+', first_188544, str_188545)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 918, 29), list_188543, result_add_188546)
        
        # Assigning a type to the variable 'firstlines' (line 918)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 16), 'firstlines', list_188543)
        
        # Assigning a List to a Name (line 919):
        
        # Assigning a List to a Name (line 919):
        
        # Obtaining an instance of the builtin type 'list' (line 919)
        list_188547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 919)
        # Adding element type (line 919)
        # Getting the type of 'second' (line 919)
        second_188548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 31), 'second')
        str_188549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 40), 'str', '\n')
        # Applying the binary operator '+' (line 919)
        result_add_188550 = python_operator(stypy.reporting.localization.Localization(__file__, 919, 31), '+', second_188548, str_188549)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 919, 30), list_188547, result_add_188550)
        
        # Assigning a type to the variable 'secondlines' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 16), 'secondlines', list_188547)
        # SSA join for if statement (line 917)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 920):
        
        # Assigning a BinOp to a Name (line 920):
        str_188551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 26), 'str', '%s != %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 920)
        tuple_188552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 920)
        # Adding element type (line 920)
        
        # Call to safe_repr(...): (line 920)
        # Processing the call arguments (line 920)
        # Getting the type of 'first' (line 920)
        first_188554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 50), 'first', False)
        # Getting the type of 'True' (line 920)
        True_188555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 57), 'True', False)
        # Processing the call keyword arguments (line 920)
        kwargs_188556 = {}
        # Getting the type of 'safe_repr' (line 920)
        safe_repr_188553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 40), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 920)
        safe_repr_call_result_188557 = invoke(stypy.reporting.localization.Localization(__file__, 920, 40), safe_repr_188553, *[first_188554, True_188555], **kwargs_188556)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 920, 40), tuple_188552, safe_repr_call_result_188557)
        # Adding element type (line 920)
        
        # Call to safe_repr(...): (line 921)
        # Processing the call arguments (line 921)
        # Getting the type of 'second' (line 921)
        second_188559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 50), 'second', False)
        # Getting the type of 'True' (line 921)
        True_188560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 58), 'True', False)
        # Processing the call keyword arguments (line 921)
        kwargs_188561 = {}
        # Getting the type of 'safe_repr' (line 921)
        safe_repr_188558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 40), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 921)
        safe_repr_call_result_188562 = invoke(stypy.reporting.localization.Localization(__file__, 921, 40), safe_repr_188558, *[second_188559, True_188560], **kwargs_188561)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 920, 40), tuple_188552, safe_repr_call_result_188562)
        
        # Applying the binary operator '%' (line 920)
        result_mod_188563 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 26), '%', str_188551, tuple_188552)
        
        # Assigning a type to the variable 'standardMsg' (line 920)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 12), 'standardMsg', result_mod_188563)
        
        # Assigning a BinOp to a Name (line 922):
        
        # Assigning a BinOp to a Name (line 922):
        str_188564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 19), 'str', '\n')
        
        # Call to join(...): (line 922)
        # Processing the call arguments (line 922)
        
        # Call to ndiff(...): (line 922)
        # Processing the call arguments (line 922)
        # Getting the type of 'firstlines' (line 922)
        firstlines_188569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 48), 'firstlines', False)
        # Getting the type of 'secondlines' (line 922)
        secondlines_188570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 60), 'secondlines', False)
        # Processing the call keyword arguments (line 922)
        kwargs_188571 = {}
        # Getting the type of 'difflib' (line 922)
        difflib_188567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 34), 'difflib', False)
        # Obtaining the member 'ndiff' of a type (line 922)
        ndiff_188568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 34), difflib_188567, 'ndiff')
        # Calling ndiff(args, kwargs) (line 922)
        ndiff_call_result_188572 = invoke(stypy.reporting.localization.Localization(__file__, 922, 34), ndiff_188568, *[firstlines_188569, secondlines_188570], **kwargs_188571)
        
        # Processing the call keyword arguments (line 922)
        kwargs_188573 = {}
        str_188565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 26), 'str', '')
        # Obtaining the member 'join' of a type (line 922)
        join_188566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 26), str_188565, 'join')
        # Calling join(args, kwargs) (line 922)
        join_call_result_188574 = invoke(stypy.reporting.localization.Localization(__file__, 922, 26), join_188566, *[ndiff_call_result_188572], **kwargs_188573)
        
        # Applying the binary operator '+' (line 922)
        result_add_188575 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 19), '+', str_188564, join_call_result_188574)
        
        # Assigning a type to the variable 'diff' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 12), 'diff', result_add_188575)
        
        # Assigning a Call to a Name (line 923):
        
        # Assigning a Call to a Name (line 923):
        
        # Call to _truncateMessage(...): (line 923)
        # Processing the call arguments (line 923)
        # Getting the type of 'standardMsg' (line 923)
        standardMsg_188578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 48), 'standardMsg', False)
        # Getting the type of 'diff' (line 923)
        diff_188579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 61), 'diff', False)
        # Processing the call keyword arguments (line 923)
        kwargs_188580 = {}
        # Getting the type of 'self' (line 923)
        self_188576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 26), 'self', False)
        # Obtaining the member '_truncateMessage' of a type (line 923)
        _truncateMessage_188577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 26), self_188576, '_truncateMessage')
        # Calling _truncateMessage(args, kwargs) (line 923)
        _truncateMessage_call_result_188581 = invoke(stypy.reporting.localization.Localization(__file__, 923, 26), _truncateMessage_188577, *[standardMsg_188578, diff_188579], **kwargs_188580)
        
        # Assigning a type to the variable 'standardMsg' (line 923)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 12), 'standardMsg', _truncateMessage_call_result_188581)
        
        # Call to fail(...): (line 924)
        # Processing the call arguments (line 924)
        
        # Call to _formatMessage(...): (line 924)
        # Processing the call arguments (line 924)
        # Getting the type of 'msg' (line 924)
        msg_188586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 924)
        standardMsg_188587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 924)
        kwargs_188588 = {}
        # Getting the type of 'self' (line 924)
        self_188584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 924)
        _formatMessage_188585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 22), self_188584, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 924)
        _formatMessage_call_result_188589 = invoke(stypy.reporting.localization.Localization(__file__, 924, 22), _formatMessage_188585, *[msg_188586, standardMsg_188587], **kwargs_188588)
        
        # Processing the call keyword arguments (line 924)
        kwargs_188590 = {}
        # Getting the type of 'self' (line 924)
        self_188582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 924)
        fail_188583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 12), self_188582, 'fail')
        # Calling fail(args, kwargs) (line 924)
        fail_call_result_188591 = invoke(stypy.reporting.localization.Localization(__file__, 924, 12), fail_188583, *[_formatMessage_call_result_188589], **kwargs_188590)
        
        # SSA join for if statement (line 910)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertMultiLineEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertMultiLineEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 903)
        stypy_return_type_188592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertMultiLineEqual'
        return stypy_return_type_188592


    @norecursion
    def assertLess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 926)
        None_188593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 35), 'None')
        defaults = [None_188593]
        # Create a new context for function 'assertLess'
        module_type_store = module_type_store.open_function_context('assertLess', 926, 4, False)
        # Assigning a type to the variable 'self' (line 927)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertLess.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertLess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertLess.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertLess.__dict__.__setitem__('stypy_function_name', 'TestCase.assertLess')
        TestCase.assertLess.__dict__.__setitem__('stypy_param_names_list', ['a', 'b', 'msg'])
        TestCase.assertLess.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertLess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertLess.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertLess.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertLess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertLess.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertLess', ['a', 'b', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertLess', localization, ['a', 'b', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertLess(...)' code ##################

        str_188594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 8), 'str', 'Just like self.assertTrue(a < b), but with a nicer default message.')
        
        
        
        # Getting the type of 'a' (line 928)
        a_188595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 15), 'a')
        # Getting the type of 'b' (line 928)
        b_188596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 19), 'b')
        # Applying the binary operator '<' (line 928)
        result_lt_188597 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 15), '<', a_188595, b_188596)
        
        # Applying the 'not' unary operator (line 928)
        result_not__188598 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 11), 'not', result_lt_188597)
        
        # Testing the type of an if condition (line 928)
        if_condition_188599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 928, 8), result_not__188598)
        # Assigning a type to the variable 'if_condition_188599' (line 928)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 8), 'if_condition_188599', if_condition_188599)
        # SSA begins for if statement (line 928)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 929):
        
        # Assigning a BinOp to a Name (line 929):
        str_188600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 26), 'str', '%s not less than %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 929)
        tuple_188601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 929)
        # Adding element type (line 929)
        
        # Call to safe_repr(...): (line 929)
        # Processing the call arguments (line 929)
        # Getting the type of 'a' (line 929)
        a_188603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 61), 'a', False)
        # Processing the call keyword arguments (line 929)
        kwargs_188604 = {}
        # Getting the type of 'safe_repr' (line 929)
        safe_repr_188602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 51), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 929)
        safe_repr_call_result_188605 = invoke(stypy.reporting.localization.Localization(__file__, 929, 51), safe_repr_188602, *[a_188603], **kwargs_188604)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 929, 51), tuple_188601, safe_repr_call_result_188605)
        # Adding element type (line 929)
        
        # Call to safe_repr(...): (line 929)
        # Processing the call arguments (line 929)
        # Getting the type of 'b' (line 929)
        b_188607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 75), 'b', False)
        # Processing the call keyword arguments (line 929)
        kwargs_188608 = {}
        # Getting the type of 'safe_repr' (line 929)
        safe_repr_188606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 65), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 929)
        safe_repr_call_result_188609 = invoke(stypy.reporting.localization.Localization(__file__, 929, 65), safe_repr_188606, *[b_188607], **kwargs_188608)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 929, 51), tuple_188601, safe_repr_call_result_188609)
        
        # Applying the binary operator '%' (line 929)
        result_mod_188610 = python_operator(stypy.reporting.localization.Localization(__file__, 929, 26), '%', str_188600, tuple_188601)
        
        # Assigning a type to the variable 'standardMsg' (line 929)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 929, 12), 'standardMsg', result_mod_188610)
        
        # Call to fail(...): (line 930)
        # Processing the call arguments (line 930)
        
        # Call to _formatMessage(...): (line 930)
        # Processing the call arguments (line 930)
        # Getting the type of 'msg' (line 930)
        msg_188615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 930)
        standardMsg_188616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 930)
        kwargs_188617 = {}
        # Getting the type of 'self' (line 930)
        self_188613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 930)
        _formatMessage_188614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 22), self_188613, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 930)
        _formatMessage_call_result_188618 = invoke(stypy.reporting.localization.Localization(__file__, 930, 22), _formatMessage_188614, *[msg_188615, standardMsg_188616], **kwargs_188617)
        
        # Processing the call keyword arguments (line 930)
        kwargs_188619 = {}
        # Getting the type of 'self' (line 930)
        self_188611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 930)
        fail_188612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 12), self_188611, 'fail')
        # Calling fail(args, kwargs) (line 930)
        fail_call_result_188620 = invoke(stypy.reporting.localization.Localization(__file__, 930, 12), fail_188612, *[_formatMessage_call_result_188618], **kwargs_188619)
        
        # SSA join for if statement (line 928)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertLess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertLess' in the type store
        # Getting the type of 'stypy_return_type' (line 926)
        stypy_return_type_188621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188621)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertLess'
        return stypy_return_type_188621


    @norecursion
    def assertLessEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 932)
        None_188622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 40), 'None')
        defaults = [None_188622]
        # Create a new context for function 'assertLessEqual'
        module_type_store = module_type_store.open_function_context('assertLessEqual', 932, 4, False)
        # Assigning a type to the variable 'self' (line 933)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertLessEqual')
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_param_names_list', ['a', 'b', 'msg'])
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertLessEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertLessEqual', ['a', 'b', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertLessEqual', localization, ['a', 'b', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertLessEqual(...)' code ##################

        str_188623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 8), 'str', 'Just like self.assertTrue(a <= b), but with a nicer default message.')
        
        
        
        # Getting the type of 'a' (line 934)
        a_188624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 15), 'a')
        # Getting the type of 'b' (line 934)
        b_188625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 20), 'b')
        # Applying the binary operator '<=' (line 934)
        result_le_188626 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 15), '<=', a_188624, b_188625)
        
        # Applying the 'not' unary operator (line 934)
        result_not__188627 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 11), 'not', result_le_188626)
        
        # Testing the type of an if condition (line 934)
        if_condition_188628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 934, 8), result_not__188627)
        # Assigning a type to the variable 'if_condition_188628' (line 934)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 934, 8), 'if_condition_188628', if_condition_188628)
        # SSA begins for if statement (line 934)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 935):
        
        # Assigning a BinOp to a Name (line 935):
        str_188629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 26), 'str', '%s not less than or equal to %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 935)
        tuple_188630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 935)
        # Adding element type (line 935)
        
        # Call to safe_repr(...): (line 935)
        # Processing the call arguments (line 935)
        # Getting the type of 'a' (line 935)
        a_188632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 73), 'a', False)
        # Processing the call keyword arguments (line 935)
        kwargs_188633 = {}
        # Getting the type of 'safe_repr' (line 935)
        safe_repr_188631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 63), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 935)
        safe_repr_call_result_188634 = invoke(stypy.reporting.localization.Localization(__file__, 935, 63), safe_repr_188631, *[a_188632], **kwargs_188633)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 935, 63), tuple_188630, safe_repr_call_result_188634)
        # Adding element type (line 935)
        
        # Call to safe_repr(...): (line 935)
        # Processing the call arguments (line 935)
        # Getting the type of 'b' (line 935)
        b_188636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 87), 'b', False)
        # Processing the call keyword arguments (line 935)
        kwargs_188637 = {}
        # Getting the type of 'safe_repr' (line 935)
        safe_repr_188635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 77), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 935)
        safe_repr_call_result_188638 = invoke(stypy.reporting.localization.Localization(__file__, 935, 77), safe_repr_188635, *[b_188636], **kwargs_188637)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 935, 63), tuple_188630, safe_repr_call_result_188638)
        
        # Applying the binary operator '%' (line 935)
        result_mod_188639 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 26), '%', str_188629, tuple_188630)
        
        # Assigning a type to the variable 'standardMsg' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 12), 'standardMsg', result_mod_188639)
        
        # Call to fail(...): (line 936)
        # Processing the call arguments (line 936)
        
        # Call to _formatMessage(...): (line 936)
        # Processing the call arguments (line 936)
        # Getting the type of 'msg' (line 936)
        msg_188644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 936)
        standardMsg_188645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 936)
        kwargs_188646 = {}
        # Getting the type of 'self' (line 936)
        self_188642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 936)
        _formatMessage_188643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 22), self_188642, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 936)
        _formatMessage_call_result_188647 = invoke(stypy.reporting.localization.Localization(__file__, 936, 22), _formatMessage_188643, *[msg_188644, standardMsg_188645], **kwargs_188646)
        
        # Processing the call keyword arguments (line 936)
        kwargs_188648 = {}
        # Getting the type of 'self' (line 936)
        self_188640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 936)
        fail_188641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 12), self_188640, 'fail')
        # Calling fail(args, kwargs) (line 936)
        fail_call_result_188649 = invoke(stypy.reporting.localization.Localization(__file__, 936, 12), fail_188641, *[_formatMessage_call_result_188647], **kwargs_188648)
        
        # SSA join for if statement (line 934)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertLessEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertLessEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 932)
        stypy_return_type_188650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188650)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertLessEqual'
        return stypy_return_type_188650


    @norecursion
    def assertGreater(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 938)
        None_188651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 38), 'None')
        defaults = [None_188651]
        # Create a new context for function 'assertGreater'
        module_type_store = module_type_store.open_function_context('assertGreater', 938, 4, False)
        # Assigning a type to the variable 'self' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertGreater.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertGreater.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertGreater.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertGreater.__dict__.__setitem__('stypy_function_name', 'TestCase.assertGreater')
        TestCase.assertGreater.__dict__.__setitem__('stypy_param_names_list', ['a', 'b', 'msg'])
        TestCase.assertGreater.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertGreater.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertGreater.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertGreater.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertGreater.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertGreater.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertGreater', ['a', 'b', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertGreater', localization, ['a', 'b', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertGreater(...)' code ##################

        str_188652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 8), 'str', 'Just like self.assertTrue(a > b), but with a nicer default message.')
        
        
        
        # Getting the type of 'a' (line 940)
        a_188653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 15), 'a')
        # Getting the type of 'b' (line 940)
        b_188654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 19), 'b')
        # Applying the binary operator '>' (line 940)
        result_gt_188655 = python_operator(stypy.reporting.localization.Localization(__file__, 940, 15), '>', a_188653, b_188654)
        
        # Applying the 'not' unary operator (line 940)
        result_not__188656 = python_operator(stypy.reporting.localization.Localization(__file__, 940, 11), 'not', result_gt_188655)
        
        # Testing the type of an if condition (line 940)
        if_condition_188657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 940, 8), result_not__188656)
        # Assigning a type to the variable 'if_condition_188657' (line 940)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 8), 'if_condition_188657', if_condition_188657)
        # SSA begins for if statement (line 940)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 941):
        
        # Assigning a BinOp to a Name (line 941):
        str_188658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 26), 'str', '%s not greater than %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 941)
        tuple_188659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 941)
        # Adding element type (line 941)
        
        # Call to safe_repr(...): (line 941)
        # Processing the call arguments (line 941)
        # Getting the type of 'a' (line 941)
        a_188661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 64), 'a', False)
        # Processing the call keyword arguments (line 941)
        kwargs_188662 = {}
        # Getting the type of 'safe_repr' (line 941)
        safe_repr_188660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 54), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 941)
        safe_repr_call_result_188663 = invoke(stypy.reporting.localization.Localization(__file__, 941, 54), safe_repr_188660, *[a_188661], **kwargs_188662)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 941, 54), tuple_188659, safe_repr_call_result_188663)
        # Adding element type (line 941)
        
        # Call to safe_repr(...): (line 941)
        # Processing the call arguments (line 941)
        # Getting the type of 'b' (line 941)
        b_188665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 78), 'b', False)
        # Processing the call keyword arguments (line 941)
        kwargs_188666 = {}
        # Getting the type of 'safe_repr' (line 941)
        safe_repr_188664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 68), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 941)
        safe_repr_call_result_188667 = invoke(stypy.reporting.localization.Localization(__file__, 941, 68), safe_repr_188664, *[b_188665], **kwargs_188666)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 941, 54), tuple_188659, safe_repr_call_result_188667)
        
        # Applying the binary operator '%' (line 941)
        result_mod_188668 = python_operator(stypy.reporting.localization.Localization(__file__, 941, 26), '%', str_188658, tuple_188659)
        
        # Assigning a type to the variable 'standardMsg' (line 941)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 12), 'standardMsg', result_mod_188668)
        
        # Call to fail(...): (line 942)
        # Processing the call arguments (line 942)
        
        # Call to _formatMessage(...): (line 942)
        # Processing the call arguments (line 942)
        # Getting the type of 'msg' (line 942)
        msg_188673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 942)
        standardMsg_188674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 942)
        kwargs_188675 = {}
        # Getting the type of 'self' (line 942)
        self_188671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 942)
        _formatMessage_188672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 22), self_188671, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 942)
        _formatMessage_call_result_188676 = invoke(stypy.reporting.localization.Localization(__file__, 942, 22), _formatMessage_188672, *[msg_188673, standardMsg_188674], **kwargs_188675)
        
        # Processing the call keyword arguments (line 942)
        kwargs_188677 = {}
        # Getting the type of 'self' (line 942)
        self_188669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 942)
        fail_188670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 12), self_188669, 'fail')
        # Calling fail(args, kwargs) (line 942)
        fail_call_result_188678 = invoke(stypy.reporting.localization.Localization(__file__, 942, 12), fail_188670, *[_formatMessage_call_result_188676], **kwargs_188677)
        
        # SSA join for if statement (line 940)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertGreater(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertGreater' in the type store
        # Getting the type of 'stypy_return_type' (line 938)
        stypy_return_type_188679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertGreater'
        return stypy_return_type_188679


    @norecursion
    def assertGreaterEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 944)
        None_188680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 43), 'None')
        defaults = [None_188680]
        # Create a new context for function 'assertGreaterEqual'
        module_type_store = module_type_store.open_function_context('assertGreaterEqual', 944, 4, False)
        # Assigning a type to the variable 'self' (line 945)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertGreaterEqual')
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_param_names_list', ['a', 'b', 'msg'])
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertGreaterEqual.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertGreaterEqual', ['a', 'b', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertGreaterEqual', localization, ['a', 'b', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertGreaterEqual(...)' code ##################

        str_188681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 8), 'str', 'Just like self.assertTrue(a >= b), but with a nicer default message.')
        
        
        
        # Getting the type of 'a' (line 946)
        a_188682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 15), 'a')
        # Getting the type of 'b' (line 946)
        b_188683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 20), 'b')
        # Applying the binary operator '>=' (line 946)
        result_ge_188684 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 15), '>=', a_188682, b_188683)
        
        # Applying the 'not' unary operator (line 946)
        result_not__188685 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 11), 'not', result_ge_188684)
        
        # Testing the type of an if condition (line 946)
        if_condition_188686 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 946, 8), result_not__188685)
        # Assigning a type to the variable 'if_condition_188686' (line 946)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 8), 'if_condition_188686', if_condition_188686)
        # SSA begins for if statement (line 946)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 947):
        
        # Assigning a BinOp to a Name (line 947):
        str_188687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 947, 26), 'str', '%s not greater than or equal to %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 947)
        tuple_188688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 947, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 947)
        # Adding element type (line 947)
        
        # Call to safe_repr(...): (line 947)
        # Processing the call arguments (line 947)
        # Getting the type of 'a' (line 947)
        a_188690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 76), 'a', False)
        # Processing the call keyword arguments (line 947)
        kwargs_188691 = {}
        # Getting the type of 'safe_repr' (line 947)
        safe_repr_188689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 66), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 947)
        safe_repr_call_result_188692 = invoke(stypy.reporting.localization.Localization(__file__, 947, 66), safe_repr_188689, *[a_188690], **kwargs_188691)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 947, 66), tuple_188688, safe_repr_call_result_188692)
        # Adding element type (line 947)
        
        # Call to safe_repr(...): (line 947)
        # Processing the call arguments (line 947)
        # Getting the type of 'b' (line 947)
        b_188694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 90), 'b', False)
        # Processing the call keyword arguments (line 947)
        kwargs_188695 = {}
        # Getting the type of 'safe_repr' (line 947)
        safe_repr_188693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 80), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 947)
        safe_repr_call_result_188696 = invoke(stypy.reporting.localization.Localization(__file__, 947, 80), safe_repr_188693, *[b_188694], **kwargs_188695)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 947, 66), tuple_188688, safe_repr_call_result_188696)
        
        # Applying the binary operator '%' (line 947)
        result_mod_188697 = python_operator(stypy.reporting.localization.Localization(__file__, 947, 26), '%', str_188687, tuple_188688)
        
        # Assigning a type to the variable 'standardMsg' (line 947)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 12), 'standardMsg', result_mod_188697)
        
        # Call to fail(...): (line 948)
        # Processing the call arguments (line 948)
        
        # Call to _formatMessage(...): (line 948)
        # Processing the call arguments (line 948)
        # Getting the type of 'msg' (line 948)
        msg_188702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 948)
        standardMsg_188703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 948)
        kwargs_188704 = {}
        # Getting the type of 'self' (line 948)
        self_188700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 948)
        _formatMessage_188701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 22), self_188700, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 948)
        _formatMessage_call_result_188705 = invoke(stypy.reporting.localization.Localization(__file__, 948, 22), _formatMessage_188701, *[msg_188702, standardMsg_188703], **kwargs_188704)
        
        # Processing the call keyword arguments (line 948)
        kwargs_188706 = {}
        # Getting the type of 'self' (line 948)
        self_188698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 948)
        fail_188699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 12), self_188698, 'fail')
        # Calling fail(args, kwargs) (line 948)
        fail_call_result_188707 = invoke(stypy.reporting.localization.Localization(__file__, 948, 12), fail_188699, *[_formatMessage_call_result_188705], **kwargs_188706)
        
        # SSA join for if statement (line 946)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertGreaterEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertGreaterEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 944)
        stypy_return_type_188708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188708)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertGreaterEqual'
        return stypy_return_type_188708


    @norecursion
    def assertIsNone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 950)
        None_188709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 36), 'None')
        defaults = [None_188709]
        # Create a new context for function 'assertIsNone'
        module_type_store = module_type_store.open_function_context('assertIsNone', 950, 4, False)
        # Assigning a type to the variable 'self' (line 951)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertIsNone.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertIsNone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertIsNone.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertIsNone.__dict__.__setitem__('stypy_function_name', 'TestCase.assertIsNone')
        TestCase.assertIsNone.__dict__.__setitem__('stypy_param_names_list', ['obj', 'msg'])
        TestCase.assertIsNone.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertIsNone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertIsNone.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertIsNone.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertIsNone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertIsNone.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertIsNone', ['obj', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertIsNone', localization, ['obj', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertIsNone(...)' code ##################

        str_188710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 8), 'str', 'Same as self.assertTrue(obj is None), with a nicer default message.')
        
        # Type idiom detected: calculating its left and rigth part (line 952)
        # Getting the type of 'obj' (line 952)
        obj_188711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'obj')
        # Getting the type of 'None' (line 952)
        None_188712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 22), 'None')
        
        (may_be_188713, more_types_in_union_188714) = may_not_be_none(obj_188711, None_188712)

        if may_be_188713:

            if more_types_in_union_188714:
                # Runtime conditional SSA (line 952)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 953):
            
            # Assigning a BinOp to a Name (line 953):
            str_188715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 26), 'str', '%s is not None')
            
            # Obtaining an instance of the builtin type 'tuple' (line 953)
            tuple_188716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 953)
            # Adding element type (line 953)
            
            # Call to safe_repr(...): (line 953)
            # Processing the call arguments (line 953)
            # Getting the type of 'obj' (line 953)
            obj_188718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 56), 'obj', False)
            # Processing the call keyword arguments (line 953)
            kwargs_188719 = {}
            # Getting the type of 'safe_repr' (line 953)
            safe_repr_188717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 46), 'safe_repr', False)
            # Calling safe_repr(args, kwargs) (line 953)
            safe_repr_call_result_188720 = invoke(stypy.reporting.localization.Localization(__file__, 953, 46), safe_repr_188717, *[obj_188718], **kwargs_188719)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 953, 46), tuple_188716, safe_repr_call_result_188720)
            
            # Applying the binary operator '%' (line 953)
            result_mod_188721 = python_operator(stypy.reporting.localization.Localization(__file__, 953, 26), '%', str_188715, tuple_188716)
            
            # Assigning a type to the variable 'standardMsg' (line 953)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 12), 'standardMsg', result_mod_188721)
            
            # Call to fail(...): (line 954)
            # Processing the call arguments (line 954)
            
            # Call to _formatMessage(...): (line 954)
            # Processing the call arguments (line 954)
            # Getting the type of 'msg' (line 954)
            msg_188726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 42), 'msg', False)
            # Getting the type of 'standardMsg' (line 954)
            standardMsg_188727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 47), 'standardMsg', False)
            # Processing the call keyword arguments (line 954)
            kwargs_188728 = {}
            # Getting the type of 'self' (line 954)
            self_188724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 22), 'self', False)
            # Obtaining the member '_formatMessage' of a type (line 954)
            _formatMessage_188725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 22), self_188724, '_formatMessage')
            # Calling _formatMessage(args, kwargs) (line 954)
            _formatMessage_call_result_188729 = invoke(stypy.reporting.localization.Localization(__file__, 954, 22), _formatMessage_188725, *[msg_188726, standardMsg_188727], **kwargs_188728)
            
            # Processing the call keyword arguments (line 954)
            kwargs_188730 = {}
            # Getting the type of 'self' (line 954)
            self_188722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 12), 'self', False)
            # Obtaining the member 'fail' of a type (line 954)
            fail_188723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 12), self_188722, 'fail')
            # Calling fail(args, kwargs) (line 954)
            fail_call_result_188731 = invoke(stypy.reporting.localization.Localization(__file__, 954, 12), fail_188723, *[_formatMessage_call_result_188729], **kwargs_188730)
            

            if more_types_in_union_188714:
                # SSA join for if statement (line 952)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'assertIsNone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertIsNone' in the type store
        # Getting the type of 'stypy_return_type' (line 950)
        stypy_return_type_188732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertIsNone'
        return stypy_return_type_188732


    @norecursion
    def assertIsNotNone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 956)
        None_188733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 39), 'None')
        defaults = [None_188733]
        # Create a new context for function 'assertIsNotNone'
        module_type_store = module_type_store.open_function_context('assertIsNotNone', 956, 4, False)
        # Assigning a type to the variable 'self' (line 957)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 957, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_function_name', 'TestCase.assertIsNotNone')
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_param_names_list', ['obj', 'msg'])
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertIsNotNone.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertIsNotNone', ['obj', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertIsNotNone', localization, ['obj', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertIsNotNone(...)' code ##################

        str_188734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, 8), 'str', 'Included for symmetry with assertIsNone.')
        
        # Type idiom detected: calculating its left and rigth part (line 958)
        # Getting the type of 'obj' (line 958)
        obj_188735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 11), 'obj')
        # Getting the type of 'None' (line 958)
        None_188736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 18), 'None')
        
        (may_be_188737, more_types_in_union_188738) = may_be_none(obj_188735, None_188736)

        if may_be_188737:

            if more_types_in_union_188738:
                # Runtime conditional SSA (line 958)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 959):
            
            # Assigning a Str to a Name (line 959):
            str_188739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 26), 'str', 'unexpectedly None')
            # Assigning a type to the variable 'standardMsg' (line 959)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 12), 'standardMsg', str_188739)
            
            # Call to fail(...): (line 960)
            # Processing the call arguments (line 960)
            
            # Call to _formatMessage(...): (line 960)
            # Processing the call arguments (line 960)
            # Getting the type of 'msg' (line 960)
            msg_188744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 42), 'msg', False)
            # Getting the type of 'standardMsg' (line 960)
            standardMsg_188745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 47), 'standardMsg', False)
            # Processing the call keyword arguments (line 960)
            kwargs_188746 = {}
            # Getting the type of 'self' (line 960)
            self_188742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 22), 'self', False)
            # Obtaining the member '_formatMessage' of a type (line 960)
            _formatMessage_188743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 22), self_188742, '_formatMessage')
            # Calling _formatMessage(args, kwargs) (line 960)
            _formatMessage_call_result_188747 = invoke(stypy.reporting.localization.Localization(__file__, 960, 22), _formatMessage_188743, *[msg_188744, standardMsg_188745], **kwargs_188746)
            
            # Processing the call keyword arguments (line 960)
            kwargs_188748 = {}
            # Getting the type of 'self' (line 960)
            self_188740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 12), 'self', False)
            # Obtaining the member 'fail' of a type (line 960)
            fail_188741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 12), self_188740, 'fail')
            # Calling fail(args, kwargs) (line 960)
            fail_call_result_188749 = invoke(stypy.reporting.localization.Localization(__file__, 960, 12), fail_188741, *[_formatMessage_call_result_188747], **kwargs_188748)
            

            if more_types_in_union_188738:
                # SSA join for if statement (line 958)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'assertIsNotNone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertIsNotNone' in the type store
        # Getting the type of 'stypy_return_type' (line 956)
        stypy_return_type_188750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188750)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertIsNotNone'
        return stypy_return_type_188750


    @norecursion
    def assertIsInstance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 962)
        None_188751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 45), 'None')
        defaults = [None_188751]
        # Create a new context for function 'assertIsInstance'
        module_type_store = module_type_store.open_function_context('assertIsInstance', 962, 4, False)
        # Assigning a type to the variable 'self' (line 963)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_function_name', 'TestCase.assertIsInstance')
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_param_names_list', ['obj', 'cls', 'msg'])
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertIsInstance.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertIsInstance', ['obj', 'cls', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertIsInstance', localization, ['obj', 'cls', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertIsInstance(...)' code ##################

        str_188752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, (-1)), 'str', 'Same as self.assertTrue(isinstance(obj, cls)), with a nicer\n        default message.')
        
        
        
        # Call to isinstance(...): (line 965)
        # Processing the call arguments (line 965)
        # Getting the type of 'obj' (line 965)
        obj_188754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 26), 'obj', False)
        # Getting the type of 'cls' (line 965)
        cls_188755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 31), 'cls', False)
        # Processing the call keyword arguments (line 965)
        kwargs_188756 = {}
        # Getting the type of 'isinstance' (line 965)
        isinstance_188753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 965)
        isinstance_call_result_188757 = invoke(stypy.reporting.localization.Localization(__file__, 965, 15), isinstance_188753, *[obj_188754, cls_188755], **kwargs_188756)
        
        # Applying the 'not' unary operator (line 965)
        result_not__188758 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 11), 'not', isinstance_call_result_188757)
        
        # Testing the type of an if condition (line 965)
        if_condition_188759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 965, 8), result_not__188758)
        # Assigning a type to the variable 'if_condition_188759' (line 965)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 8), 'if_condition_188759', if_condition_188759)
        # SSA begins for if statement (line 965)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 966):
        
        # Assigning a BinOp to a Name (line 966):
        str_188760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 26), 'str', '%s is not an instance of %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 966)
        tuple_188761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 966)
        # Adding element type (line 966)
        
        # Call to safe_repr(...): (line 966)
        # Processing the call arguments (line 966)
        # Getting the type of 'obj' (line 966)
        obj_188763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 69), 'obj', False)
        # Processing the call keyword arguments (line 966)
        kwargs_188764 = {}
        # Getting the type of 'safe_repr' (line 966)
        safe_repr_188762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 59), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 966)
        safe_repr_call_result_188765 = invoke(stypy.reporting.localization.Localization(__file__, 966, 59), safe_repr_188762, *[obj_188763], **kwargs_188764)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 966, 59), tuple_188761, safe_repr_call_result_188765)
        # Adding element type (line 966)
        # Getting the type of 'cls' (line 966)
        cls_188766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 75), 'cls')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 966, 59), tuple_188761, cls_188766)
        
        # Applying the binary operator '%' (line 966)
        result_mod_188767 = python_operator(stypy.reporting.localization.Localization(__file__, 966, 26), '%', str_188760, tuple_188761)
        
        # Assigning a type to the variable 'standardMsg' (line 966)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 12), 'standardMsg', result_mod_188767)
        
        # Call to fail(...): (line 967)
        # Processing the call arguments (line 967)
        
        # Call to _formatMessage(...): (line 967)
        # Processing the call arguments (line 967)
        # Getting the type of 'msg' (line 967)
        msg_188772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 967)
        standardMsg_188773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 967)
        kwargs_188774 = {}
        # Getting the type of 'self' (line 967)
        self_188770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 967)
        _formatMessage_188771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 967, 22), self_188770, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 967)
        _formatMessage_call_result_188775 = invoke(stypy.reporting.localization.Localization(__file__, 967, 22), _formatMessage_188771, *[msg_188772, standardMsg_188773], **kwargs_188774)
        
        # Processing the call keyword arguments (line 967)
        kwargs_188776 = {}
        # Getting the type of 'self' (line 967)
        self_188768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 967)
        fail_188769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 967, 12), self_188768, 'fail')
        # Calling fail(args, kwargs) (line 967)
        fail_call_result_188777 = invoke(stypy.reporting.localization.Localization(__file__, 967, 12), fail_188769, *[_formatMessage_call_result_188775], **kwargs_188776)
        
        # SSA join for if statement (line 965)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertIsInstance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertIsInstance' in the type store
        # Getting the type of 'stypy_return_type' (line 962)
        stypy_return_type_188778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188778)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertIsInstance'
        return stypy_return_type_188778


    @norecursion
    def assertNotIsInstance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 969)
        None_188779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 48), 'None')
        defaults = [None_188779]
        # Create a new context for function 'assertNotIsInstance'
        module_type_store = module_type_store.open_function_context('assertNotIsInstance', 969, 4, False)
        # Assigning a type to the variable 'self' (line 970)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_function_name', 'TestCase.assertNotIsInstance')
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_param_names_list', ['obj', 'cls', 'msg'])
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertNotIsInstance.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertNotIsInstance', ['obj', 'cls', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertNotIsInstance', localization, ['obj', 'cls', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertNotIsInstance(...)' code ##################

        str_188780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 8), 'str', 'Included for symmetry with assertIsInstance.')
        
        
        # Call to isinstance(...): (line 971)
        # Processing the call arguments (line 971)
        # Getting the type of 'obj' (line 971)
        obj_188782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 22), 'obj', False)
        # Getting the type of 'cls' (line 971)
        cls_188783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 27), 'cls', False)
        # Processing the call keyword arguments (line 971)
        kwargs_188784 = {}
        # Getting the type of 'isinstance' (line 971)
        isinstance_188781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 971)
        isinstance_call_result_188785 = invoke(stypy.reporting.localization.Localization(__file__, 971, 11), isinstance_188781, *[obj_188782, cls_188783], **kwargs_188784)
        
        # Testing the type of an if condition (line 971)
        if_condition_188786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 971, 8), isinstance_call_result_188785)
        # Assigning a type to the variable 'if_condition_188786' (line 971)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 8), 'if_condition_188786', if_condition_188786)
        # SSA begins for if statement (line 971)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 972):
        
        # Assigning a BinOp to a Name (line 972):
        str_188787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 26), 'str', '%s is an instance of %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 972)
        tuple_188788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 972)
        # Adding element type (line 972)
        
        # Call to safe_repr(...): (line 972)
        # Processing the call arguments (line 972)
        # Getting the type of 'obj' (line 972)
        obj_188790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 65), 'obj', False)
        # Processing the call keyword arguments (line 972)
        kwargs_188791 = {}
        # Getting the type of 'safe_repr' (line 972)
        safe_repr_188789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 55), 'safe_repr', False)
        # Calling safe_repr(args, kwargs) (line 972)
        safe_repr_call_result_188792 = invoke(stypy.reporting.localization.Localization(__file__, 972, 55), safe_repr_188789, *[obj_188790], **kwargs_188791)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 972, 55), tuple_188788, safe_repr_call_result_188792)
        # Adding element type (line 972)
        # Getting the type of 'cls' (line 972)
        cls_188793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 71), 'cls')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 972, 55), tuple_188788, cls_188793)
        
        # Applying the binary operator '%' (line 972)
        result_mod_188794 = python_operator(stypy.reporting.localization.Localization(__file__, 972, 26), '%', str_188787, tuple_188788)
        
        # Assigning a type to the variable 'standardMsg' (line 972)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 12), 'standardMsg', result_mod_188794)
        
        # Call to fail(...): (line 973)
        # Processing the call arguments (line 973)
        
        # Call to _formatMessage(...): (line 973)
        # Processing the call arguments (line 973)
        # Getting the type of 'msg' (line 973)
        msg_188799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 42), 'msg', False)
        # Getting the type of 'standardMsg' (line 973)
        standardMsg_188800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 47), 'standardMsg', False)
        # Processing the call keyword arguments (line 973)
        kwargs_188801 = {}
        # Getting the type of 'self' (line 973)
        self_188797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 22), 'self', False)
        # Obtaining the member '_formatMessage' of a type (line 973)
        _formatMessage_188798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 22), self_188797, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 973)
        _formatMessage_call_result_188802 = invoke(stypy.reporting.localization.Localization(__file__, 973, 22), _formatMessage_188798, *[msg_188799, standardMsg_188800], **kwargs_188801)
        
        # Processing the call keyword arguments (line 973)
        kwargs_188803 = {}
        # Getting the type of 'self' (line 973)
        self_188795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 973)
        fail_188796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 12), self_188795, 'fail')
        # Calling fail(args, kwargs) (line 973)
        fail_call_result_188804 = invoke(stypy.reporting.localization.Localization(__file__, 973, 12), fail_188796, *[_formatMessage_call_result_188802], **kwargs_188803)
        
        # SSA join for if statement (line 971)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertNotIsInstance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertNotIsInstance' in the type store
        # Getting the type of 'stypy_return_type' (line 969)
        stypy_return_type_188805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertNotIsInstance'
        return stypy_return_type_188805


    @norecursion
    def assertRaisesRegexp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 976)
        None_188806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 40), 'None')
        defaults = [None_188806]
        # Create a new context for function 'assertRaisesRegexp'
        module_type_store = module_type_store.open_function_context('assertRaisesRegexp', 975, 4, False)
        # Assigning a type to the variable 'self' (line 976)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_function_name', 'TestCase.assertRaisesRegexp')
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_param_names_list', ['expected_exception', 'expected_regexp', 'callable_obj'])
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertRaisesRegexp.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertRaisesRegexp', ['expected_exception', 'expected_regexp', 'callable_obj'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertRaisesRegexp', localization, ['expected_exception', 'expected_regexp', 'callable_obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertRaisesRegexp(...)' code ##################

        str_188807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, (-1)), 'str', 'Asserts that the message in a raised exception matches a regexp.\n\n        Args:\n            expected_exception: Exception class expected to be raised.\n            expected_regexp: Regexp (re pattern object or string) expected\n                    to be found in error message.\n            callable_obj: Function to be called.\n            args: Extra args.\n            kwargs: Extra kwargs.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 987)
        # Getting the type of 'expected_regexp' (line 987)
        expected_regexp_188808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 8), 'expected_regexp')
        # Getting the type of 'None' (line 987)
        None_188809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 34), 'None')
        
        (may_be_188810, more_types_in_union_188811) = may_not_be_none(expected_regexp_188808, None_188809)

        if may_be_188810:

            if more_types_in_union_188811:
                # Runtime conditional SSA (line 987)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 988):
            
            # Assigning a Call to a Name (line 988):
            
            # Call to compile(...): (line 988)
            # Processing the call arguments (line 988)
            # Getting the type of 'expected_regexp' (line 988)
            expected_regexp_188814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 41), 'expected_regexp', False)
            # Processing the call keyword arguments (line 988)
            kwargs_188815 = {}
            # Getting the type of 're' (line 988)
            re_188812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 30), 're', False)
            # Obtaining the member 'compile' of a type (line 988)
            compile_188813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 30), re_188812, 'compile')
            # Calling compile(args, kwargs) (line 988)
            compile_call_result_188816 = invoke(stypy.reporting.localization.Localization(__file__, 988, 30), compile_188813, *[expected_regexp_188814], **kwargs_188815)
            
            # Assigning a type to the variable 'expected_regexp' (line 988)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 12), 'expected_regexp', compile_call_result_188816)

            if more_types_in_union_188811:
                # SSA join for if statement (line 987)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 989):
        
        # Assigning a Call to a Name (line 989):
        
        # Call to _AssertRaisesContext(...): (line 989)
        # Processing the call arguments (line 989)
        # Getting the type of 'expected_exception' (line 989)
        expected_exception_188818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 39), 'expected_exception', False)
        # Getting the type of 'self' (line 989)
        self_188819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 59), 'self', False)
        # Getting the type of 'expected_regexp' (line 989)
        expected_regexp_188820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 65), 'expected_regexp', False)
        # Processing the call keyword arguments (line 989)
        kwargs_188821 = {}
        # Getting the type of '_AssertRaisesContext' (line 989)
        _AssertRaisesContext_188817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 18), '_AssertRaisesContext', False)
        # Calling _AssertRaisesContext(args, kwargs) (line 989)
        _AssertRaisesContext_call_result_188822 = invoke(stypy.reporting.localization.Localization(__file__, 989, 18), _AssertRaisesContext_188817, *[expected_exception_188818, self_188819, expected_regexp_188820], **kwargs_188821)
        
        # Assigning a type to the variable 'context' (line 989)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 989, 8), 'context', _AssertRaisesContext_call_result_188822)
        
        # Type idiom detected: calculating its left and rigth part (line 990)
        # Getting the type of 'callable_obj' (line 990)
        callable_obj_188823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 11), 'callable_obj')
        # Getting the type of 'None' (line 990)
        None_188824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 27), 'None')
        
        (may_be_188825, more_types_in_union_188826) = may_be_none(callable_obj_188823, None_188824)

        if may_be_188825:

            if more_types_in_union_188826:
                # Runtime conditional SSA (line 990)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'context' (line 991)
            context_188827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 19), 'context')
            # Assigning a type to the variable 'stypy_return_type' (line 991)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 12), 'stypy_return_type', context_188827)

            if more_types_in_union_188826:
                # SSA join for if statement (line 990)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'context' (line 992)
        context_188828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 13), 'context')
        with_188829 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 992, 13), context_188828, 'with parameter', '__enter__', '__exit__')

        if with_188829:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 992)
            enter___188830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 992, 13), context_188828, '__enter__')
            with_enter_188831 = invoke(stypy.reporting.localization.Localization(__file__, 992, 13), enter___188830)
            
            # Call to callable_obj(...): (line 993)
            # Getting the type of 'args' (line 993)
            args_188833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 26), 'args', False)
            # Processing the call keyword arguments (line 993)
            # Getting the type of 'kwargs' (line 993)
            kwargs_188834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 34), 'kwargs', False)
            kwargs_188835 = {'kwargs_188834': kwargs_188834}
            # Getting the type of 'callable_obj' (line 993)
            callable_obj_188832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 12), 'callable_obj', False)
            # Calling callable_obj(args, kwargs) (line 993)
            callable_obj_call_result_188836 = invoke(stypy.reporting.localization.Localization(__file__, 993, 12), callable_obj_188832, *[args_188833], **kwargs_188835)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 992)
            exit___188837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 992, 13), context_188828, '__exit__')
            with_exit_188838 = invoke(stypy.reporting.localization.Localization(__file__, 992, 13), exit___188837, None, None, None)

        
        # ################# End of 'assertRaisesRegexp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertRaisesRegexp' in the type store
        # Getting the type of 'stypy_return_type' (line 975)
        stypy_return_type_188839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188839)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertRaisesRegexp'
        return stypy_return_type_188839


    @norecursion
    def assertRegexpMatches(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 995)
        None_188840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 61), 'None')
        defaults = [None_188840]
        # Create a new context for function 'assertRegexpMatches'
        module_type_store = module_type_store.open_function_context('assertRegexpMatches', 995, 4, False)
        # Assigning a type to the variable 'self' (line 996)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 996, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_function_name', 'TestCase.assertRegexpMatches')
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_param_names_list', ['text', 'expected_regexp', 'msg'])
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertRegexpMatches.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertRegexpMatches', ['text', 'expected_regexp', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertRegexpMatches', localization, ['text', 'expected_regexp', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertRegexpMatches(...)' code ##################

        str_188841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 8), 'str', 'Fail the test unless the text matches the regular expression.')
        
        # Type idiom detected: calculating its left and rigth part (line 997)
        # Getting the type of 'basestring' (line 997)
        basestring_188842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 39), 'basestring')
        # Getting the type of 'expected_regexp' (line 997)
        expected_regexp_188843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 22), 'expected_regexp')
        
        (may_be_188844, more_types_in_union_188845) = may_be_subtype(basestring_188842, expected_regexp_188843)

        if may_be_188844:

            if more_types_in_union_188845:
                # Runtime conditional SSA (line 997)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'expected_regexp' (line 997)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 8), 'expected_regexp', remove_not_subtype_from_union(expected_regexp_188843, basestring))
            
            # Assigning a Call to a Name (line 998):
            
            # Assigning a Call to a Name (line 998):
            
            # Call to compile(...): (line 998)
            # Processing the call arguments (line 998)
            # Getting the type of 'expected_regexp' (line 998)
            expected_regexp_188848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 41), 'expected_regexp', False)
            # Processing the call keyword arguments (line 998)
            kwargs_188849 = {}
            # Getting the type of 're' (line 998)
            re_188846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 30), 're', False)
            # Obtaining the member 'compile' of a type (line 998)
            compile_188847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 998, 30), re_188846, 'compile')
            # Calling compile(args, kwargs) (line 998)
            compile_call_result_188850 = invoke(stypy.reporting.localization.Localization(__file__, 998, 30), compile_188847, *[expected_regexp_188848], **kwargs_188849)
            
            # Assigning a type to the variable 'expected_regexp' (line 998)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 998, 12), 'expected_regexp', compile_call_result_188850)

            if more_types_in_union_188845:
                # SSA join for if statement (line 997)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to search(...): (line 999)
        # Processing the call arguments (line 999)
        # Getting the type of 'text' (line 999)
        text_188853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 38), 'text', False)
        # Processing the call keyword arguments (line 999)
        kwargs_188854 = {}
        # Getting the type of 'expected_regexp' (line 999)
        expected_regexp_188851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 15), 'expected_regexp', False)
        # Obtaining the member 'search' of a type (line 999)
        search_188852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 999, 15), expected_regexp_188851, 'search')
        # Calling search(args, kwargs) (line 999)
        search_call_result_188855 = invoke(stypy.reporting.localization.Localization(__file__, 999, 15), search_188852, *[text_188853], **kwargs_188854)
        
        # Applying the 'not' unary operator (line 999)
        result_not__188856 = python_operator(stypy.reporting.localization.Localization(__file__, 999, 11), 'not', search_call_result_188855)
        
        # Testing the type of an if condition (line 999)
        if_condition_188857 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 999, 8), result_not__188856)
        # Assigning a type to the variable 'if_condition_188857' (line 999)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 8), 'if_condition_188857', if_condition_188857)
        # SSA begins for if statement (line 999)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BoolOp to a Name (line 1000):
        
        # Assigning a BoolOp to a Name (line 1000):
        
        # Evaluating a boolean operation
        # Getting the type of 'msg' (line 1000)
        msg_188858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 18), 'msg')
        str_188859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1000, 25), 'str', "Regexp didn't match")
        # Applying the binary operator 'or' (line 1000)
        result_or_keyword_188860 = python_operator(stypy.reporting.localization.Localization(__file__, 1000, 18), 'or', msg_188858, str_188859)
        
        # Assigning a type to the variable 'msg' (line 1000)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 12), 'msg', result_or_keyword_188860)
        
        # Assigning a BinOp to a Name (line 1001):
        
        # Assigning a BinOp to a Name (line 1001):
        str_188861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 18), 'str', '%s: %r not found in %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1001)
        tuple_188862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1001)
        # Adding element type (line 1001)
        # Getting the type of 'msg' (line 1001)
        msg_188863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 46), 'msg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1001, 46), tuple_188862, msg_188863)
        # Adding element type (line 1001)
        # Getting the type of 'expected_regexp' (line 1001)
        expected_regexp_188864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 51), 'expected_regexp')
        # Obtaining the member 'pattern' of a type (line 1001)
        pattern_188865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 51), expected_regexp_188864, 'pattern')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1001, 46), tuple_188862, pattern_188865)
        # Adding element type (line 1001)
        # Getting the type of 'text' (line 1001)
        text_188866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 76), 'text')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1001, 46), tuple_188862, text_188866)
        
        # Applying the binary operator '%' (line 1001)
        result_mod_188867 = python_operator(stypy.reporting.localization.Localization(__file__, 1001, 18), '%', str_188861, tuple_188862)
        
        # Assigning a type to the variable 'msg' (line 1001)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 12), 'msg', result_mod_188867)
        
        # Call to failureException(...): (line 1002)
        # Processing the call arguments (line 1002)
        # Getting the type of 'msg' (line 1002)
        msg_188870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 40), 'msg', False)
        # Processing the call keyword arguments (line 1002)
        kwargs_188871 = {}
        # Getting the type of 'self' (line 1002)
        self_188868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 18), 'self', False)
        # Obtaining the member 'failureException' of a type (line 1002)
        failureException_188869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 18), self_188868, 'failureException')
        # Calling failureException(args, kwargs) (line 1002)
        failureException_call_result_188872 = invoke(stypy.reporting.localization.Localization(__file__, 1002, 18), failureException_188869, *[msg_188870], **kwargs_188871)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1002, 12), failureException_call_result_188872, 'raise parameter', BaseException)
        # SSA join for if statement (line 999)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertRegexpMatches(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertRegexpMatches' in the type store
        # Getting the type of 'stypy_return_type' (line 995)
        stypy_return_type_188873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188873)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertRegexpMatches'
        return stypy_return_type_188873


    @norecursion
    def assertNotRegexpMatches(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 1004)
        None_188874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 66), 'None')
        defaults = [None_188874]
        # Create a new context for function 'assertNotRegexpMatches'
        module_type_store = module_type_store.open_function_context('assertNotRegexpMatches', 1004, 4, False)
        # Assigning a type to the variable 'self' (line 1005)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_function_name', 'TestCase.assertNotRegexpMatches')
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_param_names_list', ['text', 'unexpected_regexp', 'msg'])
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertNotRegexpMatches.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertNotRegexpMatches', ['text', 'unexpected_regexp', 'msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertNotRegexpMatches', localization, ['text', 'unexpected_regexp', 'msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertNotRegexpMatches(...)' code ##################

        str_188875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 8), 'str', 'Fail the test if the text matches the regular expression.')
        
        # Type idiom detected: calculating its left and rigth part (line 1006)
        # Getting the type of 'basestring' (line 1006)
        basestring_188876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 41), 'basestring')
        # Getting the type of 'unexpected_regexp' (line 1006)
        unexpected_regexp_188877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 22), 'unexpected_regexp')
        
        (may_be_188878, more_types_in_union_188879) = may_be_subtype(basestring_188876, unexpected_regexp_188877)

        if may_be_188878:

            if more_types_in_union_188879:
                # Runtime conditional SSA (line 1006)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'unexpected_regexp' (line 1006)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 8), 'unexpected_regexp', remove_not_subtype_from_union(unexpected_regexp_188877, basestring))
            
            # Assigning a Call to a Name (line 1007):
            
            # Assigning a Call to a Name (line 1007):
            
            # Call to compile(...): (line 1007)
            # Processing the call arguments (line 1007)
            # Getting the type of 'unexpected_regexp' (line 1007)
            unexpected_regexp_188882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 43), 'unexpected_regexp', False)
            # Processing the call keyword arguments (line 1007)
            kwargs_188883 = {}
            # Getting the type of 're' (line 1007)
            re_188880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 32), 're', False)
            # Obtaining the member 'compile' of a type (line 1007)
            compile_188881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 32), re_188880, 'compile')
            # Calling compile(args, kwargs) (line 1007)
            compile_call_result_188884 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 32), compile_188881, *[unexpected_regexp_188882], **kwargs_188883)
            
            # Assigning a type to the variable 'unexpected_regexp' (line 1007)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 12), 'unexpected_regexp', compile_call_result_188884)

            if more_types_in_union_188879:
                # SSA join for if statement (line 1006)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 1008):
        
        # Assigning a Call to a Name (line 1008):
        
        # Call to search(...): (line 1008)
        # Processing the call arguments (line 1008)
        # Getting the type of 'text' (line 1008)
        text_188887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 41), 'text', False)
        # Processing the call keyword arguments (line 1008)
        kwargs_188888 = {}
        # Getting the type of 'unexpected_regexp' (line 1008)
        unexpected_regexp_188885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'unexpected_regexp', False)
        # Obtaining the member 'search' of a type (line 1008)
        search_188886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 16), unexpected_regexp_188885, 'search')
        # Calling search(args, kwargs) (line 1008)
        search_call_result_188889 = invoke(stypy.reporting.localization.Localization(__file__, 1008, 16), search_188886, *[text_188887], **kwargs_188888)
        
        # Assigning a type to the variable 'match' (line 1008)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'match', search_call_result_188889)
        
        # Getting the type of 'match' (line 1009)
        match_188890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 11), 'match')
        # Testing the type of an if condition (line 1009)
        if_condition_188891 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1009, 8), match_188890)
        # Assigning a type to the variable 'if_condition_188891' (line 1009)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'if_condition_188891', if_condition_188891)
        # SSA begins for if statement (line 1009)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BoolOp to a Name (line 1010):
        
        # Assigning a BoolOp to a Name (line 1010):
        
        # Evaluating a boolean operation
        # Getting the type of 'msg' (line 1010)
        msg_188892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 18), 'msg')
        str_188893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1010, 25), 'str', 'Regexp matched')
        # Applying the binary operator 'or' (line 1010)
        result_or_keyword_188894 = python_operator(stypy.reporting.localization.Localization(__file__, 1010, 18), 'or', msg_188892, str_188893)
        
        # Assigning a type to the variable 'msg' (line 1010)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 12), 'msg', result_or_keyword_188894)
        
        # Assigning a BinOp to a Name (line 1011):
        
        # Assigning a BinOp to a Name (line 1011):
        str_188895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 18), 'str', '%s: %r matches %r in %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1011)
        tuple_188896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1011)
        # Adding element type (line 1011)
        # Getting the type of 'msg' (line 1011)
        msg_188897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 47), 'msg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 47), tuple_188896, msg_188897)
        # Adding element type (line 1011)
        
        # Obtaining the type of the subscript
        
        # Call to start(...): (line 1012)
        # Processing the call keyword arguments (line 1012)
        kwargs_188900 = {}
        # Getting the type of 'match' (line 1012)
        match_188898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 52), 'match', False)
        # Obtaining the member 'start' of a type (line 1012)
        start_188899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 52), match_188898, 'start')
        # Calling start(args, kwargs) (line 1012)
        start_call_result_188901 = invoke(stypy.reporting.localization.Localization(__file__, 1012, 52), start_188899, *[], **kwargs_188900)
        
        
        # Call to end(...): (line 1012)
        # Processing the call keyword arguments (line 1012)
        kwargs_188904 = {}
        # Getting the type of 'match' (line 1012)
        match_188902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 66), 'match', False)
        # Obtaining the member 'end' of a type (line 1012)
        end_188903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 66), match_188902, 'end')
        # Calling end(args, kwargs) (line 1012)
        end_call_result_188905 = invoke(stypy.reporting.localization.Localization(__file__, 1012, 66), end_188903, *[], **kwargs_188904)
        
        slice_188906 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1012, 47), start_call_result_188901, end_call_result_188905, None)
        # Getting the type of 'text' (line 1012)
        text_188907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 47), 'text')
        # Obtaining the member '__getitem__' of a type (line 1012)
        getitem___188908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 47), text_188907, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1012)
        subscript_call_result_188909 = invoke(stypy.reporting.localization.Localization(__file__, 1012, 47), getitem___188908, slice_188906)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 47), tuple_188896, subscript_call_result_188909)
        # Adding element type (line 1011)
        # Getting the type of 'unexpected_regexp' (line 1013)
        unexpected_regexp_188910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 47), 'unexpected_regexp')
        # Obtaining the member 'pattern' of a type (line 1013)
        pattern_188911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1013, 47), unexpected_regexp_188910, 'pattern')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 47), tuple_188896, pattern_188911)
        # Adding element type (line 1011)
        # Getting the type of 'text' (line 1014)
        text_188912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 47), 'text')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 47), tuple_188896, text_188912)
        
        # Applying the binary operator '%' (line 1011)
        result_mod_188913 = python_operator(stypy.reporting.localization.Localization(__file__, 1011, 18), '%', str_188895, tuple_188896)
        
        # Assigning a type to the variable 'msg' (line 1011)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1011, 12), 'msg', result_mod_188913)
        
        # Call to failureException(...): (line 1015)
        # Processing the call arguments (line 1015)
        # Getting the type of 'msg' (line 1015)
        msg_188916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 40), 'msg', False)
        # Processing the call keyword arguments (line 1015)
        kwargs_188917 = {}
        # Getting the type of 'self' (line 1015)
        self_188914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 18), 'self', False)
        # Obtaining the member 'failureException' of a type (line 1015)
        failureException_188915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 18), self_188914, 'failureException')
        # Calling failureException(args, kwargs) (line 1015)
        failureException_call_result_188918 = invoke(stypy.reporting.localization.Localization(__file__, 1015, 18), failureException_188915, *[msg_188916], **kwargs_188917)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1015, 12), failureException_call_result_188918, 'raise parameter', BaseException)
        # SSA join for if statement (line 1009)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertNotRegexpMatches(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertNotRegexpMatches' in the type store
        # Getting the type of 'stypy_return_type' (line 1004)
        stypy_return_type_188919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_188919)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertNotRegexpMatches'
        return stypy_return_type_188919


# Assigning a type to the variable 'TestCase' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'TestCase', TestCase)

# Assigning a Name to a Name (line 164):
# Getting the type of 'AssertionError' (line 164)
AssertionError_188920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'AssertionError')
# Getting the type of 'TestCase'
TestCase_188921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'failureException' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188921, 'failureException', AssertionError_188920)

# Assigning a Name to a Name (line 166):
# Getting the type of 'False' (line 166)
False_188922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), 'False')
# Getting the type of 'TestCase'
TestCase_188923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'longMessage' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188923, 'longMessage', False_188922)

# Assigning a BinOp to a Name (line 168):
int_188924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 14), 'int')
int_188925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 17), 'int')
# Applying the binary operator '*' (line 168)
result_mul_188926 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 14), '*', int_188924, int_188925)

# Getting the type of 'TestCase'
TestCase_188927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'maxDiff' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188927, 'maxDiff', result_mul_188926)

# Assigning a BinOp to a Name (line 172):
int_188928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 21), 'int')
int_188929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'int')
# Applying the binary operator '**' (line 172)
result_pow_188930 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 21), '**', int_188928, int_188929)

# Getting the type of 'TestCase'
TestCase_188931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member '_diffThreshold' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188931, '_diffThreshold', result_pow_188930)

# Assigning a Name to a Name (line 176):
# Getting the type of 'False' (line 176)
False_188932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'False')
# Getting the type of 'TestCase'
TestCase_188933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member '_classSetupFailed' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188933, '_classSetupFailed', False_188932)

# Assigning a Name to a Name (line 599):
# Getting the type of 'TestCase'
TestCase_188934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Obtaining the member 'assertEqual' of a type
assertEqual_188935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188934, 'assertEqual')
# Getting the type of 'TestCase'
TestCase_188936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'assertEquals' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188936, 'assertEquals', assertEqual_188935)

# Assigning a Name to a Name (line 600):
# Getting the type of 'TestCase'
TestCase_188937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Obtaining the member 'assertNotEqual' of a type
assertNotEqual_188938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188937, 'assertNotEqual')
# Getting the type of 'TestCase'
TestCase_188939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'assertNotEquals' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188939, 'assertNotEquals', assertNotEqual_188938)

# Assigning a Name to a Name (line 601):
# Getting the type of 'TestCase'
TestCase_188940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Obtaining the member 'assertAlmostEqual' of a type
assertAlmostEqual_188941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188940, 'assertAlmostEqual')
# Getting the type of 'TestCase'
TestCase_188942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'assertAlmostEquals' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188942, 'assertAlmostEquals', assertAlmostEqual_188941)

# Assigning a Name to a Name (line 602):
# Getting the type of 'TestCase'
TestCase_188943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Obtaining the member 'assertNotAlmostEqual' of a type
assertNotAlmostEqual_188944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188943, 'assertNotAlmostEqual')
# Getting the type of 'TestCase'
TestCase_188945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'assertNotAlmostEquals' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188945, 'assertNotAlmostEquals', assertNotAlmostEqual_188944)

# Assigning a Name to a Name (line 603):
# Getting the type of 'TestCase'
TestCase_188946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Obtaining the member 'assertTrue' of a type
assertTrue_188947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188946, 'assertTrue')
# Getting the type of 'TestCase'
TestCase_188948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'assert_' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188948, 'assert_', assertTrue_188947)

# Assigning a Call to a Name (line 615):

# Call to _deprecate(...): (line 615)
# Processing the call arguments (line 615)
# Getting the type of 'TestCase'
TestCase_188951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member 'assertEqual' of a type
assertEqual_188952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188951, 'assertEqual')
# Processing the call keyword arguments (line 615)
kwargs_188953 = {}
# Getting the type of 'TestCase'
TestCase_188949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member '_deprecate' of a type
_deprecate_188950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188949, '_deprecate')
# Calling _deprecate(args, kwargs) (line 615)
_deprecate_call_result_188954 = invoke(stypy.reporting.localization.Localization(__file__, 615, 22), _deprecate_188950, *[assertEqual_188952], **kwargs_188953)

# Getting the type of 'TestCase'
TestCase_188955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'failUnlessEqual' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188955, 'failUnlessEqual', _deprecate_call_result_188954)

# Assigning a Call to a Name (line 616):

# Call to _deprecate(...): (line 616)
# Processing the call arguments (line 616)
# Getting the type of 'TestCase'
TestCase_188958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member 'assertNotEqual' of a type
assertNotEqual_188959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188958, 'assertNotEqual')
# Processing the call keyword arguments (line 616)
kwargs_188960 = {}
# Getting the type of 'TestCase'
TestCase_188956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member '_deprecate' of a type
_deprecate_188957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188956, '_deprecate')
# Calling _deprecate(args, kwargs) (line 616)
_deprecate_call_result_188961 = invoke(stypy.reporting.localization.Localization(__file__, 616, 18), _deprecate_188957, *[assertNotEqual_188959], **kwargs_188960)

# Getting the type of 'TestCase'
TestCase_188962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'failIfEqual' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188962, 'failIfEqual', _deprecate_call_result_188961)

# Assigning a Call to a Name (line 617):

# Call to _deprecate(...): (line 617)
# Processing the call arguments (line 617)
# Getting the type of 'TestCase'
TestCase_188965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member 'assertAlmostEqual' of a type
assertAlmostEqual_188966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188965, 'assertAlmostEqual')
# Processing the call keyword arguments (line 617)
kwargs_188967 = {}
# Getting the type of 'TestCase'
TestCase_188963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member '_deprecate' of a type
_deprecate_188964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188963, '_deprecate')
# Calling _deprecate(args, kwargs) (line 617)
_deprecate_call_result_188968 = invoke(stypy.reporting.localization.Localization(__file__, 617, 28), _deprecate_188964, *[assertAlmostEqual_188966], **kwargs_188967)

# Getting the type of 'TestCase'
TestCase_188969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'failUnlessAlmostEqual' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188969, 'failUnlessAlmostEqual', _deprecate_call_result_188968)

# Assigning a Call to a Name (line 618):

# Call to _deprecate(...): (line 618)
# Processing the call arguments (line 618)
# Getting the type of 'TestCase'
TestCase_188972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member 'assertNotAlmostEqual' of a type
assertNotAlmostEqual_188973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188972, 'assertNotAlmostEqual')
# Processing the call keyword arguments (line 618)
kwargs_188974 = {}
# Getting the type of 'TestCase'
TestCase_188970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member '_deprecate' of a type
_deprecate_188971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188970, '_deprecate')
# Calling _deprecate(args, kwargs) (line 618)
_deprecate_call_result_188975 = invoke(stypy.reporting.localization.Localization(__file__, 618, 24), _deprecate_188971, *[assertNotAlmostEqual_188973], **kwargs_188974)

# Getting the type of 'TestCase'
TestCase_188976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'failIfAlmostEqual' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188976, 'failIfAlmostEqual', _deprecate_call_result_188975)

# Assigning a Call to a Name (line 619):

# Call to _deprecate(...): (line 619)
# Processing the call arguments (line 619)
# Getting the type of 'TestCase'
TestCase_188979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member 'assertTrue' of a type
assertTrue_188980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188979, 'assertTrue')
# Processing the call keyword arguments (line 619)
kwargs_188981 = {}
# Getting the type of 'TestCase'
TestCase_188977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member '_deprecate' of a type
_deprecate_188978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188977, '_deprecate')
# Calling _deprecate(args, kwargs) (line 619)
_deprecate_call_result_188982 = invoke(stypy.reporting.localization.Localization(__file__, 619, 17), _deprecate_188978, *[assertTrue_188980], **kwargs_188981)

# Getting the type of 'TestCase'
TestCase_188983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'failUnless' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188983, 'failUnless', _deprecate_call_result_188982)

# Assigning a Call to a Name (line 620):

# Call to _deprecate(...): (line 620)
# Processing the call arguments (line 620)
# Getting the type of 'TestCase'
TestCase_188986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member 'assertRaises' of a type
assertRaises_188987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188986, 'assertRaises')
# Processing the call keyword arguments (line 620)
kwargs_188988 = {}
# Getting the type of 'TestCase'
TestCase_188984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member '_deprecate' of a type
_deprecate_188985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188984, '_deprecate')
# Calling _deprecate(args, kwargs) (line 620)
_deprecate_call_result_188989 = invoke(stypy.reporting.localization.Localization(__file__, 620, 23), _deprecate_188985, *[assertRaises_188987], **kwargs_188988)

# Getting the type of 'TestCase'
TestCase_188990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'failUnlessRaises' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188990, 'failUnlessRaises', _deprecate_call_result_188989)

# Assigning a Call to a Name (line 621):

# Call to _deprecate(...): (line 621)
# Processing the call arguments (line 621)
# Getting the type of 'TestCase'
TestCase_188993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member 'assertFalse' of a type
assertFalse_188994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188993, 'assertFalse')
# Processing the call keyword arguments (line 621)
kwargs_188995 = {}
# Getting the type of 'TestCase'
TestCase_188991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member '_deprecate' of a type
_deprecate_188992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188991, '_deprecate')
# Calling _deprecate(args, kwargs) (line 621)
_deprecate_call_result_188996 = invoke(stypy.reporting.localization.Localization(__file__, 621, 13), _deprecate_188992, *[assertFalse_188994], **kwargs_188995)

# Getting the type of 'TestCase'
TestCase_188997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'failIf' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_188997, 'failIf', _deprecate_call_result_188996)
# Declaration of the 'FunctionTestCase' class
# Getting the type of 'TestCase' (line 1018)
TestCase_188998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 23), 'TestCase')

class FunctionTestCase(TestCase_188998, ):
    str_188999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, (-1)), 'str', "A test case that wraps a test function.\n\n    This is useful for slipping pre-existing test functions into the\n    unittest framework. Optionally, set-up and tidy-up functions can be\n    supplied. As with TestCase, the tidy-up ('tearDown') function will\n    always be called if the set-up ('setUp') function ran successfully.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 1027)
        None_189000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 39), 'None')
        # Getting the type of 'None' (line 1027)
        None_189001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 54), 'None')
        # Getting the type of 'None' (line 1027)
        None_189002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 72), 'None')
        defaults = [None_189000, None_189001, None_189002]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1027, 4, False)
        # Assigning a type to the variable 'self' (line 1028)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1028, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.__init__', ['testFunc', 'setUp', 'tearDown', 'description'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['testFunc', 'setUp', 'tearDown', 'description'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 1028)
        # Processing the call keyword arguments (line 1028)
        kwargs_189009 = {}
        
        # Call to super(...): (line 1028)
        # Processing the call arguments (line 1028)
        # Getting the type of 'FunctionTestCase' (line 1028)
        FunctionTestCase_189004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 14), 'FunctionTestCase', False)
        # Getting the type of 'self' (line 1028)
        self_189005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 32), 'self', False)
        # Processing the call keyword arguments (line 1028)
        kwargs_189006 = {}
        # Getting the type of 'super' (line 1028)
        super_189003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 8), 'super', False)
        # Calling super(args, kwargs) (line 1028)
        super_call_result_189007 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 8), super_189003, *[FunctionTestCase_189004, self_189005], **kwargs_189006)
        
        # Obtaining the member '__init__' of a type (line 1028)
        init___189008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 8), super_call_result_189007, '__init__')
        # Calling __init__(args, kwargs) (line 1028)
        init___call_result_189010 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 8), init___189008, *[], **kwargs_189009)
        
        
        # Assigning a Name to a Attribute (line 1029):
        
        # Assigning a Name to a Attribute (line 1029):
        # Getting the type of 'setUp' (line 1029)
        setUp_189011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 26), 'setUp')
        # Getting the type of 'self' (line 1029)
        self_189012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 8), 'self')
        # Setting the type of the member '_setUpFunc' of a type (line 1029)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1029, 8), self_189012, '_setUpFunc', setUp_189011)
        
        # Assigning a Name to a Attribute (line 1030):
        
        # Assigning a Name to a Attribute (line 1030):
        # Getting the type of 'tearDown' (line 1030)
        tearDown_189013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 29), 'tearDown')
        # Getting the type of 'self' (line 1030)
        self_189014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'self')
        # Setting the type of the member '_tearDownFunc' of a type (line 1030)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1030, 8), self_189014, '_tearDownFunc', tearDown_189013)
        
        # Assigning a Name to a Attribute (line 1031):
        
        # Assigning a Name to a Attribute (line 1031):
        # Getting the type of 'testFunc' (line 1031)
        testFunc_189015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 25), 'testFunc')
        # Getting the type of 'self' (line 1031)
        self_189016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'self')
        # Setting the type of the member '_testFunc' of a type (line 1031)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1031, 8), self_189016, '_testFunc', testFunc_189015)
        
        # Assigning a Name to a Attribute (line 1032):
        
        # Assigning a Name to a Attribute (line 1032):
        # Getting the type of 'description' (line 1032)
        description_189017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 28), 'description')
        # Getting the type of 'self' (line 1032)
        self_189018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 8), 'self')
        # Setting the type of the member '_description' of a type (line 1032)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 8), self_189018, '_description', description_189017)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 1034, 4, False)
        # Assigning a type to the variable 'self' (line 1035)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.setUp')
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.setUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setUp(...)' code ##################

        
        
        # Getting the type of 'self' (line 1035)
        self_189019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 11), 'self')
        # Obtaining the member '_setUpFunc' of a type (line 1035)
        _setUpFunc_189020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 11), self_189019, '_setUpFunc')
        # Getting the type of 'None' (line 1035)
        None_189021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 34), 'None')
        # Applying the binary operator 'isnot' (line 1035)
        result_is_not_189022 = python_operator(stypy.reporting.localization.Localization(__file__, 1035, 11), 'isnot', _setUpFunc_189020, None_189021)
        
        # Testing the type of an if condition (line 1035)
        if_condition_189023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1035, 8), result_is_not_189022)
        # Assigning a type to the variable 'if_condition_189023' (line 1035)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 8), 'if_condition_189023', if_condition_189023)
        # SSA begins for if statement (line 1035)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _setUpFunc(...): (line 1036)
        # Processing the call keyword arguments (line 1036)
        kwargs_189026 = {}
        # Getting the type of 'self' (line 1036)
        self_189024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 12), 'self', False)
        # Obtaining the member '_setUpFunc' of a type (line 1036)
        _setUpFunc_189025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1036, 12), self_189024, '_setUpFunc')
        # Calling _setUpFunc(args, kwargs) (line 1036)
        _setUpFunc_call_result_189027 = invoke(stypy.reporting.localization.Localization(__file__, 1036, 12), _setUpFunc_189025, *[], **kwargs_189026)
        
        # SSA join for if statement (line 1035)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 1034)
        stypy_return_type_189028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_189028


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 1038, 4, False)
        # Assigning a type to the variable 'self' (line 1039)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.tearDown')
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tearDown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tearDown(...)' code ##################

        
        
        # Getting the type of 'self' (line 1039)
        self_189029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 11), 'self')
        # Obtaining the member '_tearDownFunc' of a type (line 1039)
        _tearDownFunc_189030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 11), self_189029, '_tearDownFunc')
        # Getting the type of 'None' (line 1039)
        None_189031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 37), 'None')
        # Applying the binary operator 'isnot' (line 1039)
        result_is_not_189032 = python_operator(stypy.reporting.localization.Localization(__file__, 1039, 11), 'isnot', _tearDownFunc_189030, None_189031)
        
        # Testing the type of an if condition (line 1039)
        if_condition_189033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1039, 8), result_is_not_189032)
        # Assigning a type to the variable 'if_condition_189033' (line 1039)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'if_condition_189033', if_condition_189033)
        # SSA begins for if statement (line 1039)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _tearDownFunc(...): (line 1040)
        # Processing the call keyword arguments (line 1040)
        kwargs_189036 = {}
        # Getting the type of 'self' (line 1040)
        self_189034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 12), 'self', False)
        # Obtaining the member '_tearDownFunc' of a type (line 1040)
        _tearDownFunc_189035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 12), self_189034, '_tearDownFunc')
        # Calling _tearDownFunc(args, kwargs) (line 1040)
        _tearDownFunc_call_result_189037 = invoke(stypy.reporting.localization.Localization(__file__, 1040, 12), _tearDownFunc_189035, *[], **kwargs_189036)
        
        # SSA join for if statement (line 1039)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 1038)
        stypy_return_type_189038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189038)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_189038


    @norecursion
    def runTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runTest'
        module_type_store = module_type_store.open_function_context('runTest', 1042, 4, False)
        # Assigning a type to the variable 'self' (line 1043)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1043, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.runTest')
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.runTest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.runTest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runTest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runTest(...)' code ##################

        
        # Call to _testFunc(...): (line 1043)
        # Processing the call keyword arguments (line 1043)
        kwargs_189041 = {}
        # Getting the type of 'self' (line 1043)
        self_189039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 8), 'self', False)
        # Obtaining the member '_testFunc' of a type (line 1043)
        _testFunc_189040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 8), self_189039, '_testFunc')
        # Calling _testFunc(args, kwargs) (line 1043)
        _testFunc_call_result_189042 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 8), _testFunc_189040, *[], **kwargs_189041)
        
        
        # ################# End of 'runTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runTest' in the type store
        # Getting the type of 'stypy_return_type' (line 1042)
        stypy_return_type_189043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189043)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runTest'
        return stypy_return_type_189043


    @norecursion
    def id(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'id'
        module_type_store = module_type_store.open_function_context('id', 1045, 4, False)
        # Assigning a type to the variable 'self' (line 1046)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.id.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.id.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.id.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.id.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.id')
        FunctionTestCase.id.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionTestCase.id.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.id.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.id.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.id.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.id.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.id.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.id', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'id', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'id(...)' code ##################

        # Getting the type of 'self' (line 1046)
        self_189044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 15), 'self')
        # Obtaining the member '_testFunc' of a type (line 1046)
        _testFunc_189045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1046, 15), self_189044, '_testFunc')
        # Obtaining the member '__name__' of a type (line 1046)
        name___189046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1046, 15), _testFunc_189045, '__name__')
        # Assigning a type to the variable 'stypy_return_type' (line 1046)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 8), 'stypy_return_type', name___189046)
        
        # ################# End of 'id(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'id' in the type store
        # Getting the type of 'stypy_return_type' (line 1045)
        stypy_return_type_189047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189047)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'id'
        return stypy_return_type_189047


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 1048, 4, False)
        # Assigning a type to the variable 'self' (line 1049)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.__eq__')
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Call to isinstance(...): (line 1049)
        # Processing the call arguments (line 1049)
        # Getting the type of 'other' (line 1049)
        other_189049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 26), 'other', False)
        # Getting the type of 'self' (line 1049)
        self_189050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 33), 'self', False)
        # Obtaining the member '__class__' of a type (line 1049)
        class___189051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 33), self_189050, '__class__')
        # Processing the call keyword arguments (line 1049)
        kwargs_189052 = {}
        # Getting the type of 'isinstance' (line 1049)
        isinstance_189048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1049)
        isinstance_call_result_189053 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 15), isinstance_189048, *[other_189049, class___189051], **kwargs_189052)
        
        # Applying the 'not' unary operator (line 1049)
        result_not__189054 = python_operator(stypy.reporting.localization.Localization(__file__, 1049, 11), 'not', isinstance_call_result_189053)
        
        # Testing the type of an if condition (line 1049)
        if_condition_189055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1049, 8), result_not__189054)
        # Assigning a type to the variable 'if_condition_189055' (line 1049)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'if_condition_189055', if_condition_189055)
        # SSA begins for if statement (line 1049)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 1050)
        NotImplemented_189056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 1050)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 12), 'stypy_return_type', NotImplemented_189056)
        # SSA join for if statement (line 1049)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 1052)
        self_189057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 15), 'self')
        # Obtaining the member '_setUpFunc' of a type (line 1052)
        _setUpFunc_189058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 15), self_189057, '_setUpFunc')
        # Getting the type of 'other' (line 1052)
        other_189059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 34), 'other')
        # Obtaining the member '_setUpFunc' of a type (line 1052)
        _setUpFunc_189060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 34), other_189059, '_setUpFunc')
        # Applying the binary operator '==' (line 1052)
        result_eq_189061 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 15), '==', _setUpFunc_189058, _setUpFunc_189060)
        
        
        # Getting the type of 'self' (line 1053)
        self_189062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 15), 'self')
        # Obtaining the member '_tearDownFunc' of a type (line 1053)
        _tearDownFunc_189063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 15), self_189062, '_tearDownFunc')
        # Getting the type of 'other' (line 1053)
        other_189064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 37), 'other')
        # Obtaining the member '_tearDownFunc' of a type (line 1053)
        _tearDownFunc_189065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 37), other_189064, '_tearDownFunc')
        # Applying the binary operator '==' (line 1053)
        result_eq_189066 = python_operator(stypy.reporting.localization.Localization(__file__, 1053, 15), '==', _tearDownFunc_189063, _tearDownFunc_189065)
        
        # Applying the binary operator 'and' (line 1052)
        result_and_keyword_189067 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 15), 'and', result_eq_189061, result_eq_189066)
        
        # Getting the type of 'self' (line 1054)
        self_189068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 15), 'self')
        # Obtaining the member '_testFunc' of a type (line 1054)
        _testFunc_189069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 15), self_189068, '_testFunc')
        # Getting the type of 'other' (line 1054)
        other_189070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 33), 'other')
        # Obtaining the member '_testFunc' of a type (line 1054)
        _testFunc_189071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 33), other_189070, '_testFunc')
        # Applying the binary operator '==' (line 1054)
        result_eq_189072 = python_operator(stypy.reporting.localization.Localization(__file__, 1054, 15), '==', _testFunc_189069, _testFunc_189071)
        
        # Applying the binary operator 'and' (line 1052)
        result_and_keyword_189073 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 15), 'and', result_and_keyword_189067, result_eq_189072)
        
        # Getting the type of 'self' (line 1055)
        self_189074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 15), 'self')
        # Obtaining the member '_description' of a type (line 1055)
        _description_189075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 15), self_189074, '_description')
        # Getting the type of 'other' (line 1055)
        other_189076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 36), 'other')
        # Obtaining the member '_description' of a type (line 1055)
        _description_189077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 36), other_189076, '_description')
        # Applying the binary operator '==' (line 1055)
        result_eq_189078 = python_operator(stypy.reporting.localization.Localization(__file__, 1055, 15), '==', _description_189075, _description_189077)
        
        # Applying the binary operator 'and' (line 1052)
        result_and_keyword_189079 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 15), 'and', result_and_keyword_189073, result_eq_189078)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1052)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'stypy_return_type', result_and_keyword_189079)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 1048)
        stypy_return_type_189080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189080)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_189080


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 1057, 4, False)
        # Assigning a type to the variable 'self' (line 1058)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1058, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.__ne__')
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.__ne__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 1058)
        self_189081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 19), 'self')
        # Getting the type of 'other' (line 1058)
        other_189082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 27), 'other')
        # Applying the binary operator '==' (line 1058)
        result_eq_189083 = python_operator(stypy.reporting.localization.Localization(__file__, 1058, 19), '==', self_189081, other_189082)
        
        # Applying the 'not' unary operator (line 1058)
        result_not__189084 = python_operator(stypy.reporting.localization.Localization(__file__, 1058, 15), 'not', result_eq_189083)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1058)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1058, 8), 'stypy_return_type', result_not__189084)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 1057)
        stypy_return_type_189085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_189085


    @norecursion
    def stypy__hash__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__hash__'
        module_type_store = module_type_store.open_function_context('__hash__', 1060, 4, False)
        # Assigning a type to the variable 'self' (line 1061)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.__hash__')
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.stypy__hash__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.__hash__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__hash__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__hash__(...)' code ##################

        
        # Call to hash(...): (line 1061)
        # Processing the call arguments (line 1061)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1061)
        tuple_189087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1061)
        # Adding element type (line 1061)
        
        # Call to type(...): (line 1061)
        # Processing the call arguments (line 1061)
        # Getting the type of 'self' (line 1061)
        self_189089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 26), 'self', False)
        # Processing the call keyword arguments (line 1061)
        kwargs_189090 = {}
        # Getting the type of 'type' (line 1061)
        type_189088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 21), 'type', False)
        # Calling type(args, kwargs) (line 1061)
        type_call_result_189091 = invoke(stypy.reporting.localization.Localization(__file__, 1061, 21), type_189088, *[self_189089], **kwargs_189090)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 21), tuple_189087, type_call_result_189091)
        # Adding element type (line 1061)
        # Getting the type of 'self' (line 1061)
        self_189092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 33), 'self', False)
        # Obtaining the member '_setUpFunc' of a type (line 1061)
        _setUpFunc_189093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1061, 33), self_189092, '_setUpFunc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 21), tuple_189087, _setUpFunc_189093)
        # Adding element type (line 1061)
        # Getting the type of 'self' (line 1061)
        self_189094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 50), 'self', False)
        # Obtaining the member '_tearDownFunc' of a type (line 1061)
        _tearDownFunc_189095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1061, 50), self_189094, '_tearDownFunc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 21), tuple_189087, _tearDownFunc_189095)
        # Adding element type (line 1061)
        # Getting the type of 'self' (line 1062)
        self_189096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 21), 'self', False)
        # Obtaining the member '_testFunc' of a type (line 1062)
        _testFunc_189097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1062, 21), self_189096, '_testFunc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 21), tuple_189087, _testFunc_189097)
        # Adding element type (line 1061)
        # Getting the type of 'self' (line 1062)
        self_189098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 37), 'self', False)
        # Obtaining the member '_description' of a type (line 1062)
        _description_189099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1062, 37), self_189098, '_description')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1061, 21), tuple_189087, _description_189099)
        
        # Processing the call keyword arguments (line 1061)
        kwargs_189100 = {}
        # Getting the type of 'hash' (line 1061)
        hash_189086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 15), 'hash', False)
        # Calling hash(args, kwargs) (line 1061)
        hash_call_result_189101 = invoke(stypy.reporting.localization.Localization(__file__, 1061, 15), hash_189086, *[tuple_189087], **kwargs_189100)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1061)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 8), 'stypy_return_type', hash_call_result_189101)
        
        # ################# End of '__hash__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__hash__' in the type store
        # Getting the type of 'stypy_return_type' (line 1060)
        stypy_return_type_189102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__hash__'
        return stypy_return_type_189102


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 1064, 4, False)
        # Assigning a type to the variable 'self' (line 1065)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.__str__')
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.__str__', [], None, None, defaults, varargs, kwargs)

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

        str_189103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, 15), 'str', '%s (%s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1065)
        tuple_189104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1065)
        # Adding element type (line 1065)
        
        # Call to strclass(...): (line 1065)
        # Processing the call arguments (line 1065)
        # Getting the type of 'self' (line 1065)
        self_189106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 37), 'self', False)
        # Obtaining the member '__class__' of a type (line 1065)
        class___189107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 37), self_189106, '__class__')
        # Processing the call keyword arguments (line 1065)
        kwargs_189108 = {}
        # Getting the type of 'strclass' (line 1065)
        strclass_189105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 28), 'strclass', False)
        # Calling strclass(args, kwargs) (line 1065)
        strclass_call_result_189109 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 28), strclass_189105, *[class___189107], **kwargs_189108)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1065, 28), tuple_189104, strclass_call_result_189109)
        # Adding element type (line 1065)
        # Getting the type of 'self' (line 1066)
        self_189110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 28), 'self')
        # Obtaining the member '_testFunc' of a type (line 1066)
        _testFunc_189111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 28), self_189110, '_testFunc')
        # Obtaining the member '__name__' of a type (line 1066)
        name___189112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 28), _testFunc_189111, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1065, 28), tuple_189104, name___189112)
        
        # Applying the binary operator '%' (line 1065)
        result_mod_189113 = python_operator(stypy.reporting.localization.Localization(__file__, 1065, 15), '%', str_189103, tuple_189104)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1065)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 8), 'stypy_return_type', result_mod_189113)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 1064)
        stypy_return_type_189114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189114)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_189114


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 1068, 4, False)
        # Assigning a type to the variable 'self' (line 1069)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.__repr__')
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_189115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 15), 'str', '<%s tec=%s>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1069)
        tuple_189116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1069)
        # Adding element type (line 1069)
        
        # Call to strclass(...): (line 1069)
        # Processing the call arguments (line 1069)
        # Getting the type of 'self' (line 1069)
        self_189118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 41), 'self', False)
        # Obtaining the member '__class__' of a type (line 1069)
        class___189119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 41), self_189118, '__class__')
        # Processing the call keyword arguments (line 1069)
        kwargs_189120 = {}
        # Getting the type of 'strclass' (line 1069)
        strclass_189117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 32), 'strclass', False)
        # Calling strclass(args, kwargs) (line 1069)
        strclass_call_result_189121 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 32), strclass_189117, *[class___189119], **kwargs_189120)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 32), tuple_189116, strclass_call_result_189121)
        # Adding element type (line 1069)
        # Getting the type of 'self' (line 1070)
        self_189122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 37), 'self')
        # Obtaining the member '_testFunc' of a type (line 1070)
        _testFunc_189123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1070, 37), self_189122, '_testFunc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 32), tuple_189116, _testFunc_189123)
        
        # Applying the binary operator '%' (line 1069)
        result_mod_189124 = python_operator(stypy.reporting.localization.Localization(__file__, 1069, 15), '%', str_189115, tuple_189116)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1069)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 8), 'stypy_return_type', result_mod_189124)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 1068)
        stypy_return_type_189125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189125)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_189125


    @norecursion
    def shortDescription(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shortDescription'
        module_type_store = module_type_store.open_function_context('shortDescription', 1072, 4, False)
        # Assigning a type to the variable 'self' (line 1073)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_localization', localization)
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_function_name', 'FunctionTestCase.shortDescription')
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionTestCase.shortDescription.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionTestCase.shortDescription', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shortDescription', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shortDescription(...)' code ##################

        
        
        # Getting the type of 'self' (line 1073)
        self_189126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 11), 'self')
        # Obtaining the member '_description' of a type (line 1073)
        _description_189127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 11), self_189126, '_description')
        # Getting the type of 'None' (line 1073)
        None_189128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 36), 'None')
        # Applying the binary operator 'isnot' (line 1073)
        result_is_not_189129 = python_operator(stypy.reporting.localization.Localization(__file__, 1073, 11), 'isnot', _description_189127, None_189128)
        
        # Testing the type of an if condition (line 1073)
        if_condition_189130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1073, 8), result_is_not_189129)
        # Assigning a type to the variable 'if_condition_189130' (line 1073)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'if_condition_189130', if_condition_189130)
        # SSA begins for if statement (line 1073)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 1074)
        self_189131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 19), 'self')
        # Obtaining the member '_description' of a type (line 1074)
        _description_189132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1074, 19), self_189131, '_description')
        # Assigning a type to the variable 'stypy_return_type' (line 1074)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1074, 12), 'stypy_return_type', _description_189132)
        # SSA join for if statement (line 1073)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 1075):
        
        # Assigning a Attribute to a Name (line 1075):
        # Getting the type of 'self' (line 1075)
        self_189133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 14), 'self')
        # Obtaining the member '_testFunc' of a type (line 1075)
        _testFunc_189134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 14), self_189133, '_testFunc')
        # Obtaining the member '__doc__' of a type (line 1075)
        doc___189135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 14), _testFunc_189134, '__doc__')
        # Assigning a type to the variable 'doc' (line 1075)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'doc', doc___189135)
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'doc' (line 1076)
        doc_189136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 15), 'doc')
        
        # Call to strip(...): (line 1076)
        # Processing the call keyword arguments (line 1076)
        kwargs_189146 = {}
        
        # Obtaining the type of the subscript
        int_189137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 39), 'int')
        
        # Call to split(...): (line 1076)
        # Processing the call arguments (line 1076)
        str_189140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 33), 'str', '\n')
        # Processing the call keyword arguments (line 1076)
        kwargs_189141 = {}
        # Getting the type of 'doc' (line 1076)
        doc_189138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 23), 'doc', False)
        # Obtaining the member 'split' of a type (line 1076)
        split_189139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1076, 23), doc_189138, 'split')
        # Calling split(args, kwargs) (line 1076)
        split_call_result_189142 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 23), split_189139, *[str_189140], **kwargs_189141)
        
        # Obtaining the member '__getitem__' of a type (line 1076)
        getitem___189143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1076, 23), split_call_result_189142, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1076)
        subscript_call_result_189144 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 23), getitem___189143, int_189137)
        
        # Obtaining the member 'strip' of a type (line 1076)
        strip_189145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1076, 23), subscript_call_result_189144, 'strip')
        # Calling strip(args, kwargs) (line 1076)
        strip_call_result_189147 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 23), strip_189145, *[], **kwargs_189146)
        
        # Applying the binary operator 'and' (line 1076)
        result_and_keyword_189148 = python_operator(stypy.reporting.localization.Localization(__file__, 1076, 15), 'and', doc_189136, strip_call_result_189147)
        
        # Getting the type of 'None' (line 1076)
        None_189149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 53), 'None')
        # Applying the binary operator 'or' (line 1076)
        result_or_keyword_189150 = python_operator(stypy.reporting.localization.Localization(__file__, 1076, 15), 'or', result_and_keyword_189148, None_189149)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1076)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1076, 8), 'stypy_return_type', result_or_keyword_189150)
        
        # ################# End of 'shortDescription(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shortDescription' in the type store
        # Getting the type of 'stypy_return_type' (line 1072)
        stypy_return_type_189151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189151)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shortDescription'
        return stypy_return_type_189151


# Assigning a type to the variable 'FunctionTestCase' (line 1018)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1018, 0), 'FunctionTestCase', FunctionTestCase)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
