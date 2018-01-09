
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import sys
2: import textwrap
3: from StringIO import StringIO
4: from test import test_support
5: 
6: import traceback
7: import unittest
8: 
9: 
10: class Test_TestResult(unittest.TestCase):
11:     # Note: there are not separate tests for TestResult.wasSuccessful(),
12:     # TestResult.errors, TestResult.failures, TestResult.testsRun or
13:     # TestResult.shouldStop because these only have meaning in terms of
14:     # other TestResult methods.
15:     #
16:     # Accordingly, tests for the aforenamed attributes are incorporated
17:     # in with the tests for the defining methods.
18:     ################################################################
19: 
20:     def test_init(self):
21:         result = unittest.TestResult()
22: 
23:         self.assertTrue(result.wasSuccessful())
24:         self.assertEqual(len(result.errors), 0)
25:         self.assertEqual(len(result.failures), 0)
26:         self.assertEqual(result.testsRun, 0)
27:         self.assertEqual(result.shouldStop, False)
28:         self.assertIsNone(result._stdout_buffer)
29:         self.assertIsNone(result._stderr_buffer)
30: 
31: 
32:     # "This method can be called to signal that the set of tests being
33:     # run should be aborted by setting the TestResult's shouldStop
34:     # attribute to True."
35:     def test_stop(self):
36:         result = unittest.TestResult()
37: 
38:         result.stop()
39: 
40:         self.assertEqual(result.shouldStop, True)
41: 
42:     # "Called when the test case test is about to be run. The default
43:     # implementation simply increments the instance's testsRun counter."
44:     def test_startTest(self):
45:         class Foo(unittest.TestCase):
46:             def test_1(self):
47:                 pass
48: 
49:         test = Foo('test_1')
50: 
51:         result = unittest.TestResult()
52: 
53:         result.startTest(test)
54: 
55:         self.assertTrue(result.wasSuccessful())
56:         self.assertEqual(len(result.errors), 0)
57:         self.assertEqual(len(result.failures), 0)
58:         self.assertEqual(result.testsRun, 1)
59:         self.assertEqual(result.shouldStop, False)
60: 
61:         result.stopTest(test)
62: 
63:     # "Called after the test case test has been executed, regardless of
64:     # the outcome. The default implementation does nothing."
65:     def test_stopTest(self):
66:         class Foo(unittest.TestCase):
67:             def test_1(self):
68:                 pass
69: 
70:         test = Foo('test_1')
71: 
72:         result = unittest.TestResult()
73: 
74:         result.startTest(test)
75: 
76:         self.assertTrue(result.wasSuccessful())
77:         self.assertEqual(len(result.errors), 0)
78:         self.assertEqual(len(result.failures), 0)
79:         self.assertEqual(result.testsRun, 1)
80:         self.assertEqual(result.shouldStop, False)
81: 
82:         result.stopTest(test)
83: 
84:         # Same tests as above; make sure nothing has changed
85:         self.assertTrue(result.wasSuccessful())
86:         self.assertEqual(len(result.errors), 0)
87:         self.assertEqual(len(result.failures), 0)
88:         self.assertEqual(result.testsRun, 1)
89:         self.assertEqual(result.shouldStop, False)
90: 
91:     # "Called before and after tests are run. The default implementation does nothing."
92:     def test_startTestRun_stopTestRun(self):
93:         result = unittest.TestResult()
94:         result.startTestRun()
95:         result.stopTestRun()
96: 
97:     # "addSuccess(test)"
98:     # ...
99:     # "Called when the test case test succeeds"
100:     # ...
101:     # "wasSuccessful() - Returns True if all tests run so far have passed,
102:     # otherwise returns False"
103:     # ...
104:     # "testsRun - The total number of tests run so far."
105:     # ...
106:     # "errors - A list containing 2-tuples of TestCase instances and
107:     # formatted tracebacks. Each tuple represents a test which raised an
108:     # unexpected exception. Contains formatted
109:     # tracebacks instead of sys.exc_info() results."
110:     # ...
111:     # "failures - A list containing 2-tuples of TestCase instances and
112:     # formatted tracebacks. Each tuple represents a test where a failure was
113:     # explicitly signalled using the TestCase.fail*() or TestCase.assert*()
114:     # methods. Contains formatted tracebacks instead
115:     # of sys.exc_info() results."
116:     def test_addSuccess(self):
117:         class Foo(unittest.TestCase):
118:             def test_1(self):
119:                 pass
120: 
121:         test = Foo('test_1')
122: 
123:         result = unittest.TestResult()
124: 
125:         result.startTest(test)
126:         result.addSuccess(test)
127:         result.stopTest(test)
128: 
129:         self.assertTrue(result.wasSuccessful())
130:         self.assertEqual(len(result.errors), 0)
131:         self.assertEqual(len(result.failures), 0)
132:         self.assertEqual(result.testsRun, 1)
133:         self.assertEqual(result.shouldStop, False)
134: 
135:     # "addFailure(test, err)"
136:     # ...
137:     # "Called when the test case test signals a failure. err is a tuple of
138:     # the form returned by sys.exc_info(): (type, value, traceback)"
139:     # ...
140:     # "wasSuccessful() - Returns True if all tests run so far have passed,
141:     # otherwise returns False"
142:     # ...
143:     # "testsRun - The total number of tests run so far."
144:     # ...
145:     # "errors - A list containing 2-tuples of TestCase instances and
146:     # formatted tracebacks. Each tuple represents a test which raised an
147:     # unexpected exception. Contains formatted
148:     # tracebacks instead of sys.exc_info() results."
149:     # ...
150:     # "failures - A list containing 2-tuples of TestCase instances and
151:     # formatted tracebacks. Each tuple represents a test where a failure was
152:     # explicitly signalled using the TestCase.fail*() or TestCase.assert*()
153:     # methods. Contains formatted tracebacks instead
154:     # of sys.exc_info() results."
155:     def test_addFailure(self):
156:         class Foo(unittest.TestCase):
157:             def test_1(self):
158:                 pass
159: 
160:         test = Foo('test_1')
161:         try:
162:             test.fail("foo")
163:         except:
164:             exc_info_tuple = sys.exc_info()
165: 
166:         result = unittest.TestResult()
167: 
168:         result.startTest(test)
169:         result.addFailure(test, exc_info_tuple)
170:         result.stopTest(test)
171: 
172:         self.assertFalse(result.wasSuccessful())
173:         self.assertEqual(len(result.errors), 0)
174:         self.assertEqual(len(result.failures), 1)
175:         self.assertEqual(result.testsRun, 1)
176:         self.assertEqual(result.shouldStop, False)
177: 
178:         test_case, formatted_exc = result.failures[0]
179:         self.assertIs(test_case, test)
180:         self.assertIsInstance(formatted_exc, str)
181: 
182:     # "addError(test, err)"
183:     # ...
184:     # "Called when the test case test raises an unexpected exception err
185:     # is a tuple of the form returned by sys.exc_info():
186:     # (type, value, traceback)"
187:     # ...
188:     # "wasSuccessful() - Returns True if all tests run so far have passed,
189:     # otherwise returns False"
190:     # ...
191:     # "testsRun - The total number of tests run so far."
192:     # ...
193:     # "errors - A list containing 2-tuples of TestCase instances and
194:     # formatted tracebacks. Each tuple represents a test which raised an
195:     # unexpected exception. Contains formatted
196:     # tracebacks instead of sys.exc_info() results."
197:     # ...
198:     # "failures - A list containing 2-tuples of TestCase instances and
199:     # formatted tracebacks. Each tuple represents a test where a failure was
200:     # explicitly signalled using the TestCase.fail*() or TestCase.assert*()
201:     # methods. Contains formatted tracebacks instead
202:     # of sys.exc_info() results."
203:     def test_addError(self):
204:         class Foo(unittest.TestCase):
205:             def test_1(self):
206:                 pass
207: 
208:         test = Foo('test_1')
209:         try:
210:             raise TypeError()
211:         except:
212:             exc_info_tuple = sys.exc_info()
213: 
214:         result = unittest.TestResult()
215: 
216:         result.startTest(test)
217:         result.addError(test, exc_info_tuple)
218:         result.stopTest(test)
219: 
220:         self.assertFalse(result.wasSuccessful())
221:         self.assertEqual(len(result.errors), 1)
222:         self.assertEqual(len(result.failures), 0)
223:         self.assertEqual(result.testsRun, 1)
224:         self.assertEqual(result.shouldStop, False)
225: 
226:         test_case, formatted_exc = result.errors[0]
227:         self.assertIs(test_case, test)
228:         self.assertIsInstance(formatted_exc, str)
229: 
230:     def testGetDescriptionWithoutDocstring(self):
231:         result = unittest.TextTestResult(None, True, 1)
232:         self.assertEqual(
233:                 result.getDescription(self),
234:                 'testGetDescriptionWithoutDocstring (' + __name__ +
235:                 '.Test_TestResult)')
236: 
237:     @unittest.skipIf(sys.flags.optimize >= 2,
238:                      "Docstrings are omitted with -O2 and above")
239:     def testGetDescriptionWithOneLineDocstring(self):
240:         '''Tests getDescription() for a method with a docstring.'''
241:         result = unittest.TextTestResult(None, True, 1)
242:         self.assertEqual(
243:                 result.getDescription(self),
244:                ('testGetDescriptionWithOneLineDocstring '
245:                 '(' + __name__ + '.Test_TestResult)\n'
246:                 'Tests getDescription() for a method with a docstring.'))
247: 
248:     @unittest.skipIf(sys.flags.optimize >= 2,
249:                      "Docstrings are omitted with -O2 and above")
250:     def testGetDescriptionWithMultiLineDocstring(self):
251:         '''Tests getDescription() for a method with a longer docstring.
252:         The second line of the docstring.
253:         '''
254:         result = unittest.TextTestResult(None, True, 1)
255:         self.assertEqual(
256:                 result.getDescription(self),
257:                ('testGetDescriptionWithMultiLineDocstring '
258:                 '(' + __name__ + '.Test_TestResult)\n'
259:                 'Tests getDescription() for a method with a longer '
260:                 'docstring.'))
261: 
262:     def testStackFrameTrimming(self):
263:         class Frame(object):
264:             class tb_frame(object):
265:                 f_globals = {}
266:         result = unittest.TestResult()
267:         self.assertFalse(result._is_relevant_tb_level(Frame))
268: 
269:         Frame.tb_frame.f_globals['__unittest'] = True
270:         self.assertTrue(result._is_relevant_tb_level(Frame))
271: 
272:     def testFailFast(self):
273:         result = unittest.TestResult()
274:         result._exc_info_to_string = lambda *_: ''
275:         result.failfast = True
276:         result.addError(None, None)
277:         self.assertTrue(result.shouldStop)
278: 
279:         result = unittest.TestResult()
280:         result._exc_info_to_string = lambda *_: ''
281:         result.failfast = True
282:         result.addFailure(None, None)
283:         self.assertTrue(result.shouldStop)
284: 
285:         result = unittest.TestResult()
286:         result._exc_info_to_string = lambda *_: ''
287:         result.failfast = True
288:         result.addUnexpectedSuccess(None)
289:         self.assertTrue(result.shouldStop)
290: 
291:     def testFailFastSetByRunner(self):
292:         runner = unittest.TextTestRunner(stream=StringIO(), failfast=True)
293:         def test(result):
294:             self.assertTrue(result.failfast)
295:         runner.run(test)
296: 
297: 
298: classDict = dict(unittest.TestResult.__dict__)
299: for m in ('addSkip', 'addExpectedFailure', 'addUnexpectedSuccess',
300:            '__init__'):
301:     del classDict[m]
302: 
303: def __init__(self, stream=None, descriptions=None, verbosity=None):
304:     self.failures = []
305:     self.errors = []
306:     self.testsRun = 0
307:     self.shouldStop = False
308:     self.buffer = False
309: 
310: classDict['__init__'] = __init__
311: OldResult = type('OldResult', (object,), classDict)
312: 
313: class Test_OldTestResult(unittest.TestCase):
314: 
315:     def assertOldResultWarning(self, test, failures):
316:         with test_support.check_warnings(("TestResult has no add.+ method,",
317:                                           RuntimeWarning)):
318:             result = OldResult()
319:             test.run(result)
320:             self.assertEqual(len(result.failures), failures)
321: 
322:     def testOldTestResult(self):
323:         class Test(unittest.TestCase):
324:             def testSkip(self):
325:                 self.skipTest('foobar')
326:             @unittest.expectedFailure
327:             def testExpectedFail(self):
328:                 raise TypeError
329:             @unittest.expectedFailure
330:             def testUnexpectedSuccess(self):
331:                 pass
332: 
333:         for test_name, should_pass in (('testSkip', True),
334:                                        ('testExpectedFail', True),
335:                                        ('testUnexpectedSuccess', False)):
336:             test = Test(test_name)
337:             self.assertOldResultWarning(test, int(not should_pass))
338: 
339:     def testOldTestTesultSetup(self):
340:         class Test(unittest.TestCase):
341:             def setUp(self):
342:                 self.skipTest('no reason')
343:             def testFoo(self):
344:                 pass
345:         self.assertOldResultWarning(Test('testFoo'), 0)
346: 
347:     def testOldTestResultClass(self):
348:         @unittest.skip('no reason')
349:         class Test(unittest.TestCase):
350:             def testFoo(self):
351:                 pass
352:         self.assertOldResultWarning(Test('testFoo'), 0)
353: 
354:     def testOldResultWithRunner(self):
355:         class Test(unittest.TestCase):
356:             def testFoo(self):
357:                 pass
358:         runner = unittest.TextTestRunner(resultclass=OldResult,
359:                                           stream=StringIO())
360:         # This will raise an exception if TextTestRunner can't handle old
361:         # test result objects
362:         runner.run(Test('testFoo'))
363: 
364: 
365: class MockTraceback(object):
366:     @staticmethod
367:     def format_exception(*_):
368:         return ['A traceback']
369: 
370: def restore_traceback():
371:     unittest.result.traceback = traceback
372: 
373: 
374: class TestOutputBuffering(unittest.TestCase):
375: 
376:     def setUp(self):
377:         self._real_out = sys.stdout
378:         self._real_err = sys.stderr
379: 
380:     def tearDown(self):
381:         sys.stdout = self._real_out
382:         sys.stderr = self._real_err
383: 
384:     def testBufferOutputOff(self):
385:         real_out = self._real_out
386:         real_err = self._real_err
387: 
388:         result = unittest.TestResult()
389:         self.assertFalse(result.buffer)
390: 
391:         self.assertIs(real_out, sys.stdout)
392:         self.assertIs(real_err, sys.stderr)
393: 
394:         result.startTest(self)
395: 
396:         self.assertIs(real_out, sys.stdout)
397:         self.assertIs(real_err, sys.stderr)
398: 
399:     def testBufferOutputStartTestAddSuccess(self):
400:         real_out = self._real_out
401:         real_err = self._real_err
402: 
403:         result = unittest.TestResult()
404:         self.assertFalse(result.buffer)
405: 
406:         result.buffer = True
407: 
408:         self.assertIs(real_out, sys.stdout)
409:         self.assertIs(real_err, sys.stderr)
410: 
411:         result.startTest(self)
412: 
413:         self.assertIsNot(real_out, sys.stdout)
414:         self.assertIsNot(real_err, sys.stderr)
415:         self.assertIsInstance(sys.stdout, StringIO)
416:         self.assertIsInstance(sys.stderr, StringIO)
417:         self.assertIsNot(sys.stdout, sys.stderr)
418: 
419:         out_stream = sys.stdout
420:         err_stream = sys.stderr
421: 
422:         result._original_stdout = StringIO()
423:         result._original_stderr = StringIO()
424: 
425:         print 'foo'
426:         print >> sys.stderr, 'bar'
427: 
428:         self.assertEqual(out_stream.getvalue(), 'foo\n')
429:         self.assertEqual(err_stream.getvalue(), 'bar\n')
430: 
431:         self.assertEqual(result._original_stdout.getvalue(), '')
432:         self.assertEqual(result._original_stderr.getvalue(), '')
433: 
434:         result.addSuccess(self)
435:         result.stopTest(self)
436: 
437:         self.assertIs(sys.stdout, result._original_stdout)
438:         self.assertIs(sys.stderr, result._original_stderr)
439: 
440:         self.assertEqual(result._original_stdout.getvalue(), '')
441:         self.assertEqual(result._original_stderr.getvalue(), '')
442: 
443:         self.assertEqual(out_stream.getvalue(), '')
444:         self.assertEqual(err_stream.getvalue(), '')
445: 
446: 
447:     def getStartedResult(self):
448:         result = unittest.TestResult()
449:         result.buffer = True
450:         result.startTest(self)
451:         return result
452: 
453:     def testBufferOutputAddErrorOrFailure(self):
454:         unittest.result.traceback = MockTraceback
455:         self.addCleanup(restore_traceback)
456: 
457:         for message_attr, add_attr, include_error in [
458:             ('errors', 'addError', True),
459:             ('failures', 'addFailure', False),
460:             ('errors', 'addError', True),
461:             ('failures', 'addFailure', False)
462:         ]:
463:             result = self.getStartedResult()
464:             buffered_out = sys.stdout
465:             buffered_err = sys.stderr
466:             result._original_stdout = StringIO()
467:             result._original_stderr = StringIO()
468: 
469:             print >> sys.stdout, 'foo'
470:             if include_error:
471:                 print >> sys.stderr, 'bar'
472: 
473: 
474:             addFunction = getattr(result, add_attr)
475:             addFunction(self, (None, None, None))
476:             result.stopTest(self)
477: 
478:             result_list = getattr(result, message_attr)
479:             self.assertEqual(len(result_list), 1)
480: 
481:             test, message = result_list[0]
482:             expectedOutMessage = textwrap.dedent('''
483:                 Stdout:
484:                 foo
485:             ''')
486:             expectedErrMessage = ''
487:             if include_error:
488:                 expectedErrMessage = textwrap.dedent('''
489:                 Stderr:
490:                 bar
491:             ''')
492:             expectedFullMessage = 'A traceback%s%s' % (expectedOutMessage, expectedErrMessage)
493: 
494:             self.assertIs(test, self)
495:             self.assertEqual(result._original_stdout.getvalue(), expectedOutMessage)
496:             self.assertEqual(result._original_stderr.getvalue(), expectedErrMessage)
497:             self.assertMultiLineEqual(message, expectedFullMessage)
498: 
499:     def testBufferSetupClass(self):
500:         result = unittest.TestResult()
501:         result.buffer = True
502: 
503:         class Foo(unittest.TestCase):
504:             @classmethod
505:             def setUpClass(cls):
506:                 1//0
507:             def test_foo(self):
508:                 pass
509:         suite = unittest.TestSuite([Foo('test_foo')])
510:         suite(result)
511:         self.assertEqual(len(result.errors), 1)
512: 
513:     def testBufferTearDownClass(self):
514:         result = unittest.TestResult()
515:         result.buffer = True
516: 
517:         class Foo(unittest.TestCase):
518:             @classmethod
519:             def tearDownClass(cls):
520:                 1//0
521:             def test_foo(self):
522:                 pass
523:         suite = unittest.TestSuite([Foo('test_foo')])
524:         suite(result)
525:         self.assertEqual(len(result.errors), 1)
526: 
527:     def testBufferSetUpModule(self):
528:         result = unittest.TestResult()
529:         result.buffer = True
530: 
531:         class Foo(unittest.TestCase):
532:             def test_foo(self):
533:                 pass
534:         class Module(object):
535:             @staticmethod
536:             def setUpModule():
537:                 1//0
538: 
539:         Foo.__module__ = 'Module'
540:         sys.modules['Module'] = Module
541:         self.addCleanup(sys.modules.pop, 'Module')
542:         suite = unittest.TestSuite([Foo('test_foo')])
543:         suite(result)
544:         self.assertEqual(len(result.errors), 1)
545: 
546:     def testBufferTearDownModule(self):
547:         result = unittest.TestResult()
548:         result.buffer = True
549: 
550:         class Foo(unittest.TestCase):
551:             def test_foo(self):
552:                 pass
553:         class Module(object):
554:             @staticmethod
555:             def tearDownModule():
556:                 1//0
557: 
558:         Foo.__module__ = 'Module'
559:         sys.modules['Module'] = Module
560:         self.addCleanup(sys.modules.pop, 'Module')
561:         suite = unittest.TestSuite([Foo('test_foo')])
562:         suite(result)
563:         self.assertEqual(len(result.errors), 1)
564: 
565: 
566: if __name__ == '__main__':
567:     unittest.main()
568: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import sys' statement (line 1)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import textwrap' statement (line 2)
import textwrap

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'textwrap', textwrap, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from StringIO import StringIO' statement (line 3)
from StringIO import StringIO

import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'StringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from test import test_support' statement (line 4)
from test import test_support

import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'test', None, module_type_store, ['test_support'], [test_support])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import traceback' statement (line 6)
import traceback

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'traceback', traceback, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import unittest' statement (line 7)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'unittest', unittest, module_type_store)

# Declaration of the 'Test_TestResult' class
# Getting the type of 'unittest' (line 10)
unittest_204016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 22), 'unittest')
# Obtaining the member 'TestCase' of a type (line 10)
TestCase_204017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 22), unittest_204016, 'TestCase')

class Test_TestResult(TestCase_204017, ):

    @norecursion
    def test_init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_init'
        module_type_store = module_type_store.open_function_context('test_init', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.test_init.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.test_init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.test_init.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.test_init.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.test_init')
        Test_TestResult.test_init.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.test_init.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.test_init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.test_init.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.test_init.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.test_init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.test_init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.test_init', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_init', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_init(...)' code ##################

        
        # Assigning a Call to a Name (line 21):
        
        # Assigning a Call to a Name (line 21):
        
        # Call to TestResult(...): (line 21)
        # Processing the call keyword arguments (line 21)
        kwargs_204020 = {}
        # Getting the type of 'unittest' (line 21)
        unittest_204018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 21)
        TestResult_204019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 17), unittest_204018, 'TestResult')
        # Calling TestResult(args, kwargs) (line 21)
        TestResult_call_result_204021 = invoke(stypy.reporting.localization.Localization(__file__, 21, 17), TestResult_204019, *[], **kwargs_204020)
        
        # Assigning a type to the variable 'result' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'result', TestResult_call_result_204021)
        
        # Call to assertTrue(...): (line 23)
        # Processing the call arguments (line 23)
        
        # Call to wasSuccessful(...): (line 23)
        # Processing the call keyword arguments (line 23)
        kwargs_204026 = {}
        # Getting the type of 'result' (line 23)
        result_204024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 23)
        wasSuccessful_204025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 24), result_204024, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 23)
        wasSuccessful_call_result_204027 = invoke(stypy.reporting.localization.Localization(__file__, 23, 24), wasSuccessful_204025, *[], **kwargs_204026)
        
        # Processing the call keyword arguments (line 23)
        kwargs_204028 = {}
        # Getting the type of 'self' (line 23)
        self_204022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 23)
        assertTrue_204023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_204022, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 23)
        assertTrue_call_result_204029 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), assertTrue_204023, *[wasSuccessful_call_result_204027], **kwargs_204028)
        
        
        # Call to assertEqual(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Call to len(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'result' (line 24)
        result_204033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 24)
        errors_204034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 29), result_204033, 'errors')
        # Processing the call keyword arguments (line 24)
        kwargs_204035 = {}
        # Getting the type of 'len' (line 24)
        len_204032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'len', False)
        # Calling len(args, kwargs) (line 24)
        len_call_result_204036 = invoke(stypy.reporting.localization.Localization(__file__, 24, 25), len_204032, *[errors_204034], **kwargs_204035)
        
        int_204037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 45), 'int')
        # Processing the call keyword arguments (line 24)
        kwargs_204038 = {}
        # Getting the type of 'self' (line 24)
        self_204030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 24)
        assertEqual_204031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_204030, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 24)
        assertEqual_call_result_204039 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), assertEqual_204031, *[len_call_result_204036, int_204037], **kwargs_204038)
        
        
        # Call to assertEqual(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Call to len(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'result' (line 25)
        result_204043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 29), 'result', False)
        # Obtaining the member 'failures' of a type (line 25)
        failures_204044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 29), result_204043, 'failures')
        # Processing the call keyword arguments (line 25)
        kwargs_204045 = {}
        # Getting the type of 'len' (line 25)
        len_204042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'len', False)
        # Calling len(args, kwargs) (line 25)
        len_call_result_204046 = invoke(stypy.reporting.localization.Localization(__file__, 25, 25), len_204042, *[failures_204044], **kwargs_204045)
        
        int_204047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 47), 'int')
        # Processing the call keyword arguments (line 25)
        kwargs_204048 = {}
        # Getting the type of 'self' (line 25)
        self_204040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 25)
        assertEqual_204041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_204040, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 25)
        assertEqual_call_result_204049 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assertEqual_204041, *[len_call_result_204046, int_204047], **kwargs_204048)
        
        
        # Call to assertEqual(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'result' (line 26)
        result_204052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 26)
        testsRun_204053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 25), result_204052, 'testsRun')
        int_204054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 42), 'int')
        # Processing the call keyword arguments (line 26)
        kwargs_204055 = {}
        # Getting the type of 'self' (line 26)
        self_204050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 26)
        assertEqual_204051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_204050, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 26)
        assertEqual_call_result_204056 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assertEqual_204051, *[testsRun_204053, int_204054], **kwargs_204055)
        
        
        # Call to assertEqual(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'result' (line 27)
        result_204059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 27)
        shouldStop_204060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 25), result_204059, 'shouldStop')
        # Getting the type of 'False' (line 27)
        False_204061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 44), 'False', False)
        # Processing the call keyword arguments (line 27)
        kwargs_204062 = {}
        # Getting the type of 'self' (line 27)
        self_204057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 27)
        assertEqual_204058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_204057, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 27)
        assertEqual_call_result_204063 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assertEqual_204058, *[shouldStop_204060, False_204061], **kwargs_204062)
        
        
        # Call to assertIsNone(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'result' (line 28)
        result_204066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'result', False)
        # Obtaining the member '_stdout_buffer' of a type (line 28)
        _stdout_buffer_204067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 26), result_204066, '_stdout_buffer')
        # Processing the call keyword arguments (line 28)
        kwargs_204068 = {}
        # Getting the type of 'self' (line 28)
        self_204064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 28)
        assertIsNone_204065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_204064, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 28)
        assertIsNone_call_result_204069 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assertIsNone_204065, *[_stdout_buffer_204067], **kwargs_204068)
        
        
        # Call to assertIsNone(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'result' (line 29)
        result_204072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 26), 'result', False)
        # Obtaining the member '_stderr_buffer' of a type (line 29)
        _stderr_buffer_204073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 26), result_204072, '_stderr_buffer')
        # Processing the call keyword arguments (line 29)
        kwargs_204074 = {}
        # Getting the type of 'self' (line 29)
        self_204070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 29)
        assertIsNone_204071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_204070, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 29)
        assertIsNone_call_result_204075 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assertIsNone_204071, *[_stderr_buffer_204073], **kwargs_204074)
        
        
        # ################# End of 'test_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_init' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_204076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_init'
        return stypy_return_type_204076


    @norecursion
    def test_stop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_stop'
        module_type_store = module_type_store.open_function_context('test_stop', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.test_stop')
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.test_stop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.test_stop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_stop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_stop(...)' code ##################

        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to TestResult(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_204079 = {}
        # Getting the type of 'unittest' (line 36)
        unittest_204077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 36)
        TestResult_204078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 17), unittest_204077, 'TestResult')
        # Calling TestResult(args, kwargs) (line 36)
        TestResult_call_result_204080 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), TestResult_204078, *[], **kwargs_204079)
        
        # Assigning a type to the variable 'result' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'result', TestResult_call_result_204080)
        
        # Call to stop(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_204083 = {}
        # Getting the type of 'result' (line 38)
        result_204081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'result', False)
        # Obtaining the member 'stop' of a type (line 38)
        stop_204082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), result_204081, 'stop')
        # Calling stop(args, kwargs) (line 38)
        stop_call_result_204084 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), stop_204082, *[], **kwargs_204083)
        
        
        # Call to assertEqual(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'result' (line 40)
        result_204087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 40)
        shouldStop_204088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 25), result_204087, 'shouldStop')
        # Getting the type of 'True' (line 40)
        True_204089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'True', False)
        # Processing the call keyword arguments (line 40)
        kwargs_204090 = {}
        # Getting the type of 'self' (line 40)
        self_204085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 40)
        assertEqual_204086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_204085, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 40)
        assertEqual_call_result_204091 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assertEqual_204086, *[shouldStop_204088, True_204089], **kwargs_204090)
        
        
        # ################# End of 'test_stop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_stop' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_204092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204092)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_stop'
        return stypy_return_type_204092


    @norecursion
    def test_startTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_startTest'
        module_type_store = module_type_store.open_function_context('test_startTest', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.test_startTest')
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.test_startTest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.test_startTest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_startTest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_startTest(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 45)
        unittest_204093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 45)
        TestCase_204094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 18), unittest_204093, 'TestCase')

        class Foo(TestCase_204094, ):

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 46, 12, False)
                # Assigning a type to the variable 'self' (line 47)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_1.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_1.__dict__.__setitem__('stypy_function_name', 'Foo.test_1')
                Foo.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                pass
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 46)
                stypy_return_type_204095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204095)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_204095

        
        # Assigning a type to the variable 'Foo' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to Foo(...): (line 49)
        # Processing the call arguments (line 49)
        str_204097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 49)
        kwargs_204098 = {}
        # Getting the type of 'Foo' (line 49)
        Foo_204096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 49)
        Foo_call_result_204099 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), Foo_204096, *[str_204097], **kwargs_204098)
        
        # Assigning a type to the variable 'test' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'test', Foo_call_result_204099)
        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Call to TestResult(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_204102 = {}
        # Getting the type of 'unittest' (line 51)
        unittest_204100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 51)
        TestResult_204101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 17), unittest_204100, 'TestResult')
        # Calling TestResult(args, kwargs) (line 51)
        TestResult_call_result_204103 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), TestResult_204101, *[], **kwargs_204102)
        
        # Assigning a type to the variable 'result' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'result', TestResult_call_result_204103)
        
        # Call to startTest(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'test' (line 53)
        test_204106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'test', False)
        # Processing the call keyword arguments (line 53)
        kwargs_204107 = {}
        # Getting the type of 'result' (line 53)
        result_204104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'result', False)
        # Obtaining the member 'startTest' of a type (line 53)
        startTest_204105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), result_204104, 'startTest')
        # Calling startTest(args, kwargs) (line 53)
        startTest_call_result_204108 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), startTest_204105, *[test_204106], **kwargs_204107)
        
        
        # Call to assertTrue(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Call to wasSuccessful(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_204113 = {}
        # Getting the type of 'result' (line 55)
        result_204111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 55)
        wasSuccessful_204112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), result_204111, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 55)
        wasSuccessful_call_result_204114 = invoke(stypy.reporting.localization.Localization(__file__, 55, 24), wasSuccessful_204112, *[], **kwargs_204113)
        
        # Processing the call keyword arguments (line 55)
        kwargs_204115 = {}
        # Getting the type of 'self' (line 55)
        self_204109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 55)
        assertTrue_204110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_204109, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 55)
        assertTrue_call_result_204116 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assertTrue_204110, *[wasSuccessful_call_result_204114], **kwargs_204115)
        
        
        # Call to assertEqual(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Call to len(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'result' (line 56)
        result_204120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 56)
        errors_204121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 29), result_204120, 'errors')
        # Processing the call keyword arguments (line 56)
        kwargs_204122 = {}
        # Getting the type of 'len' (line 56)
        len_204119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'len', False)
        # Calling len(args, kwargs) (line 56)
        len_call_result_204123 = invoke(stypy.reporting.localization.Localization(__file__, 56, 25), len_204119, *[errors_204121], **kwargs_204122)
        
        int_204124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 45), 'int')
        # Processing the call keyword arguments (line 56)
        kwargs_204125 = {}
        # Getting the type of 'self' (line 56)
        self_204117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 56)
        assertEqual_204118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_204117, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 56)
        assertEqual_call_result_204126 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assertEqual_204118, *[len_call_result_204123, int_204124], **kwargs_204125)
        
        
        # Call to assertEqual(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to len(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'result' (line 57)
        result_204130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'result', False)
        # Obtaining the member 'failures' of a type (line 57)
        failures_204131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 29), result_204130, 'failures')
        # Processing the call keyword arguments (line 57)
        kwargs_204132 = {}
        # Getting the type of 'len' (line 57)
        len_204129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'len', False)
        # Calling len(args, kwargs) (line 57)
        len_call_result_204133 = invoke(stypy.reporting.localization.Localization(__file__, 57, 25), len_204129, *[failures_204131], **kwargs_204132)
        
        int_204134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 47), 'int')
        # Processing the call keyword arguments (line 57)
        kwargs_204135 = {}
        # Getting the type of 'self' (line 57)
        self_204127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 57)
        assertEqual_204128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_204127, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 57)
        assertEqual_call_result_204136 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), assertEqual_204128, *[len_call_result_204133, int_204134], **kwargs_204135)
        
        
        # Call to assertEqual(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'result' (line 58)
        result_204139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 58)
        testsRun_204140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 25), result_204139, 'testsRun')
        int_204141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 42), 'int')
        # Processing the call keyword arguments (line 58)
        kwargs_204142 = {}
        # Getting the type of 'self' (line 58)
        self_204137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 58)
        assertEqual_204138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_204137, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 58)
        assertEqual_call_result_204143 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assertEqual_204138, *[testsRun_204140, int_204141], **kwargs_204142)
        
        
        # Call to assertEqual(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'result' (line 59)
        result_204146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 59)
        shouldStop_204147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 25), result_204146, 'shouldStop')
        # Getting the type of 'False' (line 59)
        False_204148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 44), 'False', False)
        # Processing the call keyword arguments (line 59)
        kwargs_204149 = {}
        # Getting the type of 'self' (line 59)
        self_204144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 59)
        assertEqual_204145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_204144, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 59)
        assertEqual_call_result_204150 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assertEqual_204145, *[shouldStop_204147, False_204148], **kwargs_204149)
        
        
        # Call to stopTest(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'test' (line 61)
        test_204153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'test', False)
        # Processing the call keyword arguments (line 61)
        kwargs_204154 = {}
        # Getting the type of 'result' (line 61)
        result_204151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'result', False)
        # Obtaining the member 'stopTest' of a type (line 61)
        stopTest_204152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), result_204151, 'stopTest')
        # Calling stopTest(args, kwargs) (line 61)
        stopTest_call_result_204155 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), stopTest_204152, *[test_204153], **kwargs_204154)
        
        
        # ################# End of 'test_startTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_startTest' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_204156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_startTest'
        return stypy_return_type_204156


    @norecursion
    def test_stopTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_stopTest'
        module_type_store = module_type_store.open_function_context('test_stopTest', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.test_stopTest')
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.test_stopTest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.test_stopTest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_stopTest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_stopTest(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 66)
        unittest_204157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 66)
        TestCase_204158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 18), unittest_204157, 'TestCase')

        class Foo(TestCase_204158, ):

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 67, 12, False)
                # Assigning a type to the variable 'self' (line 68)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_1.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_1.__dict__.__setitem__('stypy_function_name', 'Foo.test_1')
                Foo.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                pass
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 67)
                stypy_return_type_204159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204159)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_204159

        
        # Assigning a type to the variable 'Foo' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to Foo(...): (line 70)
        # Processing the call arguments (line 70)
        str_204161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 70)
        kwargs_204162 = {}
        # Getting the type of 'Foo' (line 70)
        Foo_204160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 70)
        Foo_call_result_204163 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), Foo_204160, *[str_204161], **kwargs_204162)
        
        # Assigning a type to the variable 'test' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'test', Foo_call_result_204163)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to TestResult(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_204166 = {}
        # Getting the type of 'unittest' (line 72)
        unittest_204164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 72)
        TestResult_204165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 17), unittest_204164, 'TestResult')
        # Calling TestResult(args, kwargs) (line 72)
        TestResult_call_result_204167 = invoke(stypy.reporting.localization.Localization(__file__, 72, 17), TestResult_204165, *[], **kwargs_204166)
        
        # Assigning a type to the variable 'result' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'result', TestResult_call_result_204167)
        
        # Call to startTest(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'test' (line 74)
        test_204170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'test', False)
        # Processing the call keyword arguments (line 74)
        kwargs_204171 = {}
        # Getting the type of 'result' (line 74)
        result_204168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'result', False)
        # Obtaining the member 'startTest' of a type (line 74)
        startTest_204169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), result_204168, 'startTest')
        # Calling startTest(args, kwargs) (line 74)
        startTest_call_result_204172 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), startTest_204169, *[test_204170], **kwargs_204171)
        
        
        # Call to assertTrue(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to wasSuccessful(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_204177 = {}
        # Getting the type of 'result' (line 76)
        result_204175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 76)
        wasSuccessful_204176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 24), result_204175, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 76)
        wasSuccessful_call_result_204178 = invoke(stypy.reporting.localization.Localization(__file__, 76, 24), wasSuccessful_204176, *[], **kwargs_204177)
        
        # Processing the call keyword arguments (line 76)
        kwargs_204179 = {}
        # Getting the type of 'self' (line 76)
        self_204173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 76)
        assertTrue_204174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_204173, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 76)
        assertTrue_call_result_204180 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assertTrue_204174, *[wasSuccessful_call_result_204178], **kwargs_204179)
        
        
        # Call to assertEqual(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to len(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'result' (line 77)
        result_204184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 77)
        errors_204185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 29), result_204184, 'errors')
        # Processing the call keyword arguments (line 77)
        kwargs_204186 = {}
        # Getting the type of 'len' (line 77)
        len_204183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'len', False)
        # Calling len(args, kwargs) (line 77)
        len_call_result_204187 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), len_204183, *[errors_204185], **kwargs_204186)
        
        int_204188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 45), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_204189 = {}
        # Getting the type of 'self' (line 77)
        self_204181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 77)
        assertEqual_204182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_204181, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 77)
        assertEqual_call_result_204190 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), assertEqual_204182, *[len_call_result_204187, int_204188], **kwargs_204189)
        
        
        # Call to assertEqual(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Call to len(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'result' (line 78)
        result_204194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 29), 'result', False)
        # Obtaining the member 'failures' of a type (line 78)
        failures_204195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 29), result_204194, 'failures')
        # Processing the call keyword arguments (line 78)
        kwargs_204196 = {}
        # Getting the type of 'len' (line 78)
        len_204193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'len', False)
        # Calling len(args, kwargs) (line 78)
        len_call_result_204197 = invoke(stypy.reporting.localization.Localization(__file__, 78, 25), len_204193, *[failures_204195], **kwargs_204196)
        
        int_204198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 47), 'int')
        # Processing the call keyword arguments (line 78)
        kwargs_204199 = {}
        # Getting the type of 'self' (line 78)
        self_204191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 78)
        assertEqual_204192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_204191, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 78)
        assertEqual_call_result_204200 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assertEqual_204192, *[len_call_result_204197, int_204198], **kwargs_204199)
        
        
        # Call to assertEqual(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'result' (line 79)
        result_204203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 79)
        testsRun_204204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), result_204203, 'testsRun')
        int_204205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 42), 'int')
        # Processing the call keyword arguments (line 79)
        kwargs_204206 = {}
        # Getting the type of 'self' (line 79)
        self_204201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 79)
        assertEqual_204202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_204201, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 79)
        assertEqual_call_result_204207 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assertEqual_204202, *[testsRun_204204, int_204205], **kwargs_204206)
        
        
        # Call to assertEqual(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'result' (line 80)
        result_204210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 80)
        shouldStop_204211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 25), result_204210, 'shouldStop')
        # Getting the type of 'False' (line 80)
        False_204212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'False', False)
        # Processing the call keyword arguments (line 80)
        kwargs_204213 = {}
        # Getting the type of 'self' (line 80)
        self_204208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 80)
        assertEqual_204209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_204208, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 80)
        assertEqual_call_result_204214 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assertEqual_204209, *[shouldStop_204211, False_204212], **kwargs_204213)
        
        
        # Call to stopTest(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'test' (line 82)
        test_204217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'test', False)
        # Processing the call keyword arguments (line 82)
        kwargs_204218 = {}
        # Getting the type of 'result' (line 82)
        result_204215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'result', False)
        # Obtaining the member 'stopTest' of a type (line 82)
        stopTest_204216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), result_204215, 'stopTest')
        # Calling stopTest(args, kwargs) (line 82)
        stopTest_call_result_204219 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), stopTest_204216, *[test_204217], **kwargs_204218)
        
        
        # Call to assertTrue(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to wasSuccessful(...): (line 85)
        # Processing the call keyword arguments (line 85)
        kwargs_204224 = {}
        # Getting the type of 'result' (line 85)
        result_204222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 85)
        wasSuccessful_204223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), result_204222, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 85)
        wasSuccessful_call_result_204225 = invoke(stypy.reporting.localization.Localization(__file__, 85, 24), wasSuccessful_204223, *[], **kwargs_204224)
        
        # Processing the call keyword arguments (line 85)
        kwargs_204226 = {}
        # Getting the type of 'self' (line 85)
        self_204220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 85)
        assertTrue_204221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_204220, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 85)
        assertTrue_call_result_204227 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assertTrue_204221, *[wasSuccessful_call_result_204225], **kwargs_204226)
        
        
        # Call to assertEqual(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to len(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'result' (line 86)
        result_204231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 86)
        errors_204232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 29), result_204231, 'errors')
        # Processing the call keyword arguments (line 86)
        kwargs_204233 = {}
        # Getting the type of 'len' (line 86)
        len_204230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'len', False)
        # Calling len(args, kwargs) (line 86)
        len_call_result_204234 = invoke(stypy.reporting.localization.Localization(__file__, 86, 25), len_204230, *[errors_204232], **kwargs_204233)
        
        int_204235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 45), 'int')
        # Processing the call keyword arguments (line 86)
        kwargs_204236 = {}
        # Getting the type of 'self' (line 86)
        self_204228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 86)
        assertEqual_204229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_204228, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 86)
        assertEqual_call_result_204237 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), assertEqual_204229, *[len_call_result_204234, int_204235], **kwargs_204236)
        
        
        # Call to assertEqual(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to len(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'result' (line 87)
        result_204241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 29), 'result', False)
        # Obtaining the member 'failures' of a type (line 87)
        failures_204242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 29), result_204241, 'failures')
        # Processing the call keyword arguments (line 87)
        kwargs_204243 = {}
        # Getting the type of 'len' (line 87)
        len_204240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'len', False)
        # Calling len(args, kwargs) (line 87)
        len_call_result_204244 = invoke(stypy.reporting.localization.Localization(__file__, 87, 25), len_204240, *[failures_204242], **kwargs_204243)
        
        int_204245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 47), 'int')
        # Processing the call keyword arguments (line 87)
        kwargs_204246 = {}
        # Getting the type of 'self' (line 87)
        self_204238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 87)
        assertEqual_204239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_204238, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 87)
        assertEqual_call_result_204247 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), assertEqual_204239, *[len_call_result_204244, int_204245], **kwargs_204246)
        
        
        # Call to assertEqual(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'result' (line 88)
        result_204250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 88)
        testsRun_204251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 25), result_204250, 'testsRun')
        int_204252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 42), 'int')
        # Processing the call keyword arguments (line 88)
        kwargs_204253 = {}
        # Getting the type of 'self' (line 88)
        self_204248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 88)
        assertEqual_204249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_204248, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 88)
        assertEqual_call_result_204254 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), assertEqual_204249, *[testsRun_204251, int_204252], **kwargs_204253)
        
        
        # Call to assertEqual(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'result' (line 89)
        result_204257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 89)
        shouldStop_204258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 25), result_204257, 'shouldStop')
        # Getting the type of 'False' (line 89)
        False_204259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), 'False', False)
        # Processing the call keyword arguments (line 89)
        kwargs_204260 = {}
        # Getting the type of 'self' (line 89)
        self_204255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 89)
        assertEqual_204256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_204255, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 89)
        assertEqual_call_result_204261 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assertEqual_204256, *[shouldStop_204258, False_204259], **kwargs_204260)
        
        
        # ################# End of 'test_stopTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_stopTest' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_204262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204262)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_stopTest'
        return stypy_return_type_204262


    @norecursion
    def test_startTestRun_stopTestRun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_startTestRun_stopTestRun'
        module_type_store = module_type_store.open_function_context('test_startTestRun_stopTestRun', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.test_startTestRun_stopTestRun')
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.test_startTestRun_stopTestRun.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.test_startTestRun_stopTestRun', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_startTestRun_stopTestRun', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_startTestRun_stopTestRun(...)' code ##################

        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to TestResult(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_204265 = {}
        # Getting the type of 'unittest' (line 93)
        unittest_204263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 93)
        TestResult_204264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), unittest_204263, 'TestResult')
        # Calling TestResult(args, kwargs) (line 93)
        TestResult_call_result_204266 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), TestResult_204264, *[], **kwargs_204265)
        
        # Assigning a type to the variable 'result' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'result', TestResult_call_result_204266)
        
        # Call to startTestRun(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_204269 = {}
        # Getting the type of 'result' (line 94)
        result_204267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'result', False)
        # Obtaining the member 'startTestRun' of a type (line 94)
        startTestRun_204268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), result_204267, 'startTestRun')
        # Calling startTestRun(args, kwargs) (line 94)
        startTestRun_call_result_204270 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), startTestRun_204268, *[], **kwargs_204269)
        
        
        # Call to stopTestRun(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_204273 = {}
        # Getting the type of 'result' (line 95)
        result_204271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'result', False)
        # Obtaining the member 'stopTestRun' of a type (line 95)
        stopTestRun_204272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), result_204271, 'stopTestRun')
        # Calling stopTestRun(args, kwargs) (line 95)
        stopTestRun_call_result_204274 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), stopTestRun_204272, *[], **kwargs_204273)
        
        
        # ################# End of 'test_startTestRun_stopTestRun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_startTestRun_stopTestRun' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_204275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204275)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_startTestRun_stopTestRun'
        return stypy_return_type_204275


    @norecursion
    def test_addSuccess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addSuccess'
        module_type_store = module_type_store.open_function_context('test_addSuccess', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.test_addSuccess')
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.test_addSuccess.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.test_addSuccess', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addSuccess', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addSuccess(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 117)
        unittest_204276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 117)
        TestCase_204277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 18), unittest_204276, 'TestCase')

        class Foo(TestCase_204277, ):

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 118, 12, False)
                # Assigning a type to the variable 'self' (line 119)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_1.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_1.__dict__.__setitem__('stypy_function_name', 'Foo.test_1')
                Foo.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                pass
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 118)
                stypy_return_type_204278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204278)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_204278

        
        # Assigning a type to the variable 'Foo' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to Foo(...): (line 121)
        # Processing the call arguments (line 121)
        str_204280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 121)
        kwargs_204281 = {}
        # Getting the type of 'Foo' (line 121)
        Foo_204279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 121)
        Foo_call_result_204282 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), Foo_204279, *[str_204280], **kwargs_204281)
        
        # Assigning a type to the variable 'test' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'test', Foo_call_result_204282)
        
        # Assigning a Call to a Name (line 123):
        
        # Assigning a Call to a Name (line 123):
        
        # Call to TestResult(...): (line 123)
        # Processing the call keyword arguments (line 123)
        kwargs_204285 = {}
        # Getting the type of 'unittest' (line 123)
        unittest_204283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 123)
        TestResult_204284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 17), unittest_204283, 'TestResult')
        # Calling TestResult(args, kwargs) (line 123)
        TestResult_call_result_204286 = invoke(stypy.reporting.localization.Localization(__file__, 123, 17), TestResult_204284, *[], **kwargs_204285)
        
        # Assigning a type to the variable 'result' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'result', TestResult_call_result_204286)
        
        # Call to startTest(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'test' (line 125)
        test_204289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'test', False)
        # Processing the call keyword arguments (line 125)
        kwargs_204290 = {}
        # Getting the type of 'result' (line 125)
        result_204287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'result', False)
        # Obtaining the member 'startTest' of a type (line 125)
        startTest_204288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), result_204287, 'startTest')
        # Calling startTest(args, kwargs) (line 125)
        startTest_call_result_204291 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), startTest_204288, *[test_204289], **kwargs_204290)
        
        
        # Call to addSuccess(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'test' (line 126)
        test_204294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'test', False)
        # Processing the call keyword arguments (line 126)
        kwargs_204295 = {}
        # Getting the type of 'result' (line 126)
        result_204292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'result', False)
        # Obtaining the member 'addSuccess' of a type (line 126)
        addSuccess_204293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), result_204292, 'addSuccess')
        # Calling addSuccess(args, kwargs) (line 126)
        addSuccess_call_result_204296 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), addSuccess_204293, *[test_204294], **kwargs_204295)
        
        
        # Call to stopTest(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'test' (line 127)
        test_204299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'test', False)
        # Processing the call keyword arguments (line 127)
        kwargs_204300 = {}
        # Getting the type of 'result' (line 127)
        result_204297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'result', False)
        # Obtaining the member 'stopTest' of a type (line 127)
        stopTest_204298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), result_204297, 'stopTest')
        # Calling stopTest(args, kwargs) (line 127)
        stopTest_call_result_204301 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), stopTest_204298, *[test_204299], **kwargs_204300)
        
        
        # Call to assertTrue(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to wasSuccessful(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_204306 = {}
        # Getting the type of 'result' (line 129)
        result_204304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 129)
        wasSuccessful_204305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 24), result_204304, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 129)
        wasSuccessful_call_result_204307 = invoke(stypy.reporting.localization.Localization(__file__, 129, 24), wasSuccessful_204305, *[], **kwargs_204306)
        
        # Processing the call keyword arguments (line 129)
        kwargs_204308 = {}
        # Getting the type of 'self' (line 129)
        self_204302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 129)
        assertTrue_204303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_204302, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 129)
        assertTrue_call_result_204309 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assertTrue_204303, *[wasSuccessful_call_result_204307], **kwargs_204308)
        
        
        # Call to assertEqual(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Call to len(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'result' (line 130)
        result_204313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 130)
        errors_204314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 29), result_204313, 'errors')
        # Processing the call keyword arguments (line 130)
        kwargs_204315 = {}
        # Getting the type of 'len' (line 130)
        len_204312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'len', False)
        # Calling len(args, kwargs) (line 130)
        len_call_result_204316 = invoke(stypy.reporting.localization.Localization(__file__, 130, 25), len_204312, *[errors_204314], **kwargs_204315)
        
        int_204317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 45), 'int')
        # Processing the call keyword arguments (line 130)
        kwargs_204318 = {}
        # Getting the type of 'self' (line 130)
        self_204310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 130)
        assertEqual_204311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_204310, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 130)
        assertEqual_call_result_204319 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), assertEqual_204311, *[len_call_result_204316, int_204317], **kwargs_204318)
        
        
        # Call to assertEqual(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Call to len(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'result' (line 131)
        result_204323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'result', False)
        # Obtaining the member 'failures' of a type (line 131)
        failures_204324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 29), result_204323, 'failures')
        # Processing the call keyword arguments (line 131)
        kwargs_204325 = {}
        # Getting the type of 'len' (line 131)
        len_204322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'len', False)
        # Calling len(args, kwargs) (line 131)
        len_call_result_204326 = invoke(stypy.reporting.localization.Localization(__file__, 131, 25), len_204322, *[failures_204324], **kwargs_204325)
        
        int_204327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 47), 'int')
        # Processing the call keyword arguments (line 131)
        kwargs_204328 = {}
        # Getting the type of 'self' (line 131)
        self_204320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 131)
        assertEqual_204321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_204320, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 131)
        assertEqual_call_result_204329 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assertEqual_204321, *[len_call_result_204326, int_204327], **kwargs_204328)
        
        
        # Call to assertEqual(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'result' (line 132)
        result_204332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 132)
        testsRun_204333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 25), result_204332, 'testsRun')
        int_204334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 42), 'int')
        # Processing the call keyword arguments (line 132)
        kwargs_204335 = {}
        # Getting the type of 'self' (line 132)
        self_204330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 132)
        assertEqual_204331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_204330, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 132)
        assertEqual_call_result_204336 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), assertEqual_204331, *[testsRun_204333, int_204334], **kwargs_204335)
        
        
        # Call to assertEqual(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'result' (line 133)
        result_204339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 133)
        shouldStop_204340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 25), result_204339, 'shouldStop')
        # Getting the type of 'False' (line 133)
        False_204341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 44), 'False', False)
        # Processing the call keyword arguments (line 133)
        kwargs_204342 = {}
        # Getting the type of 'self' (line 133)
        self_204337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 133)
        assertEqual_204338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_204337, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 133)
        assertEqual_call_result_204343 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assertEqual_204338, *[shouldStop_204340, False_204341], **kwargs_204342)
        
        
        # ################# End of 'test_addSuccess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addSuccess' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_204344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204344)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addSuccess'
        return stypy_return_type_204344


    @norecursion
    def test_addFailure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addFailure'
        module_type_store = module_type_store.open_function_context('test_addFailure', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.test_addFailure')
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.test_addFailure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.test_addFailure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addFailure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addFailure(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 156)
        unittest_204345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 156)
        TestCase_204346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 18), unittest_204345, 'TestCase')

        class Foo(TestCase_204346, ):

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 157, 12, False)
                # Assigning a type to the variable 'self' (line 158)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_1.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_1.__dict__.__setitem__('stypy_function_name', 'Foo.test_1')
                Foo.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                pass
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 157)
                stypy_return_type_204347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204347)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_204347

        
        # Assigning a type to the variable 'Foo' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to Foo(...): (line 160)
        # Processing the call arguments (line 160)
        str_204349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 160)
        kwargs_204350 = {}
        # Getting the type of 'Foo' (line 160)
        Foo_204348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 160)
        Foo_call_result_204351 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), Foo_204348, *[str_204349], **kwargs_204350)
        
        # Assigning a type to the variable 'test' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'test', Foo_call_result_204351)
        
        
        # SSA begins for try-except statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to fail(...): (line 162)
        # Processing the call arguments (line 162)
        str_204354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 22), 'str', 'foo')
        # Processing the call keyword arguments (line 162)
        kwargs_204355 = {}
        # Getting the type of 'test' (line 162)
        test_204352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'test', False)
        # Obtaining the member 'fail' of a type (line 162)
        fail_204353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), test_204352, 'fail')
        # Calling fail(args, kwargs) (line 162)
        fail_call_result_204356 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), fail_204353, *[str_204354], **kwargs_204355)
        
        # SSA branch for the except part of a try statement (line 161)
        # SSA branch for the except '<any exception>' branch of a try statement (line 161)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to exc_info(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_204359 = {}
        # Getting the type of 'sys' (line 164)
        sys_204357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 29), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 164)
        exc_info_204358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 29), sys_204357, 'exc_info')
        # Calling exc_info(args, kwargs) (line 164)
        exc_info_call_result_204360 = invoke(stypy.reporting.localization.Localization(__file__, 164, 29), exc_info_204358, *[], **kwargs_204359)
        
        # Assigning a type to the variable 'exc_info_tuple' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'exc_info_tuple', exc_info_call_result_204360)
        # SSA join for try-except statement (line 161)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to TestResult(...): (line 166)
        # Processing the call keyword arguments (line 166)
        kwargs_204363 = {}
        # Getting the type of 'unittest' (line 166)
        unittest_204361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 166)
        TestResult_204362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 17), unittest_204361, 'TestResult')
        # Calling TestResult(args, kwargs) (line 166)
        TestResult_call_result_204364 = invoke(stypy.reporting.localization.Localization(__file__, 166, 17), TestResult_204362, *[], **kwargs_204363)
        
        # Assigning a type to the variable 'result' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'result', TestResult_call_result_204364)
        
        # Call to startTest(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'test' (line 168)
        test_204367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'test', False)
        # Processing the call keyword arguments (line 168)
        kwargs_204368 = {}
        # Getting the type of 'result' (line 168)
        result_204365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'result', False)
        # Obtaining the member 'startTest' of a type (line 168)
        startTest_204366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), result_204365, 'startTest')
        # Calling startTest(args, kwargs) (line 168)
        startTest_call_result_204369 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), startTest_204366, *[test_204367], **kwargs_204368)
        
        
        # Call to addFailure(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'test' (line 169)
        test_204372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'test', False)
        # Getting the type of 'exc_info_tuple' (line 169)
        exc_info_tuple_204373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 32), 'exc_info_tuple', False)
        # Processing the call keyword arguments (line 169)
        kwargs_204374 = {}
        # Getting the type of 'result' (line 169)
        result_204370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'result', False)
        # Obtaining the member 'addFailure' of a type (line 169)
        addFailure_204371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), result_204370, 'addFailure')
        # Calling addFailure(args, kwargs) (line 169)
        addFailure_call_result_204375 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), addFailure_204371, *[test_204372, exc_info_tuple_204373], **kwargs_204374)
        
        
        # Call to stopTest(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'test' (line 170)
        test_204378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'test', False)
        # Processing the call keyword arguments (line 170)
        kwargs_204379 = {}
        # Getting the type of 'result' (line 170)
        result_204376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'result', False)
        # Obtaining the member 'stopTest' of a type (line 170)
        stopTest_204377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), result_204376, 'stopTest')
        # Calling stopTest(args, kwargs) (line 170)
        stopTest_call_result_204380 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), stopTest_204377, *[test_204378], **kwargs_204379)
        
        
        # Call to assertFalse(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Call to wasSuccessful(...): (line 172)
        # Processing the call keyword arguments (line 172)
        kwargs_204385 = {}
        # Getting the type of 'result' (line 172)
        result_204383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 172)
        wasSuccessful_204384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), result_204383, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 172)
        wasSuccessful_call_result_204386 = invoke(stypy.reporting.localization.Localization(__file__, 172, 25), wasSuccessful_204384, *[], **kwargs_204385)
        
        # Processing the call keyword arguments (line 172)
        kwargs_204387 = {}
        # Getting the type of 'self' (line 172)
        self_204381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 172)
        assertFalse_204382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_204381, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 172)
        assertFalse_call_result_204388 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), assertFalse_204382, *[wasSuccessful_call_result_204386], **kwargs_204387)
        
        
        # Call to assertEqual(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Call to len(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'result' (line 173)
        result_204392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 173)
        errors_204393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 29), result_204392, 'errors')
        # Processing the call keyword arguments (line 173)
        kwargs_204394 = {}
        # Getting the type of 'len' (line 173)
        len_204391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 25), 'len', False)
        # Calling len(args, kwargs) (line 173)
        len_call_result_204395 = invoke(stypy.reporting.localization.Localization(__file__, 173, 25), len_204391, *[errors_204393], **kwargs_204394)
        
        int_204396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 45), 'int')
        # Processing the call keyword arguments (line 173)
        kwargs_204397 = {}
        # Getting the type of 'self' (line 173)
        self_204389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 173)
        assertEqual_204390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_204389, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 173)
        assertEqual_call_result_204398 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), assertEqual_204390, *[len_call_result_204395, int_204396], **kwargs_204397)
        
        
        # Call to assertEqual(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Call to len(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'result' (line 174)
        result_204402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'result', False)
        # Obtaining the member 'failures' of a type (line 174)
        failures_204403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 29), result_204402, 'failures')
        # Processing the call keyword arguments (line 174)
        kwargs_204404 = {}
        # Getting the type of 'len' (line 174)
        len_204401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'len', False)
        # Calling len(args, kwargs) (line 174)
        len_call_result_204405 = invoke(stypy.reporting.localization.Localization(__file__, 174, 25), len_204401, *[failures_204403], **kwargs_204404)
        
        int_204406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 47), 'int')
        # Processing the call keyword arguments (line 174)
        kwargs_204407 = {}
        # Getting the type of 'self' (line 174)
        self_204399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 174)
        assertEqual_204400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_204399, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 174)
        assertEqual_call_result_204408 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), assertEqual_204400, *[len_call_result_204405, int_204406], **kwargs_204407)
        
        
        # Call to assertEqual(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'result' (line 175)
        result_204411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 175)
        testsRun_204412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 25), result_204411, 'testsRun')
        int_204413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 42), 'int')
        # Processing the call keyword arguments (line 175)
        kwargs_204414 = {}
        # Getting the type of 'self' (line 175)
        self_204409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 175)
        assertEqual_204410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_204409, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 175)
        assertEqual_call_result_204415 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), assertEqual_204410, *[testsRun_204412, int_204413], **kwargs_204414)
        
        
        # Call to assertEqual(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'result' (line 176)
        result_204418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 176)
        shouldStop_204419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), result_204418, 'shouldStop')
        # Getting the type of 'False' (line 176)
        False_204420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 44), 'False', False)
        # Processing the call keyword arguments (line 176)
        kwargs_204421 = {}
        # Getting the type of 'self' (line 176)
        self_204416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 176)
        assertEqual_204417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_204416, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 176)
        assertEqual_call_result_204422 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assertEqual_204417, *[shouldStop_204419, False_204420], **kwargs_204421)
        
        
        # Assigning a Subscript to a Tuple (line 178):
        
        # Assigning a Subscript to a Name (line 178):
        
        # Obtaining the type of the subscript
        int_204423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 8), 'int')
        
        # Obtaining the type of the subscript
        int_204424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 51), 'int')
        # Getting the type of 'result' (line 178)
        result_204425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 35), 'result')
        # Obtaining the member 'failures' of a type (line 178)
        failures_204426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 35), result_204425, 'failures')
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___204427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 35), failures_204426, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_204428 = invoke(stypy.reporting.localization.Localization(__file__, 178, 35), getitem___204427, int_204424)
        
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___204429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), subscript_call_result_204428, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_204430 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), getitem___204429, int_204423)
        
        # Assigning a type to the variable 'tuple_var_assignment_204010' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'tuple_var_assignment_204010', subscript_call_result_204430)
        
        # Assigning a Subscript to a Name (line 178):
        
        # Obtaining the type of the subscript
        int_204431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 8), 'int')
        
        # Obtaining the type of the subscript
        int_204432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 51), 'int')
        # Getting the type of 'result' (line 178)
        result_204433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 35), 'result')
        # Obtaining the member 'failures' of a type (line 178)
        failures_204434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 35), result_204433, 'failures')
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___204435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 35), failures_204434, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_204436 = invoke(stypy.reporting.localization.Localization(__file__, 178, 35), getitem___204435, int_204432)
        
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___204437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), subscript_call_result_204436, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_204438 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), getitem___204437, int_204431)
        
        # Assigning a type to the variable 'tuple_var_assignment_204011' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'tuple_var_assignment_204011', subscript_call_result_204438)
        
        # Assigning a Name to a Name (line 178):
        # Getting the type of 'tuple_var_assignment_204010' (line 178)
        tuple_var_assignment_204010_204439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'tuple_var_assignment_204010')
        # Assigning a type to the variable 'test_case' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'test_case', tuple_var_assignment_204010_204439)
        
        # Assigning a Name to a Name (line 178):
        # Getting the type of 'tuple_var_assignment_204011' (line 178)
        tuple_var_assignment_204011_204440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'tuple_var_assignment_204011')
        # Assigning a type to the variable 'formatted_exc' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'formatted_exc', tuple_var_assignment_204011_204440)
        
        # Call to assertIs(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'test_case' (line 179)
        test_case_204443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'test_case', False)
        # Getting the type of 'test' (line 179)
        test_204444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 33), 'test', False)
        # Processing the call keyword arguments (line 179)
        kwargs_204445 = {}
        # Getting the type of 'self' (line 179)
        self_204441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 179)
        assertIs_204442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_204441, 'assertIs')
        # Calling assertIs(args, kwargs) (line 179)
        assertIs_call_result_204446 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), assertIs_204442, *[test_case_204443, test_204444], **kwargs_204445)
        
        
        # Call to assertIsInstance(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'formatted_exc' (line 180)
        formatted_exc_204449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'formatted_exc', False)
        # Getting the type of 'str' (line 180)
        str_204450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 45), 'str', False)
        # Processing the call keyword arguments (line 180)
        kwargs_204451 = {}
        # Getting the type of 'self' (line 180)
        self_204447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 180)
        assertIsInstance_204448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_204447, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 180)
        assertIsInstance_call_result_204452 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), assertIsInstance_204448, *[formatted_exc_204449, str_204450], **kwargs_204451)
        
        
        # ################# End of 'test_addFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_204453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204453)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addFailure'
        return stypy_return_type_204453


    @norecursion
    def test_addError(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addError'
        module_type_store = module_type_store.open_function_context('test_addError', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.test_addError')
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.test_addError.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.test_addError', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addError', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addError(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 204)
        unittest_204454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 204)
        TestCase_204455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 18), unittest_204454, 'TestCase')

        class Foo(TestCase_204455, ):

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 205, 12, False)
                # Assigning a type to the variable 'self' (line 206)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_1.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_1.__dict__.__setitem__('stypy_function_name', 'Foo.test_1')
                Foo.test_1.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_1(...)' code ##################

                pass
                
                # ################# End of 'test_1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_1' in the type store
                # Getting the type of 'stypy_return_type' (line 205)
                stypy_return_type_204456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204456)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_204456

        
        # Assigning a type to the variable 'Foo' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to Foo(...): (line 208)
        # Processing the call arguments (line 208)
        str_204458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 19), 'str', 'test_1')
        # Processing the call keyword arguments (line 208)
        kwargs_204459 = {}
        # Getting the type of 'Foo' (line 208)
        Foo_204457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 208)
        Foo_call_result_204460 = invoke(stypy.reporting.localization.Localization(__file__, 208, 15), Foo_204457, *[str_204458], **kwargs_204459)
        
        # Assigning a type to the variable 'test' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'test', Foo_call_result_204460)
        
        
        # SSA begins for try-except statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to TypeError(...): (line 210)
        # Processing the call keyword arguments (line 210)
        kwargs_204462 = {}
        # Getting the type of 'TypeError' (line 210)
        TypeError_204461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 210)
        TypeError_call_result_204463 = invoke(stypy.reporting.localization.Localization(__file__, 210, 18), TypeError_204461, *[], **kwargs_204462)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 210, 12), TypeError_call_result_204463, 'raise parameter', BaseException)
        # SSA branch for the except part of a try statement (line 209)
        # SSA branch for the except '<any exception>' branch of a try statement (line 209)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to exc_info(...): (line 212)
        # Processing the call keyword arguments (line 212)
        kwargs_204466 = {}
        # Getting the type of 'sys' (line 212)
        sys_204464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 212)
        exc_info_204465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 29), sys_204464, 'exc_info')
        # Calling exc_info(args, kwargs) (line 212)
        exc_info_call_result_204467 = invoke(stypy.reporting.localization.Localization(__file__, 212, 29), exc_info_204465, *[], **kwargs_204466)
        
        # Assigning a type to the variable 'exc_info_tuple' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'exc_info_tuple', exc_info_call_result_204467)
        # SSA join for try-except statement (line 209)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to TestResult(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_204470 = {}
        # Getting the type of 'unittest' (line 214)
        unittest_204468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 214)
        TestResult_204469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 17), unittest_204468, 'TestResult')
        # Calling TestResult(args, kwargs) (line 214)
        TestResult_call_result_204471 = invoke(stypy.reporting.localization.Localization(__file__, 214, 17), TestResult_204469, *[], **kwargs_204470)
        
        # Assigning a type to the variable 'result' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'result', TestResult_call_result_204471)
        
        # Call to startTest(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'test' (line 216)
        test_204474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'test', False)
        # Processing the call keyword arguments (line 216)
        kwargs_204475 = {}
        # Getting the type of 'result' (line 216)
        result_204472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'result', False)
        # Obtaining the member 'startTest' of a type (line 216)
        startTest_204473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), result_204472, 'startTest')
        # Calling startTest(args, kwargs) (line 216)
        startTest_call_result_204476 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), startTest_204473, *[test_204474], **kwargs_204475)
        
        
        # Call to addError(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'test' (line 217)
        test_204479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'test', False)
        # Getting the type of 'exc_info_tuple' (line 217)
        exc_info_tuple_204480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 30), 'exc_info_tuple', False)
        # Processing the call keyword arguments (line 217)
        kwargs_204481 = {}
        # Getting the type of 'result' (line 217)
        result_204477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'result', False)
        # Obtaining the member 'addError' of a type (line 217)
        addError_204478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), result_204477, 'addError')
        # Calling addError(args, kwargs) (line 217)
        addError_call_result_204482 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), addError_204478, *[test_204479, exc_info_tuple_204480], **kwargs_204481)
        
        
        # Call to stopTest(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'test' (line 218)
        test_204485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'test', False)
        # Processing the call keyword arguments (line 218)
        kwargs_204486 = {}
        # Getting the type of 'result' (line 218)
        result_204483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'result', False)
        # Obtaining the member 'stopTest' of a type (line 218)
        stopTest_204484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), result_204483, 'stopTest')
        # Calling stopTest(args, kwargs) (line 218)
        stopTest_call_result_204487 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), stopTest_204484, *[test_204485], **kwargs_204486)
        
        
        # Call to assertFalse(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to wasSuccessful(...): (line 220)
        # Processing the call keyword arguments (line 220)
        kwargs_204492 = {}
        # Getting the type of 'result' (line 220)
        result_204490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'result', False)
        # Obtaining the member 'wasSuccessful' of a type (line 220)
        wasSuccessful_204491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 25), result_204490, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 220)
        wasSuccessful_call_result_204493 = invoke(stypy.reporting.localization.Localization(__file__, 220, 25), wasSuccessful_204491, *[], **kwargs_204492)
        
        # Processing the call keyword arguments (line 220)
        kwargs_204494 = {}
        # Getting the type of 'self' (line 220)
        self_204488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 220)
        assertFalse_204489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_204488, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 220)
        assertFalse_call_result_204495 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), assertFalse_204489, *[wasSuccessful_call_result_204493], **kwargs_204494)
        
        
        # Call to assertEqual(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Call to len(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'result' (line 221)
        result_204499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 221)
        errors_204500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 29), result_204499, 'errors')
        # Processing the call keyword arguments (line 221)
        kwargs_204501 = {}
        # Getting the type of 'len' (line 221)
        len_204498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 25), 'len', False)
        # Calling len(args, kwargs) (line 221)
        len_call_result_204502 = invoke(stypy.reporting.localization.Localization(__file__, 221, 25), len_204498, *[errors_204500], **kwargs_204501)
        
        int_204503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 45), 'int')
        # Processing the call keyword arguments (line 221)
        kwargs_204504 = {}
        # Getting the type of 'self' (line 221)
        self_204496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 221)
        assertEqual_204497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_204496, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 221)
        assertEqual_call_result_204505 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), assertEqual_204497, *[len_call_result_204502, int_204503], **kwargs_204504)
        
        
        # Call to assertEqual(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Call to len(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'result' (line 222)
        result_204509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 29), 'result', False)
        # Obtaining the member 'failures' of a type (line 222)
        failures_204510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 29), result_204509, 'failures')
        # Processing the call keyword arguments (line 222)
        kwargs_204511 = {}
        # Getting the type of 'len' (line 222)
        len_204508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'len', False)
        # Calling len(args, kwargs) (line 222)
        len_call_result_204512 = invoke(stypy.reporting.localization.Localization(__file__, 222, 25), len_204508, *[failures_204510], **kwargs_204511)
        
        int_204513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 47), 'int')
        # Processing the call keyword arguments (line 222)
        kwargs_204514 = {}
        # Getting the type of 'self' (line 222)
        self_204506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 222)
        assertEqual_204507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_204506, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 222)
        assertEqual_call_result_204515 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), assertEqual_204507, *[len_call_result_204512, int_204513], **kwargs_204514)
        
        
        # Call to assertEqual(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'result' (line 223)
        result_204518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 223)
        testsRun_204519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 25), result_204518, 'testsRun')
        int_204520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 42), 'int')
        # Processing the call keyword arguments (line 223)
        kwargs_204521 = {}
        # Getting the type of 'self' (line 223)
        self_204516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 223)
        assertEqual_204517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_204516, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 223)
        assertEqual_call_result_204522 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), assertEqual_204517, *[testsRun_204519, int_204520], **kwargs_204521)
        
        
        # Call to assertEqual(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'result' (line 224)
        result_204525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 25), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 224)
        shouldStop_204526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 25), result_204525, 'shouldStop')
        # Getting the type of 'False' (line 224)
        False_204527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), 'False', False)
        # Processing the call keyword arguments (line 224)
        kwargs_204528 = {}
        # Getting the type of 'self' (line 224)
        self_204523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 224)
        assertEqual_204524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_204523, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 224)
        assertEqual_call_result_204529 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), assertEqual_204524, *[shouldStop_204526, False_204527], **kwargs_204528)
        
        
        # Assigning a Subscript to a Tuple (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_204530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Obtaining the type of the subscript
        int_204531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 49), 'int')
        # Getting the type of 'result' (line 226)
        result_204532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 35), 'result')
        # Obtaining the member 'errors' of a type (line 226)
        errors_204533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 35), result_204532, 'errors')
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___204534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 35), errors_204533, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_204535 = invoke(stypy.reporting.localization.Localization(__file__, 226, 35), getitem___204534, int_204531)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___204536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), subscript_call_result_204535, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_204537 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___204536, int_204530)
        
        # Assigning a type to the variable 'tuple_var_assignment_204012' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_204012', subscript_call_result_204537)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_204538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Obtaining the type of the subscript
        int_204539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 49), 'int')
        # Getting the type of 'result' (line 226)
        result_204540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 35), 'result')
        # Obtaining the member 'errors' of a type (line 226)
        errors_204541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 35), result_204540, 'errors')
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___204542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 35), errors_204541, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_204543 = invoke(stypy.reporting.localization.Localization(__file__, 226, 35), getitem___204542, int_204539)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___204544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), subscript_call_result_204543, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_204545 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___204544, int_204538)
        
        # Assigning a type to the variable 'tuple_var_assignment_204013' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_204013', subscript_call_result_204545)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_204012' (line 226)
        tuple_var_assignment_204012_204546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_204012')
        # Assigning a type to the variable 'test_case' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'test_case', tuple_var_assignment_204012_204546)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_204013' (line 226)
        tuple_var_assignment_204013_204547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_204013')
        # Assigning a type to the variable 'formatted_exc' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'formatted_exc', tuple_var_assignment_204013_204547)
        
        # Call to assertIs(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'test_case' (line 227)
        test_case_204550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'test_case', False)
        # Getting the type of 'test' (line 227)
        test_204551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'test', False)
        # Processing the call keyword arguments (line 227)
        kwargs_204552 = {}
        # Getting the type of 'self' (line 227)
        self_204548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 227)
        assertIs_204549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_204548, 'assertIs')
        # Calling assertIs(args, kwargs) (line 227)
        assertIs_call_result_204553 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), assertIs_204549, *[test_case_204550, test_204551], **kwargs_204552)
        
        
        # Call to assertIsInstance(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'formatted_exc' (line 228)
        formatted_exc_204556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'formatted_exc', False)
        # Getting the type of 'str' (line 228)
        str_204557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 45), 'str', False)
        # Processing the call keyword arguments (line 228)
        kwargs_204558 = {}
        # Getting the type of 'self' (line 228)
        self_204554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 228)
        assertIsInstance_204555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_204554, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 228)
        assertIsInstance_call_result_204559 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), assertIsInstance_204555, *[formatted_exc_204556, str_204557], **kwargs_204558)
        
        
        # ################# End of 'test_addError(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addError' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_204560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204560)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addError'
        return stypy_return_type_204560


    @norecursion
    def testGetDescriptionWithoutDocstring(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testGetDescriptionWithoutDocstring'
        module_type_store = module_type_store.open_function_context('testGetDescriptionWithoutDocstring', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.testGetDescriptionWithoutDocstring')
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.testGetDescriptionWithoutDocstring.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.testGetDescriptionWithoutDocstring', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testGetDescriptionWithoutDocstring', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testGetDescriptionWithoutDocstring(...)' code ##################

        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to TextTestResult(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'None' (line 231)
        None_204563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 41), 'None', False)
        # Getting the type of 'True' (line 231)
        True_204564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 47), 'True', False)
        int_204565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 53), 'int')
        # Processing the call keyword arguments (line 231)
        kwargs_204566 = {}
        # Getting the type of 'unittest' (line 231)
        unittest_204561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'unittest', False)
        # Obtaining the member 'TextTestResult' of a type (line 231)
        TextTestResult_204562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 17), unittest_204561, 'TextTestResult')
        # Calling TextTestResult(args, kwargs) (line 231)
        TextTestResult_call_result_204567 = invoke(stypy.reporting.localization.Localization(__file__, 231, 17), TextTestResult_204562, *[None_204563, True_204564, int_204565], **kwargs_204566)
        
        # Assigning a type to the variable 'result' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'result', TextTestResult_call_result_204567)
        
        # Call to assertEqual(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to getDescription(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'self' (line 233)
        self_204572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 38), 'self', False)
        # Processing the call keyword arguments (line 233)
        kwargs_204573 = {}
        # Getting the type of 'result' (line 233)
        result_204570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'result', False)
        # Obtaining the member 'getDescription' of a type (line 233)
        getDescription_204571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), result_204570, 'getDescription')
        # Calling getDescription(args, kwargs) (line 233)
        getDescription_call_result_204574 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), getDescription_204571, *[self_204572], **kwargs_204573)
        
        str_204575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 16), 'str', 'testGetDescriptionWithoutDocstring (')
        # Getting the type of '__name__' (line 234)
        name___204576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 57), '__name__', False)
        # Applying the binary operator '+' (line 234)
        result_add_204577 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 16), '+', str_204575, name___204576)
        
        str_204578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 16), 'str', '.Test_TestResult)')
        # Applying the binary operator '+' (line 234)
        result_add_204579 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 66), '+', result_add_204577, str_204578)
        
        # Processing the call keyword arguments (line 232)
        kwargs_204580 = {}
        # Getting the type of 'self' (line 232)
        self_204568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 232)
        assertEqual_204569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_204568, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 232)
        assertEqual_call_result_204581 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), assertEqual_204569, *[getDescription_call_result_204574, result_add_204579], **kwargs_204580)
        
        
        # ################# End of 'testGetDescriptionWithoutDocstring(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testGetDescriptionWithoutDocstring' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_204582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204582)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testGetDescriptionWithoutDocstring'
        return stypy_return_type_204582


    @norecursion
    def testGetDescriptionWithOneLineDocstring(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testGetDescriptionWithOneLineDocstring'
        module_type_store = module_type_store.open_function_context('testGetDescriptionWithOneLineDocstring', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.testGetDescriptionWithOneLineDocstring')
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.testGetDescriptionWithOneLineDocstring.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.testGetDescriptionWithOneLineDocstring', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testGetDescriptionWithOneLineDocstring', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testGetDescriptionWithOneLineDocstring(...)' code ##################

        str_204583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'str', 'Tests getDescription() for a method with a docstring.')
        
        # Assigning a Call to a Name (line 241):
        
        # Assigning a Call to a Name (line 241):
        
        # Call to TextTestResult(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'None' (line 241)
        None_204586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 41), 'None', False)
        # Getting the type of 'True' (line 241)
        True_204587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 47), 'True', False)
        int_204588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 53), 'int')
        # Processing the call keyword arguments (line 241)
        kwargs_204589 = {}
        # Getting the type of 'unittest' (line 241)
        unittest_204584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'unittest', False)
        # Obtaining the member 'TextTestResult' of a type (line 241)
        TextTestResult_204585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 17), unittest_204584, 'TextTestResult')
        # Calling TextTestResult(args, kwargs) (line 241)
        TextTestResult_call_result_204590 = invoke(stypy.reporting.localization.Localization(__file__, 241, 17), TextTestResult_204585, *[None_204586, True_204587, int_204588], **kwargs_204589)
        
        # Assigning a type to the variable 'result' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'result', TextTestResult_call_result_204590)
        
        # Call to assertEqual(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Call to getDescription(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'self' (line 243)
        self_204595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 38), 'self', False)
        # Processing the call keyword arguments (line 243)
        kwargs_204596 = {}
        # Getting the type of 'result' (line 243)
        result_204593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'result', False)
        # Obtaining the member 'getDescription' of a type (line 243)
        getDescription_204594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 16), result_204593, 'getDescription')
        # Calling getDescription(args, kwargs) (line 243)
        getDescription_call_result_204597 = invoke(stypy.reporting.localization.Localization(__file__, 243, 16), getDescription_204594, *[self_204595], **kwargs_204596)
        
        str_204598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 16), 'str', 'testGetDescriptionWithOneLineDocstring (')
        # Getting the type of '__name__' (line 245)
        name___204599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 22), '__name__', False)
        # Applying the binary operator '+' (line 244)
        result_add_204600 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 16), '+', str_204598, name___204599)
        
        str_204601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 33), 'str', '.Test_TestResult)\nTests getDescription() for a method with a docstring.')
        # Applying the binary operator '+' (line 245)
        result_add_204602 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 31), '+', result_add_204600, str_204601)
        
        # Processing the call keyword arguments (line 242)
        kwargs_204603 = {}
        # Getting the type of 'self' (line 242)
        self_204591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 242)
        assertEqual_204592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_204591, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 242)
        assertEqual_call_result_204604 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assertEqual_204592, *[getDescription_call_result_204597, result_add_204602], **kwargs_204603)
        
        
        # ################# End of 'testGetDescriptionWithOneLineDocstring(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testGetDescriptionWithOneLineDocstring' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_204605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204605)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testGetDescriptionWithOneLineDocstring'
        return stypy_return_type_204605


    @norecursion
    def testGetDescriptionWithMultiLineDocstring(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testGetDescriptionWithMultiLineDocstring'
        module_type_store = module_type_store.open_function_context('testGetDescriptionWithMultiLineDocstring', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.testGetDescriptionWithMultiLineDocstring')
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.testGetDescriptionWithMultiLineDocstring.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.testGetDescriptionWithMultiLineDocstring', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testGetDescriptionWithMultiLineDocstring', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testGetDescriptionWithMultiLineDocstring(...)' code ##################

        str_204606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', 'Tests getDescription() for a method with a longer docstring.\n        The second line of the docstring.\n        ')
        
        # Assigning a Call to a Name (line 254):
        
        # Assigning a Call to a Name (line 254):
        
        # Call to TextTestResult(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'None' (line 254)
        None_204609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 41), 'None', False)
        # Getting the type of 'True' (line 254)
        True_204610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 47), 'True', False)
        int_204611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 53), 'int')
        # Processing the call keyword arguments (line 254)
        kwargs_204612 = {}
        # Getting the type of 'unittest' (line 254)
        unittest_204607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'unittest', False)
        # Obtaining the member 'TextTestResult' of a type (line 254)
        TextTestResult_204608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 17), unittest_204607, 'TextTestResult')
        # Calling TextTestResult(args, kwargs) (line 254)
        TextTestResult_call_result_204613 = invoke(stypy.reporting.localization.Localization(__file__, 254, 17), TextTestResult_204608, *[None_204609, True_204610, int_204611], **kwargs_204612)
        
        # Assigning a type to the variable 'result' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'result', TextTestResult_call_result_204613)
        
        # Call to assertEqual(...): (line 255)
        # Processing the call arguments (line 255)
        
        # Call to getDescription(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'self' (line 256)
        self_204618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 38), 'self', False)
        # Processing the call keyword arguments (line 256)
        kwargs_204619 = {}
        # Getting the type of 'result' (line 256)
        result_204616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'result', False)
        # Obtaining the member 'getDescription' of a type (line 256)
        getDescription_204617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 16), result_204616, 'getDescription')
        # Calling getDescription(args, kwargs) (line 256)
        getDescription_call_result_204620 = invoke(stypy.reporting.localization.Localization(__file__, 256, 16), getDescription_204617, *[self_204618], **kwargs_204619)
        
        str_204621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 16), 'str', 'testGetDescriptionWithMultiLineDocstring (')
        # Getting the type of '__name__' (line 258)
        name___204622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 22), '__name__', False)
        # Applying the binary operator '+' (line 257)
        result_add_204623 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 16), '+', str_204621, name___204622)
        
        str_204624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 33), 'str', '.Test_TestResult)\nTests getDescription() for a method with a longer docstring.')
        # Applying the binary operator '+' (line 258)
        result_add_204625 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 31), '+', result_add_204623, str_204624)
        
        # Processing the call keyword arguments (line 255)
        kwargs_204626 = {}
        # Getting the type of 'self' (line 255)
        self_204614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 255)
        assertEqual_204615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_204614, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 255)
        assertEqual_call_result_204627 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assertEqual_204615, *[getDescription_call_result_204620, result_add_204625], **kwargs_204626)
        
        
        # ################# End of 'testGetDescriptionWithMultiLineDocstring(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testGetDescriptionWithMultiLineDocstring' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_204628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204628)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testGetDescriptionWithMultiLineDocstring'
        return stypy_return_type_204628


    @norecursion
    def testStackFrameTrimming(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testStackFrameTrimming'
        module_type_store = module_type_store.open_function_context('testStackFrameTrimming', 262, 4, False)
        # Assigning a type to the variable 'self' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.testStackFrameTrimming')
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.testStackFrameTrimming.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.testStackFrameTrimming', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testStackFrameTrimming', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testStackFrameTrimming(...)' code ##################

        # Declaration of the 'Frame' class

        class Frame(object, ):
            # Declaration of the 'tb_frame' class

            class tb_frame(object, ):
                
                # Assigning a Dict to a Name (line 265):
                
                # Assigning a Dict to a Name (line 265):
                
                # Obtaining an instance of the builtin type 'dict' (line 265)
                dict_204629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 28), 'dict')
                # Adding type elements to the builtin type 'dict' instance (line 265)
                
                # Assigning a type to the variable 'f_globals' (line 265)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'f_globals', dict_204629)
            
            # Assigning a type to the variable 'tb_frame' (line 264)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'tb_frame', tb_frame)
        
        # Assigning a type to the variable 'Frame' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'Frame', Frame)
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to TestResult(...): (line 266)
        # Processing the call keyword arguments (line 266)
        kwargs_204632 = {}
        # Getting the type of 'unittest' (line 266)
        unittest_204630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 266)
        TestResult_204631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 17), unittest_204630, 'TestResult')
        # Calling TestResult(args, kwargs) (line 266)
        TestResult_call_result_204633 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), TestResult_204631, *[], **kwargs_204632)
        
        # Assigning a type to the variable 'result' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'result', TestResult_call_result_204633)
        
        # Call to assertFalse(...): (line 267)
        # Processing the call arguments (line 267)
        
        # Call to _is_relevant_tb_level(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'Frame' (line 267)
        Frame_204638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 54), 'Frame', False)
        # Processing the call keyword arguments (line 267)
        kwargs_204639 = {}
        # Getting the type of 'result' (line 267)
        result_204636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'result', False)
        # Obtaining the member '_is_relevant_tb_level' of a type (line 267)
        _is_relevant_tb_level_204637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 25), result_204636, '_is_relevant_tb_level')
        # Calling _is_relevant_tb_level(args, kwargs) (line 267)
        _is_relevant_tb_level_call_result_204640 = invoke(stypy.reporting.localization.Localization(__file__, 267, 25), _is_relevant_tb_level_204637, *[Frame_204638], **kwargs_204639)
        
        # Processing the call keyword arguments (line 267)
        kwargs_204641 = {}
        # Getting the type of 'self' (line 267)
        self_204634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 267)
        assertFalse_204635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_204634, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 267)
        assertFalse_call_result_204642 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), assertFalse_204635, *[_is_relevant_tb_level_call_result_204640], **kwargs_204641)
        
        
        # Assigning a Name to a Subscript (line 269):
        
        # Assigning a Name to a Subscript (line 269):
        # Getting the type of 'True' (line 269)
        True_204643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 49), 'True')
        # Getting the type of 'Frame' (line 269)
        Frame_204644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'Frame')
        # Obtaining the member 'tb_frame' of a type (line 269)
        tb_frame_204645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), Frame_204644, 'tb_frame')
        # Obtaining the member 'f_globals' of a type (line 269)
        f_globals_204646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), tb_frame_204645, 'f_globals')
        str_204647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 33), 'str', '__unittest')
        # Storing an element on a container (line 269)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 8), f_globals_204646, (str_204647, True_204643))
        
        # Call to assertTrue(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Call to _is_relevant_tb_level(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'Frame' (line 270)
        Frame_204652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 53), 'Frame', False)
        # Processing the call keyword arguments (line 270)
        kwargs_204653 = {}
        # Getting the type of 'result' (line 270)
        result_204650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 24), 'result', False)
        # Obtaining the member '_is_relevant_tb_level' of a type (line 270)
        _is_relevant_tb_level_204651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 24), result_204650, '_is_relevant_tb_level')
        # Calling _is_relevant_tb_level(args, kwargs) (line 270)
        _is_relevant_tb_level_call_result_204654 = invoke(stypy.reporting.localization.Localization(__file__, 270, 24), _is_relevant_tb_level_204651, *[Frame_204652], **kwargs_204653)
        
        # Processing the call keyword arguments (line 270)
        kwargs_204655 = {}
        # Getting the type of 'self' (line 270)
        self_204648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 270)
        assertTrue_204649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_204648, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 270)
        assertTrue_call_result_204656 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), assertTrue_204649, *[_is_relevant_tb_level_call_result_204654], **kwargs_204655)
        
        
        # ################# End of 'testStackFrameTrimming(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testStackFrameTrimming' in the type store
        # Getting the type of 'stypy_return_type' (line 262)
        stypy_return_type_204657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204657)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testStackFrameTrimming'
        return stypy_return_type_204657


    @norecursion
    def testFailFast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testFailFast'
        module_type_store = module_type_store.open_function_context('testFailFast', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.testFailFast')
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.testFailFast.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.testFailFast', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testFailFast', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testFailFast(...)' code ##################

        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to TestResult(...): (line 273)
        # Processing the call keyword arguments (line 273)
        kwargs_204660 = {}
        # Getting the type of 'unittest' (line 273)
        unittest_204658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 273)
        TestResult_204659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 17), unittest_204658, 'TestResult')
        # Calling TestResult(args, kwargs) (line 273)
        TestResult_call_result_204661 = invoke(stypy.reporting.localization.Localization(__file__, 273, 17), TestResult_204659, *[], **kwargs_204660)
        
        # Assigning a type to the variable 'result' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'result', TestResult_call_result_204661)
        
        # Assigning a Lambda to a Attribute (line 274):
        
        # Assigning a Lambda to a Attribute (line 274):

        @norecursion
        def _stypy_temp_lambda_91(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_91'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_91', 274, 37, True)
            # Passed parameters checking function
            _stypy_temp_lambda_91.stypy_localization = localization
            _stypy_temp_lambda_91.stypy_type_of_self = None
            _stypy_temp_lambda_91.stypy_type_store = module_type_store
            _stypy_temp_lambda_91.stypy_function_name = '_stypy_temp_lambda_91'
            _stypy_temp_lambda_91.stypy_param_names_list = []
            _stypy_temp_lambda_91.stypy_varargs_param_name = '_'
            _stypy_temp_lambda_91.stypy_kwargs_param_name = None
            _stypy_temp_lambda_91.stypy_call_defaults = defaults
            _stypy_temp_lambda_91.stypy_call_varargs = varargs
            _stypy_temp_lambda_91.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_91', [], '_', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_91', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            str_204662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 48), 'str', '')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), 'stypy_return_type', str_204662)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_91' in the type store
            # Getting the type of 'stypy_return_type' (line 274)
            stypy_return_type_204663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_204663)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_91'
            return stypy_return_type_204663

        # Assigning a type to the variable '_stypy_temp_lambda_91' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), '_stypy_temp_lambda_91', _stypy_temp_lambda_91)
        # Getting the type of '_stypy_temp_lambda_91' (line 274)
        _stypy_temp_lambda_91_204664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), '_stypy_temp_lambda_91')
        # Getting the type of 'result' (line 274)
        result_204665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'result')
        # Setting the type of the member '_exc_info_to_string' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), result_204665, '_exc_info_to_string', _stypy_temp_lambda_91_204664)
        
        # Assigning a Name to a Attribute (line 275):
        
        # Assigning a Name to a Attribute (line 275):
        # Getting the type of 'True' (line 275)
        True_204666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'True')
        # Getting the type of 'result' (line 275)
        result_204667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'result')
        # Setting the type of the member 'failfast' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), result_204667, 'failfast', True_204666)
        
        # Call to addError(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'None' (line 276)
        None_204670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 24), 'None', False)
        # Getting the type of 'None' (line 276)
        None_204671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 30), 'None', False)
        # Processing the call keyword arguments (line 276)
        kwargs_204672 = {}
        # Getting the type of 'result' (line 276)
        result_204668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'result', False)
        # Obtaining the member 'addError' of a type (line 276)
        addError_204669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), result_204668, 'addError')
        # Calling addError(args, kwargs) (line 276)
        addError_call_result_204673 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), addError_204669, *[None_204670, None_204671], **kwargs_204672)
        
        
        # Call to assertTrue(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'result' (line 277)
        result_204676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 24), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 277)
        shouldStop_204677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 24), result_204676, 'shouldStop')
        # Processing the call keyword arguments (line 277)
        kwargs_204678 = {}
        # Getting the type of 'self' (line 277)
        self_204674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 277)
        assertTrue_204675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), self_204674, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 277)
        assertTrue_call_result_204679 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), assertTrue_204675, *[shouldStop_204677], **kwargs_204678)
        
        
        # Assigning a Call to a Name (line 279):
        
        # Assigning a Call to a Name (line 279):
        
        # Call to TestResult(...): (line 279)
        # Processing the call keyword arguments (line 279)
        kwargs_204682 = {}
        # Getting the type of 'unittest' (line 279)
        unittest_204680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 279)
        TestResult_204681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 17), unittest_204680, 'TestResult')
        # Calling TestResult(args, kwargs) (line 279)
        TestResult_call_result_204683 = invoke(stypy.reporting.localization.Localization(__file__, 279, 17), TestResult_204681, *[], **kwargs_204682)
        
        # Assigning a type to the variable 'result' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'result', TestResult_call_result_204683)
        
        # Assigning a Lambda to a Attribute (line 280):
        
        # Assigning a Lambda to a Attribute (line 280):

        @norecursion
        def _stypy_temp_lambda_92(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_92'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_92', 280, 37, True)
            # Passed parameters checking function
            _stypy_temp_lambda_92.stypy_localization = localization
            _stypy_temp_lambda_92.stypy_type_of_self = None
            _stypy_temp_lambda_92.stypy_type_store = module_type_store
            _stypy_temp_lambda_92.stypy_function_name = '_stypy_temp_lambda_92'
            _stypy_temp_lambda_92.stypy_param_names_list = []
            _stypy_temp_lambda_92.stypy_varargs_param_name = '_'
            _stypy_temp_lambda_92.stypy_kwargs_param_name = None
            _stypy_temp_lambda_92.stypy_call_defaults = defaults
            _stypy_temp_lambda_92.stypy_call_varargs = varargs
            _stypy_temp_lambda_92.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_92', [], '_', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_92', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            str_204684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 48), 'str', '')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), 'stypy_return_type', str_204684)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_92' in the type store
            # Getting the type of 'stypy_return_type' (line 280)
            stypy_return_type_204685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_204685)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_92'
            return stypy_return_type_204685

        # Assigning a type to the variable '_stypy_temp_lambda_92' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), '_stypy_temp_lambda_92', _stypy_temp_lambda_92)
        # Getting the type of '_stypy_temp_lambda_92' (line 280)
        _stypy_temp_lambda_92_204686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 37), '_stypy_temp_lambda_92')
        # Getting the type of 'result' (line 280)
        result_204687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'result')
        # Setting the type of the member '_exc_info_to_string' of a type (line 280)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), result_204687, '_exc_info_to_string', _stypy_temp_lambda_92_204686)
        
        # Assigning a Name to a Attribute (line 281):
        
        # Assigning a Name to a Attribute (line 281):
        # Getting the type of 'True' (line 281)
        True_204688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'True')
        # Getting the type of 'result' (line 281)
        result_204689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'result')
        # Setting the type of the member 'failfast' of a type (line 281)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), result_204689, 'failfast', True_204688)
        
        # Call to addFailure(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'None' (line 282)
        None_204692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'None', False)
        # Getting the type of 'None' (line 282)
        None_204693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 32), 'None', False)
        # Processing the call keyword arguments (line 282)
        kwargs_204694 = {}
        # Getting the type of 'result' (line 282)
        result_204690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'result', False)
        # Obtaining the member 'addFailure' of a type (line 282)
        addFailure_204691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), result_204690, 'addFailure')
        # Calling addFailure(args, kwargs) (line 282)
        addFailure_call_result_204695 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), addFailure_204691, *[None_204692, None_204693], **kwargs_204694)
        
        
        # Call to assertTrue(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'result' (line 283)
        result_204698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 283)
        shouldStop_204699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 24), result_204698, 'shouldStop')
        # Processing the call keyword arguments (line 283)
        kwargs_204700 = {}
        # Getting the type of 'self' (line 283)
        self_204696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 283)
        assertTrue_204697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_204696, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 283)
        assertTrue_call_result_204701 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assertTrue_204697, *[shouldStop_204699], **kwargs_204700)
        
        
        # Assigning a Call to a Name (line 285):
        
        # Assigning a Call to a Name (line 285):
        
        # Call to TestResult(...): (line 285)
        # Processing the call keyword arguments (line 285)
        kwargs_204704 = {}
        # Getting the type of 'unittest' (line 285)
        unittest_204702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 285)
        TestResult_204703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 17), unittest_204702, 'TestResult')
        # Calling TestResult(args, kwargs) (line 285)
        TestResult_call_result_204705 = invoke(stypy.reporting.localization.Localization(__file__, 285, 17), TestResult_204703, *[], **kwargs_204704)
        
        # Assigning a type to the variable 'result' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'result', TestResult_call_result_204705)
        
        # Assigning a Lambda to a Attribute (line 286):
        
        # Assigning a Lambda to a Attribute (line 286):

        @norecursion
        def _stypy_temp_lambda_93(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_93'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_93', 286, 37, True)
            # Passed parameters checking function
            _stypy_temp_lambda_93.stypy_localization = localization
            _stypy_temp_lambda_93.stypy_type_of_self = None
            _stypy_temp_lambda_93.stypy_type_store = module_type_store
            _stypy_temp_lambda_93.stypy_function_name = '_stypy_temp_lambda_93'
            _stypy_temp_lambda_93.stypy_param_names_list = []
            _stypy_temp_lambda_93.stypy_varargs_param_name = '_'
            _stypy_temp_lambda_93.stypy_kwargs_param_name = None
            _stypy_temp_lambda_93.stypy_call_defaults = defaults
            _stypy_temp_lambda_93.stypy_call_varargs = varargs
            _stypy_temp_lambda_93.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_93', [], '_', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_93', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            str_204706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 48), 'str', '')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), 'stypy_return_type', str_204706)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_93' in the type store
            # Getting the type of 'stypy_return_type' (line 286)
            stypy_return_type_204707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_204707)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_93'
            return stypy_return_type_204707

        # Assigning a type to the variable '_stypy_temp_lambda_93' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), '_stypy_temp_lambda_93', _stypy_temp_lambda_93)
        # Getting the type of '_stypy_temp_lambda_93' (line 286)
        _stypy_temp_lambda_93_204708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), '_stypy_temp_lambda_93')
        # Getting the type of 'result' (line 286)
        result_204709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'result')
        # Setting the type of the member '_exc_info_to_string' of a type (line 286)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), result_204709, '_exc_info_to_string', _stypy_temp_lambda_93_204708)
        
        # Assigning a Name to a Attribute (line 287):
        
        # Assigning a Name to a Attribute (line 287):
        # Getting the type of 'True' (line 287)
        True_204710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'True')
        # Getting the type of 'result' (line 287)
        result_204711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'result')
        # Setting the type of the member 'failfast' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), result_204711, 'failfast', True_204710)
        
        # Call to addUnexpectedSuccess(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'None' (line 288)
        None_204714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 36), 'None', False)
        # Processing the call keyword arguments (line 288)
        kwargs_204715 = {}
        # Getting the type of 'result' (line 288)
        result_204712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'result', False)
        # Obtaining the member 'addUnexpectedSuccess' of a type (line 288)
        addUnexpectedSuccess_204713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), result_204712, 'addUnexpectedSuccess')
        # Calling addUnexpectedSuccess(args, kwargs) (line 288)
        addUnexpectedSuccess_call_result_204716 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), addUnexpectedSuccess_204713, *[None_204714], **kwargs_204715)
        
        
        # Call to assertTrue(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'result' (line 289)
        result_204719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 24), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 289)
        shouldStop_204720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 24), result_204719, 'shouldStop')
        # Processing the call keyword arguments (line 289)
        kwargs_204721 = {}
        # Getting the type of 'self' (line 289)
        self_204717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 289)
        assertTrue_204718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), self_204717, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 289)
        assertTrue_call_result_204722 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), assertTrue_204718, *[shouldStop_204720], **kwargs_204721)
        
        
        # ################# End of 'testFailFast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testFailFast' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_204723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204723)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testFailFast'
        return stypy_return_type_204723


    @norecursion
    def testFailFastSetByRunner(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testFailFastSetByRunner'
        module_type_store = module_type_store.open_function_context('testFailFastSetByRunner', 291, 4, False)
        # Assigning a type to the variable 'self' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_localization', localization)
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_function_name', 'Test_TestResult.testFailFastSetByRunner')
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestResult.testFailFastSetByRunner.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.testFailFastSetByRunner', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testFailFastSetByRunner', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testFailFastSetByRunner(...)' code ##################

        
        # Assigning a Call to a Name (line 292):
        
        # Assigning a Call to a Name (line 292):
        
        # Call to TextTestRunner(...): (line 292)
        # Processing the call keyword arguments (line 292)
        
        # Call to StringIO(...): (line 292)
        # Processing the call keyword arguments (line 292)
        kwargs_204727 = {}
        # Getting the type of 'StringIO' (line 292)
        StringIO_204726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 48), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 292)
        StringIO_call_result_204728 = invoke(stypy.reporting.localization.Localization(__file__, 292, 48), StringIO_204726, *[], **kwargs_204727)
        
        keyword_204729 = StringIO_call_result_204728
        # Getting the type of 'True' (line 292)
        True_204730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 69), 'True', False)
        keyword_204731 = True_204730
        kwargs_204732 = {'failfast': keyword_204731, 'stream': keyword_204729}
        # Getting the type of 'unittest' (line 292)
        unittest_204724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 17), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 292)
        TextTestRunner_204725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 17), unittest_204724, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 292)
        TextTestRunner_call_result_204733 = invoke(stypy.reporting.localization.Localization(__file__, 292, 17), TextTestRunner_204725, *[], **kwargs_204732)
        
        # Assigning a type to the variable 'runner' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'runner', TextTestRunner_call_result_204733)

        @norecursion
        def test(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test'
            module_type_store = module_type_store.open_function_context('test', 293, 8, False)
            
            # Passed parameters checking function
            test.stypy_localization = localization
            test.stypy_type_of_self = None
            test.stypy_type_store = module_type_store
            test.stypy_function_name = 'test'
            test.stypy_param_names_list = ['result']
            test.stypy_varargs_param_name = None
            test.stypy_kwargs_param_name = None
            test.stypy_call_defaults = defaults
            test.stypy_call_varargs = varargs
            test.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'test', ['result'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'test', localization, ['result'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'test(...)' code ##################

            
            # Call to assertTrue(...): (line 294)
            # Processing the call arguments (line 294)
            # Getting the type of 'result' (line 294)
            result_204736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 28), 'result', False)
            # Obtaining the member 'failfast' of a type (line 294)
            failfast_204737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 28), result_204736, 'failfast')
            # Processing the call keyword arguments (line 294)
            kwargs_204738 = {}
            # Getting the type of 'self' (line 294)
            self_204734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'self', False)
            # Obtaining the member 'assertTrue' of a type (line 294)
            assertTrue_204735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), self_204734, 'assertTrue')
            # Calling assertTrue(args, kwargs) (line 294)
            assertTrue_call_result_204739 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), assertTrue_204735, *[failfast_204737], **kwargs_204738)
            
            
            # ################# End of 'test(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test' in the type store
            # Getting the type of 'stypy_return_type' (line 293)
            stypy_return_type_204740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_204740)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test'
            return stypy_return_type_204740

        # Assigning a type to the variable 'test' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'test', test)
        
        # Call to run(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'test' (line 295)
        test_204743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'test', False)
        # Processing the call keyword arguments (line 295)
        kwargs_204744 = {}
        # Getting the type of 'runner' (line 295)
        runner_204741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'runner', False)
        # Obtaining the member 'run' of a type (line 295)
        run_204742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), runner_204741, 'run')
        # Calling run(args, kwargs) (line 295)
        run_call_result_204745 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), run_204742, *[test_204743], **kwargs_204744)
        
        
        # ################# End of 'testFailFastSetByRunner(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testFailFastSetByRunner' in the type store
        # Getting the type of 'stypy_return_type' (line 291)
        stypy_return_type_204746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204746)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testFailFastSetByRunner'
        return stypy_return_type_204746


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestResult.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_TestResult' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Test_TestResult', Test_TestResult)

# Assigning a Call to a Name (line 298):

# Assigning a Call to a Name (line 298):

# Call to dict(...): (line 298)
# Processing the call arguments (line 298)
# Getting the type of 'unittest' (line 298)
unittest_204748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 17), 'unittest', False)
# Obtaining the member 'TestResult' of a type (line 298)
TestResult_204749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 17), unittest_204748, 'TestResult')
# Obtaining the member '__dict__' of a type (line 298)
dict___204750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 17), TestResult_204749, '__dict__')
# Processing the call keyword arguments (line 298)
kwargs_204751 = {}
# Getting the type of 'dict' (line 298)
dict_204747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'dict', False)
# Calling dict(args, kwargs) (line 298)
dict_call_result_204752 = invoke(stypy.reporting.localization.Localization(__file__, 298, 12), dict_204747, *[dict___204750], **kwargs_204751)

# Assigning a type to the variable 'classDict' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'classDict', dict_call_result_204752)


# Obtaining an instance of the builtin type 'tuple' (line 299)
tuple_204753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 299)
# Adding element type (line 299)
str_204754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 10), 'str', 'addSkip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 10), tuple_204753, str_204754)
# Adding element type (line 299)
str_204755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 21), 'str', 'addExpectedFailure')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 10), tuple_204753, str_204755)
# Adding element type (line 299)
str_204756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 43), 'str', 'addUnexpectedSuccess')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 10), tuple_204753, str_204756)
# Adding element type (line 299)
str_204757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 11), 'str', '__init__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 10), tuple_204753, str_204757)

# Testing the type of a for loop iterable (line 299)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 299, 0), tuple_204753)
# Getting the type of the for loop variable (line 299)
for_loop_var_204758 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 299, 0), tuple_204753)
# Assigning a type to the variable 'm' (line 299)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), 'm', for_loop_var_204758)
# SSA begins for a for statement (line 299)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
# Deleting a member
# Getting the type of 'classDict' (line 301)
classDict_204759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'classDict')

# Obtaining the type of the subscript
# Getting the type of 'm' (line 301)
m_204760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 18), 'm')
# Getting the type of 'classDict' (line 301)
classDict_204761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'classDict')
# Obtaining the member '__getitem__' of a type (line 301)
getitem___204762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), classDict_204761, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 301)
subscript_call_result_204763 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), getitem___204762, m_204760)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 4), classDict_204759, subscript_call_result_204763)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


@norecursion
def __init__(type_of_self, localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 303)
    None_204764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 26), 'None')
    # Getting the type of 'None' (line 303)
    None_204765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 45), 'None')
    # Getting the type of 'None' (line 303)
    None_204766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 61), 'None')
    defaults = [None_204764, None_204765, None_204766]
    # Create a new context for function '__init__'
    module_type_store = module_type_store.open_function_context('__init__', 303, 0, False)
    
    # Passed parameters checking function
    arguments = process_argument_values(localization, None, module_type_store, '__init__', ['self', 'stream', 'descriptions', 'verbosity'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return

    # Initialize method data
    init_call_information(module_type_store, '__init__', localization, ['self', 'stream', 'descriptions', 'verbosity'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__init__(...)' code ##################

    
    # Assigning a List to a Attribute (line 304):
    
    # Assigning a List to a Attribute (line 304):
    
    # Obtaining an instance of the builtin type 'list' (line 304)
    list_204767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 304)
    
    # Getting the type of 'self' (line 304)
    self_204768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self')
    # Setting the type of the member 'failures' of a type (line 304)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 4), self_204768, 'failures', list_204767)
    
    # Assigning a List to a Attribute (line 305):
    
    # Assigning a List to a Attribute (line 305):
    
    # Obtaining an instance of the builtin type 'list' (line 305)
    list_204769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 305)
    
    # Getting the type of 'self' (line 305)
    self_204770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self')
    # Setting the type of the member 'errors' of a type (line 305)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 4), self_204770, 'errors', list_204769)
    
    # Assigning a Num to a Attribute (line 306):
    
    # Assigning a Num to a Attribute (line 306):
    int_204771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 20), 'int')
    # Getting the type of 'self' (line 306)
    self_204772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self')
    # Setting the type of the member 'testsRun' of a type (line 306)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 4), self_204772, 'testsRun', int_204771)
    
    # Assigning a Name to a Attribute (line 307):
    
    # Assigning a Name to a Attribute (line 307):
    # Getting the type of 'False' (line 307)
    False_204773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'False')
    # Getting the type of 'self' (line 307)
    self_204774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'self')
    # Setting the type of the member 'shouldStop' of a type (line 307)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 4), self_204774, 'shouldStop', False_204773)
    
    # Assigning a Name to a Attribute (line 308):
    
    # Assigning a Name to a Attribute (line 308):
    # Getting the type of 'False' (line 308)
    False_204775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 18), 'False')
    # Getting the type of 'self' (line 308)
    self_204776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'self')
    # Setting the type of the member 'buffer' of a type (line 308)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 4), self_204776, 'buffer', False_204775)
    
    # ################# End of '__init__(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()

# Assigning a type to the variable '__init__' (line 303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), '__init__', __init__)

# Assigning a Name to a Subscript (line 310):

# Assigning a Name to a Subscript (line 310):
# Getting the type of '__init__' (line 310)
init___204777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 24), '__init__')
# Getting the type of 'classDict' (line 310)
classDict_204778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'classDict')
str_204779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 10), 'str', '__init__')
# Storing an element on a container (line 310)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 0), classDict_204778, (str_204779, init___204777))

# Assigning a Call to a Name (line 311):

# Assigning a Call to a Name (line 311):

# Call to type(...): (line 311)
# Processing the call arguments (line 311)
str_204781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 17), 'str', 'OldResult')

# Obtaining an instance of the builtin type 'tuple' (line 311)
tuple_204782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 311)
# Adding element type (line 311)
# Getting the type of 'object' (line 311)
object_204783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 31), 'object', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 31), tuple_204782, object_204783)

# Getting the type of 'classDict' (line 311)
classDict_204784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 41), 'classDict', False)
# Processing the call keyword arguments (line 311)
kwargs_204785 = {}
# Getting the type of 'type' (line 311)
type_204780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'type', False)
# Calling type(args, kwargs) (line 311)
type_call_result_204786 = invoke(stypy.reporting.localization.Localization(__file__, 311, 12), type_204780, *[str_204781, tuple_204782, classDict_204784], **kwargs_204785)

# Assigning a type to the variable 'OldResult' (line 311)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 0), 'OldResult', type_call_result_204786)
# Declaration of the 'Test_OldTestResult' class
# Getting the type of 'unittest' (line 313)
unittest_204787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 25), 'unittest')
# Obtaining the member 'TestCase' of a type (line 313)
TestCase_204788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 25), unittest_204787, 'TestCase')

class Test_OldTestResult(TestCase_204788, ):

    @norecursion
    def assertOldResultWarning(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'assertOldResultWarning'
        module_type_store = module_type_store.open_function_context('assertOldResultWarning', 315, 4, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_localization', localization)
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_function_name', 'Test_OldTestResult.assertOldResultWarning')
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_param_names_list', ['test', 'failures'])
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_OldTestResult.assertOldResultWarning.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_OldTestResult.assertOldResultWarning', ['test', 'failures'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertOldResultWarning', localization, ['test', 'failures'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertOldResultWarning(...)' code ##################

        
        # Call to check_warnings(...): (line 316)
        # Processing the call arguments (line 316)
        
        # Obtaining an instance of the builtin type 'tuple' (line 316)
        tuple_204791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 316)
        # Adding element type (line 316)
        str_204792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 42), 'str', 'TestResult has no add.+ method,')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 42), tuple_204791, str_204792)
        # Adding element type (line 316)
        # Getting the type of 'RuntimeWarning' (line 317)
        RuntimeWarning_204793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 42), 'RuntimeWarning', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 42), tuple_204791, RuntimeWarning_204793)
        
        # Processing the call keyword arguments (line 316)
        kwargs_204794 = {}
        # Getting the type of 'test_support' (line 316)
        test_support_204789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'test_support', False)
        # Obtaining the member 'check_warnings' of a type (line 316)
        check_warnings_204790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 13), test_support_204789, 'check_warnings')
        # Calling check_warnings(args, kwargs) (line 316)
        check_warnings_call_result_204795 = invoke(stypy.reporting.localization.Localization(__file__, 316, 13), check_warnings_204790, *[tuple_204791], **kwargs_204794)
        
        with_204796 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 316, 13), check_warnings_call_result_204795, 'with parameter', '__enter__', '__exit__')

        if with_204796:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 316)
            enter___204797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 13), check_warnings_call_result_204795, '__enter__')
            with_enter_204798 = invoke(stypy.reporting.localization.Localization(__file__, 316, 13), enter___204797)
            
            # Assigning a Call to a Name (line 318):
            
            # Assigning a Call to a Name (line 318):
            
            # Call to OldResult(...): (line 318)
            # Processing the call keyword arguments (line 318)
            kwargs_204800 = {}
            # Getting the type of 'OldResult' (line 318)
            OldResult_204799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'OldResult', False)
            # Calling OldResult(args, kwargs) (line 318)
            OldResult_call_result_204801 = invoke(stypy.reporting.localization.Localization(__file__, 318, 21), OldResult_204799, *[], **kwargs_204800)
            
            # Assigning a type to the variable 'result' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'result', OldResult_call_result_204801)
            
            # Call to run(...): (line 319)
            # Processing the call arguments (line 319)
            # Getting the type of 'result' (line 319)
            result_204804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 21), 'result', False)
            # Processing the call keyword arguments (line 319)
            kwargs_204805 = {}
            # Getting the type of 'test' (line 319)
            test_204802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'test', False)
            # Obtaining the member 'run' of a type (line 319)
            run_204803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), test_204802, 'run')
            # Calling run(args, kwargs) (line 319)
            run_call_result_204806 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), run_204803, *[result_204804], **kwargs_204805)
            
            
            # Call to assertEqual(...): (line 320)
            # Processing the call arguments (line 320)
            
            # Call to len(...): (line 320)
            # Processing the call arguments (line 320)
            # Getting the type of 'result' (line 320)
            result_204810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 33), 'result', False)
            # Obtaining the member 'failures' of a type (line 320)
            failures_204811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 33), result_204810, 'failures')
            # Processing the call keyword arguments (line 320)
            kwargs_204812 = {}
            # Getting the type of 'len' (line 320)
            len_204809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 29), 'len', False)
            # Calling len(args, kwargs) (line 320)
            len_call_result_204813 = invoke(stypy.reporting.localization.Localization(__file__, 320, 29), len_204809, *[failures_204811], **kwargs_204812)
            
            # Getting the type of 'failures' (line 320)
            failures_204814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 51), 'failures', False)
            # Processing the call keyword arguments (line 320)
            kwargs_204815 = {}
            # Getting the type of 'self' (line 320)
            self_204807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 320)
            assertEqual_204808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), self_204807, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 320)
            assertEqual_call_result_204816 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), assertEqual_204808, *[len_call_result_204813, failures_204814], **kwargs_204815)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 316)
            exit___204817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 13), check_warnings_call_result_204795, '__exit__')
            with_exit_204818 = invoke(stypy.reporting.localization.Localization(__file__, 316, 13), exit___204817, None, None, None)

        
        # ################# End of 'assertOldResultWarning(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertOldResultWarning' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_204819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204819)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertOldResultWarning'
        return stypy_return_type_204819


    @norecursion
    def testOldTestResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testOldTestResult'
        module_type_store = module_type_store.open_function_context('testOldTestResult', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_localization', localization)
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_function_name', 'Test_OldTestResult.testOldTestResult')
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_param_names_list', [])
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_OldTestResult.testOldTestResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_OldTestResult.testOldTestResult', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testOldTestResult', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testOldTestResult(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 323)
        unittest_204820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 323)
        TestCase_204821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 19), unittest_204820, 'TestCase')

        class Test(TestCase_204821, ):

            @norecursion
            def testSkip(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testSkip'
                module_type_store = module_type_store.open_function_context('testSkip', 324, 12, False)
                # Assigning a type to the variable 'self' (line 325)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.testSkip.__dict__.__setitem__('stypy_localization', localization)
                Test.testSkip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.testSkip.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.testSkip.__dict__.__setitem__('stypy_function_name', 'Test.testSkip')
                Test.testSkip.__dict__.__setitem__('stypy_param_names_list', [])
                Test.testSkip.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.testSkip.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.testSkip.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.testSkip.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.testSkip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.testSkip.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.testSkip', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testSkip', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testSkip(...)' code ##################

                
                # Call to skipTest(...): (line 325)
                # Processing the call arguments (line 325)
                str_204824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 30), 'str', 'foobar')
                # Processing the call keyword arguments (line 325)
                kwargs_204825 = {}
                # Getting the type of 'self' (line 325)
                self_204822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'self', False)
                # Obtaining the member 'skipTest' of a type (line 325)
                skipTest_204823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 16), self_204822, 'skipTest')
                # Calling skipTest(args, kwargs) (line 325)
                skipTest_call_result_204826 = invoke(stypy.reporting.localization.Localization(__file__, 325, 16), skipTest_204823, *[str_204824], **kwargs_204825)
                
                
                # ################# End of 'testSkip(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testSkip' in the type store
                # Getting the type of 'stypy_return_type' (line 324)
                stypy_return_type_204827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204827)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testSkip'
                return stypy_return_type_204827


            @norecursion
            def testExpectedFail(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testExpectedFail'
                module_type_store = module_type_store.open_function_context('testExpectedFail', 326, 12, False)
                # Assigning a type to the variable 'self' (line 327)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.testExpectedFail.__dict__.__setitem__('stypy_localization', localization)
                Test.testExpectedFail.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.testExpectedFail.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.testExpectedFail.__dict__.__setitem__('stypy_function_name', 'Test.testExpectedFail')
                Test.testExpectedFail.__dict__.__setitem__('stypy_param_names_list', [])
                Test.testExpectedFail.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.testExpectedFail.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.testExpectedFail.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.testExpectedFail.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.testExpectedFail.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.testExpectedFail.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.testExpectedFail', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testExpectedFail', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testExpectedFail(...)' code ##################

                # Getting the type of 'TypeError' (line 328)
                TypeError_204828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 22), 'TypeError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 328, 16), TypeError_204828, 'raise parameter', BaseException)
                
                # ################# End of 'testExpectedFail(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testExpectedFail' in the type store
                # Getting the type of 'stypy_return_type' (line 326)
                stypy_return_type_204829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204829)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testExpectedFail'
                return stypy_return_type_204829


            @norecursion
            def testUnexpectedSuccess(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testUnexpectedSuccess'
                module_type_store = module_type_store.open_function_context('testUnexpectedSuccess', 329, 12, False)
                # Assigning a type to the variable 'self' (line 330)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_localization', localization)
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_function_name', 'Test.testUnexpectedSuccess')
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_param_names_list', [])
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.testUnexpectedSuccess.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.testUnexpectedSuccess', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testUnexpectedSuccess', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testUnexpectedSuccess(...)' code ##################

                pass
                
                # ################# End of 'testUnexpectedSuccess(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testUnexpectedSuccess' in the type store
                # Getting the type of 'stypy_return_type' (line 329)
                stypy_return_type_204830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204830)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testUnexpectedSuccess'
                return stypy_return_type_204830

        
        # Assigning a type to the variable 'Test' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'Test', Test)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_204831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_204832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        str_204833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 40), 'str', 'testSkip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 40), tuple_204832, str_204833)
        # Adding element type (line 333)
        # Getting the type of 'True' (line 333)
        True_204834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 52), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 40), tuple_204832, True_204834)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 39), tuple_204831, tuple_204832)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'tuple' (line 334)
        tuple_204835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 334)
        # Adding element type (line 334)
        str_204836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 40), 'str', 'testExpectedFail')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 40), tuple_204835, str_204836)
        # Adding element type (line 334)
        # Getting the type of 'True' (line 334)
        True_204837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 60), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 40), tuple_204835, True_204837)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 39), tuple_204831, tuple_204835)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'tuple' (line 335)
        tuple_204838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 335)
        # Adding element type (line 335)
        str_204839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 40), 'str', 'testUnexpectedSuccess')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 40), tuple_204838, str_204839)
        # Adding element type (line 335)
        # Getting the type of 'False' (line 335)
        False_204840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 65), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 40), tuple_204838, False_204840)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 39), tuple_204831, tuple_204838)
        
        # Testing the type of a for loop iterable (line 333)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 333, 8), tuple_204831)
        # Getting the type of the for loop variable (line 333)
        for_loop_var_204841 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 333, 8), tuple_204831)
        # Assigning a type to the variable 'test_name' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'test_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 8), for_loop_var_204841))
        # Assigning a type to the variable 'should_pass' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'should_pass', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 8), for_loop_var_204841))
        # SSA begins for a for statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 336):
        
        # Assigning a Call to a Name (line 336):
        
        # Call to Test(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'test_name' (line 336)
        test_name_204843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'test_name', False)
        # Processing the call keyword arguments (line 336)
        kwargs_204844 = {}
        # Getting the type of 'Test' (line 336)
        Test_204842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'Test', False)
        # Calling Test(args, kwargs) (line 336)
        Test_call_result_204845 = invoke(stypy.reporting.localization.Localization(__file__, 336, 19), Test_204842, *[test_name_204843], **kwargs_204844)
        
        # Assigning a type to the variable 'test' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'test', Test_call_result_204845)
        
        # Call to assertOldResultWarning(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'test' (line 337)
        test_204848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 40), 'test', False)
        
        # Call to int(...): (line 337)
        # Processing the call arguments (line 337)
        
        # Getting the type of 'should_pass' (line 337)
        should_pass_204850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 54), 'should_pass', False)
        # Applying the 'not' unary operator (line 337)
        result_not__204851 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 50), 'not', should_pass_204850)
        
        # Processing the call keyword arguments (line 337)
        kwargs_204852 = {}
        # Getting the type of 'int' (line 337)
        int_204849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 46), 'int', False)
        # Calling int(args, kwargs) (line 337)
        int_call_result_204853 = invoke(stypy.reporting.localization.Localization(__file__, 337, 46), int_204849, *[result_not__204851], **kwargs_204852)
        
        # Processing the call keyword arguments (line 337)
        kwargs_204854 = {}
        # Getting the type of 'self' (line 337)
        self_204846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'self', False)
        # Obtaining the member 'assertOldResultWarning' of a type (line 337)
        assertOldResultWarning_204847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), self_204846, 'assertOldResultWarning')
        # Calling assertOldResultWarning(args, kwargs) (line 337)
        assertOldResultWarning_call_result_204855 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), assertOldResultWarning_204847, *[test_204848, int_call_result_204853], **kwargs_204854)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'testOldTestResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testOldTestResult' in the type store
        # Getting the type of 'stypy_return_type' (line 322)
        stypy_return_type_204856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204856)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testOldTestResult'
        return stypy_return_type_204856


    @norecursion
    def testOldTestTesultSetup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testOldTestTesultSetup'
        module_type_store = module_type_store.open_function_context('testOldTestTesultSetup', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_localization', localization)
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_function_name', 'Test_OldTestResult.testOldTestTesultSetup')
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_param_names_list', [])
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_OldTestResult.testOldTestTesultSetup.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_OldTestResult.testOldTestTesultSetup', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testOldTestTesultSetup', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testOldTestTesultSetup(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 340)
        unittest_204857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 340)
        TestCase_204858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 19), unittest_204857, 'TestCase')

        class Test(TestCase_204858, ):

            @norecursion
            def setUp(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUp'
                module_type_store = module_type_store.open_function_context('setUp', 341, 12, False)
                # Assigning a type to the variable 'self' (line 342)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.setUp.__dict__.__setitem__('stypy_localization', localization)
                Test.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.setUp.__dict__.__setitem__('stypy_function_name', 'Test.setUp')
                Test.setUp.__dict__.__setitem__('stypy_param_names_list', [])
                Test.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.setUp', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to skipTest(...): (line 342)
                # Processing the call arguments (line 342)
                str_204861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 30), 'str', 'no reason')
                # Processing the call keyword arguments (line 342)
                kwargs_204862 = {}
                # Getting the type of 'self' (line 342)
                self_204859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'self', False)
                # Obtaining the member 'skipTest' of a type (line 342)
                skipTest_204860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 16), self_204859, 'skipTest')
                # Calling skipTest(args, kwargs) (line 342)
                skipTest_call_result_204863 = invoke(stypy.reporting.localization.Localization(__file__, 342, 16), skipTest_204860, *[str_204861], **kwargs_204862)
                
                
                # ################# End of 'setUp(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUp' in the type store
                # Getting the type of 'stypy_return_type' (line 341)
                stypy_return_type_204864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204864)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUp'
                return stypy_return_type_204864


            @norecursion
            def testFoo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testFoo'
                module_type_store = module_type_store.open_function_context('testFoo', 343, 12, False)
                # Assigning a type to the variable 'self' (line 344)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.testFoo.__dict__.__setitem__('stypy_localization', localization)
                Test.testFoo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.testFoo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.testFoo.__dict__.__setitem__('stypy_function_name', 'Test.testFoo')
                Test.testFoo.__dict__.__setitem__('stypy_param_names_list', [])
                Test.testFoo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.testFoo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.testFoo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.testFoo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.testFoo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.testFoo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.testFoo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testFoo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testFoo(...)' code ##################

                pass
                
                # ################# End of 'testFoo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testFoo' in the type store
                # Getting the type of 'stypy_return_type' (line 343)
                stypy_return_type_204865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204865)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testFoo'
                return stypy_return_type_204865

        
        # Assigning a type to the variable 'Test' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'Test', Test)
        
        # Call to assertOldResultWarning(...): (line 345)
        # Processing the call arguments (line 345)
        
        # Call to Test(...): (line 345)
        # Processing the call arguments (line 345)
        str_204869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 41), 'str', 'testFoo')
        # Processing the call keyword arguments (line 345)
        kwargs_204870 = {}
        # Getting the type of 'Test' (line 345)
        Test_204868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 36), 'Test', False)
        # Calling Test(args, kwargs) (line 345)
        Test_call_result_204871 = invoke(stypy.reporting.localization.Localization(__file__, 345, 36), Test_204868, *[str_204869], **kwargs_204870)
        
        int_204872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 53), 'int')
        # Processing the call keyword arguments (line 345)
        kwargs_204873 = {}
        # Getting the type of 'self' (line 345)
        self_204866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'self', False)
        # Obtaining the member 'assertOldResultWarning' of a type (line 345)
        assertOldResultWarning_204867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), self_204866, 'assertOldResultWarning')
        # Calling assertOldResultWarning(args, kwargs) (line 345)
        assertOldResultWarning_call_result_204874 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assertOldResultWarning_204867, *[Test_call_result_204871, int_204872], **kwargs_204873)
        
        
        # ################# End of 'testOldTestTesultSetup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testOldTestTesultSetup' in the type store
        # Getting the type of 'stypy_return_type' (line 339)
        stypy_return_type_204875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testOldTestTesultSetup'
        return stypy_return_type_204875


    @norecursion
    def testOldTestResultClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testOldTestResultClass'
        module_type_store = module_type_store.open_function_context('testOldTestResultClass', 347, 4, False)
        # Assigning a type to the variable 'self' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_localization', localization)
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_function_name', 'Test_OldTestResult.testOldTestResultClass')
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_param_names_list', [])
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_OldTestResult.testOldTestResultClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_OldTestResult.testOldTestResultClass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testOldTestResultClass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testOldTestResultClass(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 349)
        unittest_204876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 349)
        TestCase_204877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 19), unittest_204876, 'TestCase')

        class Test(TestCase_204877, ):

            @norecursion
            def testFoo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testFoo'
                module_type_store = module_type_store.open_function_context('testFoo', 350, 12, False)
                # Assigning a type to the variable 'self' (line 351)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.testFoo.__dict__.__setitem__('stypy_localization', localization)
                Test.testFoo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.testFoo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.testFoo.__dict__.__setitem__('stypy_function_name', 'Test.testFoo')
                Test.testFoo.__dict__.__setitem__('stypy_param_names_list', [])
                Test.testFoo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.testFoo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.testFoo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.testFoo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.testFoo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.testFoo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.testFoo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testFoo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testFoo(...)' code ##################

                pass
                
                # ################# End of 'testFoo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testFoo' in the type store
                # Getting the type of 'stypy_return_type' (line 350)
                stypy_return_type_204878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204878)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testFoo'
                return stypy_return_type_204878

        
        # Assigning a type to the variable 'Test' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'Test', Test)
        
        # Call to assertOldResultWarning(...): (line 352)
        # Processing the call arguments (line 352)
        
        # Call to Test(...): (line 352)
        # Processing the call arguments (line 352)
        str_204882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 41), 'str', 'testFoo')
        # Processing the call keyword arguments (line 352)
        kwargs_204883 = {}
        # Getting the type of 'Test' (line 352)
        Test_204881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 36), 'Test', False)
        # Calling Test(args, kwargs) (line 352)
        Test_call_result_204884 = invoke(stypy.reporting.localization.Localization(__file__, 352, 36), Test_204881, *[str_204882], **kwargs_204883)
        
        int_204885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 53), 'int')
        # Processing the call keyword arguments (line 352)
        kwargs_204886 = {}
        # Getting the type of 'self' (line 352)
        self_204879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'self', False)
        # Obtaining the member 'assertOldResultWarning' of a type (line 352)
        assertOldResultWarning_204880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), self_204879, 'assertOldResultWarning')
        # Calling assertOldResultWarning(args, kwargs) (line 352)
        assertOldResultWarning_call_result_204887 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), assertOldResultWarning_204880, *[Test_call_result_204884, int_204885], **kwargs_204886)
        
        
        # ################# End of 'testOldTestResultClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testOldTestResultClass' in the type store
        # Getting the type of 'stypy_return_type' (line 347)
        stypy_return_type_204888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204888)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testOldTestResultClass'
        return stypy_return_type_204888


    @norecursion
    def testOldResultWithRunner(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testOldResultWithRunner'
        module_type_store = module_type_store.open_function_context('testOldResultWithRunner', 354, 4, False)
        # Assigning a type to the variable 'self' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_localization', localization)
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_function_name', 'Test_OldTestResult.testOldResultWithRunner')
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_param_names_list', [])
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_OldTestResult.testOldResultWithRunner.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_OldTestResult.testOldResultWithRunner', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testOldResultWithRunner', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testOldResultWithRunner(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 355)
        unittest_204889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 355)
        TestCase_204890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 19), unittest_204889, 'TestCase')

        class Test(TestCase_204890, ):

            @norecursion
            def testFoo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testFoo'
                module_type_store = module_type_store.open_function_context('testFoo', 356, 12, False)
                # Assigning a type to the variable 'self' (line 357)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.testFoo.__dict__.__setitem__('stypy_localization', localization)
                Test.testFoo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.testFoo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.testFoo.__dict__.__setitem__('stypy_function_name', 'Test.testFoo')
                Test.testFoo.__dict__.__setitem__('stypy_param_names_list', [])
                Test.testFoo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.testFoo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.testFoo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.testFoo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.testFoo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.testFoo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.testFoo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testFoo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testFoo(...)' code ##################

                pass
                
                # ################# End of 'testFoo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testFoo' in the type store
                # Getting the type of 'stypy_return_type' (line 356)
                stypy_return_type_204891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_204891)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testFoo'
                return stypy_return_type_204891

        
        # Assigning a type to the variable 'Test' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'Test', Test)
        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to TextTestRunner(...): (line 358)
        # Processing the call keyword arguments (line 358)
        # Getting the type of 'OldResult' (line 358)
        OldResult_204894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 53), 'OldResult', False)
        keyword_204895 = OldResult_204894
        
        # Call to StringIO(...): (line 359)
        # Processing the call keyword arguments (line 359)
        kwargs_204897 = {}
        # Getting the type of 'StringIO' (line 359)
        StringIO_204896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 49), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 359)
        StringIO_call_result_204898 = invoke(stypy.reporting.localization.Localization(__file__, 359, 49), StringIO_204896, *[], **kwargs_204897)
        
        keyword_204899 = StringIO_call_result_204898
        kwargs_204900 = {'resultclass': keyword_204895, 'stream': keyword_204899}
        # Getting the type of 'unittest' (line 358)
        unittest_204892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 358)
        TextTestRunner_204893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 17), unittest_204892, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 358)
        TextTestRunner_call_result_204901 = invoke(stypy.reporting.localization.Localization(__file__, 358, 17), TextTestRunner_204893, *[], **kwargs_204900)
        
        # Assigning a type to the variable 'runner' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'runner', TextTestRunner_call_result_204901)
        
        # Call to run(...): (line 362)
        # Processing the call arguments (line 362)
        
        # Call to Test(...): (line 362)
        # Processing the call arguments (line 362)
        str_204905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 24), 'str', 'testFoo')
        # Processing the call keyword arguments (line 362)
        kwargs_204906 = {}
        # Getting the type of 'Test' (line 362)
        Test_204904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'Test', False)
        # Calling Test(args, kwargs) (line 362)
        Test_call_result_204907 = invoke(stypy.reporting.localization.Localization(__file__, 362, 19), Test_204904, *[str_204905], **kwargs_204906)
        
        # Processing the call keyword arguments (line 362)
        kwargs_204908 = {}
        # Getting the type of 'runner' (line 362)
        runner_204902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'runner', False)
        # Obtaining the member 'run' of a type (line 362)
        run_204903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), runner_204902, 'run')
        # Calling run(args, kwargs) (line 362)
        run_call_result_204909 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), run_204903, *[Test_call_result_204907], **kwargs_204908)
        
        
        # ################# End of 'testOldResultWithRunner(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testOldResultWithRunner' in the type store
        # Getting the type of 'stypy_return_type' (line 354)
        stypy_return_type_204910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204910)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testOldResultWithRunner'
        return stypy_return_type_204910


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 313, 0, False)
        # Assigning a type to the variable 'self' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_OldTestResult.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_OldTestResult' (line 313)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 0), 'Test_OldTestResult', Test_OldTestResult)
# Declaration of the 'MockTraceback' class

class MockTraceback(object, ):

    @staticmethod
    @norecursion
    def format_exception(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'format_exception'
        module_type_store = module_type_store.open_function_context('format_exception', 366, 4, False)
        
        # Passed parameters checking function
        MockTraceback.format_exception.__dict__.__setitem__('stypy_localization', localization)
        MockTraceback.format_exception.__dict__.__setitem__('stypy_type_of_self', None)
        MockTraceback.format_exception.__dict__.__setitem__('stypy_type_store', module_type_store)
        MockTraceback.format_exception.__dict__.__setitem__('stypy_function_name', 'format_exception')
        MockTraceback.format_exception.__dict__.__setitem__('stypy_param_names_list', [])
        MockTraceback.format_exception.__dict__.__setitem__('stypy_varargs_param_name', '_')
        MockTraceback.format_exception.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MockTraceback.format_exception.__dict__.__setitem__('stypy_call_defaults', defaults)
        MockTraceback.format_exception.__dict__.__setitem__('stypy_call_varargs', varargs)
        MockTraceback.format_exception.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MockTraceback.format_exception.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'format_exception', [], '_', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'format_exception', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'format_exception(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 368)
        list_204911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 368)
        # Adding element type (line 368)
        str_204912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 16), 'str', 'A traceback')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 15), list_204911, str_204912)
        
        # Assigning a type to the variable 'stypy_return_type' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'stypy_return_type', list_204911)
        
        # ################# End of 'format_exception(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format_exception' in the type store
        # Getting the type of 'stypy_return_type' (line 366)
        stypy_return_type_204913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204913)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format_exception'
        return stypy_return_type_204913


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 365, 0, False)
        # Assigning a type to the variable 'self' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MockTraceback.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MockTraceback' (line 365)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 0), 'MockTraceback', MockTraceback)

@norecursion
def restore_traceback(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'restore_traceback'
    module_type_store = module_type_store.open_function_context('restore_traceback', 370, 0, False)
    
    # Passed parameters checking function
    restore_traceback.stypy_localization = localization
    restore_traceback.stypy_type_of_self = None
    restore_traceback.stypy_type_store = module_type_store
    restore_traceback.stypy_function_name = 'restore_traceback'
    restore_traceback.stypy_param_names_list = []
    restore_traceback.stypy_varargs_param_name = None
    restore_traceback.stypy_kwargs_param_name = None
    restore_traceback.stypy_call_defaults = defaults
    restore_traceback.stypy_call_varargs = varargs
    restore_traceback.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'restore_traceback', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'restore_traceback', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'restore_traceback(...)' code ##################

    
    # Assigning a Name to a Attribute (line 371):
    
    # Assigning a Name to a Attribute (line 371):
    # Getting the type of 'traceback' (line 371)
    traceback_204914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 32), 'traceback')
    # Getting the type of 'unittest' (line 371)
    unittest_204915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'unittest')
    # Obtaining the member 'result' of a type (line 371)
    result_204916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 4), unittest_204915, 'result')
    # Setting the type of the member 'traceback' of a type (line 371)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 4), result_204916, 'traceback', traceback_204914)
    
    # ################# End of 'restore_traceback(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'restore_traceback' in the type store
    # Getting the type of 'stypy_return_type' (line 370)
    stypy_return_type_204917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_204917)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'restore_traceback'
    return stypy_return_type_204917

# Assigning a type to the variable 'restore_traceback' (line 370)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 0), 'restore_traceback', restore_traceback)
# Declaration of the 'TestOutputBuffering' class
# Getting the type of 'unittest' (line 374)
unittest_204918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 26), 'unittest')
# Obtaining the member 'TestCase' of a type (line 374)
TestCase_204919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 26), unittest_204918, 'TestCase')

class TestOutputBuffering(TestCase_204919, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 376, 4, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.setUp')
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 377):
        
        # Assigning a Attribute to a Attribute (line 377):
        # Getting the type of 'sys' (line 377)
        sys_204920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 25), 'sys')
        # Obtaining the member 'stdout' of a type (line 377)
        stdout_204921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 25), sys_204920, 'stdout')
        # Getting the type of 'self' (line 377)
        self_204922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'self')
        # Setting the type of the member '_real_out' of a type (line 377)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), self_204922, '_real_out', stdout_204921)
        
        # Assigning a Attribute to a Attribute (line 378):
        
        # Assigning a Attribute to a Attribute (line 378):
        # Getting the type of 'sys' (line 378)
        sys_204923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 25), 'sys')
        # Obtaining the member 'stderr' of a type (line 378)
        stderr_204924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 25), sys_204923, 'stderr')
        # Getting the type of 'self' (line 378)
        self_204925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self')
        # Setting the type of the member '_real_err' of a type (line 378)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_204925, '_real_err', stderr_204924)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 376)
        stypy_return_type_204926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204926)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_204926


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 380, 4, False)
        # Assigning a type to the variable 'self' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.tearDown')
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Attribute (line 381):
        
        # Assigning a Attribute to a Attribute (line 381):
        # Getting the type of 'self' (line 381)
        self_204927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'self')
        # Obtaining the member '_real_out' of a type (line 381)
        _real_out_204928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), self_204927, '_real_out')
        # Getting the type of 'sys' (line 381)
        sys_204929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'sys')
        # Setting the type of the member 'stdout' of a type (line 381)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), sys_204929, 'stdout', _real_out_204928)
        
        # Assigning a Attribute to a Attribute (line 382):
        
        # Assigning a Attribute to a Attribute (line 382):
        # Getting the type of 'self' (line 382)
        self_204930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 21), 'self')
        # Obtaining the member '_real_err' of a type (line 382)
        _real_err_204931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 21), self_204930, '_real_err')
        # Getting the type of 'sys' (line 382)
        sys_204932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'sys')
        # Setting the type of the member 'stderr' of a type (line 382)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), sys_204932, 'stderr', _real_err_204931)
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 380)
        stypy_return_type_204933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204933)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_204933


    @norecursion
    def testBufferOutputOff(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBufferOutputOff'
        module_type_store = module_type_store.open_function_context('testBufferOutputOff', 384, 4, False)
        # Assigning a type to the variable 'self' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.testBufferOutputOff')
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.testBufferOutputOff.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.testBufferOutputOff', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBufferOutputOff', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBufferOutputOff(...)' code ##################

        
        # Assigning a Attribute to a Name (line 385):
        
        # Assigning a Attribute to a Name (line 385):
        # Getting the type of 'self' (line 385)
        self_204934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'self')
        # Obtaining the member '_real_out' of a type (line 385)
        _real_out_204935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 19), self_204934, '_real_out')
        # Assigning a type to the variable 'real_out' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'real_out', _real_out_204935)
        
        # Assigning a Attribute to a Name (line 386):
        
        # Assigning a Attribute to a Name (line 386):
        # Getting the type of 'self' (line 386)
        self_204936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 19), 'self')
        # Obtaining the member '_real_err' of a type (line 386)
        _real_err_204937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 19), self_204936, '_real_err')
        # Assigning a type to the variable 'real_err' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'real_err', _real_err_204937)
        
        # Assigning a Call to a Name (line 388):
        
        # Assigning a Call to a Name (line 388):
        
        # Call to TestResult(...): (line 388)
        # Processing the call keyword arguments (line 388)
        kwargs_204940 = {}
        # Getting the type of 'unittest' (line 388)
        unittest_204938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 388)
        TestResult_204939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 17), unittest_204938, 'TestResult')
        # Calling TestResult(args, kwargs) (line 388)
        TestResult_call_result_204941 = invoke(stypy.reporting.localization.Localization(__file__, 388, 17), TestResult_204939, *[], **kwargs_204940)
        
        # Assigning a type to the variable 'result' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'result', TestResult_call_result_204941)
        
        # Call to assertFalse(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'result' (line 389)
        result_204944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 25), 'result', False)
        # Obtaining the member 'buffer' of a type (line 389)
        buffer_204945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 25), result_204944, 'buffer')
        # Processing the call keyword arguments (line 389)
        kwargs_204946 = {}
        # Getting the type of 'self' (line 389)
        self_204942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 389)
        assertFalse_204943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_204942, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 389)
        assertFalse_call_result_204947 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), assertFalse_204943, *[buffer_204945], **kwargs_204946)
        
        
        # Call to assertIs(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'real_out' (line 391)
        real_out_204950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 22), 'real_out', False)
        # Getting the type of 'sys' (line 391)
        sys_204951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 32), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 391)
        stdout_204952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 32), sys_204951, 'stdout')
        # Processing the call keyword arguments (line 391)
        kwargs_204953 = {}
        # Getting the type of 'self' (line 391)
        self_204948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 391)
        assertIs_204949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), self_204948, 'assertIs')
        # Calling assertIs(args, kwargs) (line 391)
        assertIs_call_result_204954 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), assertIs_204949, *[real_out_204950, stdout_204952], **kwargs_204953)
        
        
        # Call to assertIs(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'real_err' (line 392)
        real_err_204957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 22), 'real_err', False)
        # Getting the type of 'sys' (line 392)
        sys_204958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 32), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 392)
        stderr_204959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 32), sys_204958, 'stderr')
        # Processing the call keyword arguments (line 392)
        kwargs_204960 = {}
        # Getting the type of 'self' (line 392)
        self_204955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 392)
        assertIs_204956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), self_204955, 'assertIs')
        # Calling assertIs(args, kwargs) (line 392)
        assertIs_call_result_204961 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), assertIs_204956, *[real_err_204957, stderr_204959], **kwargs_204960)
        
        
        # Call to startTest(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'self' (line 394)
        self_204964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 25), 'self', False)
        # Processing the call keyword arguments (line 394)
        kwargs_204965 = {}
        # Getting the type of 'result' (line 394)
        result_204962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'result', False)
        # Obtaining the member 'startTest' of a type (line 394)
        startTest_204963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), result_204962, 'startTest')
        # Calling startTest(args, kwargs) (line 394)
        startTest_call_result_204966 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), startTest_204963, *[self_204964], **kwargs_204965)
        
        
        # Call to assertIs(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'real_out' (line 396)
        real_out_204969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'real_out', False)
        # Getting the type of 'sys' (line 396)
        sys_204970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 32), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 396)
        stdout_204971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 32), sys_204970, 'stdout')
        # Processing the call keyword arguments (line 396)
        kwargs_204972 = {}
        # Getting the type of 'self' (line 396)
        self_204967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 396)
        assertIs_204968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_204967, 'assertIs')
        # Calling assertIs(args, kwargs) (line 396)
        assertIs_call_result_204973 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), assertIs_204968, *[real_out_204969, stdout_204971], **kwargs_204972)
        
        
        # Call to assertIs(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'real_err' (line 397)
        real_err_204976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 22), 'real_err', False)
        # Getting the type of 'sys' (line 397)
        sys_204977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 32), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 397)
        stderr_204978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 32), sys_204977, 'stderr')
        # Processing the call keyword arguments (line 397)
        kwargs_204979 = {}
        # Getting the type of 'self' (line 397)
        self_204974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 397)
        assertIs_204975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_204974, 'assertIs')
        # Calling assertIs(args, kwargs) (line 397)
        assertIs_call_result_204980 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), assertIs_204975, *[real_err_204976, stderr_204978], **kwargs_204979)
        
        
        # ################# End of 'testBufferOutputOff(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBufferOutputOff' in the type store
        # Getting the type of 'stypy_return_type' (line 384)
        stypy_return_type_204981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204981)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBufferOutputOff'
        return stypy_return_type_204981


    @norecursion
    def testBufferOutputStartTestAddSuccess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBufferOutputStartTestAddSuccess'
        module_type_store = module_type_store.open_function_context('testBufferOutputStartTestAddSuccess', 399, 4, False)
        # Assigning a type to the variable 'self' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.testBufferOutputStartTestAddSuccess')
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.testBufferOutputStartTestAddSuccess.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.testBufferOutputStartTestAddSuccess', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBufferOutputStartTestAddSuccess', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBufferOutputStartTestAddSuccess(...)' code ##################

        
        # Assigning a Attribute to a Name (line 400):
        
        # Assigning a Attribute to a Name (line 400):
        # Getting the type of 'self' (line 400)
        self_204982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 19), 'self')
        # Obtaining the member '_real_out' of a type (line 400)
        _real_out_204983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 19), self_204982, '_real_out')
        # Assigning a type to the variable 'real_out' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'real_out', _real_out_204983)
        
        # Assigning a Attribute to a Name (line 401):
        
        # Assigning a Attribute to a Name (line 401):
        # Getting the type of 'self' (line 401)
        self_204984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 19), 'self')
        # Obtaining the member '_real_err' of a type (line 401)
        _real_err_204985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 19), self_204984, '_real_err')
        # Assigning a type to the variable 'real_err' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'real_err', _real_err_204985)
        
        # Assigning a Call to a Name (line 403):
        
        # Assigning a Call to a Name (line 403):
        
        # Call to TestResult(...): (line 403)
        # Processing the call keyword arguments (line 403)
        kwargs_204988 = {}
        # Getting the type of 'unittest' (line 403)
        unittest_204986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 403)
        TestResult_204987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 17), unittest_204986, 'TestResult')
        # Calling TestResult(args, kwargs) (line 403)
        TestResult_call_result_204989 = invoke(stypy.reporting.localization.Localization(__file__, 403, 17), TestResult_204987, *[], **kwargs_204988)
        
        # Assigning a type to the variable 'result' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'result', TestResult_call_result_204989)
        
        # Call to assertFalse(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'result' (line 404)
        result_204992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 25), 'result', False)
        # Obtaining the member 'buffer' of a type (line 404)
        buffer_204993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 25), result_204992, 'buffer')
        # Processing the call keyword arguments (line 404)
        kwargs_204994 = {}
        # Getting the type of 'self' (line 404)
        self_204990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 404)
        assertFalse_204991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), self_204990, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 404)
        assertFalse_call_result_204995 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), assertFalse_204991, *[buffer_204993], **kwargs_204994)
        
        
        # Assigning a Name to a Attribute (line 406):
        
        # Assigning a Name to a Attribute (line 406):
        # Getting the type of 'True' (line 406)
        True_204996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 24), 'True')
        # Getting the type of 'result' (line 406)
        result_204997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'result')
        # Setting the type of the member 'buffer' of a type (line 406)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), result_204997, 'buffer', True_204996)
        
        # Call to assertIs(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'real_out' (line 408)
        real_out_205000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 22), 'real_out', False)
        # Getting the type of 'sys' (line 408)
        sys_205001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 32), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 408)
        stdout_205002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 32), sys_205001, 'stdout')
        # Processing the call keyword arguments (line 408)
        kwargs_205003 = {}
        # Getting the type of 'self' (line 408)
        self_204998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 408)
        assertIs_204999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 8), self_204998, 'assertIs')
        # Calling assertIs(args, kwargs) (line 408)
        assertIs_call_result_205004 = invoke(stypy.reporting.localization.Localization(__file__, 408, 8), assertIs_204999, *[real_out_205000, stdout_205002], **kwargs_205003)
        
        
        # Call to assertIs(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'real_err' (line 409)
        real_err_205007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'real_err', False)
        # Getting the type of 'sys' (line 409)
        sys_205008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 32), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 409)
        stderr_205009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 32), sys_205008, 'stderr')
        # Processing the call keyword arguments (line 409)
        kwargs_205010 = {}
        # Getting the type of 'self' (line 409)
        self_205005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 409)
        assertIs_205006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), self_205005, 'assertIs')
        # Calling assertIs(args, kwargs) (line 409)
        assertIs_call_result_205011 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), assertIs_205006, *[real_err_205007, stderr_205009], **kwargs_205010)
        
        
        # Call to startTest(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'self' (line 411)
        self_205014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 25), 'self', False)
        # Processing the call keyword arguments (line 411)
        kwargs_205015 = {}
        # Getting the type of 'result' (line 411)
        result_205012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'result', False)
        # Obtaining the member 'startTest' of a type (line 411)
        startTest_205013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), result_205012, 'startTest')
        # Calling startTest(args, kwargs) (line 411)
        startTest_call_result_205016 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), startTest_205013, *[self_205014], **kwargs_205015)
        
        
        # Call to assertIsNot(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'real_out' (line 413)
        real_out_205019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 25), 'real_out', False)
        # Getting the type of 'sys' (line 413)
        sys_205020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 35), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 413)
        stdout_205021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 35), sys_205020, 'stdout')
        # Processing the call keyword arguments (line 413)
        kwargs_205022 = {}
        # Getting the type of 'self' (line 413)
        self_205017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'self', False)
        # Obtaining the member 'assertIsNot' of a type (line 413)
        assertIsNot_205018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 8), self_205017, 'assertIsNot')
        # Calling assertIsNot(args, kwargs) (line 413)
        assertIsNot_call_result_205023 = invoke(stypy.reporting.localization.Localization(__file__, 413, 8), assertIsNot_205018, *[real_out_205019, stdout_205021], **kwargs_205022)
        
        
        # Call to assertIsNot(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'real_err' (line 414)
        real_err_205026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 25), 'real_err', False)
        # Getting the type of 'sys' (line 414)
        sys_205027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 35), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 414)
        stderr_205028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 35), sys_205027, 'stderr')
        # Processing the call keyword arguments (line 414)
        kwargs_205029 = {}
        # Getting the type of 'self' (line 414)
        self_205024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'self', False)
        # Obtaining the member 'assertIsNot' of a type (line 414)
        assertIsNot_205025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), self_205024, 'assertIsNot')
        # Calling assertIsNot(args, kwargs) (line 414)
        assertIsNot_call_result_205030 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), assertIsNot_205025, *[real_err_205026, stderr_205028], **kwargs_205029)
        
        
        # Call to assertIsInstance(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'sys' (line 415)
        sys_205033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 30), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 415)
        stdout_205034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 30), sys_205033, 'stdout')
        # Getting the type of 'StringIO' (line 415)
        StringIO_205035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 42), 'StringIO', False)
        # Processing the call keyword arguments (line 415)
        kwargs_205036 = {}
        # Getting the type of 'self' (line 415)
        self_205031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 415)
        assertIsInstance_205032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), self_205031, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 415)
        assertIsInstance_call_result_205037 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), assertIsInstance_205032, *[stdout_205034, StringIO_205035], **kwargs_205036)
        
        
        # Call to assertIsInstance(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'sys' (line 416)
        sys_205040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 30), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 416)
        stderr_205041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 30), sys_205040, 'stderr')
        # Getting the type of 'StringIO' (line 416)
        StringIO_205042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 42), 'StringIO', False)
        # Processing the call keyword arguments (line 416)
        kwargs_205043 = {}
        # Getting the type of 'self' (line 416)
        self_205038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 416)
        assertIsInstance_205039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), self_205038, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 416)
        assertIsInstance_call_result_205044 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), assertIsInstance_205039, *[stderr_205041, StringIO_205042], **kwargs_205043)
        
        
        # Call to assertIsNot(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'sys' (line 417)
        sys_205047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 25), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 417)
        stdout_205048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 25), sys_205047, 'stdout')
        # Getting the type of 'sys' (line 417)
        sys_205049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 37), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 417)
        stderr_205050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 37), sys_205049, 'stderr')
        # Processing the call keyword arguments (line 417)
        kwargs_205051 = {}
        # Getting the type of 'self' (line 417)
        self_205045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'self', False)
        # Obtaining the member 'assertIsNot' of a type (line 417)
        assertIsNot_205046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), self_205045, 'assertIsNot')
        # Calling assertIsNot(args, kwargs) (line 417)
        assertIsNot_call_result_205052 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), assertIsNot_205046, *[stdout_205048, stderr_205050], **kwargs_205051)
        
        
        # Assigning a Attribute to a Name (line 419):
        
        # Assigning a Attribute to a Name (line 419):
        # Getting the type of 'sys' (line 419)
        sys_205053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 21), 'sys')
        # Obtaining the member 'stdout' of a type (line 419)
        stdout_205054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 21), sys_205053, 'stdout')
        # Assigning a type to the variable 'out_stream' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'out_stream', stdout_205054)
        
        # Assigning a Attribute to a Name (line 420):
        
        # Assigning a Attribute to a Name (line 420):
        # Getting the type of 'sys' (line 420)
        sys_205055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 21), 'sys')
        # Obtaining the member 'stderr' of a type (line 420)
        stderr_205056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 21), sys_205055, 'stderr')
        # Assigning a type to the variable 'err_stream' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'err_stream', stderr_205056)
        
        # Assigning a Call to a Attribute (line 422):
        
        # Assigning a Call to a Attribute (line 422):
        
        # Call to StringIO(...): (line 422)
        # Processing the call keyword arguments (line 422)
        kwargs_205058 = {}
        # Getting the type of 'StringIO' (line 422)
        StringIO_205057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 34), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 422)
        StringIO_call_result_205059 = invoke(stypy.reporting.localization.Localization(__file__, 422, 34), StringIO_205057, *[], **kwargs_205058)
        
        # Getting the type of 'result' (line 422)
        result_205060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'result')
        # Setting the type of the member '_original_stdout' of a type (line 422)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), result_205060, '_original_stdout', StringIO_call_result_205059)
        
        # Assigning a Call to a Attribute (line 423):
        
        # Assigning a Call to a Attribute (line 423):
        
        # Call to StringIO(...): (line 423)
        # Processing the call keyword arguments (line 423)
        kwargs_205062 = {}
        # Getting the type of 'StringIO' (line 423)
        StringIO_205061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 34), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 423)
        StringIO_call_result_205063 = invoke(stypy.reporting.localization.Localization(__file__, 423, 34), StringIO_205061, *[], **kwargs_205062)
        
        # Getting the type of 'result' (line 423)
        result_205064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'result')
        # Setting the type of the member '_original_stderr' of a type (line 423)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), result_205064, '_original_stderr', StringIO_call_result_205063)
        str_205065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 14), 'str', 'foo')
        str_205066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 29), 'str', 'bar')
        
        # Call to assertEqual(...): (line 428)
        # Processing the call arguments (line 428)
        
        # Call to getvalue(...): (line 428)
        # Processing the call keyword arguments (line 428)
        kwargs_205071 = {}
        # Getting the type of 'out_stream' (line 428)
        out_stream_205069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 25), 'out_stream', False)
        # Obtaining the member 'getvalue' of a type (line 428)
        getvalue_205070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 25), out_stream_205069, 'getvalue')
        # Calling getvalue(args, kwargs) (line 428)
        getvalue_call_result_205072 = invoke(stypy.reporting.localization.Localization(__file__, 428, 25), getvalue_205070, *[], **kwargs_205071)
        
        str_205073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 48), 'str', 'foo\n')
        # Processing the call keyword arguments (line 428)
        kwargs_205074 = {}
        # Getting the type of 'self' (line 428)
        self_205067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 428)
        assertEqual_205068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), self_205067, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 428)
        assertEqual_call_result_205075 = invoke(stypy.reporting.localization.Localization(__file__, 428, 8), assertEqual_205068, *[getvalue_call_result_205072, str_205073], **kwargs_205074)
        
        
        # Call to assertEqual(...): (line 429)
        # Processing the call arguments (line 429)
        
        # Call to getvalue(...): (line 429)
        # Processing the call keyword arguments (line 429)
        kwargs_205080 = {}
        # Getting the type of 'err_stream' (line 429)
        err_stream_205078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 25), 'err_stream', False)
        # Obtaining the member 'getvalue' of a type (line 429)
        getvalue_205079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 25), err_stream_205078, 'getvalue')
        # Calling getvalue(args, kwargs) (line 429)
        getvalue_call_result_205081 = invoke(stypy.reporting.localization.Localization(__file__, 429, 25), getvalue_205079, *[], **kwargs_205080)
        
        str_205082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 48), 'str', 'bar\n')
        # Processing the call keyword arguments (line 429)
        kwargs_205083 = {}
        # Getting the type of 'self' (line 429)
        self_205076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 429)
        assertEqual_205077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), self_205076, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 429)
        assertEqual_call_result_205084 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), assertEqual_205077, *[getvalue_call_result_205081, str_205082], **kwargs_205083)
        
        
        # Call to assertEqual(...): (line 431)
        # Processing the call arguments (line 431)
        
        # Call to getvalue(...): (line 431)
        # Processing the call keyword arguments (line 431)
        kwargs_205090 = {}
        # Getting the type of 'result' (line 431)
        result_205087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 25), 'result', False)
        # Obtaining the member '_original_stdout' of a type (line 431)
        _original_stdout_205088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 25), result_205087, '_original_stdout')
        # Obtaining the member 'getvalue' of a type (line 431)
        getvalue_205089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 25), _original_stdout_205088, 'getvalue')
        # Calling getvalue(args, kwargs) (line 431)
        getvalue_call_result_205091 = invoke(stypy.reporting.localization.Localization(__file__, 431, 25), getvalue_205089, *[], **kwargs_205090)
        
        str_205092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 61), 'str', '')
        # Processing the call keyword arguments (line 431)
        kwargs_205093 = {}
        # Getting the type of 'self' (line 431)
        self_205085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 431)
        assertEqual_205086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), self_205085, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 431)
        assertEqual_call_result_205094 = invoke(stypy.reporting.localization.Localization(__file__, 431, 8), assertEqual_205086, *[getvalue_call_result_205091, str_205092], **kwargs_205093)
        
        
        # Call to assertEqual(...): (line 432)
        # Processing the call arguments (line 432)
        
        # Call to getvalue(...): (line 432)
        # Processing the call keyword arguments (line 432)
        kwargs_205100 = {}
        # Getting the type of 'result' (line 432)
        result_205097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 25), 'result', False)
        # Obtaining the member '_original_stderr' of a type (line 432)
        _original_stderr_205098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 25), result_205097, '_original_stderr')
        # Obtaining the member 'getvalue' of a type (line 432)
        getvalue_205099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 25), _original_stderr_205098, 'getvalue')
        # Calling getvalue(args, kwargs) (line 432)
        getvalue_call_result_205101 = invoke(stypy.reporting.localization.Localization(__file__, 432, 25), getvalue_205099, *[], **kwargs_205100)
        
        str_205102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 61), 'str', '')
        # Processing the call keyword arguments (line 432)
        kwargs_205103 = {}
        # Getting the type of 'self' (line 432)
        self_205095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 432)
        assertEqual_205096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), self_205095, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 432)
        assertEqual_call_result_205104 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), assertEqual_205096, *[getvalue_call_result_205101, str_205102], **kwargs_205103)
        
        
        # Call to addSuccess(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'self' (line 434)
        self_205107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 26), 'self', False)
        # Processing the call keyword arguments (line 434)
        kwargs_205108 = {}
        # Getting the type of 'result' (line 434)
        result_205105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'result', False)
        # Obtaining the member 'addSuccess' of a type (line 434)
        addSuccess_205106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), result_205105, 'addSuccess')
        # Calling addSuccess(args, kwargs) (line 434)
        addSuccess_call_result_205109 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), addSuccess_205106, *[self_205107], **kwargs_205108)
        
        
        # Call to stopTest(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'self' (line 435)
        self_205112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 24), 'self', False)
        # Processing the call keyword arguments (line 435)
        kwargs_205113 = {}
        # Getting the type of 'result' (line 435)
        result_205110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'result', False)
        # Obtaining the member 'stopTest' of a type (line 435)
        stopTest_205111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 8), result_205110, 'stopTest')
        # Calling stopTest(args, kwargs) (line 435)
        stopTest_call_result_205114 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), stopTest_205111, *[self_205112], **kwargs_205113)
        
        
        # Call to assertIs(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'sys' (line 437)
        sys_205117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 22), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 437)
        stdout_205118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 22), sys_205117, 'stdout')
        # Getting the type of 'result' (line 437)
        result_205119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 34), 'result', False)
        # Obtaining the member '_original_stdout' of a type (line 437)
        _original_stdout_205120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 34), result_205119, '_original_stdout')
        # Processing the call keyword arguments (line 437)
        kwargs_205121 = {}
        # Getting the type of 'self' (line 437)
        self_205115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 437)
        assertIs_205116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), self_205115, 'assertIs')
        # Calling assertIs(args, kwargs) (line 437)
        assertIs_call_result_205122 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), assertIs_205116, *[stdout_205118, _original_stdout_205120], **kwargs_205121)
        
        
        # Call to assertIs(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'sys' (line 438)
        sys_205125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 438)
        stderr_205126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 22), sys_205125, 'stderr')
        # Getting the type of 'result' (line 438)
        result_205127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 34), 'result', False)
        # Obtaining the member '_original_stderr' of a type (line 438)
        _original_stderr_205128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 34), result_205127, '_original_stderr')
        # Processing the call keyword arguments (line 438)
        kwargs_205129 = {}
        # Getting the type of 'self' (line 438)
        self_205123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 438)
        assertIs_205124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), self_205123, 'assertIs')
        # Calling assertIs(args, kwargs) (line 438)
        assertIs_call_result_205130 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), assertIs_205124, *[stderr_205126, _original_stderr_205128], **kwargs_205129)
        
        
        # Call to assertEqual(...): (line 440)
        # Processing the call arguments (line 440)
        
        # Call to getvalue(...): (line 440)
        # Processing the call keyword arguments (line 440)
        kwargs_205136 = {}
        # Getting the type of 'result' (line 440)
        result_205133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'result', False)
        # Obtaining the member '_original_stdout' of a type (line 440)
        _original_stdout_205134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 25), result_205133, '_original_stdout')
        # Obtaining the member 'getvalue' of a type (line 440)
        getvalue_205135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 25), _original_stdout_205134, 'getvalue')
        # Calling getvalue(args, kwargs) (line 440)
        getvalue_call_result_205137 = invoke(stypy.reporting.localization.Localization(__file__, 440, 25), getvalue_205135, *[], **kwargs_205136)
        
        str_205138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 61), 'str', '')
        # Processing the call keyword arguments (line 440)
        kwargs_205139 = {}
        # Getting the type of 'self' (line 440)
        self_205131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 440)
        assertEqual_205132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), self_205131, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 440)
        assertEqual_call_result_205140 = invoke(stypy.reporting.localization.Localization(__file__, 440, 8), assertEqual_205132, *[getvalue_call_result_205137, str_205138], **kwargs_205139)
        
        
        # Call to assertEqual(...): (line 441)
        # Processing the call arguments (line 441)
        
        # Call to getvalue(...): (line 441)
        # Processing the call keyword arguments (line 441)
        kwargs_205146 = {}
        # Getting the type of 'result' (line 441)
        result_205143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 25), 'result', False)
        # Obtaining the member '_original_stderr' of a type (line 441)
        _original_stderr_205144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 25), result_205143, '_original_stderr')
        # Obtaining the member 'getvalue' of a type (line 441)
        getvalue_205145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 25), _original_stderr_205144, 'getvalue')
        # Calling getvalue(args, kwargs) (line 441)
        getvalue_call_result_205147 = invoke(stypy.reporting.localization.Localization(__file__, 441, 25), getvalue_205145, *[], **kwargs_205146)
        
        str_205148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 61), 'str', '')
        # Processing the call keyword arguments (line 441)
        kwargs_205149 = {}
        # Getting the type of 'self' (line 441)
        self_205141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 441)
        assertEqual_205142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), self_205141, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 441)
        assertEqual_call_result_205150 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), assertEqual_205142, *[getvalue_call_result_205147, str_205148], **kwargs_205149)
        
        
        # Call to assertEqual(...): (line 443)
        # Processing the call arguments (line 443)
        
        # Call to getvalue(...): (line 443)
        # Processing the call keyword arguments (line 443)
        kwargs_205155 = {}
        # Getting the type of 'out_stream' (line 443)
        out_stream_205153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 25), 'out_stream', False)
        # Obtaining the member 'getvalue' of a type (line 443)
        getvalue_205154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 25), out_stream_205153, 'getvalue')
        # Calling getvalue(args, kwargs) (line 443)
        getvalue_call_result_205156 = invoke(stypy.reporting.localization.Localization(__file__, 443, 25), getvalue_205154, *[], **kwargs_205155)
        
        str_205157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 48), 'str', '')
        # Processing the call keyword arguments (line 443)
        kwargs_205158 = {}
        # Getting the type of 'self' (line 443)
        self_205151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 443)
        assertEqual_205152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), self_205151, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 443)
        assertEqual_call_result_205159 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), assertEqual_205152, *[getvalue_call_result_205156, str_205157], **kwargs_205158)
        
        
        # Call to assertEqual(...): (line 444)
        # Processing the call arguments (line 444)
        
        # Call to getvalue(...): (line 444)
        # Processing the call keyword arguments (line 444)
        kwargs_205164 = {}
        # Getting the type of 'err_stream' (line 444)
        err_stream_205162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 25), 'err_stream', False)
        # Obtaining the member 'getvalue' of a type (line 444)
        getvalue_205163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 25), err_stream_205162, 'getvalue')
        # Calling getvalue(args, kwargs) (line 444)
        getvalue_call_result_205165 = invoke(stypy.reporting.localization.Localization(__file__, 444, 25), getvalue_205163, *[], **kwargs_205164)
        
        str_205166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 48), 'str', '')
        # Processing the call keyword arguments (line 444)
        kwargs_205167 = {}
        # Getting the type of 'self' (line 444)
        self_205160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 444)
        assertEqual_205161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), self_205160, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 444)
        assertEqual_call_result_205168 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), assertEqual_205161, *[getvalue_call_result_205165, str_205166], **kwargs_205167)
        
        
        # ################# End of 'testBufferOutputStartTestAddSuccess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBufferOutputStartTestAddSuccess' in the type store
        # Getting the type of 'stypy_return_type' (line 399)
        stypy_return_type_205169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBufferOutputStartTestAddSuccess'
        return stypy_return_type_205169


    @norecursion
    def getStartedResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getStartedResult'
        module_type_store = module_type_store.open_function_context('getStartedResult', 447, 4, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.getStartedResult')
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.getStartedResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.getStartedResult', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getStartedResult', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getStartedResult(...)' code ##################

        
        # Assigning a Call to a Name (line 448):
        
        # Assigning a Call to a Name (line 448):
        
        # Call to TestResult(...): (line 448)
        # Processing the call keyword arguments (line 448)
        kwargs_205172 = {}
        # Getting the type of 'unittest' (line 448)
        unittest_205170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 448)
        TestResult_205171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 17), unittest_205170, 'TestResult')
        # Calling TestResult(args, kwargs) (line 448)
        TestResult_call_result_205173 = invoke(stypy.reporting.localization.Localization(__file__, 448, 17), TestResult_205171, *[], **kwargs_205172)
        
        # Assigning a type to the variable 'result' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'result', TestResult_call_result_205173)
        
        # Assigning a Name to a Attribute (line 449):
        
        # Assigning a Name to a Attribute (line 449):
        # Getting the type of 'True' (line 449)
        True_205174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 24), 'True')
        # Getting the type of 'result' (line 449)
        result_205175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'result')
        # Setting the type of the member 'buffer' of a type (line 449)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), result_205175, 'buffer', True_205174)
        
        # Call to startTest(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'self' (line 450)
        self_205178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 25), 'self', False)
        # Processing the call keyword arguments (line 450)
        kwargs_205179 = {}
        # Getting the type of 'result' (line 450)
        result_205176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'result', False)
        # Obtaining the member 'startTest' of a type (line 450)
        startTest_205177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), result_205176, 'startTest')
        # Calling startTest(args, kwargs) (line 450)
        startTest_call_result_205180 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), startTest_205177, *[self_205178], **kwargs_205179)
        
        # Getting the type of 'result' (line 451)
        result_205181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'stypy_return_type', result_205181)
        
        # ################# End of 'getStartedResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getStartedResult' in the type store
        # Getting the type of 'stypy_return_type' (line 447)
        stypy_return_type_205182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205182)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getStartedResult'
        return stypy_return_type_205182


    @norecursion
    def testBufferOutputAddErrorOrFailure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBufferOutputAddErrorOrFailure'
        module_type_store = module_type_store.open_function_context('testBufferOutputAddErrorOrFailure', 453, 4, False)
        # Assigning a type to the variable 'self' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.testBufferOutputAddErrorOrFailure')
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.testBufferOutputAddErrorOrFailure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.testBufferOutputAddErrorOrFailure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBufferOutputAddErrorOrFailure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBufferOutputAddErrorOrFailure(...)' code ##################

        
        # Assigning a Name to a Attribute (line 454):
        
        # Assigning a Name to a Attribute (line 454):
        # Getting the type of 'MockTraceback' (line 454)
        MockTraceback_205183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 36), 'MockTraceback')
        # Getting the type of 'unittest' (line 454)
        unittest_205184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'unittest')
        # Obtaining the member 'result' of a type (line 454)
        result_205185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), unittest_205184, 'result')
        # Setting the type of the member 'traceback' of a type (line 454)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), result_205185, 'traceback', MockTraceback_205183)
        
        # Call to addCleanup(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 'restore_traceback' (line 455)
        restore_traceback_205188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 24), 'restore_traceback', False)
        # Processing the call keyword arguments (line 455)
        kwargs_205189 = {}
        # Getting the type of 'self' (line 455)
        self_205186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 455)
        addCleanup_205187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), self_205186, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 455)
        addCleanup_call_result_205190 = invoke(stypy.reporting.localization.Localization(__file__, 455, 8), addCleanup_205187, *[restore_traceback_205188], **kwargs_205189)
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 457)
        list_205191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 457)
        # Adding element type (line 457)
        
        # Obtaining an instance of the builtin type 'tuple' (line 458)
        tuple_205192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 458)
        # Adding element type (line 458)
        str_205193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 13), 'str', 'errors')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 13), tuple_205192, str_205193)
        # Adding element type (line 458)
        str_205194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 23), 'str', 'addError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 13), tuple_205192, str_205194)
        # Adding element type (line 458)
        # Getting the type of 'True' (line 458)
        True_205195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 35), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 13), tuple_205192, True_205195)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 53), list_205191, tuple_205192)
        # Adding element type (line 457)
        
        # Obtaining an instance of the builtin type 'tuple' (line 459)
        tuple_205196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 459)
        # Adding element type (line 459)
        str_205197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 13), 'str', 'failures')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 13), tuple_205196, str_205197)
        # Adding element type (line 459)
        str_205198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 25), 'str', 'addFailure')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 13), tuple_205196, str_205198)
        # Adding element type (line 459)
        # Getting the type of 'False' (line 459)
        False_205199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 39), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 13), tuple_205196, False_205199)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 53), list_205191, tuple_205196)
        # Adding element type (line 457)
        
        # Obtaining an instance of the builtin type 'tuple' (line 460)
        tuple_205200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 460)
        # Adding element type (line 460)
        str_205201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 13), 'str', 'errors')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 13), tuple_205200, str_205201)
        # Adding element type (line 460)
        str_205202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 23), 'str', 'addError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 13), tuple_205200, str_205202)
        # Adding element type (line 460)
        # Getting the type of 'True' (line 460)
        True_205203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 35), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 13), tuple_205200, True_205203)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 53), list_205191, tuple_205200)
        # Adding element type (line 457)
        
        # Obtaining an instance of the builtin type 'tuple' (line 461)
        tuple_205204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 461)
        # Adding element type (line 461)
        str_205205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 13), 'str', 'failures')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 13), tuple_205204, str_205205)
        # Adding element type (line 461)
        str_205206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 25), 'str', 'addFailure')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 13), tuple_205204, str_205206)
        # Adding element type (line 461)
        # Getting the type of 'False' (line 461)
        False_205207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 39), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 13), tuple_205204, False_205207)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 53), list_205191, tuple_205204)
        
        # Testing the type of a for loop iterable (line 457)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 457, 8), list_205191)
        # Getting the type of the for loop variable (line 457)
        for_loop_var_205208 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 457, 8), list_205191)
        # Assigning a type to the variable 'message_attr' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'message_attr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 8), for_loop_var_205208))
        # Assigning a type to the variable 'add_attr' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'add_attr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 8), for_loop_var_205208))
        # Assigning a type to the variable 'include_error' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'include_error', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 8), for_loop_var_205208))
        # SSA begins for a for statement (line 457)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to getStartedResult(...): (line 463)
        # Processing the call keyword arguments (line 463)
        kwargs_205211 = {}
        # Getting the type of 'self' (line 463)
        self_205209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 21), 'self', False)
        # Obtaining the member 'getStartedResult' of a type (line 463)
        getStartedResult_205210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 21), self_205209, 'getStartedResult')
        # Calling getStartedResult(args, kwargs) (line 463)
        getStartedResult_call_result_205212 = invoke(stypy.reporting.localization.Localization(__file__, 463, 21), getStartedResult_205210, *[], **kwargs_205211)
        
        # Assigning a type to the variable 'result' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'result', getStartedResult_call_result_205212)
        
        # Assigning a Attribute to a Name (line 464):
        
        # Assigning a Attribute to a Name (line 464):
        # Getting the type of 'sys' (line 464)
        sys_205213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 27), 'sys')
        # Obtaining the member 'stdout' of a type (line 464)
        stdout_205214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 27), sys_205213, 'stdout')
        # Assigning a type to the variable 'buffered_out' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'buffered_out', stdout_205214)
        
        # Assigning a Attribute to a Name (line 465):
        
        # Assigning a Attribute to a Name (line 465):
        # Getting the type of 'sys' (line 465)
        sys_205215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 27), 'sys')
        # Obtaining the member 'stderr' of a type (line 465)
        stderr_205216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 27), sys_205215, 'stderr')
        # Assigning a type to the variable 'buffered_err' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'buffered_err', stderr_205216)
        
        # Assigning a Call to a Attribute (line 466):
        
        # Assigning a Call to a Attribute (line 466):
        
        # Call to StringIO(...): (line 466)
        # Processing the call keyword arguments (line 466)
        kwargs_205218 = {}
        # Getting the type of 'StringIO' (line 466)
        StringIO_205217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 38), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 466)
        StringIO_call_result_205219 = invoke(stypy.reporting.localization.Localization(__file__, 466, 38), StringIO_205217, *[], **kwargs_205218)
        
        # Getting the type of 'result' (line 466)
        result_205220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'result')
        # Setting the type of the member '_original_stdout' of a type (line 466)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), result_205220, '_original_stdout', StringIO_call_result_205219)
        
        # Assigning a Call to a Attribute (line 467):
        
        # Assigning a Call to a Attribute (line 467):
        
        # Call to StringIO(...): (line 467)
        # Processing the call keyword arguments (line 467)
        kwargs_205222 = {}
        # Getting the type of 'StringIO' (line 467)
        StringIO_205221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 38), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 467)
        StringIO_call_result_205223 = invoke(stypy.reporting.localization.Localization(__file__, 467, 38), StringIO_205221, *[], **kwargs_205222)
        
        # Getting the type of 'result' (line 467)
        result_205224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'result')
        # Setting the type of the member '_original_stderr' of a type (line 467)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 12), result_205224, '_original_stderr', StringIO_call_result_205223)
        str_205225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 33), 'str', 'foo')
        
        # Getting the type of 'include_error' (line 470)
        include_error_205226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'include_error')
        # Testing the type of an if condition (line 470)
        if_condition_205227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 12), include_error_205226)
        # Assigning a type to the variable 'if_condition_205227' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'if_condition_205227', if_condition_205227)
        # SSA begins for if statement (line 470)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_205228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 37), 'str', 'bar')
        # SSA join for if statement (line 470)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 474):
        
        # Assigning a Call to a Name (line 474):
        
        # Call to getattr(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'result' (line 474)
        result_205230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 34), 'result', False)
        # Getting the type of 'add_attr' (line 474)
        add_attr_205231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 42), 'add_attr', False)
        # Processing the call keyword arguments (line 474)
        kwargs_205232 = {}
        # Getting the type of 'getattr' (line 474)
        getattr_205229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 26), 'getattr', False)
        # Calling getattr(args, kwargs) (line 474)
        getattr_call_result_205233 = invoke(stypy.reporting.localization.Localization(__file__, 474, 26), getattr_205229, *[result_205230, add_attr_205231], **kwargs_205232)
        
        # Assigning a type to the variable 'addFunction' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'addFunction', getattr_call_result_205233)
        
        # Call to addFunction(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'self' (line 475)
        self_205235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 24), 'self', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 475)
        tuple_205236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 475)
        # Adding element type (line 475)
        # Getting the type of 'None' (line 475)
        None_205237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 31), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 31), tuple_205236, None_205237)
        # Adding element type (line 475)
        # Getting the type of 'None' (line 475)
        None_205238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 37), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 31), tuple_205236, None_205238)
        # Adding element type (line 475)
        # Getting the type of 'None' (line 475)
        None_205239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 43), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 31), tuple_205236, None_205239)
        
        # Processing the call keyword arguments (line 475)
        kwargs_205240 = {}
        # Getting the type of 'addFunction' (line 475)
        addFunction_205234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'addFunction', False)
        # Calling addFunction(args, kwargs) (line 475)
        addFunction_call_result_205241 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), addFunction_205234, *[self_205235, tuple_205236], **kwargs_205240)
        
        
        # Call to stopTest(...): (line 476)
        # Processing the call arguments (line 476)
        # Getting the type of 'self' (line 476)
        self_205244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 28), 'self', False)
        # Processing the call keyword arguments (line 476)
        kwargs_205245 = {}
        # Getting the type of 'result' (line 476)
        result_205242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'result', False)
        # Obtaining the member 'stopTest' of a type (line 476)
        stopTest_205243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 12), result_205242, 'stopTest')
        # Calling stopTest(args, kwargs) (line 476)
        stopTest_call_result_205246 = invoke(stypy.reporting.localization.Localization(__file__, 476, 12), stopTest_205243, *[self_205244], **kwargs_205245)
        
        
        # Assigning a Call to a Name (line 478):
        
        # Assigning a Call to a Name (line 478):
        
        # Call to getattr(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'result' (line 478)
        result_205248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 34), 'result', False)
        # Getting the type of 'message_attr' (line 478)
        message_attr_205249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 42), 'message_attr', False)
        # Processing the call keyword arguments (line 478)
        kwargs_205250 = {}
        # Getting the type of 'getattr' (line 478)
        getattr_205247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 26), 'getattr', False)
        # Calling getattr(args, kwargs) (line 478)
        getattr_call_result_205251 = invoke(stypy.reporting.localization.Localization(__file__, 478, 26), getattr_205247, *[result_205248, message_attr_205249], **kwargs_205250)
        
        # Assigning a type to the variable 'result_list' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'result_list', getattr_call_result_205251)
        
        # Call to assertEqual(...): (line 479)
        # Processing the call arguments (line 479)
        
        # Call to len(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'result_list' (line 479)
        result_list_205255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 33), 'result_list', False)
        # Processing the call keyword arguments (line 479)
        kwargs_205256 = {}
        # Getting the type of 'len' (line 479)
        len_205254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 29), 'len', False)
        # Calling len(args, kwargs) (line 479)
        len_call_result_205257 = invoke(stypy.reporting.localization.Localization(__file__, 479, 29), len_205254, *[result_list_205255], **kwargs_205256)
        
        int_205258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 47), 'int')
        # Processing the call keyword arguments (line 479)
        kwargs_205259 = {}
        # Getting the type of 'self' (line 479)
        self_205252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 479)
        assertEqual_205253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), self_205252, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 479)
        assertEqual_call_result_205260 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), assertEqual_205253, *[len_call_result_205257, int_205258], **kwargs_205259)
        
        
        # Assigning a Subscript to a Tuple (line 481):
        
        # Assigning a Subscript to a Name (line 481):
        
        # Obtaining the type of the subscript
        int_205261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 12), 'int')
        
        # Obtaining the type of the subscript
        int_205262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 40), 'int')
        # Getting the type of 'result_list' (line 481)
        result_list_205263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 28), 'result_list')
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___205264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 28), result_list_205263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_205265 = invoke(stypy.reporting.localization.Localization(__file__, 481, 28), getitem___205264, int_205262)
        
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___205266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), subscript_call_result_205265, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_205267 = invoke(stypy.reporting.localization.Localization(__file__, 481, 12), getitem___205266, int_205261)
        
        # Assigning a type to the variable 'tuple_var_assignment_204014' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'tuple_var_assignment_204014', subscript_call_result_205267)
        
        # Assigning a Subscript to a Name (line 481):
        
        # Obtaining the type of the subscript
        int_205268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 12), 'int')
        
        # Obtaining the type of the subscript
        int_205269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 40), 'int')
        # Getting the type of 'result_list' (line 481)
        result_list_205270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 28), 'result_list')
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___205271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 28), result_list_205270, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_205272 = invoke(stypy.reporting.localization.Localization(__file__, 481, 28), getitem___205271, int_205269)
        
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___205273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 12), subscript_call_result_205272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_205274 = invoke(stypy.reporting.localization.Localization(__file__, 481, 12), getitem___205273, int_205268)
        
        # Assigning a type to the variable 'tuple_var_assignment_204015' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'tuple_var_assignment_204015', subscript_call_result_205274)
        
        # Assigning a Name to a Name (line 481):
        # Getting the type of 'tuple_var_assignment_204014' (line 481)
        tuple_var_assignment_204014_205275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'tuple_var_assignment_204014')
        # Assigning a type to the variable 'test' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'test', tuple_var_assignment_204014_205275)
        
        # Assigning a Name to a Name (line 481):
        # Getting the type of 'tuple_var_assignment_204015' (line 481)
        tuple_var_assignment_204015_205276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'tuple_var_assignment_204015')
        # Assigning a type to the variable 'message' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 18), 'message', tuple_var_assignment_204015_205276)
        
        # Assigning a Call to a Name (line 482):
        
        # Assigning a Call to a Name (line 482):
        
        # Call to dedent(...): (line 482)
        # Processing the call arguments (line 482)
        str_205279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, (-1)), 'str', '\n                Stdout:\n                foo\n            ')
        # Processing the call keyword arguments (line 482)
        kwargs_205280 = {}
        # Getting the type of 'textwrap' (line 482)
        textwrap_205277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 33), 'textwrap', False)
        # Obtaining the member 'dedent' of a type (line 482)
        dedent_205278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 33), textwrap_205277, 'dedent')
        # Calling dedent(args, kwargs) (line 482)
        dedent_call_result_205281 = invoke(stypy.reporting.localization.Localization(__file__, 482, 33), dedent_205278, *[str_205279], **kwargs_205280)
        
        # Assigning a type to the variable 'expectedOutMessage' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'expectedOutMessage', dedent_call_result_205281)
        
        # Assigning a Str to a Name (line 486):
        
        # Assigning a Str to a Name (line 486):
        str_205282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 33), 'str', '')
        # Assigning a type to the variable 'expectedErrMessage' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'expectedErrMessage', str_205282)
        
        # Getting the type of 'include_error' (line 487)
        include_error_205283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 15), 'include_error')
        # Testing the type of an if condition (line 487)
        if_condition_205284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 487, 12), include_error_205283)
        # Assigning a type to the variable 'if_condition_205284' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'if_condition_205284', if_condition_205284)
        # SSA begins for if statement (line 487)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 488):
        
        # Assigning a Call to a Name (line 488):
        
        # Call to dedent(...): (line 488)
        # Processing the call arguments (line 488)
        str_205287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, (-1)), 'str', '\n                Stderr:\n                bar\n            ')
        # Processing the call keyword arguments (line 488)
        kwargs_205288 = {}
        # Getting the type of 'textwrap' (line 488)
        textwrap_205285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 37), 'textwrap', False)
        # Obtaining the member 'dedent' of a type (line 488)
        dedent_205286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 37), textwrap_205285, 'dedent')
        # Calling dedent(args, kwargs) (line 488)
        dedent_call_result_205289 = invoke(stypy.reporting.localization.Localization(__file__, 488, 37), dedent_205286, *[str_205287], **kwargs_205288)
        
        # Assigning a type to the variable 'expectedErrMessage' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 16), 'expectedErrMessage', dedent_call_result_205289)
        # SSA join for if statement (line 487)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 492):
        
        # Assigning a BinOp to a Name (line 492):
        str_205290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 34), 'str', 'A traceback%s%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 492)
        tuple_205291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 492)
        # Adding element type (line 492)
        # Getting the type of 'expectedOutMessage' (line 492)
        expectedOutMessage_205292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 55), 'expectedOutMessage')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 55), tuple_205291, expectedOutMessage_205292)
        # Adding element type (line 492)
        # Getting the type of 'expectedErrMessage' (line 492)
        expectedErrMessage_205293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 75), 'expectedErrMessage')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 492, 55), tuple_205291, expectedErrMessage_205293)
        
        # Applying the binary operator '%' (line 492)
        result_mod_205294 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 34), '%', str_205290, tuple_205291)
        
        # Assigning a type to the variable 'expectedFullMessage' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'expectedFullMessage', result_mod_205294)
        
        # Call to assertIs(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'test' (line 494)
        test_205297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 26), 'test', False)
        # Getting the type of 'self' (line 494)
        self_205298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 32), 'self', False)
        # Processing the call keyword arguments (line 494)
        kwargs_205299 = {}
        # Getting the type of 'self' (line 494)
        self_205295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 494)
        assertIs_205296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 12), self_205295, 'assertIs')
        # Calling assertIs(args, kwargs) (line 494)
        assertIs_call_result_205300 = invoke(stypy.reporting.localization.Localization(__file__, 494, 12), assertIs_205296, *[test_205297, self_205298], **kwargs_205299)
        
        
        # Call to assertEqual(...): (line 495)
        # Processing the call arguments (line 495)
        
        # Call to getvalue(...): (line 495)
        # Processing the call keyword arguments (line 495)
        kwargs_205306 = {}
        # Getting the type of 'result' (line 495)
        result_205303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 29), 'result', False)
        # Obtaining the member '_original_stdout' of a type (line 495)
        _original_stdout_205304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 29), result_205303, '_original_stdout')
        # Obtaining the member 'getvalue' of a type (line 495)
        getvalue_205305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 29), _original_stdout_205304, 'getvalue')
        # Calling getvalue(args, kwargs) (line 495)
        getvalue_call_result_205307 = invoke(stypy.reporting.localization.Localization(__file__, 495, 29), getvalue_205305, *[], **kwargs_205306)
        
        # Getting the type of 'expectedOutMessage' (line 495)
        expectedOutMessage_205308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 65), 'expectedOutMessage', False)
        # Processing the call keyword arguments (line 495)
        kwargs_205309 = {}
        # Getting the type of 'self' (line 495)
        self_205301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 495)
        assertEqual_205302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 12), self_205301, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 495)
        assertEqual_call_result_205310 = invoke(stypy.reporting.localization.Localization(__file__, 495, 12), assertEqual_205302, *[getvalue_call_result_205307, expectedOutMessage_205308], **kwargs_205309)
        
        
        # Call to assertEqual(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Call to getvalue(...): (line 496)
        # Processing the call keyword arguments (line 496)
        kwargs_205316 = {}
        # Getting the type of 'result' (line 496)
        result_205313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 29), 'result', False)
        # Obtaining the member '_original_stderr' of a type (line 496)
        _original_stderr_205314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 29), result_205313, '_original_stderr')
        # Obtaining the member 'getvalue' of a type (line 496)
        getvalue_205315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 29), _original_stderr_205314, 'getvalue')
        # Calling getvalue(args, kwargs) (line 496)
        getvalue_call_result_205317 = invoke(stypy.reporting.localization.Localization(__file__, 496, 29), getvalue_205315, *[], **kwargs_205316)
        
        # Getting the type of 'expectedErrMessage' (line 496)
        expectedErrMessage_205318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 65), 'expectedErrMessage', False)
        # Processing the call keyword arguments (line 496)
        kwargs_205319 = {}
        # Getting the type of 'self' (line 496)
        self_205311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 496)
        assertEqual_205312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 12), self_205311, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 496)
        assertEqual_call_result_205320 = invoke(stypy.reporting.localization.Localization(__file__, 496, 12), assertEqual_205312, *[getvalue_call_result_205317, expectedErrMessage_205318], **kwargs_205319)
        
        
        # Call to assertMultiLineEqual(...): (line 497)
        # Processing the call arguments (line 497)
        # Getting the type of 'message' (line 497)
        message_205323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 38), 'message', False)
        # Getting the type of 'expectedFullMessage' (line 497)
        expectedFullMessage_205324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 47), 'expectedFullMessage', False)
        # Processing the call keyword arguments (line 497)
        kwargs_205325 = {}
        # Getting the type of 'self' (line 497)
        self_205321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'self', False)
        # Obtaining the member 'assertMultiLineEqual' of a type (line 497)
        assertMultiLineEqual_205322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 12), self_205321, 'assertMultiLineEqual')
        # Calling assertMultiLineEqual(args, kwargs) (line 497)
        assertMultiLineEqual_call_result_205326 = invoke(stypy.reporting.localization.Localization(__file__, 497, 12), assertMultiLineEqual_205322, *[message_205323, expectedFullMessage_205324], **kwargs_205325)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'testBufferOutputAddErrorOrFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBufferOutputAddErrorOrFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 453)
        stypy_return_type_205327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205327)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBufferOutputAddErrorOrFailure'
        return stypy_return_type_205327


    @norecursion
    def testBufferSetupClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBufferSetupClass'
        module_type_store = module_type_store.open_function_context('testBufferSetupClass', 499, 4, False)
        # Assigning a type to the variable 'self' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.testBufferSetupClass')
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.testBufferSetupClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.testBufferSetupClass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBufferSetupClass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBufferSetupClass(...)' code ##################

        
        # Assigning a Call to a Name (line 500):
        
        # Assigning a Call to a Name (line 500):
        
        # Call to TestResult(...): (line 500)
        # Processing the call keyword arguments (line 500)
        kwargs_205330 = {}
        # Getting the type of 'unittest' (line 500)
        unittest_205328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 500)
        TestResult_205329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 17), unittest_205328, 'TestResult')
        # Calling TestResult(args, kwargs) (line 500)
        TestResult_call_result_205331 = invoke(stypy.reporting.localization.Localization(__file__, 500, 17), TestResult_205329, *[], **kwargs_205330)
        
        # Assigning a type to the variable 'result' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'result', TestResult_call_result_205331)
        
        # Assigning a Name to a Attribute (line 501):
        
        # Assigning a Name to a Attribute (line 501):
        # Getting the type of 'True' (line 501)
        True_205332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 24), 'True')
        # Getting the type of 'result' (line 501)
        result_205333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'result')
        # Setting the type of the member 'buffer' of a type (line 501)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), result_205333, 'buffer', True_205332)
        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 503)
        unittest_205334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 503)
        TestCase_205335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 18), unittest_205334, 'TestCase')

        class Foo(TestCase_205335, ):

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 504, 12, False)
                # Assigning a type to the variable 'self' (line 505)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Foo.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.setUpClass.__dict__.__setitem__('stypy_function_name', 'Foo.setUpClass')
                Foo.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                int_205336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 16), 'int')
                int_205337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 19), 'int')
                # Applying the binary operator '//' (line 506)
                result_floordiv_205338 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 16), '//', int_205336, int_205337)
                
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 504)
                stypy_return_type_205339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205339)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_205339


            @norecursion
            def test_foo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_foo'
                module_type_store = module_type_store.open_function_context('test_foo', 507, 12, False)
                # Assigning a type to the variable 'self' (line 508)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_foo.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_foo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_foo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_foo.__dict__.__setitem__('stypy_function_name', 'Foo.test_foo')
                Foo.test_foo.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_foo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_foo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_foo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_foo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_foo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_foo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_foo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_foo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_foo(...)' code ##################

                pass
                
                # ################# End of 'test_foo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_foo' in the type store
                # Getting the type of 'stypy_return_type' (line 507)
                stypy_return_type_205340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205340)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_foo'
                return stypy_return_type_205340

        
        # Assigning a type to the variable 'Foo' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 509):
        
        # Assigning a Call to a Name (line 509):
        
        # Call to TestSuite(...): (line 509)
        # Processing the call arguments (line 509)
        
        # Obtaining an instance of the builtin type 'list' (line 509)
        list_205343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 509)
        # Adding element type (line 509)
        
        # Call to Foo(...): (line 509)
        # Processing the call arguments (line 509)
        str_205345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 40), 'str', 'test_foo')
        # Processing the call keyword arguments (line 509)
        kwargs_205346 = {}
        # Getting the type of 'Foo' (line 509)
        Foo_205344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 36), 'Foo', False)
        # Calling Foo(args, kwargs) (line 509)
        Foo_call_result_205347 = invoke(stypy.reporting.localization.Localization(__file__, 509, 36), Foo_205344, *[str_205345], **kwargs_205346)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 35), list_205343, Foo_call_result_205347)
        
        # Processing the call keyword arguments (line 509)
        kwargs_205348 = {}
        # Getting the type of 'unittest' (line 509)
        unittest_205341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 509)
        TestSuite_205342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 16), unittest_205341, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 509)
        TestSuite_call_result_205349 = invoke(stypy.reporting.localization.Localization(__file__, 509, 16), TestSuite_205342, *[list_205343], **kwargs_205348)
        
        # Assigning a type to the variable 'suite' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'suite', TestSuite_call_result_205349)
        
        # Call to suite(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'result' (line 510)
        result_205351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 14), 'result', False)
        # Processing the call keyword arguments (line 510)
        kwargs_205352 = {}
        # Getting the type of 'suite' (line 510)
        suite_205350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'suite', False)
        # Calling suite(args, kwargs) (line 510)
        suite_call_result_205353 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), suite_205350, *[result_205351], **kwargs_205352)
        
        
        # Call to assertEqual(...): (line 511)
        # Processing the call arguments (line 511)
        
        # Call to len(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'result' (line 511)
        result_205357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 511)
        errors_205358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 29), result_205357, 'errors')
        # Processing the call keyword arguments (line 511)
        kwargs_205359 = {}
        # Getting the type of 'len' (line 511)
        len_205356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 25), 'len', False)
        # Calling len(args, kwargs) (line 511)
        len_call_result_205360 = invoke(stypy.reporting.localization.Localization(__file__, 511, 25), len_205356, *[errors_205358], **kwargs_205359)
        
        int_205361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 45), 'int')
        # Processing the call keyword arguments (line 511)
        kwargs_205362 = {}
        # Getting the type of 'self' (line 511)
        self_205354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 511)
        assertEqual_205355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 8), self_205354, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 511)
        assertEqual_call_result_205363 = invoke(stypy.reporting.localization.Localization(__file__, 511, 8), assertEqual_205355, *[len_call_result_205360, int_205361], **kwargs_205362)
        
        
        # ################# End of 'testBufferSetupClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBufferSetupClass' in the type store
        # Getting the type of 'stypy_return_type' (line 499)
        stypy_return_type_205364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBufferSetupClass'
        return stypy_return_type_205364


    @norecursion
    def testBufferTearDownClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBufferTearDownClass'
        module_type_store = module_type_store.open_function_context('testBufferTearDownClass', 513, 4, False)
        # Assigning a type to the variable 'self' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.testBufferTearDownClass')
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.testBufferTearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.testBufferTearDownClass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBufferTearDownClass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBufferTearDownClass(...)' code ##################

        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Call to TestResult(...): (line 514)
        # Processing the call keyword arguments (line 514)
        kwargs_205367 = {}
        # Getting the type of 'unittest' (line 514)
        unittest_205365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 514)
        TestResult_205366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 17), unittest_205365, 'TestResult')
        # Calling TestResult(args, kwargs) (line 514)
        TestResult_call_result_205368 = invoke(stypy.reporting.localization.Localization(__file__, 514, 17), TestResult_205366, *[], **kwargs_205367)
        
        # Assigning a type to the variable 'result' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'result', TestResult_call_result_205368)
        
        # Assigning a Name to a Attribute (line 515):
        
        # Assigning a Name to a Attribute (line 515):
        # Getting the type of 'True' (line 515)
        True_205369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 24), 'True')
        # Getting the type of 'result' (line 515)
        result_205370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'result')
        # Setting the type of the member 'buffer' of a type (line 515)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 8), result_205370, 'buffer', True_205369)
        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 517)
        unittest_205371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 517)
        TestCase_205372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 18), unittest_205371, 'TestCase')

        class Foo(TestCase_205372, ):

            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 518, 12, False)
                # Assigning a type to the variable 'self' (line 519)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Foo.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Foo.tearDownClass')
                Foo.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                int_205373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 16), 'int')
                int_205374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 19), 'int')
                # Applying the binary operator '//' (line 520)
                result_floordiv_205375 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 16), '//', int_205373, int_205374)
                
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 518)
                stypy_return_type_205376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205376)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_205376


            @norecursion
            def test_foo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_foo'
                module_type_store = module_type_store.open_function_context('test_foo', 521, 12, False)
                # Assigning a type to the variable 'self' (line 522)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_foo.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_foo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_foo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_foo.__dict__.__setitem__('stypy_function_name', 'Foo.test_foo')
                Foo.test_foo.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_foo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_foo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_foo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_foo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_foo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_foo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_foo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_foo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_foo(...)' code ##################

                pass
                
                # ################# End of 'test_foo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_foo' in the type store
                # Getting the type of 'stypy_return_type' (line 521)
                stypy_return_type_205377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205377)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_foo'
                return stypy_return_type_205377

        
        # Assigning a type to the variable 'Foo' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 523):
        
        # Assigning a Call to a Name (line 523):
        
        # Call to TestSuite(...): (line 523)
        # Processing the call arguments (line 523)
        
        # Obtaining an instance of the builtin type 'list' (line 523)
        list_205380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 523)
        # Adding element type (line 523)
        
        # Call to Foo(...): (line 523)
        # Processing the call arguments (line 523)
        str_205382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 40), 'str', 'test_foo')
        # Processing the call keyword arguments (line 523)
        kwargs_205383 = {}
        # Getting the type of 'Foo' (line 523)
        Foo_205381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 36), 'Foo', False)
        # Calling Foo(args, kwargs) (line 523)
        Foo_call_result_205384 = invoke(stypy.reporting.localization.Localization(__file__, 523, 36), Foo_205381, *[str_205382], **kwargs_205383)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 35), list_205380, Foo_call_result_205384)
        
        # Processing the call keyword arguments (line 523)
        kwargs_205385 = {}
        # Getting the type of 'unittest' (line 523)
        unittest_205378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 523)
        TestSuite_205379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 16), unittest_205378, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 523)
        TestSuite_call_result_205386 = invoke(stypy.reporting.localization.Localization(__file__, 523, 16), TestSuite_205379, *[list_205380], **kwargs_205385)
        
        # Assigning a type to the variable 'suite' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'suite', TestSuite_call_result_205386)
        
        # Call to suite(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'result' (line 524)
        result_205388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 14), 'result', False)
        # Processing the call keyword arguments (line 524)
        kwargs_205389 = {}
        # Getting the type of 'suite' (line 524)
        suite_205387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'suite', False)
        # Calling suite(args, kwargs) (line 524)
        suite_call_result_205390 = invoke(stypy.reporting.localization.Localization(__file__, 524, 8), suite_205387, *[result_205388], **kwargs_205389)
        
        
        # Call to assertEqual(...): (line 525)
        # Processing the call arguments (line 525)
        
        # Call to len(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'result' (line 525)
        result_205394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 525)
        errors_205395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 29), result_205394, 'errors')
        # Processing the call keyword arguments (line 525)
        kwargs_205396 = {}
        # Getting the type of 'len' (line 525)
        len_205393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'len', False)
        # Calling len(args, kwargs) (line 525)
        len_call_result_205397 = invoke(stypy.reporting.localization.Localization(__file__, 525, 25), len_205393, *[errors_205395], **kwargs_205396)
        
        int_205398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 45), 'int')
        # Processing the call keyword arguments (line 525)
        kwargs_205399 = {}
        # Getting the type of 'self' (line 525)
        self_205391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 525)
        assertEqual_205392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 8), self_205391, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 525)
        assertEqual_call_result_205400 = invoke(stypy.reporting.localization.Localization(__file__, 525, 8), assertEqual_205392, *[len_call_result_205397, int_205398], **kwargs_205399)
        
        
        # ################# End of 'testBufferTearDownClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBufferTearDownClass' in the type store
        # Getting the type of 'stypy_return_type' (line 513)
        stypy_return_type_205401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205401)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBufferTearDownClass'
        return stypy_return_type_205401


    @norecursion
    def testBufferSetUpModule(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBufferSetUpModule'
        module_type_store = module_type_store.open_function_context('testBufferSetUpModule', 527, 4, False)
        # Assigning a type to the variable 'self' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.testBufferSetUpModule')
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.testBufferSetUpModule.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.testBufferSetUpModule', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBufferSetUpModule', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBufferSetUpModule(...)' code ##################

        
        # Assigning a Call to a Name (line 528):
        
        # Assigning a Call to a Name (line 528):
        
        # Call to TestResult(...): (line 528)
        # Processing the call keyword arguments (line 528)
        kwargs_205404 = {}
        # Getting the type of 'unittest' (line 528)
        unittest_205402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 528)
        TestResult_205403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 17), unittest_205402, 'TestResult')
        # Calling TestResult(args, kwargs) (line 528)
        TestResult_call_result_205405 = invoke(stypy.reporting.localization.Localization(__file__, 528, 17), TestResult_205403, *[], **kwargs_205404)
        
        # Assigning a type to the variable 'result' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'result', TestResult_call_result_205405)
        
        # Assigning a Name to a Attribute (line 529):
        
        # Assigning a Name to a Attribute (line 529):
        # Getting the type of 'True' (line 529)
        True_205406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 24), 'True')
        # Getting the type of 'result' (line 529)
        result_205407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'result')
        # Setting the type of the member 'buffer' of a type (line 529)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 8), result_205407, 'buffer', True_205406)
        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 531)
        unittest_205408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 531)
        TestCase_205409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 18), unittest_205408, 'TestCase')

        class Foo(TestCase_205409, ):

            @norecursion
            def test_foo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_foo'
                module_type_store = module_type_store.open_function_context('test_foo', 532, 12, False)
                # Assigning a type to the variable 'self' (line 533)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_foo.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_foo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_foo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_foo.__dict__.__setitem__('stypy_function_name', 'Foo.test_foo')
                Foo.test_foo.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_foo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_foo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_foo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_foo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_foo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_foo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_foo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_foo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_foo(...)' code ##################

                pass
                
                # ################# End of 'test_foo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_foo' in the type store
                # Getting the type of 'stypy_return_type' (line 532)
                stypy_return_type_205410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205410)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_foo'
                return stypy_return_type_205410

        
        # Assigning a type to the variable 'Foo' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'Foo', Foo)
        # Declaration of the 'Module' class

        class Module(object, ):

            @staticmethod
            @norecursion
            def setUpModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpModule'
                module_type_store = module_type_store.open_function_context('setUpModule', 535, 12, False)
                
                # Passed parameters checking function
                Module.setUpModule.__dict__.__setitem__('stypy_localization', localization)
                Module.setUpModule.__dict__.__setitem__('stypy_type_of_self', None)
                Module.setUpModule.__dict__.__setitem__('stypy_type_store', module_type_store)
                Module.setUpModule.__dict__.__setitem__('stypy_function_name', 'setUpModule')
                Module.setUpModule.__dict__.__setitem__('stypy_param_names_list', [])
                Module.setUpModule.__dict__.__setitem__('stypy_varargs_param_name', None)
                Module.setUpModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Module.setUpModule.__dict__.__setitem__('stypy_call_defaults', defaults)
                Module.setUpModule.__dict__.__setitem__('stypy_call_varargs', varargs)
                Module.setUpModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Module.setUpModule.__dict__.__setitem__('stypy_declared_arg_number', 0)
                arguments = process_argument_values(localization, None, module_type_store, 'setUpModule', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'setUpModule', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'setUpModule(...)' code ##################

                int_205411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 16), 'int')
                int_205412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 19), 'int')
                # Applying the binary operator '//' (line 537)
                result_floordiv_205413 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 16), '//', int_205411, int_205412)
                
                
                # ################# End of 'setUpModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpModule' in the type store
                # Getting the type of 'stypy_return_type' (line 535)
                stypy_return_type_205414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205414)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpModule'
                return stypy_return_type_205414

        
        # Assigning a type to the variable 'Module' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'Module', Module)
        
        # Assigning a Str to a Attribute (line 539):
        
        # Assigning a Str to a Attribute (line 539):
        str_205415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 25), 'str', 'Module')
        # Getting the type of 'Foo' (line 539)
        Foo_205416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'Foo')
        # Setting the type of the member '__module__' of a type (line 539)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 8), Foo_205416, '__module__', str_205415)
        
        # Assigning a Name to a Subscript (line 540):
        
        # Assigning a Name to a Subscript (line 540):
        # Getting the type of 'Module' (line 540)
        Module_205417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 32), 'Module')
        # Getting the type of 'sys' (line 540)
        sys_205418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 540)
        modules_205419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 8), sys_205418, 'modules')
        str_205420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 20), 'str', 'Module')
        # Storing an element on a container (line 540)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 8), modules_205419, (str_205420, Module_205417))
        
        # Call to addCleanup(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'sys' (line 541)
        sys_205423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 24), 'sys', False)
        # Obtaining the member 'modules' of a type (line 541)
        modules_205424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 24), sys_205423, 'modules')
        # Obtaining the member 'pop' of a type (line 541)
        pop_205425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 24), modules_205424, 'pop')
        str_205426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 41), 'str', 'Module')
        # Processing the call keyword arguments (line 541)
        kwargs_205427 = {}
        # Getting the type of 'self' (line 541)
        self_205421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 541)
        addCleanup_205422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 8), self_205421, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 541)
        addCleanup_call_result_205428 = invoke(stypy.reporting.localization.Localization(__file__, 541, 8), addCleanup_205422, *[pop_205425, str_205426], **kwargs_205427)
        
        
        # Assigning a Call to a Name (line 542):
        
        # Assigning a Call to a Name (line 542):
        
        # Call to TestSuite(...): (line 542)
        # Processing the call arguments (line 542)
        
        # Obtaining an instance of the builtin type 'list' (line 542)
        list_205431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 542)
        # Adding element type (line 542)
        
        # Call to Foo(...): (line 542)
        # Processing the call arguments (line 542)
        str_205433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 40), 'str', 'test_foo')
        # Processing the call keyword arguments (line 542)
        kwargs_205434 = {}
        # Getting the type of 'Foo' (line 542)
        Foo_205432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 36), 'Foo', False)
        # Calling Foo(args, kwargs) (line 542)
        Foo_call_result_205435 = invoke(stypy.reporting.localization.Localization(__file__, 542, 36), Foo_205432, *[str_205433], **kwargs_205434)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 35), list_205431, Foo_call_result_205435)
        
        # Processing the call keyword arguments (line 542)
        kwargs_205436 = {}
        # Getting the type of 'unittest' (line 542)
        unittest_205429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 542)
        TestSuite_205430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 16), unittest_205429, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 542)
        TestSuite_call_result_205437 = invoke(stypy.reporting.localization.Localization(__file__, 542, 16), TestSuite_205430, *[list_205431], **kwargs_205436)
        
        # Assigning a type to the variable 'suite' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'suite', TestSuite_call_result_205437)
        
        # Call to suite(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'result' (line 543)
        result_205439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 14), 'result', False)
        # Processing the call keyword arguments (line 543)
        kwargs_205440 = {}
        # Getting the type of 'suite' (line 543)
        suite_205438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'suite', False)
        # Calling suite(args, kwargs) (line 543)
        suite_call_result_205441 = invoke(stypy.reporting.localization.Localization(__file__, 543, 8), suite_205438, *[result_205439], **kwargs_205440)
        
        
        # Call to assertEqual(...): (line 544)
        # Processing the call arguments (line 544)
        
        # Call to len(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'result' (line 544)
        result_205445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 544)
        errors_205446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 29), result_205445, 'errors')
        # Processing the call keyword arguments (line 544)
        kwargs_205447 = {}
        # Getting the type of 'len' (line 544)
        len_205444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 25), 'len', False)
        # Calling len(args, kwargs) (line 544)
        len_call_result_205448 = invoke(stypy.reporting.localization.Localization(__file__, 544, 25), len_205444, *[errors_205446], **kwargs_205447)
        
        int_205449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 45), 'int')
        # Processing the call keyword arguments (line 544)
        kwargs_205450 = {}
        # Getting the type of 'self' (line 544)
        self_205442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 544)
        assertEqual_205443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 8), self_205442, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 544)
        assertEqual_call_result_205451 = invoke(stypy.reporting.localization.Localization(__file__, 544, 8), assertEqual_205443, *[len_call_result_205448, int_205449], **kwargs_205450)
        
        
        # ################# End of 'testBufferSetUpModule(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBufferSetUpModule' in the type store
        # Getting the type of 'stypy_return_type' (line 527)
        stypy_return_type_205452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205452)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBufferSetUpModule'
        return stypy_return_type_205452


    @norecursion
    def testBufferTearDownModule(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBufferTearDownModule'
        module_type_store = module_type_store.open_function_context('testBufferTearDownModule', 546, 4, False)
        # Assigning a type to the variable 'self' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_localization', localization)
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_function_name', 'TestOutputBuffering.testBufferTearDownModule')
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_param_names_list', [])
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOutputBuffering.testBufferTearDownModule.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.testBufferTearDownModule', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBufferTearDownModule', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBufferTearDownModule(...)' code ##################

        
        # Assigning a Call to a Name (line 547):
        
        # Assigning a Call to a Name (line 547):
        
        # Call to TestResult(...): (line 547)
        # Processing the call keyword arguments (line 547)
        kwargs_205455 = {}
        # Getting the type of 'unittest' (line 547)
        unittest_205453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 547)
        TestResult_205454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 17), unittest_205453, 'TestResult')
        # Calling TestResult(args, kwargs) (line 547)
        TestResult_call_result_205456 = invoke(stypy.reporting.localization.Localization(__file__, 547, 17), TestResult_205454, *[], **kwargs_205455)
        
        # Assigning a type to the variable 'result' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'result', TestResult_call_result_205456)
        
        # Assigning a Name to a Attribute (line 548):
        
        # Assigning a Name to a Attribute (line 548):
        # Getting the type of 'True' (line 548)
        True_205457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 24), 'True')
        # Getting the type of 'result' (line 548)
        result_205458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'result')
        # Setting the type of the member 'buffer' of a type (line 548)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 8), result_205458, 'buffer', True_205457)
        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 550)
        unittest_205459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 550)
        TestCase_205460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 18), unittest_205459, 'TestCase')

        class Foo(TestCase_205460, ):

            @norecursion
            def test_foo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_foo'
                module_type_store = module_type_store.open_function_context('test_foo', 551, 12, False)
                # Assigning a type to the variable 'self' (line 552)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_foo.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_foo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_foo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_foo.__dict__.__setitem__('stypy_function_name', 'Foo.test_foo')
                Foo.test_foo.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_foo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_foo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_foo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_foo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_foo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_foo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_foo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_foo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_foo(...)' code ##################

                pass
                
                # ################# End of 'test_foo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_foo' in the type store
                # Getting the type of 'stypy_return_type' (line 551)
                stypy_return_type_205461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205461)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_foo'
                return stypy_return_type_205461

        
        # Assigning a type to the variable 'Foo' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'Foo', Foo)
        # Declaration of the 'Module' class

        class Module(object, ):

            @staticmethod
            @norecursion
            def tearDownModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownModule'
                module_type_store = module_type_store.open_function_context('tearDownModule', 554, 12, False)
                
                # Passed parameters checking function
                Module.tearDownModule.__dict__.__setitem__('stypy_localization', localization)
                Module.tearDownModule.__dict__.__setitem__('stypy_type_of_self', None)
                Module.tearDownModule.__dict__.__setitem__('stypy_type_store', module_type_store)
                Module.tearDownModule.__dict__.__setitem__('stypy_function_name', 'tearDownModule')
                Module.tearDownModule.__dict__.__setitem__('stypy_param_names_list', [])
                Module.tearDownModule.__dict__.__setitem__('stypy_varargs_param_name', None)
                Module.tearDownModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Module.tearDownModule.__dict__.__setitem__('stypy_call_defaults', defaults)
                Module.tearDownModule.__dict__.__setitem__('stypy_call_varargs', varargs)
                Module.tearDownModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Module.tearDownModule.__dict__.__setitem__('stypy_declared_arg_number', 0)
                arguments = process_argument_values(localization, None, module_type_store, 'tearDownModule', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'tearDownModule', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'tearDownModule(...)' code ##################

                int_205462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 16), 'int')
                int_205463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 19), 'int')
                # Applying the binary operator '//' (line 556)
                result_floordiv_205464 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 16), '//', int_205462, int_205463)
                
                
                # ################# End of 'tearDownModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownModule' in the type store
                # Getting the type of 'stypy_return_type' (line 554)
                stypy_return_type_205465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205465)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownModule'
                return stypy_return_type_205465

        
        # Assigning a type to the variable 'Module' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'Module', Module)
        
        # Assigning a Str to a Attribute (line 558):
        
        # Assigning a Str to a Attribute (line 558):
        str_205466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 25), 'str', 'Module')
        # Getting the type of 'Foo' (line 558)
        Foo_205467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'Foo')
        # Setting the type of the member '__module__' of a type (line 558)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 8), Foo_205467, '__module__', str_205466)
        
        # Assigning a Name to a Subscript (line 559):
        
        # Assigning a Name to a Subscript (line 559):
        # Getting the type of 'Module' (line 559)
        Module_205468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 32), 'Module')
        # Getting the type of 'sys' (line 559)
        sys_205469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 559)
        modules_205470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), sys_205469, 'modules')
        str_205471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 20), 'str', 'Module')
        # Storing an element on a container (line 559)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 559, 8), modules_205470, (str_205471, Module_205468))
        
        # Call to addCleanup(...): (line 560)
        # Processing the call arguments (line 560)
        # Getting the type of 'sys' (line 560)
        sys_205474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 24), 'sys', False)
        # Obtaining the member 'modules' of a type (line 560)
        modules_205475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 24), sys_205474, 'modules')
        # Obtaining the member 'pop' of a type (line 560)
        pop_205476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 24), modules_205475, 'pop')
        str_205477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 41), 'str', 'Module')
        # Processing the call keyword arguments (line 560)
        kwargs_205478 = {}
        # Getting the type of 'self' (line 560)
        self_205472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 560)
        addCleanup_205473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 8), self_205472, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 560)
        addCleanup_call_result_205479 = invoke(stypy.reporting.localization.Localization(__file__, 560, 8), addCleanup_205473, *[pop_205476, str_205477], **kwargs_205478)
        
        
        # Assigning a Call to a Name (line 561):
        
        # Assigning a Call to a Name (line 561):
        
        # Call to TestSuite(...): (line 561)
        # Processing the call arguments (line 561)
        
        # Obtaining an instance of the builtin type 'list' (line 561)
        list_205482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 561)
        # Adding element type (line 561)
        
        # Call to Foo(...): (line 561)
        # Processing the call arguments (line 561)
        str_205484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 40), 'str', 'test_foo')
        # Processing the call keyword arguments (line 561)
        kwargs_205485 = {}
        # Getting the type of 'Foo' (line 561)
        Foo_205483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 36), 'Foo', False)
        # Calling Foo(args, kwargs) (line 561)
        Foo_call_result_205486 = invoke(stypy.reporting.localization.Localization(__file__, 561, 36), Foo_205483, *[str_205484], **kwargs_205485)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 35), list_205482, Foo_call_result_205486)
        
        # Processing the call keyword arguments (line 561)
        kwargs_205487 = {}
        # Getting the type of 'unittest' (line 561)
        unittest_205480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 561)
        TestSuite_205481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 16), unittest_205480, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 561)
        TestSuite_call_result_205488 = invoke(stypy.reporting.localization.Localization(__file__, 561, 16), TestSuite_205481, *[list_205482], **kwargs_205487)
        
        # Assigning a type to the variable 'suite' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'suite', TestSuite_call_result_205488)
        
        # Call to suite(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'result' (line 562)
        result_205490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 14), 'result', False)
        # Processing the call keyword arguments (line 562)
        kwargs_205491 = {}
        # Getting the type of 'suite' (line 562)
        suite_205489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'suite', False)
        # Calling suite(args, kwargs) (line 562)
        suite_call_result_205492 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), suite_205489, *[result_205490], **kwargs_205491)
        
        
        # Call to assertEqual(...): (line 563)
        # Processing the call arguments (line 563)
        
        # Call to len(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'result' (line 563)
        result_205496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 563)
        errors_205497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 29), result_205496, 'errors')
        # Processing the call keyword arguments (line 563)
        kwargs_205498 = {}
        # Getting the type of 'len' (line 563)
        len_205495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 25), 'len', False)
        # Calling len(args, kwargs) (line 563)
        len_call_result_205499 = invoke(stypy.reporting.localization.Localization(__file__, 563, 25), len_205495, *[errors_205497], **kwargs_205498)
        
        int_205500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 45), 'int')
        # Processing the call keyword arguments (line 563)
        kwargs_205501 = {}
        # Getting the type of 'self' (line 563)
        self_205493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 563)
        assertEqual_205494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 8), self_205493, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 563)
        assertEqual_call_result_205502 = invoke(stypy.reporting.localization.Localization(__file__, 563, 8), assertEqual_205494, *[len_call_result_205499, int_205500], **kwargs_205501)
        
        
        # ################# End of 'testBufferTearDownModule(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBufferTearDownModule' in the type store
        # Getting the type of 'stypy_return_type' (line 546)
        stypy_return_type_205503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBufferTearDownModule'
        return stypy_return_type_205503


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 374, 0, False)
        # Assigning a type to the variable 'self' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOutputBuffering.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestOutputBuffering' (line 374)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'TestOutputBuffering', TestOutputBuffering)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 567)
    # Processing the call keyword arguments (line 567)
    kwargs_205506 = {}
    # Getting the type of 'unittest' (line 567)
    unittest_205504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 567)
    main_205505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 4), unittest_205504, 'main')
    # Calling main(args, kwargs) (line 567)
    main_call_result_205507 = invoke(stypy.reporting.localization.Localization(__file__, 567, 4), main_205505, *[], **kwargs_205506)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
