
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import sys
2: 
3: from cStringIO import StringIO
4: 
5: import unittest
6: 
7: 
8: def resultFactory(*_):
9:     return unittest.TestResult()
10: 
11: 
12: class TestSetups(unittest.TestCase):
13: 
14:     def getRunner(self):
15:         return unittest.TextTestRunner(resultclass=resultFactory,
16:                                           stream=StringIO())
17:     def runTests(self, *cases):
18:         suite = unittest.TestSuite()
19:         for case in cases:
20:             tests = unittest.defaultTestLoader.loadTestsFromTestCase(case)
21:             suite.addTests(tests)
22: 
23:         runner = self.getRunner()
24: 
25:         # creating a nested suite exposes some potential bugs
26:         realSuite = unittest.TestSuite()
27:         realSuite.addTest(suite)
28:         # adding empty suites to the end exposes potential bugs
29:         suite.addTest(unittest.TestSuite())
30:         realSuite.addTest(unittest.TestSuite())
31:         return runner.run(realSuite)
32: 
33:     def test_setup_class(self):
34:         class Test(unittest.TestCase):
35:             setUpCalled = 0
36:             @classmethod
37:             def setUpClass(cls):
38:                 Test.setUpCalled += 1
39:                 unittest.TestCase.setUpClass()
40:             def test_one(self):
41:                 pass
42:             def test_two(self):
43:                 pass
44: 
45:         result = self.runTests(Test)
46: 
47:         self.assertEqual(Test.setUpCalled, 1)
48:         self.assertEqual(result.testsRun, 2)
49:         self.assertEqual(len(result.errors), 0)
50: 
51:     def test_teardown_class(self):
52:         class Test(unittest.TestCase):
53:             tearDownCalled = 0
54:             @classmethod
55:             def tearDownClass(cls):
56:                 Test.tearDownCalled += 1
57:                 unittest.TestCase.tearDownClass()
58:             def test_one(self):
59:                 pass
60:             def test_two(self):
61:                 pass
62: 
63:         result = self.runTests(Test)
64: 
65:         self.assertEqual(Test.tearDownCalled, 1)
66:         self.assertEqual(result.testsRun, 2)
67:         self.assertEqual(len(result.errors), 0)
68: 
69:     def test_teardown_class_two_classes(self):
70:         class Test(unittest.TestCase):
71:             tearDownCalled = 0
72:             @classmethod
73:             def tearDownClass(cls):
74:                 Test.tearDownCalled += 1
75:                 unittest.TestCase.tearDownClass()
76:             def test_one(self):
77:                 pass
78:             def test_two(self):
79:                 pass
80: 
81:         class Test2(unittest.TestCase):
82:             tearDownCalled = 0
83:             @classmethod
84:             def tearDownClass(cls):
85:                 Test2.tearDownCalled += 1
86:                 unittest.TestCase.tearDownClass()
87:             def test_one(self):
88:                 pass
89:             def test_two(self):
90:                 pass
91: 
92:         result = self.runTests(Test, Test2)
93: 
94:         self.assertEqual(Test.tearDownCalled, 1)
95:         self.assertEqual(Test2.tearDownCalled, 1)
96:         self.assertEqual(result.testsRun, 4)
97:         self.assertEqual(len(result.errors), 0)
98: 
99:     def test_error_in_setupclass(self):
100:         class BrokenTest(unittest.TestCase):
101:             @classmethod
102:             def setUpClass(cls):
103:                 raise TypeError('foo')
104:             def test_one(self):
105:                 pass
106:             def test_two(self):
107:                 pass
108: 
109:         result = self.runTests(BrokenTest)
110: 
111:         self.assertEqual(result.testsRun, 0)
112:         self.assertEqual(len(result.errors), 1)
113:         error, _ = result.errors[0]
114:         self.assertEqual(str(error),
115:                     'setUpClass (%s.BrokenTest)' % __name__)
116: 
117:     def test_error_in_teardown_class(self):
118:         class Test(unittest.TestCase):
119:             tornDown = 0
120:             @classmethod
121:             def tearDownClass(cls):
122:                 Test.tornDown += 1
123:                 raise TypeError('foo')
124:             def test_one(self):
125:                 pass
126:             def test_two(self):
127:                 pass
128: 
129:         class Test2(unittest.TestCase):
130:             tornDown = 0
131:             @classmethod
132:             def tearDownClass(cls):
133:                 Test2.tornDown += 1
134:                 raise TypeError('foo')
135:             def test_one(self):
136:                 pass
137:             def test_two(self):
138:                 pass
139: 
140:         result = self.runTests(Test, Test2)
141:         self.assertEqual(result.testsRun, 4)
142:         self.assertEqual(len(result.errors), 2)
143:         self.assertEqual(Test.tornDown, 1)
144:         self.assertEqual(Test2.tornDown, 1)
145: 
146:         error, _ = result.errors[0]
147:         self.assertEqual(str(error),
148:                     'tearDownClass (%s.Test)' % __name__)
149: 
150:     def test_class_not_torndown_when_setup_fails(self):
151:         class Test(unittest.TestCase):
152:             tornDown = False
153:             @classmethod
154:             def setUpClass(cls):
155:                 raise TypeError
156:             @classmethod
157:             def tearDownClass(cls):
158:                 Test.tornDown = True
159:                 raise TypeError('foo')
160:             def test_one(self):
161:                 pass
162: 
163:         self.runTests(Test)
164:         self.assertFalse(Test.tornDown)
165: 
166:     def test_class_not_setup_or_torndown_when_skipped(self):
167:         class Test(unittest.TestCase):
168:             classSetUp = False
169:             tornDown = False
170:             @classmethod
171:             def setUpClass(cls):
172:                 Test.classSetUp = True
173:             @classmethod
174:             def tearDownClass(cls):
175:                 Test.tornDown = True
176:             def test_one(self):
177:                 pass
178: 
179:         Test = unittest.skip("hop")(Test)
180:         self.runTests(Test)
181:         self.assertFalse(Test.classSetUp)
182:         self.assertFalse(Test.tornDown)
183: 
184:     def test_setup_teardown_order_with_pathological_suite(self):
185:         results = []
186: 
187:         class Module1(object):
188:             @staticmethod
189:             def setUpModule():
190:                 results.append('Module1.setUpModule')
191:             @staticmethod
192:             def tearDownModule():
193:                 results.append('Module1.tearDownModule')
194: 
195:         class Module2(object):
196:             @staticmethod
197:             def setUpModule():
198:                 results.append('Module2.setUpModule')
199:             @staticmethod
200:             def tearDownModule():
201:                 results.append('Module2.tearDownModule')
202: 
203:         class Test1(unittest.TestCase):
204:             @classmethod
205:             def setUpClass(cls):
206:                 results.append('setup 1')
207:             @classmethod
208:             def tearDownClass(cls):
209:                 results.append('teardown 1')
210:             def testOne(self):
211:                 results.append('Test1.testOne')
212:             def testTwo(self):
213:                 results.append('Test1.testTwo')
214: 
215:         class Test2(unittest.TestCase):
216:             @classmethod
217:             def setUpClass(cls):
218:                 results.append('setup 2')
219:             @classmethod
220:             def tearDownClass(cls):
221:                 results.append('teardown 2')
222:             def testOne(self):
223:                 results.append('Test2.testOne')
224:             def testTwo(self):
225:                 results.append('Test2.testTwo')
226: 
227:         class Test3(unittest.TestCase):
228:             @classmethod
229:             def setUpClass(cls):
230:                 results.append('setup 3')
231:             @classmethod
232:             def tearDownClass(cls):
233:                 results.append('teardown 3')
234:             def testOne(self):
235:                 results.append('Test3.testOne')
236:             def testTwo(self):
237:                 results.append('Test3.testTwo')
238: 
239:         Test1.__module__ = Test2.__module__ = 'Module'
240:         Test3.__module__ = 'Module2'
241:         sys.modules['Module'] = Module1
242:         sys.modules['Module2'] = Module2
243: 
244:         first = unittest.TestSuite((Test1('testOne'),))
245:         second = unittest.TestSuite((Test1('testTwo'),))
246:         third = unittest.TestSuite((Test2('testOne'),))
247:         fourth = unittest.TestSuite((Test2('testTwo'),))
248:         fifth = unittest.TestSuite((Test3('testOne'),))
249:         sixth = unittest.TestSuite((Test3('testTwo'),))
250:         suite = unittest.TestSuite((first, second, third, fourth, fifth, sixth))
251: 
252:         runner = self.getRunner()
253:         result = runner.run(suite)
254:         self.assertEqual(result.testsRun, 6)
255:         self.assertEqual(len(result.errors), 0)
256: 
257:         self.assertEqual(results,
258:                          ['Module1.setUpModule', 'setup 1',
259:                           'Test1.testOne', 'Test1.testTwo', 'teardown 1',
260:                           'setup 2', 'Test2.testOne', 'Test2.testTwo',
261:                           'teardown 2', 'Module1.tearDownModule',
262:                           'Module2.setUpModule', 'setup 3',
263:                           'Test3.testOne', 'Test3.testTwo',
264:                           'teardown 3', 'Module2.tearDownModule'])
265: 
266:     def test_setup_module(self):
267:         class Module(object):
268:             moduleSetup = 0
269:             @staticmethod
270:             def setUpModule():
271:                 Module.moduleSetup += 1
272: 
273:         class Test(unittest.TestCase):
274:             def test_one(self):
275:                 pass
276:             def test_two(self):
277:                 pass
278:         Test.__module__ = 'Module'
279:         sys.modules['Module'] = Module
280: 
281:         result = self.runTests(Test)
282:         self.assertEqual(Module.moduleSetup, 1)
283:         self.assertEqual(result.testsRun, 2)
284:         self.assertEqual(len(result.errors), 0)
285: 
286:     def test_error_in_setup_module(self):
287:         class Module(object):
288:             moduleSetup = 0
289:             moduleTornDown = 0
290:             @staticmethod
291:             def setUpModule():
292:                 Module.moduleSetup += 1
293:                 raise TypeError('foo')
294:             @staticmethod
295:             def tearDownModule():
296:                 Module.moduleTornDown += 1
297: 
298:         class Test(unittest.TestCase):
299:             classSetUp = False
300:             classTornDown = False
301:             @classmethod
302:             def setUpClass(cls):
303:                 Test.classSetUp = True
304:             @classmethod
305:             def tearDownClass(cls):
306:                 Test.classTornDown = True
307:             def test_one(self):
308:                 pass
309:             def test_two(self):
310:                 pass
311: 
312:         class Test2(unittest.TestCase):
313:             def test_one(self):
314:                 pass
315:             def test_two(self):
316:                 pass
317:         Test.__module__ = 'Module'
318:         Test2.__module__ = 'Module'
319:         sys.modules['Module'] = Module
320: 
321:         result = self.runTests(Test, Test2)
322:         self.assertEqual(Module.moduleSetup, 1)
323:         self.assertEqual(Module.moduleTornDown, 0)
324:         self.assertEqual(result.testsRun, 0)
325:         self.assertFalse(Test.classSetUp)
326:         self.assertFalse(Test.classTornDown)
327:         self.assertEqual(len(result.errors), 1)
328:         error, _ = result.errors[0]
329:         self.assertEqual(str(error), 'setUpModule (Module)')
330: 
331:     def test_testcase_with_missing_module(self):
332:         class Test(unittest.TestCase):
333:             def test_one(self):
334:                 pass
335:             def test_two(self):
336:                 pass
337:         Test.__module__ = 'Module'
338:         sys.modules.pop('Module', None)
339: 
340:         result = self.runTests(Test)
341:         self.assertEqual(result.testsRun, 2)
342: 
343:     def test_teardown_module(self):
344:         class Module(object):
345:             moduleTornDown = 0
346:             @staticmethod
347:             def tearDownModule():
348:                 Module.moduleTornDown += 1
349: 
350:         class Test(unittest.TestCase):
351:             def test_one(self):
352:                 pass
353:             def test_two(self):
354:                 pass
355:         Test.__module__ = 'Module'
356:         sys.modules['Module'] = Module
357: 
358:         result = self.runTests(Test)
359:         self.assertEqual(Module.moduleTornDown, 1)
360:         self.assertEqual(result.testsRun, 2)
361:         self.assertEqual(len(result.errors), 0)
362: 
363:     def test_error_in_teardown_module(self):
364:         class Module(object):
365:             moduleTornDown = 0
366:             @staticmethod
367:             def tearDownModule():
368:                 Module.moduleTornDown += 1
369:                 raise TypeError('foo')
370: 
371:         class Test(unittest.TestCase):
372:             classSetUp = False
373:             classTornDown = False
374:             @classmethod
375:             def setUpClass(cls):
376:                 Test.classSetUp = True
377:             @classmethod
378:             def tearDownClass(cls):
379:                 Test.classTornDown = True
380:             def test_one(self):
381:                 pass
382:             def test_two(self):
383:                 pass
384: 
385:         class Test2(unittest.TestCase):
386:             def test_one(self):
387:                 pass
388:             def test_two(self):
389:                 pass
390:         Test.__module__ = 'Module'
391:         Test2.__module__ = 'Module'
392:         sys.modules['Module'] = Module
393: 
394:         result = self.runTests(Test, Test2)
395:         self.assertEqual(Module.moduleTornDown, 1)
396:         self.assertEqual(result.testsRun, 4)
397:         self.assertTrue(Test.classSetUp)
398:         self.assertTrue(Test.classTornDown)
399:         self.assertEqual(len(result.errors), 1)
400:         error, _ = result.errors[0]
401:         self.assertEqual(str(error), 'tearDownModule (Module)')
402: 
403:     def test_skiptest_in_setupclass(self):
404:         class Test(unittest.TestCase):
405:             @classmethod
406:             def setUpClass(cls):
407:                 raise unittest.SkipTest('foo')
408:             def test_one(self):
409:                 pass
410:             def test_two(self):
411:                 pass
412: 
413:         result = self.runTests(Test)
414:         self.assertEqual(result.testsRun, 0)
415:         self.assertEqual(len(result.errors), 0)
416:         self.assertEqual(len(result.skipped), 1)
417:         skipped = result.skipped[0][0]
418:         self.assertEqual(str(skipped), 'setUpClass (%s.Test)' % __name__)
419: 
420:     def test_skiptest_in_setupmodule(self):
421:         class Test(unittest.TestCase):
422:             def test_one(self):
423:                 pass
424:             def test_two(self):
425:                 pass
426: 
427:         class Module(object):
428:             @staticmethod
429:             def setUpModule():
430:                 raise unittest.SkipTest('foo')
431: 
432:         Test.__module__ = 'Module'
433:         sys.modules['Module'] = Module
434: 
435:         result = self.runTests(Test)
436:         self.assertEqual(result.testsRun, 0)
437:         self.assertEqual(len(result.errors), 0)
438:         self.assertEqual(len(result.skipped), 1)
439:         skipped = result.skipped[0][0]
440:         self.assertEqual(str(skipped), 'setUpModule (Module)')
441: 
442:     def test_suite_debug_executes_setups_and_teardowns(self):
443:         ordering = []
444: 
445:         class Module(object):
446:             @staticmethod
447:             def setUpModule():
448:                 ordering.append('setUpModule')
449:             @staticmethod
450:             def tearDownModule():
451:                 ordering.append('tearDownModule')
452: 
453:         class Test(unittest.TestCase):
454:             @classmethod
455:             def setUpClass(cls):
456:                 ordering.append('setUpClass')
457:             @classmethod
458:             def tearDownClass(cls):
459:                 ordering.append('tearDownClass')
460:             def test_something(self):
461:                 ordering.append('test_something')
462: 
463:         Test.__module__ = 'Module'
464:         sys.modules['Module'] = Module
465: 
466:         suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
467:         suite.debug()
468:         expectedOrder = ['setUpModule', 'setUpClass', 'test_something', 'tearDownClass', 'tearDownModule']
469:         self.assertEqual(ordering, expectedOrder)
470: 
471:     def test_suite_debug_propagates_exceptions(self):
472:         class Module(object):
473:             @staticmethod
474:             def setUpModule():
475:                 if phase == 0:
476:                     raise Exception('setUpModule')
477:             @staticmethod
478:             def tearDownModule():
479:                 if phase == 1:
480:                     raise Exception('tearDownModule')
481: 
482:         class Test(unittest.TestCase):
483:             @classmethod
484:             def setUpClass(cls):
485:                 if phase == 2:
486:                     raise Exception('setUpClass')
487:             @classmethod
488:             def tearDownClass(cls):
489:                 if phase == 3:
490:                     raise Exception('tearDownClass')
491:             def test_something(self):
492:                 if phase == 4:
493:                     raise Exception('test_something')
494: 
495:         Test.__module__ = 'Module'
496:         sys.modules['Module'] = Module
497: 
498:         _suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
499:         suite = unittest.TestSuite()
500:         suite.addTest(_suite)
501: 
502:         messages = ('setUpModule', 'tearDownModule', 'setUpClass', 'tearDownClass', 'test_something')
503:         for phase, msg in enumerate(messages):
504:             with self.assertRaisesRegexp(Exception, msg):
505:                 suite.debug()
506: 
507: if __name__ == '__main__':
508:     unittest.main()
509: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from cStringIO import StringIO' statement (line 3)
from cStringIO import StringIO

import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import unittest' statement (line 5)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'unittest', unittest, module_type_store)


@norecursion
def resultFactory(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'resultFactory'
    module_type_store = module_type_store.open_function_context('resultFactory', 8, 0, False)
    
    # Passed parameters checking function
    resultFactory.stypy_localization = localization
    resultFactory.stypy_type_of_self = None
    resultFactory.stypy_type_store = module_type_store
    resultFactory.stypy_function_name = 'resultFactory'
    resultFactory.stypy_param_names_list = []
    resultFactory.stypy_varargs_param_name = '_'
    resultFactory.stypy_kwargs_param_name = None
    resultFactory.stypy_call_defaults = defaults
    resultFactory.stypy_call_varargs = varargs
    resultFactory.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'resultFactory', [], '_', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'resultFactory', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'resultFactory(...)' code ##################

    
    # Call to TestResult(...): (line 9)
    # Processing the call keyword arguments (line 9)
    kwargs_206299 = {}
    # Getting the type of 'unittest' (line 9)
    unittest_206297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'unittest', False)
    # Obtaining the member 'TestResult' of a type (line 9)
    TestResult_206298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 11), unittest_206297, 'TestResult')
    # Calling TestResult(args, kwargs) (line 9)
    TestResult_call_result_206300 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), TestResult_206298, *[], **kwargs_206299)
    
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', TestResult_call_result_206300)
    
    # ################# End of 'resultFactory(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'resultFactory' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_206301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_206301)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'resultFactory'
    return stypy_return_type_206301

# Assigning a type to the variable 'resultFactory' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'resultFactory', resultFactory)
# Declaration of the 'TestSetups' class
# Getting the type of 'unittest' (line 12)
unittest_206302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'unittest')
# Obtaining the member 'TestCase' of a type (line 12)
TestCase_206303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 17), unittest_206302, 'TestCase')

class TestSetups(TestCase_206303, ):

    @norecursion
    def getRunner(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getRunner'
        module_type_store = module_type_store.open_function_context('getRunner', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.getRunner.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.getRunner.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.getRunner.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.getRunner.__dict__.__setitem__('stypy_function_name', 'TestSetups.getRunner')
        TestSetups.getRunner.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.getRunner.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.getRunner.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.getRunner.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.getRunner.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.getRunner.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.getRunner.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.getRunner', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getRunner', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getRunner(...)' code ##################

        
        # Call to TextTestRunner(...): (line 15)
        # Processing the call keyword arguments (line 15)
        # Getting the type of 'resultFactory' (line 15)
        resultFactory_206306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 51), 'resultFactory', False)
        keyword_206307 = resultFactory_206306
        
        # Call to StringIO(...): (line 16)
        # Processing the call keyword arguments (line 16)
        kwargs_206309 = {}
        # Getting the type of 'StringIO' (line 16)
        StringIO_206308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 49), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 16)
        StringIO_call_result_206310 = invoke(stypy.reporting.localization.Localization(__file__, 16, 49), StringIO_206308, *[], **kwargs_206309)
        
        keyword_206311 = StringIO_call_result_206310
        kwargs_206312 = {'resultclass': keyword_206307, 'stream': keyword_206311}
        # Getting the type of 'unittest' (line 15)
        unittest_206304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 15)
        TextTestRunner_206305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 15), unittest_206304, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 15)
        TextTestRunner_call_result_206313 = invoke(stypy.reporting.localization.Localization(__file__, 15, 15), TextTestRunner_206305, *[], **kwargs_206312)
        
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', TextTestRunner_call_result_206313)
        
        # ################# End of 'getRunner(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getRunner' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_206314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206314)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getRunner'
        return stypy_return_type_206314


    @norecursion
    def runTests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runTests'
        module_type_store = module_type_store.open_function_context('runTests', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.runTests.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.runTests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.runTests.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.runTests.__dict__.__setitem__('stypy_function_name', 'TestSetups.runTests')
        TestSetups.runTests.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.runTests.__dict__.__setitem__('stypy_varargs_param_name', 'cases')
        TestSetups.runTests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.runTests.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.runTests.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.runTests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.runTests.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.runTests', [], 'cases', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runTests', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runTests(...)' code ##################

        
        # Assigning a Call to a Name (line 18):
        
        # Assigning a Call to a Name (line 18):
        
        # Call to TestSuite(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_206317 = {}
        # Getting the type of 'unittest' (line 18)
        unittest_206315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 18)
        TestSuite_206316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 16), unittest_206315, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 18)
        TestSuite_call_result_206318 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), TestSuite_206316, *[], **kwargs_206317)
        
        # Assigning a type to the variable 'suite' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'suite', TestSuite_call_result_206318)
        
        # Getting the type of 'cases' (line 19)
        cases_206319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'cases')
        # Testing the type of a for loop iterable (line 19)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 8), cases_206319)
        # Getting the type of the for loop variable (line 19)
        for_loop_var_206320 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 8), cases_206319)
        # Assigning a type to the variable 'case' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'case', for_loop_var_206320)
        # SSA begins for a for statement (line 19)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to loadTestsFromTestCase(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'case' (line 20)
        case_206324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 69), 'case', False)
        # Processing the call keyword arguments (line 20)
        kwargs_206325 = {}
        # Getting the type of 'unittest' (line 20)
        unittest_206321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'unittest', False)
        # Obtaining the member 'defaultTestLoader' of a type (line 20)
        defaultTestLoader_206322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 20), unittest_206321, 'defaultTestLoader')
        # Obtaining the member 'loadTestsFromTestCase' of a type (line 20)
        loadTestsFromTestCase_206323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 20), defaultTestLoader_206322, 'loadTestsFromTestCase')
        # Calling loadTestsFromTestCase(args, kwargs) (line 20)
        loadTestsFromTestCase_call_result_206326 = invoke(stypy.reporting.localization.Localization(__file__, 20, 20), loadTestsFromTestCase_206323, *[case_206324], **kwargs_206325)
        
        # Assigning a type to the variable 'tests' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'tests', loadTestsFromTestCase_call_result_206326)
        
        # Call to addTests(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'tests' (line 21)
        tests_206329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'tests', False)
        # Processing the call keyword arguments (line 21)
        kwargs_206330 = {}
        # Getting the type of 'suite' (line 21)
        suite_206327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'suite', False)
        # Obtaining the member 'addTests' of a type (line 21)
        addTests_206328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), suite_206327, 'addTests')
        # Calling addTests(args, kwargs) (line 21)
        addTests_call_result_206331 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), addTests_206328, *[tests_206329], **kwargs_206330)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 23):
        
        # Assigning a Call to a Name (line 23):
        
        # Call to getRunner(...): (line 23)
        # Processing the call keyword arguments (line 23)
        kwargs_206334 = {}
        # Getting the type of 'self' (line 23)
        self_206332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'self', False)
        # Obtaining the member 'getRunner' of a type (line 23)
        getRunner_206333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 17), self_206332, 'getRunner')
        # Calling getRunner(args, kwargs) (line 23)
        getRunner_call_result_206335 = invoke(stypy.reporting.localization.Localization(__file__, 23, 17), getRunner_206333, *[], **kwargs_206334)
        
        # Assigning a type to the variable 'runner' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'runner', getRunner_call_result_206335)
        
        # Assigning a Call to a Name (line 26):
        
        # Assigning a Call to a Name (line 26):
        
        # Call to TestSuite(...): (line 26)
        # Processing the call keyword arguments (line 26)
        kwargs_206338 = {}
        # Getting the type of 'unittest' (line 26)
        unittest_206336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 26)
        TestSuite_206337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 20), unittest_206336, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 26)
        TestSuite_call_result_206339 = invoke(stypy.reporting.localization.Localization(__file__, 26, 20), TestSuite_206337, *[], **kwargs_206338)
        
        # Assigning a type to the variable 'realSuite' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'realSuite', TestSuite_call_result_206339)
        
        # Call to addTest(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'suite' (line 27)
        suite_206342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'suite', False)
        # Processing the call keyword arguments (line 27)
        kwargs_206343 = {}
        # Getting the type of 'realSuite' (line 27)
        realSuite_206340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'realSuite', False)
        # Obtaining the member 'addTest' of a type (line 27)
        addTest_206341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), realSuite_206340, 'addTest')
        # Calling addTest(args, kwargs) (line 27)
        addTest_call_result_206344 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), addTest_206341, *[suite_206342], **kwargs_206343)
        
        
        # Call to addTest(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Call to TestSuite(...): (line 29)
        # Processing the call keyword arguments (line 29)
        kwargs_206349 = {}
        # Getting the type of 'unittest' (line 29)
        unittest_206347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 29)
        TestSuite_206348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 22), unittest_206347, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 29)
        TestSuite_call_result_206350 = invoke(stypy.reporting.localization.Localization(__file__, 29, 22), TestSuite_206348, *[], **kwargs_206349)
        
        # Processing the call keyword arguments (line 29)
        kwargs_206351 = {}
        # Getting the type of 'suite' (line 29)
        suite_206345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'suite', False)
        # Obtaining the member 'addTest' of a type (line 29)
        addTest_206346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), suite_206345, 'addTest')
        # Calling addTest(args, kwargs) (line 29)
        addTest_call_result_206352 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), addTest_206346, *[TestSuite_call_result_206350], **kwargs_206351)
        
        
        # Call to addTest(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to TestSuite(...): (line 30)
        # Processing the call keyword arguments (line 30)
        kwargs_206357 = {}
        # Getting the type of 'unittest' (line 30)
        unittest_206355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 30)
        TestSuite_206356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 26), unittest_206355, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 30)
        TestSuite_call_result_206358 = invoke(stypy.reporting.localization.Localization(__file__, 30, 26), TestSuite_206356, *[], **kwargs_206357)
        
        # Processing the call keyword arguments (line 30)
        kwargs_206359 = {}
        # Getting the type of 'realSuite' (line 30)
        realSuite_206353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'realSuite', False)
        # Obtaining the member 'addTest' of a type (line 30)
        addTest_206354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), realSuite_206353, 'addTest')
        # Calling addTest(args, kwargs) (line 30)
        addTest_call_result_206360 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), addTest_206354, *[TestSuite_call_result_206358], **kwargs_206359)
        
        
        # Call to run(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'realSuite' (line 31)
        realSuite_206363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'realSuite', False)
        # Processing the call keyword arguments (line 31)
        kwargs_206364 = {}
        # Getting the type of 'runner' (line 31)
        runner_206361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'runner', False)
        # Obtaining the member 'run' of a type (line 31)
        run_206362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), runner_206361, 'run')
        # Calling run(args, kwargs) (line 31)
        run_call_result_206365 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), run_206362, *[realSuite_206363], **kwargs_206364)
        
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', run_call_result_206365)
        
        # ################# End of 'runTests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runTests' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_206366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runTests'
        return stypy_return_type_206366


    @norecursion
    def test_setup_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_setup_class'
        module_type_store = module_type_store.open_function_context('test_setup_class', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_setup_class')
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_setup_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_setup_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_setup_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_setup_class(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 34)
        unittest_206367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 34)
        TestCase_206368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), unittest_206367, 'TestCase')

        class Test(TestCase_206368, ):
            
            # Assigning a Num to a Name (line 35):
            
            # Assigning a Num to a Name (line 35):
            int_206369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'int')
            # Assigning a type to the variable 'setUpCalled' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'setUpCalled', int_206369)

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 36, 12, False)
                # Assigning a type to the variable 'self' (line 37)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test.setUpClass')
                Test.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Getting the type of 'Test' (line 38)
                Test_206370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'Test')
                # Obtaining the member 'setUpCalled' of a type (line 38)
                setUpCalled_206371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), Test_206370, 'setUpCalled')
                int_206372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'int')
                # Applying the binary operator '+=' (line 38)
                result_iadd_206373 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 16), '+=', setUpCalled_206371, int_206372)
                # Getting the type of 'Test' (line 38)
                Test_206374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'Test')
                # Setting the type of the member 'setUpCalled' of a type (line 38)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), Test_206374, 'setUpCalled', result_iadd_206373)
                
                
                # Call to setUpClass(...): (line 39)
                # Processing the call keyword arguments (line 39)
                kwargs_206378 = {}
                # Getting the type of 'unittest' (line 39)
                unittest_206375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'unittest', False)
                # Obtaining the member 'TestCase' of a type (line 39)
                TestCase_206376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), unittest_206375, 'TestCase')
                # Obtaining the member 'setUpClass' of a type (line 39)
                setUpClass_206377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), TestCase_206376, 'setUpClass')
                # Calling setUpClass(args, kwargs) (line 39)
                setUpClass_call_result_206379 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), setUpClass_206377, *[], **kwargs_206378)
                
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 36)
                stypy_return_type_206380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206380)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_206380


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 40, 12, False)
                # Assigning a type to the variable 'self' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 40)
                stypy_return_type_206381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206381)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206381


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 42, 12, False)
                # Assigning a type to the variable 'self' (line 43)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 42)
                stypy_return_type_206382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206382)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_206382

        
        # Assigning a type to the variable 'Test' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'Test', Test)
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to runTests(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'Test' (line 45)
        Test_206385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'Test', False)
        # Processing the call keyword arguments (line 45)
        kwargs_206386 = {}
        # Getting the type of 'self' (line 45)
        self_206383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 45)
        runTests_206384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), self_206383, 'runTests')
        # Calling runTests(args, kwargs) (line 45)
        runTests_call_result_206387 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), runTests_206384, *[Test_206385], **kwargs_206386)
        
        # Assigning a type to the variable 'result' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'result', runTests_call_result_206387)
        
        # Call to assertEqual(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'Test' (line 47)
        Test_206390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'Test', False)
        # Obtaining the member 'setUpCalled' of a type (line 47)
        setUpCalled_206391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), Test_206390, 'setUpCalled')
        int_206392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'int')
        # Processing the call keyword arguments (line 47)
        kwargs_206393 = {}
        # Getting the type of 'self' (line 47)
        self_206388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 47)
        assertEqual_206389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_206388, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 47)
        assertEqual_call_result_206394 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assertEqual_206389, *[setUpCalled_206391, int_206392], **kwargs_206393)
        
        
        # Call to assertEqual(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'result' (line 48)
        result_206397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 48)
        testsRun_206398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 25), result_206397, 'testsRun')
        int_206399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 42), 'int')
        # Processing the call keyword arguments (line 48)
        kwargs_206400 = {}
        # Getting the type of 'self' (line 48)
        self_206395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 48)
        assertEqual_206396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_206395, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 48)
        assertEqual_call_result_206401 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assertEqual_206396, *[testsRun_206398, int_206399], **kwargs_206400)
        
        
        # Call to assertEqual(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Call to len(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'result' (line 49)
        result_206405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 49)
        errors_206406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 29), result_206405, 'errors')
        # Processing the call keyword arguments (line 49)
        kwargs_206407 = {}
        # Getting the type of 'len' (line 49)
        len_206404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'len', False)
        # Calling len(args, kwargs) (line 49)
        len_call_result_206408 = invoke(stypy.reporting.localization.Localization(__file__, 49, 25), len_206404, *[errors_206406], **kwargs_206407)
        
        int_206409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 45), 'int')
        # Processing the call keyword arguments (line 49)
        kwargs_206410 = {}
        # Getting the type of 'self' (line 49)
        self_206402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 49)
        assertEqual_206403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_206402, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 49)
        assertEqual_call_result_206411 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assertEqual_206403, *[len_call_result_206408, int_206409], **kwargs_206410)
        
        
        # ################# End of 'test_setup_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_setup_class' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_206412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206412)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_setup_class'
        return stypy_return_type_206412


    @norecursion
    def test_teardown_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_teardown_class'
        module_type_store = module_type_store.open_function_context('test_teardown_class', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_teardown_class')
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_teardown_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_teardown_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_teardown_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_teardown_class(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 52)
        unittest_206413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 52)
        TestCase_206414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 19), unittest_206413, 'TestCase')

        class Test(TestCase_206414, ):
            
            # Assigning a Num to a Name (line 53):
            
            # Assigning a Num to a Name (line 53):
            int_206415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'int')
            # Assigning a type to the variable 'tearDownCalled' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'tearDownCalled', int_206415)

            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 54, 12, False)
                # Assigning a type to the variable 'self' (line 55)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test.tearDownClass')
                Test.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Getting the type of 'Test' (line 56)
                Test_206416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'Test')
                # Obtaining the member 'tearDownCalled' of a type (line 56)
                tearDownCalled_206417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), Test_206416, 'tearDownCalled')
                int_206418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 39), 'int')
                # Applying the binary operator '+=' (line 56)
                result_iadd_206419 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 16), '+=', tearDownCalled_206417, int_206418)
                # Getting the type of 'Test' (line 56)
                Test_206420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'Test')
                # Setting the type of the member 'tearDownCalled' of a type (line 56)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), Test_206420, 'tearDownCalled', result_iadd_206419)
                
                
                # Call to tearDownClass(...): (line 57)
                # Processing the call keyword arguments (line 57)
                kwargs_206424 = {}
                # Getting the type of 'unittest' (line 57)
                unittest_206421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'unittest', False)
                # Obtaining the member 'TestCase' of a type (line 57)
                TestCase_206422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), unittest_206421, 'TestCase')
                # Obtaining the member 'tearDownClass' of a type (line 57)
                tearDownClass_206423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), TestCase_206422, 'tearDownClass')
                # Calling tearDownClass(args, kwargs) (line 57)
                tearDownClass_call_result_206425 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), tearDownClass_206423, *[], **kwargs_206424)
                
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 54)
                stypy_return_type_206426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206426)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206426


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 58, 12, False)
                # Assigning a type to the variable 'self' (line 59)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 58)
                stypy_return_type_206427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206427)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206427


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 60, 12, False)
                # Assigning a type to the variable 'self' (line 61)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 60)
                stypy_return_type_206428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206428)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_206428

        
        # Assigning a type to the variable 'Test' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'Test', Test)
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to runTests(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'Test' (line 63)
        Test_206431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'Test', False)
        # Processing the call keyword arguments (line 63)
        kwargs_206432 = {}
        # Getting the type of 'self' (line 63)
        self_206429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 63)
        runTests_206430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 17), self_206429, 'runTests')
        # Calling runTests(args, kwargs) (line 63)
        runTests_call_result_206433 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), runTests_206430, *[Test_206431], **kwargs_206432)
        
        # Assigning a type to the variable 'result' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'result', runTests_call_result_206433)
        
        # Call to assertEqual(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'Test' (line 65)
        Test_206436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'Test', False)
        # Obtaining the member 'tearDownCalled' of a type (line 65)
        tearDownCalled_206437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 25), Test_206436, 'tearDownCalled')
        int_206438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 46), 'int')
        # Processing the call keyword arguments (line 65)
        kwargs_206439 = {}
        # Getting the type of 'self' (line 65)
        self_206434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 65)
        assertEqual_206435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_206434, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 65)
        assertEqual_call_result_206440 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assertEqual_206435, *[tearDownCalled_206437, int_206438], **kwargs_206439)
        
        
        # Call to assertEqual(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'result' (line 66)
        result_206443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 66)
        testsRun_206444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), result_206443, 'testsRun')
        int_206445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 42), 'int')
        # Processing the call keyword arguments (line 66)
        kwargs_206446 = {}
        # Getting the type of 'self' (line 66)
        self_206441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 66)
        assertEqual_206442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_206441, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 66)
        assertEqual_call_result_206447 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assertEqual_206442, *[testsRun_206444, int_206445], **kwargs_206446)
        
        
        # Call to assertEqual(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to len(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'result' (line 67)
        result_206451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 67)
        errors_206452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 29), result_206451, 'errors')
        # Processing the call keyword arguments (line 67)
        kwargs_206453 = {}
        # Getting the type of 'len' (line 67)
        len_206450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'len', False)
        # Calling len(args, kwargs) (line 67)
        len_call_result_206454 = invoke(stypy.reporting.localization.Localization(__file__, 67, 25), len_206450, *[errors_206452], **kwargs_206453)
        
        int_206455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 45), 'int')
        # Processing the call keyword arguments (line 67)
        kwargs_206456 = {}
        # Getting the type of 'self' (line 67)
        self_206448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 67)
        assertEqual_206449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_206448, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 67)
        assertEqual_call_result_206457 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assertEqual_206449, *[len_call_result_206454, int_206455], **kwargs_206456)
        
        
        # ################# End of 'test_teardown_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_teardown_class' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_206458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206458)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_teardown_class'
        return stypy_return_type_206458


    @norecursion
    def test_teardown_class_two_classes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_teardown_class_two_classes'
        module_type_store = module_type_store.open_function_context('test_teardown_class_two_classes', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_teardown_class_two_classes')
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_teardown_class_two_classes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_teardown_class_two_classes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_teardown_class_two_classes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_teardown_class_two_classes(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 70)
        unittest_206459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 70)
        TestCase_206460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), unittest_206459, 'TestCase')

        class Test(TestCase_206460, ):
            
            # Assigning a Num to a Name (line 71):
            
            # Assigning a Num to a Name (line 71):
            int_206461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'int')
            # Assigning a type to the variable 'tearDownCalled' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'tearDownCalled', int_206461)

            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 72, 12, False)
                # Assigning a type to the variable 'self' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test.tearDownClass')
                Test.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Getting the type of 'Test' (line 74)
                Test_206462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'Test')
                # Obtaining the member 'tearDownCalled' of a type (line 74)
                tearDownCalled_206463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), Test_206462, 'tearDownCalled')
                int_206464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 39), 'int')
                # Applying the binary operator '+=' (line 74)
                result_iadd_206465 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 16), '+=', tearDownCalled_206463, int_206464)
                # Getting the type of 'Test' (line 74)
                Test_206466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'Test')
                # Setting the type of the member 'tearDownCalled' of a type (line 74)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), Test_206466, 'tearDownCalled', result_iadd_206465)
                
                
                # Call to tearDownClass(...): (line 75)
                # Processing the call keyword arguments (line 75)
                kwargs_206470 = {}
                # Getting the type of 'unittest' (line 75)
                unittest_206467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'unittest', False)
                # Obtaining the member 'TestCase' of a type (line 75)
                TestCase_206468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), unittest_206467, 'TestCase')
                # Obtaining the member 'tearDownClass' of a type (line 75)
                tearDownClass_206469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), TestCase_206468, 'tearDownClass')
                # Calling tearDownClass(args, kwargs) (line 75)
                tearDownClass_call_result_206471 = invoke(stypy.reporting.localization.Localization(__file__, 75, 16), tearDownClass_206469, *[], **kwargs_206470)
                
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 72)
                stypy_return_type_206472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206472)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206472


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 76, 12, False)
                # Assigning a type to the variable 'self' (line 77)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 76)
                stypy_return_type_206473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206473)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206473


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 78, 12, False)
                # Assigning a type to the variable 'self' (line 79)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 78)
                stypy_return_type_206474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206474)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_206474

        
        # Assigning a type to the variable 'Test' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'Test', Test)
        # Declaration of the 'Test2' class
        # Getting the type of 'unittest' (line 81)
        unittest_206475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 81)
        TestCase_206476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 20), unittest_206475, 'TestCase')

        class Test2(TestCase_206476, ):
            
            # Assigning a Num to a Name (line 82):
            
            # Assigning a Num to a Name (line 82):
            int_206477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 29), 'int')
            # Assigning a type to the variable 'tearDownCalled' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'tearDownCalled', int_206477)

            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 83, 12, False)
                # Assigning a type to the variable 'self' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test2.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test2.tearDownClass')
                Test2.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Getting the type of 'Test2' (line 85)
                Test2_206478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'Test2')
                # Obtaining the member 'tearDownCalled' of a type (line 85)
                tearDownCalled_206479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 16), Test2_206478, 'tearDownCalled')
                int_206480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 40), 'int')
                # Applying the binary operator '+=' (line 85)
                result_iadd_206481 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 16), '+=', tearDownCalled_206479, int_206480)
                # Getting the type of 'Test2' (line 85)
                Test2_206482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'Test2')
                # Setting the type of the member 'tearDownCalled' of a type (line 85)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 16), Test2_206482, 'tearDownCalled', result_iadd_206481)
                
                
                # Call to tearDownClass(...): (line 86)
                # Processing the call keyword arguments (line 86)
                kwargs_206486 = {}
                # Getting the type of 'unittest' (line 86)
                unittest_206483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'unittest', False)
                # Obtaining the member 'TestCase' of a type (line 86)
                TestCase_206484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), unittest_206483, 'TestCase')
                # Obtaining the member 'tearDownClass' of a type (line 86)
                tearDownClass_206485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), TestCase_206484, 'tearDownClass')
                # Calling tearDownClass(args, kwargs) (line 86)
                tearDownClass_call_result_206487 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), tearDownClass_206485, *[], **kwargs_206486)
                
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 83)
                stypy_return_type_206488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206488)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206488


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 87, 12, False)
                # Assigning a type to the variable 'self' (line 88)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test2.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.test_one.__dict__.__setitem__('stypy_function_name', 'Test2.test_one')
                Test2.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 87)
                stypy_return_type_206489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206489)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206489


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 89, 12, False)
                # Assigning a type to the variable 'self' (line 90)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test2.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.test_two.__dict__.__setitem__('stypy_function_name', 'Test2.test_two')
                Test2.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 89)
                stypy_return_type_206490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206490)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_206490

        
        # Assigning a type to the variable 'Test2' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'Test2', Test2)
        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to runTests(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'Test' (line 92)
        Test_206493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'Test', False)
        # Getting the type of 'Test2' (line 92)
        Test2_206494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'Test2', False)
        # Processing the call keyword arguments (line 92)
        kwargs_206495 = {}
        # Getting the type of 'self' (line 92)
        self_206491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 92)
        runTests_206492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 17), self_206491, 'runTests')
        # Calling runTests(args, kwargs) (line 92)
        runTests_call_result_206496 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), runTests_206492, *[Test_206493, Test2_206494], **kwargs_206495)
        
        # Assigning a type to the variable 'result' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'result', runTests_call_result_206496)
        
        # Call to assertEqual(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'Test' (line 94)
        Test_206499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'Test', False)
        # Obtaining the member 'tearDownCalled' of a type (line 94)
        tearDownCalled_206500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), Test_206499, 'tearDownCalled')
        int_206501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 46), 'int')
        # Processing the call keyword arguments (line 94)
        kwargs_206502 = {}
        # Getting the type of 'self' (line 94)
        self_206497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 94)
        assertEqual_206498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_206497, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 94)
        assertEqual_call_result_206503 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), assertEqual_206498, *[tearDownCalled_206500, int_206501], **kwargs_206502)
        
        
        # Call to assertEqual(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'Test2' (line 95)
        Test2_206506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'Test2', False)
        # Obtaining the member 'tearDownCalled' of a type (line 95)
        tearDownCalled_206507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 25), Test2_206506, 'tearDownCalled')
        int_206508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 47), 'int')
        # Processing the call keyword arguments (line 95)
        kwargs_206509 = {}
        # Getting the type of 'self' (line 95)
        self_206504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 95)
        assertEqual_206505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_206504, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 95)
        assertEqual_call_result_206510 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), assertEqual_206505, *[tearDownCalled_206507, int_206508], **kwargs_206509)
        
        
        # Call to assertEqual(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'result' (line 96)
        result_206513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 96)
        testsRun_206514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 25), result_206513, 'testsRun')
        int_206515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'int')
        # Processing the call keyword arguments (line 96)
        kwargs_206516 = {}
        # Getting the type of 'self' (line 96)
        self_206511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 96)
        assertEqual_206512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_206511, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 96)
        assertEqual_call_result_206517 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assertEqual_206512, *[testsRun_206514, int_206515], **kwargs_206516)
        
        
        # Call to assertEqual(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to len(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'result' (line 97)
        result_206521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 97)
        errors_206522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 29), result_206521, 'errors')
        # Processing the call keyword arguments (line 97)
        kwargs_206523 = {}
        # Getting the type of 'len' (line 97)
        len_206520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'len', False)
        # Calling len(args, kwargs) (line 97)
        len_call_result_206524 = invoke(stypy.reporting.localization.Localization(__file__, 97, 25), len_206520, *[errors_206522], **kwargs_206523)
        
        int_206525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 45), 'int')
        # Processing the call keyword arguments (line 97)
        kwargs_206526 = {}
        # Getting the type of 'self' (line 97)
        self_206518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 97)
        assertEqual_206519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_206518, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 97)
        assertEqual_call_result_206527 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assertEqual_206519, *[len_call_result_206524, int_206525], **kwargs_206526)
        
        
        # ################# End of 'test_teardown_class_two_classes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_teardown_class_two_classes' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_206528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_teardown_class_two_classes'
        return stypy_return_type_206528


    @norecursion
    def test_error_in_setupclass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_error_in_setupclass'
        module_type_store = module_type_store.open_function_context('test_error_in_setupclass', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_error_in_setupclass')
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_error_in_setupclass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_error_in_setupclass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_error_in_setupclass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_error_in_setupclass(...)' code ##################

        # Declaration of the 'BrokenTest' class
        # Getting the type of 'unittest' (line 100)
        unittest_206529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 100)
        TestCase_206530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 25), unittest_206529, 'TestCase')

        class BrokenTest(TestCase_206530, ):

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 101, 12, False)
                # Assigning a type to the variable 'self' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_function_name', 'BrokenTest.setUpClass')
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                BrokenTest.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'BrokenTest.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to TypeError(...): (line 103)
                # Processing the call arguments (line 103)
                str_206532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 32), 'str', 'foo')
                # Processing the call keyword arguments (line 103)
                kwargs_206533 = {}
                # Getting the type of 'TypeError' (line 103)
                TypeError_206531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 103)
                TypeError_call_result_206534 = invoke(stypy.reporting.localization.Localization(__file__, 103, 22), TypeError_206531, *[str_206532], **kwargs_206533)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 103, 16), TypeError_call_result_206534, 'raise parameter', BaseException)
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 101)
                stypy_return_type_206535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206535)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_206535


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 104, 12, False)
                # Assigning a type to the variable 'self' (line 105)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                BrokenTest.test_one.__dict__.__setitem__('stypy_localization', localization)
                BrokenTest.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                BrokenTest.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                BrokenTest.test_one.__dict__.__setitem__('stypy_function_name', 'BrokenTest.test_one')
                BrokenTest.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                BrokenTest.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                BrokenTest.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                BrokenTest.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                BrokenTest.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                BrokenTest.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                BrokenTest.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'BrokenTest.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 104)
                stypy_return_type_206536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206536)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206536


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 106, 12, False)
                # Assigning a type to the variable 'self' (line 107)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                BrokenTest.test_two.__dict__.__setitem__('stypy_localization', localization)
                BrokenTest.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                BrokenTest.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                BrokenTest.test_two.__dict__.__setitem__('stypy_function_name', 'BrokenTest.test_two')
                BrokenTest.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                BrokenTest.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                BrokenTest.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                BrokenTest.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                BrokenTest.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                BrokenTest.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                BrokenTest.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'BrokenTest.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 106)
                stypy_return_type_206537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206537)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_206537

        
        # Assigning a type to the variable 'BrokenTest' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'BrokenTest', BrokenTest)
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to runTests(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'BrokenTest' (line 109)
        BrokenTest_206540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'BrokenTest', False)
        # Processing the call keyword arguments (line 109)
        kwargs_206541 = {}
        # Getting the type of 'self' (line 109)
        self_206538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 109)
        runTests_206539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 17), self_206538, 'runTests')
        # Calling runTests(args, kwargs) (line 109)
        runTests_call_result_206542 = invoke(stypy.reporting.localization.Localization(__file__, 109, 17), runTests_206539, *[BrokenTest_206540], **kwargs_206541)
        
        # Assigning a type to the variable 'result' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'result', runTests_call_result_206542)
        
        # Call to assertEqual(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'result' (line 111)
        result_206545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 111)
        testsRun_206546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 25), result_206545, 'testsRun')
        int_206547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 42), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_206548 = {}
        # Getting the type of 'self' (line 111)
        self_206543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 111)
        assertEqual_206544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_206543, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 111)
        assertEqual_call_result_206549 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assertEqual_206544, *[testsRun_206546, int_206547], **kwargs_206548)
        
        
        # Call to assertEqual(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to len(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'result' (line 112)
        result_206553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 112)
        errors_206554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 29), result_206553, 'errors')
        # Processing the call keyword arguments (line 112)
        kwargs_206555 = {}
        # Getting the type of 'len' (line 112)
        len_206552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 'len', False)
        # Calling len(args, kwargs) (line 112)
        len_call_result_206556 = invoke(stypy.reporting.localization.Localization(__file__, 112, 25), len_206552, *[errors_206554], **kwargs_206555)
        
        int_206557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 45), 'int')
        # Processing the call keyword arguments (line 112)
        kwargs_206558 = {}
        # Getting the type of 'self' (line 112)
        self_206550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 112)
        assertEqual_206551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_206550, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 112)
        assertEqual_call_result_206559 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assertEqual_206551, *[len_call_result_206556, int_206557], **kwargs_206558)
        
        
        # Assigning a Subscript to a Tuple (line 113):
        
        # Assigning a Subscript to a Name (line 113):
        
        # Obtaining the type of the subscript
        int_206560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'int')
        
        # Obtaining the type of the subscript
        int_206561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 33), 'int')
        # Getting the type of 'result' (line 113)
        result_206562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'result')
        # Obtaining the member 'errors' of a type (line 113)
        errors_206563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), result_206562, 'errors')
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___206564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), errors_206563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_206565 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), getitem___206564, int_206561)
        
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___206566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), subscript_call_result_206565, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_206567 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), getitem___206566, int_206560)
        
        # Assigning a type to the variable 'tuple_var_assignment_206289' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'tuple_var_assignment_206289', subscript_call_result_206567)
        
        # Assigning a Subscript to a Name (line 113):
        
        # Obtaining the type of the subscript
        int_206568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'int')
        
        # Obtaining the type of the subscript
        int_206569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 33), 'int')
        # Getting the type of 'result' (line 113)
        result_206570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'result')
        # Obtaining the member 'errors' of a type (line 113)
        errors_206571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), result_206570, 'errors')
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___206572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), errors_206571, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_206573 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), getitem___206572, int_206569)
        
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___206574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), subscript_call_result_206573, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_206575 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), getitem___206574, int_206568)
        
        # Assigning a type to the variable 'tuple_var_assignment_206290' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'tuple_var_assignment_206290', subscript_call_result_206575)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'tuple_var_assignment_206289' (line 113)
        tuple_var_assignment_206289_206576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'tuple_var_assignment_206289')
        # Assigning a type to the variable 'error' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'error', tuple_var_assignment_206289_206576)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'tuple_var_assignment_206290' (line 113)
        tuple_var_assignment_206290_206577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'tuple_var_assignment_206290')
        # Assigning a type to the variable '_' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), '_', tuple_var_assignment_206290_206577)
        
        # Call to assertEqual(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to str(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'error' (line 114)
        error_206581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'error', False)
        # Processing the call keyword arguments (line 114)
        kwargs_206582 = {}
        # Getting the type of 'str' (line 114)
        str_206580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 25), 'str', False)
        # Calling str(args, kwargs) (line 114)
        str_call_result_206583 = invoke(stypy.reporting.localization.Localization(__file__, 114, 25), str_206580, *[error_206581], **kwargs_206582)
        
        str_206584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 20), 'str', 'setUpClass (%s.BrokenTest)')
        # Getting the type of '__name__' (line 115)
        name___206585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 51), '__name__', False)
        # Applying the binary operator '%' (line 115)
        result_mod_206586 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 20), '%', str_206584, name___206585)
        
        # Processing the call keyword arguments (line 114)
        kwargs_206587 = {}
        # Getting the type of 'self' (line 114)
        self_206578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 114)
        assertEqual_206579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_206578, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 114)
        assertEqual_call_result_206588 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assertEqual_206579, *[str_call_result_206583, result_mod_206586], **kwargs_206587)
        
        
        # ################# End of 'test_error_in_setupclass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_error_in_setupclass' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_206589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206589)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_error_in_setupclass'
        return stypy_return_type_206589


    @norecursion
    def test_error_in_teardown_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_error_in_teardown_class'
        module_type_store = module_type_store.open_function_context('test_error_in_teardown_class', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_error_in_teardown_class')
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_error_in_teardown_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_error_in_teardown_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_error_in_teardown_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_error_in_teardown_class(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 118)
        unittest_206590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 118)
        TestCase_206591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), unittest_206590, 'TestCase')

        class Test(TestCase_206591, ):
            
            # Assigning a Num to a Name (line 119):
            
            # Assigning a Num to a Name (line 119):
            int_206592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'int')
            # Assigning a type to the variable 'tornDown' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'tornDown', int_206592)

            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 120, 12, False)
                # Assigning a type to the variable 'self' (line 121)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test.tearDownClass')
                Test.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Getting the type of 'Test' (line 122)
                Test_206593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'Test')
                # Obtaining the member 'tornDown' of a type (line 122)
                tornDown_206594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), Test_206593, 'tornDown')
                int_206595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 33), 'int')
                # Applying the binary operator '+=' (line 122)
                result_iadd_206596 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 16), '+=', tornDown_206594, int_206595)
                # Getting the type of 'Test' (line 122)
                Test_206597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'Test')
                # Setting the type of the member 'tornDown' of a type (line 122)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), Test_206597, 'tornDown', result_iadd_206596)
                
                
                # Call to TypeError(...): (line 123)
                # Processing the call arguments (line 123)
                str_206599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 32), 'str', 'foo')
                # Processing the call keyword arguments (line 123)
                kwargs_206600 = {}
                # Getting the type of 'TypeError' (line 123)
                TypeError_206598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 123)
                TypeError_call_result_206601 = invoke(stypy.reporting.localization.Localization(__file__, 123, 22), TypeError_206598, *[str_206599], **kwargs_206600)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 123, 16), TypeError_call_result_206601, 'raise parameter', BaseException)
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 120)
                stypy_return_type_206602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206602)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206602


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 124, 12, False)
                # Assigning a type to the variable 'self' (line 125)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 124)
                stypy_return_type_206603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206603)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206603


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 126, 12, False)
                # Assigning a type to the variable 'self' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 126)
                stypy_return_type_206604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206604)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_206604

        
        # Assigning a type to the variable 'Test' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'Test', Test)
        # Declaration of the 'Test2' class
        # Getting the type of 'unittest' (line 129)
        unittest_206605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 129)
        TestCase_206606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 20), unittest_206605, 'TestCase')

        class Test2(TestCase_206606, ):
            
            # Assigning a Num to a Name (line 130):
            
            # Assigning a Num to a Name (line 130):
            int_206607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 23), 'int')
            # Assigning a type to the variable 'tornDown' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'tornDown', int_206607)

            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 131, 12, False)
                # Assigning a type to the variable 'self' (line 132)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test2.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test2.tearDownClass')
                Test2.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Getting the type of 'Test2' (line 133)
                Test2_206608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'Test2')
                # Obtaining the member 'tornDown' of a type (line 133)
                tornDown_206609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), Test2_206608, 'tornDown')
                int_206610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 34), 'int')
                # Applying the binary operator '+=' (line 133)
                result_iadd_206611 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 16), '+=', tornDown_206609, int_206610)
                # Getting the type of 'Test2' (line 133)
                Test2_206612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'Test2')
                # Setting the type of the member 'tornDown' of a type (line 133)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), Test2_206612, 'tornDown', result_iadd_206611)
                
                
                # Call to TypeError(...): (line 134)
                # Processing the call arguments (line 134)
                str_206614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 32), 'str', 'foo')
                # Processing the call keyword arguments (line 134)
                kwargs_206615 = {}
                # Getting the type of 'TypeError' (line 134)
                TypeError_206613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 134)
                TypeError_call_result_206616 = invoke(stypy.reporting.localization.Localization(__file__, 134, 22), TypeError_206613, *[str_206614], **kwargs_206615)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 134, 16), TypeError_call_result_206616, 'raise parameter', BaseException)
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 131)
                stypy_return_type_206617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206617)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206617


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 135, 12, False)
                # Assigning a type to the variable 'self' (line 136)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test2.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.test_one.__dict__.__setitem__('stypy_function_name', 'Test2.test_one')
                Test2.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 135)
                stypy_return_type_206618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206618)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206618


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 137, 12, False)
                # Assigning a type to the variable 'self' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test2.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.test_two.__dict__.__setitem__('stypy_function_name', 'Test2.test_two')
                Test2.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 137)
                stypy_return_type_206619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206619)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_206619

        
        # Assigning a type to the variable 'Test2' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'Test2', Test2)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to runTests(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'Test' (line 140)
        Test_206622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 31), 'Test', False)
        # Getting the type of 'Test2' (line 140)
        Test2_206623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 37), 'Test2', False)
        # Processing the call keyword arguments (line 140)
        kwargs_206624 = {}
        # Getting the type of 'self' (line 140)
        self_206620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 140)
        runTests_206621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 17), self_206620, 'runTests')
        # Calling runTests(args, kwargs) (line 140)
        runTests_call_result_206625 = invoke(stypy.reporting.localization.Localization(__file__, 140, 17), runTests_206621, *[Test_206622, Test2_206623], **kwargs_206624)
        
        # Assigning a type to the variable 'result' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'result', runTests_call_result_206625)
        
        # Call to assertEqual(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'result' (line 141)
        result_206628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 141)
        testsRun_206629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 25), result_206628, 'testsRun')
        int_206630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 42), 'int')
        # Processing the call keyword arguments (line 141)
        kwargs_206631 = {}
        # Getting the type of 'self' (line 141)
        self_206626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 141)
        assertEqual_206627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), self_206626, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 141)
        assertEqual_call_result_206632 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), assertEqual_206627, *[testsRun_206629, int_206630], **kwargs_206631)
        
        
        # Call to assertEqual(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Call to len(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'result' (line 142)
        result_206636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 142)
        errors_206637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 29), result_206636, 'errors')
        # Processing the call keyword arguments (line 142)
        kwargs_206638 = {}
        # Getting the type of 'len' (line 142)
        len_206635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'len', False)
        # Calling len(args, kwargs) (line 142)
        len_call_result_206639 = invoke(stypy.reporting.localization.Localization(__file__, 142, 25), len_206635, *[errors_206637], **kwargs_206638)
        
        int_206640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 45), 'int')
        # Processing the call keyword arguments (line 142)
        kwargs_206641 = {}
        # Getting the type of 'self' (line 142)
        self_206633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 142)
        assertEqual_206634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_206633, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 142)
        assertEqual_call_result_206642 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), assertEqual_206634, *[len_call_result_206639, int_206640], **kwargs_206641)
        
        
        # Call to assertEqual(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'Test' (line 143)
        Test_206645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 'Test', False)
        # Obtaining the member 'tornDown' of a type (line 143)
        tornDown_206646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 25), Test_206645, 'tornDown')
        int_206647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 40), 'int')
        # Processing the call keyword arguments (line 143)
        kwargs_206648 = {}
        # Getting the type of 'self' (line 143)
        self_206643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 143)
        assertEqual_206644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_206643, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 143)
        assertEqual_call_result_206649 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), assertEqual_206644, *[tornDown_206646, int_206647], **kwargs_206648)
        
        
        # Call to assertEqual(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'Test2' (line 144)
        Test2_206652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'Test2', False)
        # Obtaining the member 'tornDown' of a type (line 144)
        tornDown_206653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 25), Test2_206652, 'tornDown')
        int_206654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 41), 'int')
        # Processing the call keyword arguments (line 144)
        kwargs_206655 = {}
        # Getting the type of 'self' (line 144)
        self_206650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 144)
        assertEqual_206651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_206650, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 144)
        assertEqual_call_result_206656 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assertEqual_206651, *[tornDown_206653, int_206654], **kwargs_206655)
        
        
        # Assigning a Subscript to a Tuple (line 146):
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_206657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Obtaining the type of the subscript
        int_206658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 33), 'int')
        # Getting the type of 'result' (line 146)
        result_206659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'result')
        # Obtaining the member 'errors' of a type (line 146)
        errors_206660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), result_206659, 'errors')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___206661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), errors_206660, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_206662 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), getitem___206661, int_206658)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___206663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_206662, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_206664 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___206663, int_206657)
        
        # Assigning a type to the variable 'tuple_var_assignment_206291' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_206291', subscript_call_result_206664)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_206665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Obtaining the type of the subscript
        int_206666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 33), 'int')
        # Getting the type of 'result' (line 146)
        result_206667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'result')
        # Obtaining the member 'errors' of a type (line 146)
        errors_206668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), result_206667, 'errors')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___206669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), errors_206668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_206670 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), getitem___206669, int_206666)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___206671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), subscript_call_result_206670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_206672 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___206671, int_206665)
        
        # Assigning a type to the variable 'tuple_var_assignment_206292' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_206292', subscript_call_result_206672)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_206291' (line 146)
        tuple_var_assignment_206291_206673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_206291')
        # Assigning a type to the variable 'error' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'error', tuple_var_assignment_206291_206673)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_206292' (line 146)
        tuple_var_assignment_206292_206674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_206292')
        # Assigning a type to the variable '_' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), '_', tuple_var_assignment_206292_206674)
        
        # Call to assertEqual(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Call to str(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'error' (line 147)
        error_206678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 29), 'error', False)
        # Processing the call keyword arguments (line 147)
        kwargs_206679 = {}
        # Getting the type of 'str' (line 147)
        str_206677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'str', False)
        # Calling str(args, kwargs) (line 147)
        str_call_result_206680 = invoke(stypy.reporting.localization.Localization(__file__, 147, 25), str_206677, *[error_206678], **kwargs_206679)
        
        str_206681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 20), 'str', 'tearDownClass (%s.Test)')
        # Getting the type of '__name__' (line 148)
        name___206682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), '__name__', False)
        # Applying the binary operator '%' (line 148)
        result_mod_206683 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 20), '%', str_206681, name___206682)
        
        # Processing the call keyword arguments (line 147)
        kwargs_206684 = {}
        # Getting the type of 'self' (line 147)
        self_206675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 147)
        assertEqual_206676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_206675, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 147)
        assertEqual_call_result_206685 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assertEqual_206676, *[str_call_result_206680, result_mod_206683], **kwargs_206684)
        
        
        # ################# End of 'test_error_in_teardown_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_error_in_teardown_class' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_206686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206686)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_error_in_teardown_class'
        return stypy_return_type_206686


    @norecursion
    def test_class_not_torndown_when_setup_fails(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_class_not_torndown_when_setup_fails'
        module_type_store = module_type_store.open_function_context('test_class_not_torndown_when_setup_fails', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_class_not_torndown_when_setup_fails')
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_class_not_torndown_when_setup_fails.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_class_not_torndown_when_setup_fails', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_class_not_torndown_when_setup_fails', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_class_not_torndown_when_setup_fails(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 151)
        unittest_206687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 151)
        TestCase_206688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 19), unittest_206687, 'TestCase')

        class Test(TestCase_206688, ):
            
            # Assigning a Name to a Name (line 152):
            
            # Assigning a Name to a Name (line 152):
            # Getting the type of 'False' (line 152)
            False_206689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'False')
            # Assigning a type to the variable 'tornDown' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tornDown', False_206689)

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 153, 12, False)
                # Assigning a type to the variable 'self' (line 154)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test.setUpClass')
                Test.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                # Getting the type of 'TypeError' (line 155)
                TypeError_206690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'TypeError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 155, 16), TypeError_206690, 'raise parameter', BaseException)
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 153)
                stypy_return_type_206691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206691)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_206691


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 156, 12, False)
                # Assigning a type to the variable 'self' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test.tearDownClass')
                Test.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Assigning a Name to a Attribute (line 158):
                
                # Assigning a Name to a Attribute (line 158):
                # Getting the type of 'True' (line 158)
                True_206692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'True')
                # Getting the type of 'Test' (line 158)
                Test_206693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'Test')
                # Setting the type of the member 'tornDown' of a type (line 158)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), Test_206693, 'tornDown', True_206692)
                
                # Call to TypeError(...): (line 159)
                # Processing the call arguments (line 159)
                str_206695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 32), 'str', 'foo')
                # Processing the call keyword arguments (line 159)
                kwargs_206696 = {}
                # Getting the type of 'TypeError' (line 159)
                TypeError_206694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 159)
                TypeError_call_result_206697 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), TypeError_206694, *[str_206695], **kwargs_206696)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 159, 16), TypeError_call_result_206697, 'raise parameter', BaseException)
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 156)
                stypy_return_type_206698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206698)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206698


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 160, 12, False)
                # Assigning a type to the variable 'self' (line 161)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 160)
                stypy_return_type_206699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206699)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206699

        
        # Assigning a type to the variable 'Test' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'Test', Test)
        
        # Call to runTests(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'Test' (line 163)
        Test_206702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'Test', False)
        # Processing the call keyword arguments (line 163)
        kwargs_206703 = {}
        # Getting the type of 'self' (line 163)
        self_206700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self', False)
        # Obtaining the member 'runTests' of a type (line 163)
        runTests_206701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_206700, 'runTests')
        # Calling runTests(args, kwargs) (line 163)
        runTests_call_result_206704 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), runTests_206701, *[Test_206702], **kwargs_206703)
        
        
        # Call to assertFalse(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'Test' (line 164)
        Test_206707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'Test', False)
        # Obtaining the member 'tornDown' of a type (line 164)
        tornDown_206708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 25), Test_206707, 'tornDown')
        # Processing the call keyword arguments (line 164)
        kwargs_206709 = {}
        # Getting the type of 'self' (line 164)
        self_206705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 164)
        assertFalse_206706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_206705, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 164)
        assertFalse_call_result_206710 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), assertFalse_206706, *[tornDown_206708], **kwargs_206709)
        
        
        # ################# End of 'test_class_not_torndown_when_setup_fails(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_class_not_torndown_when_setup_fails' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_206711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206711)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_class_not_torndown_when_setup_fails'
        return stypy_return_type_206711


    @norecursion
    def test_class_not_setup_or_torndown_when_skipped(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_class_not_setup_or_torndown_when_skipped'
        module_type_store = module_type_store.open_function_context('test_class_not_setup_or_torndown_when_skipped', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_class_not_setup_or_torndown_when_skipped')
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_class_not_setup_or_torndown_when_skipped.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_class_not_setup_or_torndown_when_skipped', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_class_not_setup_or_torndown_when_skipped', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_class_not_setup_or_torndown_when_skipped(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 167)
        unittest_206712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 167)
        TestCase_206713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 19), unittest_206712, 'TestCase')

        class Test(TestCase_206713, ):
            
            # Assigning a Name to a Name (line 168):
            
            # Assigning a Name to a Name (line 168):
            # Getting the type of 'False' (line 168)
            False_206714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'False')
            # Assigning a type to the variable 'classSetUp' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'classSetUp', False_206714)
            
            # Assigning a Name to a Name (line 169):
            
            # Assigning a Name to a Name (line 169):
            # Getting the type of 'False' (line 169)
            False_206715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'False')
            # Assigning a type to the variable 'tornDown' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'tornDown', False_206715)

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 170, 12, False)
                # Assigning a type to the variable 'self' (line 171)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test.setUpClass')
                Test.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Assigning a Name to a Attribute (line 172):
                
                # Assigning a Name to a Attribute (line 172):
                # Getting the type of 'True' (line 172)
                True_206716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'True')
                # Getting the type of 'Test' (line 172)
                Test_206717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'Test')
                # Setting the type of the member 'classSetUp' of a type (line 172)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), Test_206717, 'classSetUp', True_206716)
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 170)
                stypy_return_type_206718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206718)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_206718


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 173, 12, False)
                # Assigning a type to the variable 'self' (line 174)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test.tearDownClass')
                Test.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Assigning a Name to a Attribute (line 175):
                
                # Assigning a Name to a Attribute (line 175):
                # Getting the type of 'True' (line 175)
                True_206719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 32), 'True')
                # Getting the type of 'Test' (line 175)
                Test_206720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'Test')
                # Setting the type of the member 'tornDown' of a type (line 175)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 16), Test_206720, 'tornDown', True_206719)
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 173)
                stypy_return_type_206721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206721)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206721


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 176, 12, False)
                # Assigning a type to the variable 'self' (line 177)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 176)
                stypy_return_type_206722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206722)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206722

        
        # Assigning a type to the variable 'Test' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'Test', Test)
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to (...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'Test' (line 179)
        Test_206728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 36), 'Test', False)
        # Processing the call keyword arguments (line 179)
        kwargs_206729 = {}
        
        # Call to skip(...): (line 179)
        # Processing the call arguments (line 179)
        str_206725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'str', 'hop')
        # Processing the call keyword arguments (line 179)
        kwargs_206726 = {}
        # Getting the type of 'unittest' (line 179)
        unittest_206723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'unittest', False)
        # Obtaining the member 'skip' of a type (line 179)
        skip_206724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 15), unittest_206723, 'skip')
        # Calling skip(args, kwargs) (line 179)
        skip_call_result_206727 = invoke(stypy.reporting.localization.Localization(__file__, 179, 15), skip_206724, *[str_206725], **kwargs_206726)
        
        # Calling (args, kwargs) (line 179)
        _call_result_206730 = invoke(stypy.reporting.localization.Localization(__file__, 179, 15), skip_call_result_206727, *[Test_206728], **kwargs_206729)
        
        # Assigning a type to the variable 'Test' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'Test', _call_result_206730)
        
        # Call to runTests(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'Test' (line 180)
        Test_206733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'Test', False)
        # Processing the call keyword arguments (line 180)
        kwargs_206734 = {}
        # Getting the type of 'self' (line 180)
        self_206731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self', False)
        # Obtaining the member 'runTests' of a type (line 180)
        runTests_206732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_206731, 'runTests')
        # Calling runTests(args, kwargs) (line 180)
        runTests_call_result_206735 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), runTests_206732, *[Test_206733], **kwargs_206734)
        
        
        # Call to assertFalse(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'Test' (line 181)
        Test_206738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'Test', False)
        # Obtaining the member 'classSetUp' of a type (line 181)
        classSetUp_206739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 25), Test_206738, 'classSetUp')
        # Processing the call keyword arguments (line 181)
        kwargs_206740 = {}
        # Getting the type of 'self' (line 181)
        self_206736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 181)
        assertFalse_206737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_206736, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 181)
        assertFalse_call_result_206741 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assertFalse_206737, *[classSetUp_206739], **kwargs_206740)
        
        
        # Call to assertFalse(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'Test' (line 182)
        Test_206744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 25), 'Test', False)
        # Obtaining the member 'tornDown' of a type (line 182)
        tornDown_206745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 25), Test_206744, 'tornDown')
        # Processing the call keyword arguments (line 182)
        kwargs_206746 = {}
        # Getting the type of 'self' (line 182)
        self_206742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 182)
        assertFalse_206743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), self_206742, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 182)
        assertFalse_call_result_206747 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), assertFalse_206743, *[tornDown_206745], **kwargs_206746)
        
        
        # ################# End of 'test_class_not_setup_or_torndown_when_skipped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_class_not_setup_or_torndown_when_skipped' in the type store
        # Getting the type of 'stypy_return_type' (line 166)
        stypy_return_type_206748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_class_not_setup_or_torndown_when_skipped'
        return stypy_return_type_206748


    @norecursion
    def test_setup_teardown_order_with_pathological_suite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_setup_teardown_order_with_pathological_suite'
        module_type_store = module_type_store.open_function_context('test_setup_teardown_order_with_pathological_suite', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_setup_teardown_order_with_pathological_suite')
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_setup_teardown_order_with_pathological_suite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_setup_teardown_order_with_pathological_suite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_setup_teardown_order_with_pathological_suite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_setup_teardown_order_with_pathological_suite(...)' code ##################

        
        # Assigning a List to a Name (line 185):
        
        # Assigning a List to a Name (line 185):
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_206749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        
        # Assigning a type to the variable 'results' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'results', list_206749)
        # Declaration of the 'Module1' class

        class Module1(object, ):

            @staticmethod
            @norecursion
            def setUpModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpModule'
                module_type_store = module_type_store.open_function_context('setUpModule', 188, 12, False)
                
                # Passed parameters checking function
                Module1.setUpModule.__dict__.__setitem__('stypy_localization', localization)
                Module1.setUpModule.__dict__.__setitem__('stypy_type_of_self', None)
                Module1.setUpModule.__dict__.__setitem__('stypy_type_store', module_type_store)
                Module1.setUpModule.__dict__.__setitem__('stypy_function_name', 'setUpModule')
                Module1.setUpModule.__dict__.__setitem__('stypy_param_names_list', [])
                Module1.setUpModule.__dict__.__setitem__('stypy_varargs_param_name', None)
                Module1.setUpModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Module1.setUpModule.__dict__.__setitem__('stypy_call_defaults', defaults)
                Module1.setUpModule.__dict__.__setitem__('stypy_call_varargs', varargs)
                Module1.setUpModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Module1.setUpModule.__dict__.__setitem__('stypy_declared_arg_number', 0)
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

                
                # Call to append(...): (line 190)
                # Processing the call arguments (line 190)
                str_206752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 31), 'str', 'Module1.setUpModule')
                # Processing the call keyword arguments (line 190)
                kwargs_206753 = {}
                # Getting the type of 'results' (line 190)
                results_206750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 190)
                append_206751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), results_206750, 'append')
                # Calling append(args, kwargs) (line 190)
                append_call_result_206754 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), append_206751, *[str_206752], **kwargs_206753)
                
                
                # ################# End of 'setUpModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpModule' in the type store
                # Getting the type of 'stypy_return_type' (line 188)
                stypy_return_type_206755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206755)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpModule'
                return stypy_return_type_206755


            @staticmethod
            @norecursion
            def tearDownModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownModule'
                module_type_store = module_type_store.open_function_context('tearDownModule', 191, 12, False)
                
                # Passed parameters checking function
                Module1.tearDownModule.__dict__.__setitem__('stypy_localization', localization)
                Module1.tearDownModule.__dict__.__setitem__('stypy_type_of_self', None)
                Module1.tearDownModule.__dict__.__setitem__('stypy_type_store', module_type_store)
                Module1.tearDownModule.__dict__.__setitem__('stypy_function_name', 'tearDownModule')
                Module1.tearDownModule.__dict__.__setitem__('stypy_param_names_list', [])
                Module1.tearDownModule.__dict__.__setitem__('stypy_varargs_param_name', None)
                Module1.tearDownModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Module1.tearDownModule.__dict__.__setitem__('stypy_call_defaults', defaults)
                Module1.tearDownModule.__dict__.__setitem__('stypy_call_varargs', varargs)
                Module1.tearDownModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Module1.tearDownModule.__dict__.__setitem__('stypy_declared_arg_number', 0)
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

                
                # Call to append(...): (line 193)
                # Processing the call arguments (line 193)
                str_206758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 31), 'str', 'Module1.tearDownModule')
                # Processing the call keyword arguments (line 193)
                kwargs_206759 = {}
                # Getting the type of 'results' (line 193)
                results_206756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 193)
                append_206757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 16), results_206756, 'append')
                # Calling append(args, kwargs) (line 193)
                append_call_result_206760 = invoke(stypy.reporting.localization.Localization(__file__, 193, 16), append_206757, *[str_206758], **kwargs_206759)
                
                
                # ################# End of 'tearDownModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownModule' in the type store
                # Getting the type of 'stypy_return_type' (line 191)
                stypy_return_type_206761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206761)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownModule'
                return stypy_return_type_206761

        
        # Assigning a type to the variable 'Module1' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'Module1', Module1)
        # Declaration of the 'Module2' class

        class Module2(object, ):

            @staticmethod
            @norecursion
            def setUpModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpModule'
                module_type_store = module_type_store.open_function_context('setUpModule', 196, 12, False)
                
                # Passed parameters checking function
                Module2.setUpModule.__dict__.__setitem__('stypy_localization', localization)
                Module2.setUpModule.__dict__.__setitem__('stypy_type_of_self', None)
                Module2.setUpModule.__dict__.__setitem__('stypy_type_store', module_type_store)
                Module2.setUpModule.__dict__.__setitem__('stypy_function_name', 'setUpModule')
                Module2.setUpModule.__dict__.__setitem__('stypy_param_names_list', [])
                Module2.setUpModule.__dict__.__setitem__('stypy_varargs_param_name', None)
                Module2.setUpModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Module2.setUpModule.__dict__.__setitem__('stypy_call_defaults', defaults)
                Module2.setUpModule.__dict__.__setitem__('stypy_call_varargs', varargs)
                Module2.setUpModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Module2.setUpModule.__dict__.__setitem__('stypy_declared_arg_number', 0)
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

                
                # Call to append(...): (line 198)
                # Processing the call arguments (line 198)
                str_206764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 31), 'str', 'Module2.setUpModule')
                # Processing the call keyword arguments (line 198)
                kwargs_206765 = {}
                # Getting the type of 'results' (line 198)
                results_206762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 198)
                append_206763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 16), results_206762, 'append')
                # Calling append(args, kwargs) (line 198)
                append_call_result_206766 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), append_206763, *[str_206764], **kwargs_206765)
                
                
                # ################# End of 'setUpModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpModule' in the type store
                # Getting the type of 'stypy_return_type' (line 196)
                stypy_return_type_206767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206767)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpModule'
                return stypy_return_type_206767


            @staticmethod
            @norecursion
            def tearDownModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownModule'
                module_type_store = module_type_store.open_function_context('tearDownModule', 199, 12, False)
                
                # Passed parameters checking function
                Module2.tearDownModule.__dict__.__setitem__('stypy_localization', localization)
                Module2.tearDownModule.__dict__.__setitem__('stypy_type_of_self', None)
                Module2.tearDownModule.__dict__.__setitem__('stypy_type_store', module_type_store)
                Module2.tearDownModule.__dict__.__setitem__('stypy_function_name', 'tearDownModule')
                Module2.tearDownModule.__dict__.__setitem__('stypy_param_names_list', [])
                Module2.tearDownModule.__dict__.__setitem__('stypy_varargs_param_name', None)
                Module2.tearDownModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Module2.tearDownModule.__dict__.__setitem__('stypy_call_defaults', defaults)
                Module2.tearDownModule.__dict__.__setitem__('stypy_call_varargs', varargs)
                Module2.tearDownModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Module2.tearDownModule.__dict__.__setitem__('stypy_declared_arg_number', 0)
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

                
                # Call to append(...): (line 201)
                # Processing the call arguments (line 201)
                str_206770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 31), 'str', 'Module2.tearDownModule')
                # Processing the call keyword arguments (line 201)
                kwargs_206771 = {}
                # Getting the type of 'results' (line 201)
                results_206768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 201)
                append_206769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), results_206768, 'append')
                # Calling append(args, kwargs) (line 201)
                append_call_result_206772 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), append_206769, *[str_206770], **kwargs_206771)
                
                
                # ################# End of 'tearDownModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownModule' in the type store
                # Getting the type of 'stypy_return_type' (line 199)
                stypy_return_type_206773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206773)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownModule'
                return stypy_return_type_206773

        
        # Assigning a type to the variable 'Module2' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'Module2', Module2)
        # Declaration of the 'Test1' class
        # Getting the type of 'unittest' (line 203)
        unittest_206774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 203)
        TestCase_206775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 20), unittest_206774, 'TestCase')

        class Test1(TestCase_206775, ):

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 204, 12, False)
                # Assigning a type to the variable 'self' (line 205)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test1.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test1.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test1.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test1.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test1.setUpClass')
                Test1.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test1.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test1.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test1.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test1.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test1.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test1.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test1.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 206)
                # Processing the call arguments (line 206)
                str_206778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 31), 'str', 'setup 1')
                # Processing the call keyword arguments (line 206)
                kwargs_206779 = {}
                # Getting the type of 'results' (line 206)
                results_206776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 206)
                append_206777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), results_206776, 'append')
                # Calling append(args, kwargs) (line 206)
                append_call_result_206780 = invoke(stypy.reporting.localization.Localization(__file__, 206, 16), append_206777, *[str_206778], **kwargs_206779)
                
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 204)
                stypy_return_type_206781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206781)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_206781


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 207, 12, False)
                # Assigning a type to the variable 'self' (line 208)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test1.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test1.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test1.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test1.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test1.tearDownClass')
                Test1.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test1.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test1.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test1.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test1.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test1.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test1.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test1.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 209)
                # Processing the call arguments (line 209)
                str_206784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 31), 'str', 'teardown 1')
                # Processing the call keyword arguments (line 209)
                kwargs_206785 = {}
                # Getting the type of 'results' (line 209)
                results_206782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 209)
                append_206783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 16), results_206782, 'append')
                # Calling append(args, kwargs) (line 209)
                append_call_result_206786 = invoke(stypy.reporting.localization.Localization(__file__, 209, 16), append_206783, *[str_206784], **kwargs_206785)
                
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 207)
                stypy_return_type_206787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206787)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206787


            @norecursion
            def testOne(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testOne'
                module_type_store = module_type_store.open_function_context('testOne', 210, 12, False)
                # Assigning a type to the variable 'self' (line 211)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test1.testOne.__dict__.__setitem__('stypy_localization', localization)
                Test1.testOne.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test1.testOne.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test1.testOne.__dict__.__setitem__('stypy_function_name', 'Test1.testOne')
                Test1.testOne.__dict__.__setitem__('stypy_param_names_list', [])
                Test1.testOne.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test1.testOne.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test1.testOne.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test1.testOne.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test1.testOne.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test1.testOne.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test1.testOne', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testOne', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testOne(...)' code ##################

                
                # Call to append(...): (line 211)
                # Processing the call arguments (line 211)
                str_206790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 31), 'str', 'Test1.testOne')
                # Processing the call keyword arguments (line 211)
                kwargs_206791 = {}
                # Getting the type of 'results' (line 211)
                results_206788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 211)
                append_206789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 16), results_206788, 'append')
                # Calling append(args, kwargs) (line 211)
                append_call_result_206792 = invoke(stypy.reporting.localization.Localization(__file__, 211, 16), append_206789, *[str_206790], **kwargs_206791)
                
                
                # ################# End of 'testOne(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testOne' in the type store
                # Getting the type of 'stypy_return_type' (line 210)
                stypy_return_type_206793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206793)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testOne'
                return stypy_return_type_206793


            @norecursion
            def testTwo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testTwo'
                module_type_store = module_type_store.open_function_context('testTwo', 212, 12, False)
                # Assigning a type to the variable 'self' (line 213)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test1.testTwo.__dict__.__setitem__('stypy_localization', localization)
                Test1.testTwo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test1.testTwo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test1.testTwo.__dict__.__setitem__('stypy_function_name', 'Test1.testTwo')
                Test1.testTwo.__dict__.__setitem__('stypy_param_names_list', [])
                Test1.testTwo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test1.testTwo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test1.testTwo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test1.testTwo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test1.testTwo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test1.testTwo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test1.testTwo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testTwo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testTwo(...)' code ##################

                
                # Call to append(...): (line 213)
                # Processing the call arguments (line 213)
                str_206796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 31), 'str', 'Test1.testTwo')
                # Processing the call keyword arguments (line 213)
                kwargs_206797 = {}
                # Getting the type of 'results' (line 213)
                results_206794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 213)
                append_206795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), results_206794, 'append')
                # Calling append(args, kwargs) (line 213)
                append_call_result_206798 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), append_206795, *[str_206796], **kwargs_206797)
                
                
                # ################# End of 'testTwo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testTwo' in the type store
                # Getting the type of 'stypy_return_type' (line 212)
                stypy_return_type_206799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206799)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testTwo'
                return stypy_return_type_206799

        
        # Assigning a type to the variable 'Test1' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'Test1', Test1)
        # Declaration of the 'Test2' class
        # Getting the type of 'unittest' (line 215)
        unittest_206800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 215)
        TestCase_206801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 20), unittest_206800, 'TestCase')

        class Test2(TestCase_206801, ):

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 216, 12, False)
                # Assigning a type to the variable 'self' (line 217)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test2.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test2.setUpClass')
                Test2.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 218)
                # Processing the call arguments (line 218)
                str_206804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 31), 'str', 'setup 2')
                # Processing the call keyword arguments (line 218)
                kwargs_206805 = {}
                # Getting the type of 'results' (line 218)
                results_206802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 218)
                append_206803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 16), results_206802, 'append')
                # Calling append(args, kwargs) (line 218)
                append_call_result_206806 = invoke(stypy.reporting.localization.Localization(__file__, 218, 16), append_206803, *[str_206804], **kwargs_206805)
                
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 216)
                stypy_return_type_206807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206807)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_206807


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 219, 12, False)
                # Assigning a type to the variable 'self' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test2.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test2.tearDownClass')
                Test2.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 221)
                # Processing the call arguments (line 221)
                str_206810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 31), 'str', 'teardown 2')
                # Processing the call keyword arguments (line 221)
                kwargs_206811 = {}
                # Getting the type of 'results' (line 221)
                results_206808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 221)
                append_206809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), results_206808, 'append')
                # Calling append(args, kwargs) (line 221)
                append_call_result_206812 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), append_206809, *[str_206810], **kwargs_206811)
                
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 219)
                stypy_return_type_206813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206813)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206813


            @norecursion
            def testOne(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testOne'
                module_type_store = module_type_store.open_function_context('testOne', 222, 12, False)
                # Assigning a type to the variable 'self' (line 223)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.testOne.__dict__.__setitem__('stypy_localization', localization)
                Test2.testOne.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.testOne.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.testOne.__dict__.__setitem__('stypy_function_name', 'Test2.testOne')
                Test2.testOne.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.testOne.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.testOne.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.testOne.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.testOne.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.testOne.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.testOne.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.testOne', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testOne', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testOne(...)' code ##################

                
                # Call to append(...): (line 223)
                # Processing the call arguments (line 223)
                str_206816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 31), 'str', 'Test2.testOne')
                # Processing the call keyword arguments (line 223)
                kwargs_206817 = {}
                # Getting the type of 'results' (line 223)
                results_206814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 223)
                append_206815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 16), results_206814, 'append')
                # Calling append(args, kwargs) (line 223)
                append_call_result_206818 = invoke(stypy.reporting.localization.Localization(__file__, 223, 16), append_206815, *[str_206816], **kwargs_206817)
                
                
                # ################# End of 'testOne(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testOne' in the type store
                # Getting the type of 'stypy_return_type' (line 222)
                stypy_return_type_206819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206819)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testOne'
                return stypy_return_type_206819


            @norecursion
            def testTwo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testTwo'
                module_type_store = module_type_store.open_function_context('testTwo', 224, 12, False)
                # Assigning a type to the variable 'self' (line 225)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.testTwo.__dict__.__setitem__('stypy_localization', localization)
                Test2.testTwo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.testTwo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.testTwo.__dict__.__setitem__('stypy_function_name', 'Test2.testTwo')
                Test2.testTwo.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.testTwo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.testTwo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.testTwo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.testTwo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.testTwo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.testTwo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.testTwo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testTwo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testTwo(...)' code ##################

                
                # Call to append(...): (line 225)
                # Processing the call arguments (line 225)
                str_206822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 31), 'str', 'Test2.testTwo')
                # Processing the call keyword arguments (line 225)
                kwargs_206823 = {}
                # Getting the type of 'results' (line 225)
                results_206820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 225)
                append_206821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), results_206820, 'append')
                # Calling append(args, kwargs) (line 225)
                append_call_result_206824 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), append_206821, *[str_206822], **kwargs_206823)
                
                
                # ################# End of 'testTwo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testTwo' in the type store
                # Getting the type of 'stypy_return_type' (line 224)
                stypy_return_type_206825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206825)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testTwo'
                return stypy_return_type_206825

        
        # Assigning a type to the variable 'Test2' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'Test2', Test2)
        # Declaration of the 'Test3' class
        # Getting the type of 'unittest' (line 227)
        unittest_206826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 227)
        TestCase_206827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 20), unittest_206826, 'TestCase')

        class Test3(TestCase_206827, ):

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 228, 12, False)
                # Assigning a type to the variable 'self' (line 229)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test3.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test3.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test3.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test3.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test3.setUpClass')
                Test3.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test3.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test3.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test3.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test3.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test3.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test3.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test3.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 230)
                # Processing the call arguments (line 230)
                str_206830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 31), 'str', 'setup 3')
                # Processing the call keyword arguments (line 230)
                kwargs_206831 = {}
                # Getting the type of 'results' (line 230)
                results_206828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 230)
                append_206829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 16), results_206828, 'append')
                # Calling append(args, kwargs) (line 230)
                append_call_result_206832 = invoke(stypy.reporting.localization.Localization(__file__, 230, 16), append_206829, *[str_206830], **kwargs_206831)
                
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 228)
                stypy_return_type_206833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206833)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_206833


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 231, 12, False)
                # Assigning a type to the variable 'self' (line 232)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test3.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test3.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test3.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test3.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test3.tearDownClass')
                Test3.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test3.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test3.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test3.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test3.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test3.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test3.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test3.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 233)
                # Processing the call arguments (line 233)
                str_206836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 31), 'str', 'teardown 3')
                # Processing the call keyword arguments (line 233)
                kwargs_206837 = {}
                # Getting the type of 'results' (line 233)
                results_206834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 233)
                append_206835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), results_206834, 'append')
                # Calling append(args, kwargs) (line 233)
                append_call_result_206838 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), append_206835, *[str_206836], **kwargs_206837)
                
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 231)
                stypy_return_type_206839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206839)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_206839


            @norecursion
            def testOne(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testOne'
                module_type_store = module_type_store.open_function_context('testOne', 234, 12, False)
                # Assigning a type to the variable 'self' (line 235)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test3.testOne.__dict__.__setitem__('stypy_localization', localization)
                Test3.testOne.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test3.testOne.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test3.testOne.__dict__.__setitem__('stypy_function_name', 'Test3.testOne')
                Test3.testOne.__dict__.__setitem__('stypy_param_names_list', [])
                Test3.testOne.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test3.testOne.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test3.testOne.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test3.testOne.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test3.testOne.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test3.testOne.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test3.testOne', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testOne', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testOne(...)' code ##################

                
                # Call to append(...): (line 235)
                # Processing the call arguments (line 235)
                str_206842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 31), 'str', 'Test3.testOne')
                # Processing the call keyword arguments (line 235)
                kwargs_206843 = {}
                # Getting the type of 'results' (line 235)
                results_206840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 235)
                append_206841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), results_206840, 'append')
                # Calling append(args, kwargs) (line 235)
                append_call_result_206844 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), append_206841, *[str_206842], **kwargs_206843)
                
                
                # ################# End of 'testOne(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testOne' in the type store
                # Getting the type of 'stypy_return_type' (line 234)
                stypy_return_type_206845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206845)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testOne'
                return stypy_return_type_206845


            @norecursion
            def testTwo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testTwo'
                module_type_store = module_type_store.open_function_context('testTwo', 236, 12, False)
                # Assigning a type to the variable 'self' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test3.testTwo.__dict__.__setitem__('stypy_localization', localization)
                Test3.testTwo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test3.testTwo.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test3.testTwo.__dict__.__setitem__('stypy_function_name', 'Test3.testTwo')
                Test3.testTwo.__dict__.__setitem__('stypy_param_names_list', [])
                Test3.testTwo.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test3.testTwo.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test3.testTwo.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test3.testTwo.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test3.testTwo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test3.testTwo.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test3.testTwo', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testTwo', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testTwo(...)' code ##################

                
                # Call to append(...): (line 237)
                # Processing the call arguments (line 237)
                str_206848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 31), 'str', 'Test3.testTwo')
                # Processing the call keyword arguments (line 237)
                kwargs_206849 = {}
                # Getting the type of 'results' (line 237)
                results_206846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'results', False)
                # Obtaining the member 'append' of a type (line 237)
                append_206847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), results_206846, 'append')
                # Calling append(args, kwargs) (line 237)
                append_call_result_206850 = invoke(stypy.reporting.localization.Localization(__file__, 237, 16), append_206847, *[str_206848], **kwargs_206849)
                
                
                # ################# End of 'testTwo(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testTwo' in the type store
                # Getting the type of 'stypy_return_type' (line 236)
                stypy_return_type_206851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206851)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testTwo'
                return stypy_return_type_206851

        
        # Assigning a type to the variable 'Test3' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'Test3', Test3)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Str to a Attribute (line 239):
        str_206852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 46), 'str', 'Module')
        # Getting the type of 'Test2' (line 239)
        Test2_206853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'Test2')
        # Setting the type of the member '__module__' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 27), Test2_206853, '__module__', str_206852)
        
        # Assigning a Attribute to a Attribute (line 239):
        # Getting the type of 'Test2' (line 239)
        Test2_206854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'Test2')
        # Obtaining the member '__module__' of a type (line 239)
        module___206855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 27), Test2_206854, '__module__')
        # Getting the type of 'Test1' (line 239)
        Test1_206856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'Test1')
        # Setting the type of the member '__module__' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), Test1_206856, '__module__', module___206855)
        
        # Assigning a Str to a Attribute (line 240):
        
        # Assigning a Str to a Attribute (line 240):
        str_206857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 27), 'str', 'Module2')
        # Getting the type of 'Test3' (line 240)
        Test3_206858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'Test3')
        # Setting the type of the member '__module__' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), Test3_206858, '__module__', str_206857)
        
        # Assigning a Name to a Subscript (line 241):
        
        # Assigning a Name to a Subscript (line 241):
        # Getting the type of 'Module1' (line 241)
        Module1_206859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 32), 'Module1')
        # Getting the type of 'sys' (line 241)
        sys_206860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 241)
        modules_206861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), sys_206860, 'modules')
        str_206862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 20), 'str', 'Module')
        # Storing an element on a container (line 241)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 8), modules_206861, (str_206862, Module1_206859))
        
        # Assigning a Name to a Subscript (line 242):
        
        # Assigning a Name to a Subscript (line 242):
        # Getting the type of 'Module2' (line 242)
        Module2_206863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 33), 'Module2')
        # Getting the type of 'sys' (line 242)
        sys_206864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 242)
        modules_206865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), sys_206864, 'modules')
        str_206866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 20), 'str', 'Module2')
        # Storing an element on a container (line 242)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 8), modules_206865, (str_206866, Module2_206863))
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Call to TestSuite(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Obtaining an instance of the builtin type 'tuple' (line 244)
        tuple_206869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 244)
        # Adding element type (line 244)
        
        # Call to Test1(...): (line 244)
        # Processing the call arguments (line 244)
        str_206871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 42), 'str', 'testOne')
        # Processing the call keyword arguments (line 244)
        kwargs_206872 = {}
        # Getting the type of 'Test1' (line 244)
        Test1_206870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 36), 'Test1', False)
        # Calling Test1(args, kwargs) (line 244)
        Test1_call_result_206873 = invoke(stypy.reporting.localization.Localization(__file__, 244, 36), Test1_206870, *[str_206871], **kwargs_206872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 36), tuple_206869, Test1_call_result_206873)
        
        # Processing the call keyword arguments (line 244)
        kwargs_206874 = {}
        # Getting the type of 'unittest' (line 244)
        unittest_206867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 244)
        TestSuite_206868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 16), unittest_206867, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 244)
        TestSuite_call_result_206875 = invoke(stypy.reporting.localization.Localization(__file__, 244, 16), TestSuite_206868, *[tuple_206869], **kwargs_206874)
        
        # Assigning a type to the variable 'first' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'first', TestSuite_call_result_206875)
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to TestSuite(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Obtaining an instance of the builtin type 'tuple' (line 245)
        tuple_206878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 245)
        # Adding element type (line 245)
        
        # Call to Test1(...): (line 245)
        # Processing the call arguments (line 245)
        str_206880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 43), 'str', 'testTwo')
        # Processing the call keyword arguments (line 245)
        kwargs_206881 = {}
        # Getting the type of 'Test1' (line 245)
        Test1_206879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 37), 'Test1', False)
        # Calling Test1(args, kwargs) (line 245)
        Test1_call_result_206882 = invoke(stypy.reporting.localization.Localization(__file__, 245, 37), Test1_206879, *[str_206880], **kwargs_206881)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 37), tuple_206878, Test1_call_result_206882)
        
        # Processing the call keyword arguments (line 245)
        kwargs_206883 = {}
        # Getting the type of 'unittest' (line 245)
        unittest_206876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 245)
        TestSuite_206877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 17), unittest_206876, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 245)
        TestSuite_call_result_206884 = invoke(stypy.reporting.localization.Localization(__file__, 245, 17), TestSuite_206877, *[tuple_206878], **kwargs_206883)
        
        # Assigning a type to the variable 'second' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'second', TestSuite_call_result_206884)
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to TestSuite(...): (line 246)
        # Processing the call arguments (line 246)
        
        # Obtaining an instance of the builtin type 'tuple' (line 246)
        tuple_206887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 246)
        # Adding element type (line 246)
        
        # Call to Test2(...): (line 246)
        # Processing the call arguments (line 246)
        str_206889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 42), 'str', 'testOne')
        # Processing the call keyword arguments (line 246)
        kwargs_206890 = {}
        # Getting the type of 'Test2' (line 246)
        Test2_206888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 36), 'Test2', False)
        # Calling Test2(args, kwargs) (line 246)
        Test2_call_result_206891 = invoke(stypy.reporting.localization.Localization(__file__, 246, 36), Test2_206888, *[str_206889], **kwargs_206890)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 36), tuple_206887, Test2_call_result_206891)
        
        # Processing the call keyword arguments (line 246)
        kwargs_206892 = {}
        # Getting the type of 'unittest' (line 246)
        unittest_206885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 246)
        TestSuite_206886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), unittest_206885, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 246)
        TestSuite_call_result_206893 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), TestSuite_206886, *[tuple_206887], **kwargs_206892)
        
        # Assigning a type to the variable 'third' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'third', TestSuite_call_result_206893)
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to TestSuite(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Obtaining an instance of the builtin type 'tuple' (line 247)
        tuple_206896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 247)
        # Adding element type (line 247)
        
        # Call to Test2(...): (line 247)
        # Processing the call arguments (line 247)
        str_206898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 43), 'str', 'testTwo')
        # Processing the call keyword arguments (line 247)
        kwargs_206899 = {}
        # Getting the type of 'Test2' (line 247)
        Test2_206897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 37), 'Test2', False)
        # Calling Test2(args, kwargs) (line 247)
        Test2_call_result_206900 = invoke(stypy.reporting.localization.Localization(__file__, 247, 37), Test2_206897, *[str_206898], **kwargs_206899)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 37), tuple_206896, Test2_call_result_206900)
        
        # Processing the call keyword arguments (line 247)
        kwargs_206901 = {}
        # Getting the type of 'unittest' (line 247)
        unittest_206894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 247)
        TestSuite_206895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 17), unittest_206894, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 247)
        TestSuite_call_result_206902 = invoke(stypy.reporting.localization.Localization(__file__, 247, 17), TestSuite_206895, *[tuple_206896], **kwargs_206901)
        
        # Assigning a type to the variable 'fourth' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'fourth', TestSuite_call_result_206902)
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to TestSuite(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Obtaining an instance of the builtin type 'tuple' (line 248)
        tuple_206905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 248)
        # Adding element type (line 248)
        
        # Call to Test3(...): (line 248)
        # Processing the call arguments (line 248)
        str_206907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 42), 'str', 'testOne')
        # Processing the call keyword arguments (line 248)
        kwargs_206908 = {}
        # Getting the type of 'Test3' (line 248)
        Test3_206906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 36), 'Test3', False)
        # Calling Test3(args, kwargs) (line 248)
        Test3_call_result_206909 = invoke(stypy.reporting.localization.Localization(__file__, 248, 36), Test3_206906, *[str_206907], **kwargs_206908)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 36), tuple_206905, Test3_call_result_206909)
        
        # Processing the call keyword arguments (line 248)
        kwargs_206910 = {}
        # Getting the type of 'unittest' (line 248)
        unittest_206903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 248)
        TestSuite_206904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 16), unittest_206903, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 248)
        TestSuite_call_result_206911 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), TestSuite_206904, *[tuple_206905], **kwargs_206910)
        
        # Assigning a type to the variable 'fifth' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'fifth', TestSuite_call_result_206911)
        
        # Assigning a Call to a Name (line 249):
        
        # Assigning a Call to a Name (line 249):
        
        # Call to TestSuite(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Obtaining an instance of the builtin type 'tuple' (line 249)
        tuple_206914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 249)
        # Adding element type (line 249)
        
        # Call to Test3(...): (line 249)
        # Processing the call arguments (line 249)
        str_206916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 42), 'str', 'testTwo')
        # Processing the call keyword arguments (line 249)
        kwargs_206917 = {}
        # Getting the type of 'Test3' (line 249)
        Test3_206915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 36), 'Test3', False)
        # Calling Test3(args, kwargs) (line 249)
        Test3_call_result_206918 = invoke(stypy.reporting.localization.Localization(__file__, 249, 36), Test3_206915, *[str_206916], **kwargs_206917)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 36), tuple_206914, Test3_call_result_206918)
        
        # Processing the call keyword arguments (line 249)
        kwargs_206919 = {}
        # Getting the type of 'unittest' (line 249)
        unittest_206912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 249)
        TestSuite_206913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 16), unittest_206912, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 249)
        TestSuite_call_result_206920 = invoke(stypy.reporting.localization.Localization(__file__, 249, 16), TestSuite_206913, *[tuple_206914], **kwargs_206919)
        
        # Assigning a type to the variable 'sixth' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'sixth', TestSuite_call_result_206920)
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to TestSuite(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Obtaining an instance of the builtin type 'tuple' (line 250)
        tuple_206923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 250)
        # Adding element type (line 250)
        # Getting the type of 'first' (line 250)
        first_206924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 36), 'first', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 36), tuple_206923, first_206924)
        # Adding element type (line 250)
        # Getting the type of 'second' (line 250)
        second_206925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 43), 'second', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 36), tuple_206923, second_206925)
        # Adding element type (line 250)
        # Getting the type of 'third' (line 250)
        third_206926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 51), 'third', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 36), tuple_206923, third_206926)
        # Adding element type (line 250)
        # Getting the type of 'fourth' (line 250)
        fourth_206927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 58), 'fourth', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 36), tuple_206923, fourth_206927)
        # Adding element type (line 250)
        # Getting the type of 'fifth' (line 250)
        fifth_206928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 66), 'fifth', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 36), tuple_206923, fifth_206928)
        # Adding element type (line 250)
        # Getting the type of 'sixth' (line 250)
        sixth_206929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 73), 'sixth', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 36), tuple_206923, sixth_206929)
        
        # Processing the call keyword arguments (line 250)
        kwargs_206930 = {}
        # Getting the type of 'unittest' (line 250)
        unittest_206921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 250)
        TestSuite_206922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), unittest_206921, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 250)
        TestSuite_call_result_206931 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), TestSuite_206922, *[tuple_206923], **kwargs_206930)
        
        # Assigning a type to the variable 'suite' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'suite', TestSuite_call_result_206931)
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to getRunner(...): (line 252)
        # Processing the call keyword arguments (line 252)
        kwargs_206934 = {}
        # Getting the type of 'self' (line 252)
        self_206932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 17), 'self', False)
        # Obtaining the member 'getRunner' of a type (line 252)
        getRunner_206933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 17), self_206932, 'getRunner')
        # Calling getRunner(args, kwargs) (line 252)
        getRunner_call_result_206935 = invoke(stypy.reporting.localization.Localization(__file__, 252, 17), getRunner_206933, *[], **kwargs_206934)
        
        # Assigning a type to the variable 'runner' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'runner', getRunner_call_result_206935)
        
        # Assigning a Call to a Name (line 253):
        
        # Assigning a Call to a Name (line 253):
        
        # Call to run(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'suite' (line 253)
        suite_206938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'suite', False)
        # Processing the call keyword arguments (line 253)
        kwargs_206939 = {}
        # Getting the type of 'runner' (line 253)
        runner_206936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 17), 'runner', False)
        # Obtaining the member 'run' of a type (line 253)
        run_206937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 17), runner_206936, 'run')
        # Calling run(args, kwargs) (line 253)
        run_call_result_206940 = invoke(stypy.reporting.localization.Localization(__file__, 253, 17), run_206937, *[suite_206938], **kwargs_206939)
        
        # Assigning a type to the variable 'result' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'result', run_call_result_206940)
        
        # Call to assertEqual(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'result' (line 254)
        result_206943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 254)
        testsRun_206944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 25), result_206943, 'testsRun')
        int_206945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 42), 'int')
        # Processing the call keyword arguments (line 254)
        kwargs_206946 = {}
        # Getting the type of 'self' (line 254)
        self_206941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 254)
        assertEqual_206942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), self_206941, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 254)
        assertEqual_call_result_206947 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), assertEqual_206942, *[testsRun_206944, int_206945], **kwargs_206946)
        
        
        # Call to assertEqual(...): (line 255)
        # Processing the call arguments (line 255)
        
        # Call to len(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'result' (line 255)
        result_206951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 255)
        errors_206952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 29), result_206951, 'errors')
        # Processing the call keyword arguments (line 255)
        kwargs_206953 = {}
        # Getting the type of 'len' (line 255)
        len_206950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 25), 'len', False)
        # Calling len(args, kwargs) (line 255)
        len_call_result_206954 = invoke(stypy.reporting.localization.Localization(__file__, 255, 25), len_206950, *[errors_206952], **kwargs_206953)
        
        int_206955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 45), 'int')
        # Processing the call keyword arguments (line 255)
        kwargs_206956 = {}
        # Getting the type of 'self' (line 255)
        self_206948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 255)
        assertEqual_206949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_206948, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 255)
        assertEqual_call_result_206957 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assertEqual_206949, *[len_call_result_206954, int_206955], **kwargs_206956)
        
        
        # Call to assertEqual(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'results' (line 257)
        results_206960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'results', False)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_206961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        str_206962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 26), 'str', 'Module1.setUpModule')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206962)
        # Adding element type (line 258)
        str_206963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 49), 'str', 'setup 1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206963)
        # Adding element type (line 258)
        str_206964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 26), 'str', 'Test1.testOne')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206964)
        # Adding element type (line 258)
        str_206965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'str', 'Test1.testTwo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206965)
        # Adding element type (line 258)
        str_206966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 60), 'str', 'teardown 1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206966)
        # Adding element type (line 258)
        str_206967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 26), 'str', 'setup 2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206967)
        # Adding element type (line 258)
        str_206968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 37), 'str', 'Test2.testOne')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206968)
        # Adding element type (line 258)
        str_206969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 54), 'str', 'Test2.testTwo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206969)
        # Adding element type (line 258)
        str_206970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 26), 'str', 'teardown 2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206970)
        # Adding element type (line 258)
        str_206971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 40), 'str', 'Module1.tearDownModule')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206971)
        # Adding element type (line 258)
        str_206972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 26), 'str', 'Module2.setUpModule')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206972)
        # Adding element type (line 258)
        str_206973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 49), 'str', 'setup 3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206973)
        # Adding element type (line 258)
        str_206974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 26), 'str', 'Test3.testOne')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206974)
        # Adding element type (line 258)
        str_206975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 43), 'str', 'Test3.testTwo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206975)
        # Adding element type (line 258)
        str_206976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 26), 'str', 'teardown 3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206976)
        # Adding element type (line 258)
        str_206977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 40), 'str', 'Module2.tearDownModule')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 25), list_206961, str_206977)
        
        # Processing the call keyword arguments (line 257)
        kwargs_206978 = {}
        # Getting the type of 'self' (line 257)
        self_206958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 257)
        assertEqual_206959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_206958, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 257)
        assertEqual_call_result_206979 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), assertEqual_206959, *[results_206960, list_206961], **kwargs_206978)
        
        
        # ################# End of 'test_setup_teardown_order_with_pathological_suite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_setup_teardown_order_with_pathological_suite' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_206980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206980)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_setup_teardown_order_with_pathological_suite'
        return stypy_return_type_206980


    @norecursion
    def test_setup_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_setup_module'
        module_type_store = module_type_store.open_function_context('test_setup_module', 266, 4, False)
        # Assigning a type to the variable 'self' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_setup_module')
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_setup_module.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_setup_module', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_setup_module', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_setup_module(...)' code ##################

        # Declaration of the 'Module' class

        class Module(object, ):
            
            # Assigning a Num to a Name (line 268):
            
            # Assigning a Num to a Name (line 268):
            int_206981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 26), 'int')
            # Assigning a type to the variable 'moduleSetup' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'moduleSetup', int_206981)

            @staticmethod
            @norecursion
            def setUpModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpModule'
                module_type_store = module_type_store.open_function_context('setUpModule', 269, 12, False)
                
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

                
                # Getting the type of 'Module' (line 271)
                Module_206982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'Module')
                # Obtaining the member 'moduleSetup' of a type (line 271)
                moduleSetup_206983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), Module_206982, 'moduleSetup')
                int_206984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 38), 'int')
                # Applying the binary operator '+=' (line 271)
                result_iadd_206985 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 16), '+=', moduleSetup_206983, int_206984)
                # Getting the type of 'Module' (line 271)
                Module_206986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'Module')
                # Setting the type of the member 'moduleSetup' of a type (line 271)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), Module_206986, 'moduleSetup', result_iadd_206985)
                
                
                # ################# End of 'setUpModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpModule' in the type store
                # Getting the type of 'stypy_return_type' (line 269)
                stypy_return_type_206987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206987)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpModule'
                return stypy_return_type_206987

        
        # Assigning a type to the variable 'Module' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'Module', Module)
        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 273)
        unittest_206988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 273)
        TestCase_206989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 19), unittest_206988, 'TestCase')

        class Test(TestCase_206989, ):

            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 274, 12, False)
                # Assigning a type to the variable 'self' (line 275)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 274)
                stypy_return_type_206990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206990)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_206990


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 276, 12, False)
                # Assigning a type to the variable 'self' (line 277)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 276)
                stypy_return_type_206991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206991)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_206991

        
        # Assigning a type to the variable 'Test' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'Test', Test)
        
        # Assigning a Str to a Attribute (line 278):
        
        # Assigning a Str to a Attribute (line 278):
        str_206992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 26), 'str', 'Module')
        # Getting the type of 'Test' (line 278)
        Test_206993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'Test')
        # Setting the type of the member '__module__' of a type (line 278)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), Test_206993, '__module__', str_206992)
        
        # Assigning a Name to a Subscript (line 279):
        
        # Assigning a Name to a Subscript (line 279):
        # Getting the type of 'Module' (line 279)
        Module_206994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 32), 'Module')
        # Getting the type of 'sys' (line 279)
        sys_206995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 279)
        modules_206996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), sys_206995, 'modules')
        str_206997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 20), 'str', 'Module')
        # Storing an element on a container (line 279)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 8), modules_206996, (str_206997, Module_206994))
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to runTests(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'Test' (line 281)
        Test_207000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 31), 'Test', False)
        # Processing the call keyword arguments (line 281)
        kwargs_207001 = {}
        # Getting the type of 'self' (line 281)
        self_206998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 281)
        runTests_206999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 17), self_206998, 'runTests')
        # Calling runTests(args, kwargs) (line 281)
        runTests_call_result_207002 = invoke(stypy.reporting.localization.Localization(__file__, 281, 17), runTests_206999, *[Test_207000], **kwargs_207001)
        
        # Assigning a type to the variable 'result' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'result', runTests_call_result_207002)
        
        # Call to assertEqual(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'Module' (line 282)
        Module_207005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 25), 'Module', False)
        # Obtaining the member 'moduleSetup' of a type (line 282)
        moduleSetup_207006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 25), Module_207005, 'moduleSetup')
        int_207007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 45), 'int')
        # Processing the call keyword arguments (line 282)
        kwargs_207008 = {}
        # Getting the type of 'self' (line 282)
        self_207003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 282)
        assertEqual_207004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), self_207003, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 282)
        assertEqual_call_result_207009 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), assertEqual_207004, *[moduleSetup_207006, int_207007], **kwargs_207008)
        
        
        # Call to assertEqual(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'result' (line 283)
        result_207012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 283)
        testsRun_207013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 25), result_207012, 'testsRun')
        int_207014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 42), 'int')
        # Processing the call keyword arguments (line 283)
        kwargs_207015 = {}
        # Getting the type of 'self' (line 283)
        self_207010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 283)
        assertEqual_207011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_207010, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 283)
        assertEqual_call_result_207016 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assertEqual_207011, *[testsRun_207013, int_207014], **kwargs_207015)
        
        
        # Call to assertEqual(...): (line 284)
        # Processing the call arguments (line 284)
        
        # Call to len(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'result' (line 284)
        result_207020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 284)
        errors_207021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 29), result_207020, 'errors')
        # Processing the call keyword arguments (line 284)
        kwargs_207022 = {}
        # Getting the type of 'len' (line 284)
        len_207019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'len', False)
        # Calling len(args, kwargs) (line 284)
        len_call_result_207023 = invoke(stypy.reporting.localization.Localization(__file__, 284, 25), len_207019, *[errors_207021], **kwargs_207022)
        
        int_207024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 45), 'int')
        # Processing the call keyword arguments (line 284)
        kwargs_207025 = {}
        # Getting the type of 'self' (line 284)
        self_207017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 284)
        assertEqual_207018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_207017, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 284)
        assertEqual_call_result_207026 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), assertEqual_207018, *[len_call_result_207023, int_207024], **kwargs_207025)
        
        
        # ################# End of 'test_setup_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_setup_module' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_207027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_setup_module'
        return stypy_return_type_207027


    @norecursion
    def test_error_in_setup_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_error_in_setup_module'
        module_type_store = module_type_store.open_function_context('test_error_in_setup_module', 286, 4, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_error_in_setup_module')
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_error_in_setup_module.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_error_in_setup_module', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_error_in_setup_module', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_error_in_setup_module(...)' code ##################

        # Declaration of the 'Module' class

        class Module(object, ):
            
            # Assigning a Num to a Name (line 288):
            
            # Assigning a Num to a Name (line 288):
            int_207028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 26), 'int')
            # Assigning a type to the variable 'moduleSetup' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'moduleSetup', int_207028)
            
            # Assigning a Num to a Name (line 289):
            
            # Assigning a Num to a Name (line 289):
            int_207029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 29), 'int')
            # Assigning a type to the variable 'moduleTornDown' (line 289)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'moduleTornDown', int_207029)

            @staticmethod
            @norecursion
            def setUpModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpModule'
                module_type_store = module_type_store.open_function_context('setUpModule', 290, 12, False)
                
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

                
                # Getting the type of 'Module' (line 292)
                Module_207030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'Module')
                # Obtaining the member 'moduleSetup' of a type (line 292)
                moduleSetup_207031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 16), Module_207030, 'moduleSetup')
                int_207032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 38), 'int')
                # Applying the binary operator '+=' (line 292)
                result_iadd_207033 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 16), '+=', moduleSetup_207031, int_207032)
                # Getting the type of 'Module' (line 292)
                Module_207034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'Module')
                # Setting the type of the member 'moduleSetup' of a type (line 292)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 16), Module_207034, 'moduleSetup', result_iadd_207033)
                
                
                # Call to TypeError(...): (line 293)
                # Processing the call arguments (line 293)
                str_207036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 32), 'str', 'foo')
                # Processing the call keyword arguments (line 293)
                kwargs_207037 = {}
                # Getting the type of 'TypeError' (line 293)
                TypeError_207035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 293)
                TypeError_call_result_207038 = invoke(stypy.reporting.localization.Localization(__file__, 293, 22), TypeError_207035, *[str_207036], **kwargs_207037)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 293, 16), TypeError_call_result_207038, 'raise parameter', BaseException)
                
                # ################# End of 'setUpModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpModule' in the type store
                # Getting the type of 'stypy_return_type' (line 290)
                stypy_return_type_207039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207039)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpModule'
                return stypy_return_type_207039


            @staticmethod
            @norecursion
            def tearDownModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownModule'
                module_type_store = module_type_store.open_function_context('tearDownModule', 294, 12, False)
                
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

                
                # Getting the type of 'Module' (line 296)
                Module_207040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'Module')
                # Obtaining the member 'moduleTornDown' of a type (line 296)
                moduleTornDown_207041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 16), Module_207040, 'moduleTornDown')
                int_207042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 41), 'int')
                # Applying the binary operator '+=' (line 296)
                result_iadd_207043 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 16), '+=', moduleTornDown_207041, int_207042)
                # Getting the type of 'Module' (line 296)
                Module_207044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'Module')
                # Setting the type of the member 'moduleTornDown' of a type (line 296)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 16), Module_207044, 'moduleTornDown', result_iadd_207043)
                
                
                # ################# End of 'tearDownModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownModule' in the type store
                # Getting the type of 'stypy_return_type' (line 294)
                stypy_return_type_207045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207045)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownModule'
                return stypy_return_type_207045

        
        # Assigning a type to the variable 'Module' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'Module', Module)
        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 298)
        unittest_207046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 298)
        TestCase_207047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 19), unittest_207046, 'TestCase')

        class Test(TestCase_207047, ):
            
            # Assigning a Name to a Name (line 299):
            
            # Assigning a Name to a Name (line 299):
            # Getting the type of 'False' (line 299)
            False_207048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 25), 'False')
            # Assigning a type to the variable 'classSetUp' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'classSetUp', False_207048)
            
            # Assigning a Name to a Name (line 300):
            
            # Assigning a Name to a Name (line 300):
            # Getting the type of 'False' (line 300)
            False_207049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 28), 'False')
            # Assigning a type to the variable 'classTornDown' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'classTornDown', False_207049)

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 301, 12, False)
                # Assigning a type to the variable 'self' (line 302)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test.setUpClass')
                Test.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Assigning a Name to a Attribute (line 303):
                
                # Assigning a Name to a Attribute (line 303):
                # Getting the type of 'True' (line 303)
                True_207050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 34), 'True')
                # Getting the type of 'Test' (line 303)
                Test_207051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'Test')
                # Setting the type of the member 'classSetUp' of a type (line 303)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), Test_207051, 'classSetUp', True_207050)
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 301)
                stypy_return_type_207052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207052)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_207052


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 304, 12, False)
                # Assigning a type to the variable 'self' (line 305)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test.tearDownClass')
                Test.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Assigning a Name to a Attribute (line 306):
                
                # Assigning a Name to a Attribute (line 306):
                # Getting the type of 'True' (line 306)
                True_207053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 37), 'True')
                # Getting the type of 'Test' (line 306)
                Test_207054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'Test')
                # Setting the type of the member 'classTornDown' of a type (line 306)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 16), Test_207054, 'classTornDown', True_207053)
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 304)
                stypy_return_type_207055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207055)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_207055


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 307, 12, False)
                # Assigning a type to the variable 'self' (line 308)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 307)
                stypy_return_type_207056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207056)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_207056


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 309, 12, False)
                # Assigning a type to the variable 'self' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 309)
                stypy_return_type_207057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207057)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_207057

        
        # Assigning a type to the variable 'Test' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'Test', Test)
        # Declaration of the 'Test2' class
        # Getting the type of 'unittest' (line 312)
        unittest_207058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 312)
        TestCase_207059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 20), unittest_207058, 'TestCase')

        class Test2(TestCase_207059, ):

            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 313, 12, False)
                # Assigning a type to the variable 'self' (line 314)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test2.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.test_one.__dict__.__setitem__('stypy_function_name', 'Test2.test_one')
                Test2.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 313)
                stypy_return_type_207060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207060)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_207060


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 315, 12, False)
                # Assigning a type to the variable 'self' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test2.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.test_two.__dict__.__setitem__('stypy_function_name', 'Test2.test_two')
                Test2.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 315)
                stypy_return_type_207061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207061)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_207061

        
        # Assigning a type to the variable 'Test2' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'Test2', Test2)
        
        # Assigning a Str to a Attribute (line 317):
        
        # Assigning a Str to a Attribute (line 317):
        str_207062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 26), 'str', 'Module')
        # Getting the type of 'Test' (line 317)
        Test_207063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'Test')
        # Setting the type of the member '__module__' of a type (line 317)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), Test_207063, '__module__', str_207062)
        
        # Assigning a Str to a Attribute (line 318):
        
        # Assigning a Str to a Attribute (line 318):
        str_207064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 27), 'str', 'Module')
        # Getting the type of 'Test2' (line 318)
        Test2_207065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'Test2')
        # Setting the type of the member '__module__' of a type (line 318)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), Test2_207065, '__module__', str_207064)
        
        # Assigning a Name to a Subscript (line 319):
        
        # Assigning a Name to a Subscript (line 319):
        # Getting the type of 'Module' (line 319)
        Module_207066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 32), 'Module')
        # Getting the type of 'sys' (line 319)
        sys_207067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 319)
        modules_207068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), sys_207067, 'modules')
        str_207069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 20), 'str', 'Module')
        # Storing an element on a container (line 319)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 8), modules_207068, (str_207069, Module_207066))
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to runTests(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'Test' (line 321)
        Test_207072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 31), 'Test', False)
        # Getting the type of 'Test2' (line 321)
        Test2_207073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 37), 'Test2', False)
        # Processing the call keyword arguments (line 321)
        kwargs_207074 = {}
        # Getting the type of 'self' (line 321)
        self_207070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 321)
        runTests_207071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 17), self_207070, 'runTests')
        # Calling runTests(args, kwargs) (line 321)
        runTests_call_result_207075 = invoke(stypy.reporting.localization.Localization(__file__, 321, 17), runTests_207071, *[Test_207072, Test2_207073], **kwargs_207074)
        
        # Assigning a type to the variable 'result' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'result', runTests_call_result_207075)
        
        # Call to assertEqual(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'Module' (line 322)
        Module_207078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'Module', False)
        # Obtaining the member 'moduleSetup' of a type (line 322)
        moduleSetup_207079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 25), Module_207078, 'moduleSetup')
        int_207080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 45), 'int')
        # Processing the call keyword arguments (line 322)
        kwargs_207081 = {}
        # Getting the type of 'self' (line 322)
        self_207076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 322)
        assertEqual_207077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), self_207076, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 322)
        assertEqual_call_result_207082 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), assertEqual_207077, *[moduleSetup_207079, int_207080], **kwargs_207081)
        
        
        # Call to assertEqual(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'Module' (line 323)
        Module_207085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 25), 'Module', False)
        # Obtaining the member 'moduleTornDown' of a type (line 323)
        moduleTornDown_207086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 25), Module_207085, 'moduleTornDown')
        int_207087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 48), 'int')
        # Processing the call keyword arguments (line 323)
        kwargs_207088 = {}
        # Getting the type of 'self' (line 323)
        self_207083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 323)
        assertEqual_207084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_207083, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 323)
        assertEqual_call_result_207089 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), assertEqual_207084, *[moduleTornDown_207086, int_207087], **kwargs_207088)
        
        
        # Call to assertEqual(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'result' (line 324)
        result_207092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 324)
        testsRun_207093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 25), result_207092, 'testsRun')
        int_207094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 42), 'int')
        # Processing the call keyword arguments (line 324)
        kwargs_207095 = {}
        # Getting the type of 'self' (line 324)
        self_207090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 324)
        assertEqual_207091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), self_207090, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 324)
        assertEqual_call_result_207096 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), assertEqual_207091, *[testsRun_207093, int_207094], **kwargs_207095)
        
        
        # Call to assertFalse(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'Test' (line 325)
        Test_207099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 25), 'Test', False)
        # Obtaining the member 'classSetUp' of a type (line 325)
        classSetUp_207100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 25), Test_207099, 'classSetUp')
        # Processing the call keyword arguments (line 325)
        kwargs_207101 = {}
        # Getting the type of 'self' (line 325)
        self_207097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 325)
        assertFalse_207098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), self_207097, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 325)
        assertFalse_call_result_207102 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), assertFalse_207098, *[classSetUp_207100], **kwargs_207101)
        
        
        # Call to assertFalse(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'Test' (line 326)
        Test_207105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 25), 'Test', False)
        # Obtaining the member 'classTornDown' of a type (line 326)
        classTornDown_207106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 25), Test_207105, 'classTornDown')
        # Processing the call keyword arguments (line 326)
        kwargs_207107 = {}
        # Getting the type of 'self' (line 326)
        self_207103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 326)
        assertFalse_207104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_207103, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 326)
        assertFalse_call_result_207108 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), assertFalse_207104, *[classTornDown_207106], **kwargs_207107)
        
        
        # Call to assertEqual(...): (line 327)
        # Processing the call arguments (line 327)
        
        # Call to len(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'result' (line 327)
        result_207112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 327)
        errors_207113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 29), result_207112, 'errors')
        # Processing the call keyword arguments (line 327)
        kwargs_207114 = {}
        # Getting the type of 'len' (line 327)
        len_207111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 25), 'len', False)
        # Calling len(args, kwargs) (line 327)
        len_call_result_207115 = invoke(stypy.reporting.localization.Localization(__file__, 327, 25), len_207111, *[errors_207113], **kwargs_207114)
        
        int_207116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 45), 'int')
        # Processing the call keyword arguments (line 327)
        kwargs_207117 = {}
        # Getting the type of 'self' (line 327)
        self_207109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 327)
        assertEqual_207110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), self_207109, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 327)
        assertEqual_call_result_207118 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), assertEqual_207110, *[len_call_result_207115, int_207116], **kwargs_207117)
        
        
        # Assigning a Subscript to a Tuple (line 328):
        
        # Assigning a Subscript to a Name (line 328):
        
        # Obtaining the type of the subscript
        int_207119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 8), 'int')
        
        # Obtaining the type of the subscript
        int_207120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 33), 'int')
        # Getting the type of 'result' (line 328)
        result_207121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'result')
        # Obtaining the member 'errors' of a type (line 328)
        errors_207122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 19), result_207121, 'errors')
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___207123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 19), errors_207122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_207124 = invoke(stypy.reporting.localization.Localization(__file__, 328, 19), getitem___207123, int_207120)
        
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___207125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), subscript_call_result_207124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_207126 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), getitem___207125, int_207119)
        
        # Assigning a type to the variable 'tuple_var_assignment_206293' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_206293', subscript_call_result_207126)
        
        # Assigning a Subscript to a Name (line 328):
        
        # Obtaining the type of the subscript
        int_207127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 8), 'int')
        
        # Obtaining the type of the subscript
        int_207128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 33), 'int')
        # Getting the type of 'result' (line 328)
        result_207129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'result')
        # Obtaining the member 'errors' of a type (line 328)
        errors_207130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 19), result_207129, 'errors')
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___207131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 19), errors_207130, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_207132 = invoke(stypy.reporting.localization.Localization(__file__, 328, 19), getitem___207131, int_207128)
        
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___207133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), subscript_call_result_207132, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_207134 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), getitem___207133, int_207127)
        
        # Assigning a type to the variable 'tuple_var_assignment_206294' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_206294', subscript_call_result_207134)
        
        # Assigning a Name to a Name (line 328):
        # Getting the type of 'tuple_var_assignment_206293' (line 328)
        tuple_var_assignment_206293_207135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_206293')
        # Assigning a type to the variable 'error' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'error', tuple_var_assignment_206293_207135)
        
        # Assigning a Name to a Name (line 328):
        # Getting the type of 'tuple_var_assignment_206294' (line 328)
        tuple_var_assignment_206294_207136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_206294')
        # Assigning a type to the variable '_' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), '_', tuple_var_assignment_206294_207136)
        
        # Call to assertEqual(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Call to str(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'error' (line 329)
        error_207140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 29), 'error', False)
        # Processing the call keyword arguments (line 329)
        kwargs_207141 = {}
        # Getting the type of 'str' (line 329)
        str_207139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 25), 'str', False)
        # Calling str(args, kwargs) (line 329)
        str_call_result_207142 = invoke(stypy.reporting.localization.Localization(__file__, 329, 25), str_207139, *[error_207140], **kwargs_207141)
        
        str_207143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 37), 'str', 'setUpModule (Module)')
        # Processing the call keyword arguments (line 329)
        kwargs_207144 = {}
        # Getting the type of 'self' (line 329)
        self_207137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 329)
        assertEqual_207138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), self_207137, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 329)
        assertEqual_call_result_207145 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), assertEqual_207138, *[str_call_result_207142, str_207143], **kwargs_207144)
        
        
        # ################# End of 'test_error_in_setup_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_error_in_setup_module' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_207146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207146)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_error_in_setup_module'
        return stypy_return_type_207146


    @norecursion
    def test_testcase_with_missing_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_testcase_with_missing_module'
        module_type_store = module_type_store.open_function_context('test_testcase_with_missing_module', 331, 4, False)
        # Assigning a type to the variable 'self' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_testcase_with_missing_module')
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_testcase_with_missing_module.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_testcase_with_missing_module', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_testcase_with_missing_module', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_testcase_with_missing_module(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 332)
        unittest_207147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 332)
        TestCase_207148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), unittest_207147, 'TestCase')

        class Test(TestCase_207148, ):

            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 333, 12, False)
                # Assigning a type to the variable 'self' (line 334)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 333)
                stypy_return_type_207149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207149)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_207149


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 335, 12, False)
                # Assigning a type to the variable 'self' (line 336)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 335)
                stypy_return_type_207150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207150)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_207150

        
        # Assigning a type to the variable 'Test' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'Test', Test)
        
        # Assigning a Str to a Attribute (line 337):
        
        # Assigning a Str to a Attribute (line 337):
        str_207151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 26), 'str', 'Module')
        # Getting the type of 'Test' (line 337)
        Test_207152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'Test')
        # Setting the type of the member '__module__' of a type (line 337)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), Test_207152, '__module__', str_207151)
        
        # Call to pop(...): (line 338)
        # Processing the call arguments (line 338)
        str_207156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 24), 'str', 'Module')
        # Getting the type of 'None' (line 338)
        None_207157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 34), 'None', False)
        # Processing the call keyword arguments (line 338)
        kwargs_207158 = {}
        # Getting the type of 'sys' (line 338)
        sys_207153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'sys', False)
        # Obtaining the member 'modules' of a type (line 338)
        modules_207154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), sys_207153, 'modules')
        # Obtaining the member 'pop' of a type (line 338)
        pop_207155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), modules_207154, 'pop')
        # Calling pop(args, kwargs) (line 338)
        pop_call_result_207159 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), pop_207155, *[str_207156, None_207157], **kwargs_207158)
        
        
        # Assigning a Call to a Name (line 340):
        
        # Assigning a Call to a Name (line 340):
        
        # Call to runTests(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'Test' (line 340)
        Test_207162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 31), 'Test', False)
        # Processing the call keyword arguments (line 340)
        kwargs_207163 = {}
        # Getting the type of 'self' (line 340)
        self_207160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 340)
        runTests_207161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 17), self_207160, 'runTests')
        # Calling runTests(args, kwargs) (line 340)
        runTests_call_result_207164 = invoke(stypy.reporting.localization.Localization(__file__, 340, 17), runTests_207161, *[Test_207162], **kwargs_207163)
        
        # Assigning a type to the variable 'result' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'result', runTests_call_result_207164)
        
        # Call to assertEqual(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'result' (line 341)
        result_207167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 341)
        testsRun_207168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 25), result_207167, 'testsRun')
        int_207169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 42), 'int')
        # Processing the call keyword arguments (line 341)
        kwargs_207170 = {}
        # Getting the type of 'self' (line 341)
        self_207165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 341)
        assertEqual_207166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_207165, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 341)
        assertEqual_call_result_207171 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), assertEqual_207166, *[testsRun_207168, int_207169], **kwargs_207170)
        
        
        # ################# End of 'test_testcase_with_missing_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_testcase_with_missing_module' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_207172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207172)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_testcase_with_missing_module'
        return stypy_return_type_207172


    @norecursion
    def test_teardown_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_teardown_module'
        module_type_store = module_type_store.open_function_context('test_teardown_module', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_teardown_module')
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_teardown_module.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_teardown_module', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_teardown_module', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_teardown_module(...)' code ##################

        # Declaration of the 'Module' class

        class Module(object, ):
            
            # Assigning a Num to a Name (line 345):
            
            # Assigning a Num to a Name (line 345):
            int_207173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 29), 'int')
            # Assigning a type to the variable 'moduleTornDown' (line 345)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'moduleTornDown', int_207173)

            @staticmethod
            @norecursion
            def tearDownModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownModule'
                module_type_store = module_type_store.open_function_context('tearDownModule', 346, 12, False)
                
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

                
                # Getting the type of 'Module' (line 348)
                Module_207174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'Module')
                # Obtaining the member 'moduleTornDown' of a type (line 348)
                moduleTornDown_207175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 16), Module_207174, 'moduleTornDown')
                int_207176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 41), 'int')
                # Applying the binary operator '+=' (line 348)
                result_iadd_207177 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 16), '+=', moduleTornDown_207175, int_207176)
                # Getting the type of 'Module' (line 348)
                Module_207178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'Module')
                # Setting the type of the member 'moduleTornDown' of a type (line 348)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 16), Module_207178, 'moduleTornDown', result_iadd_207177)
                
                
                # ################# End of 'tearDownModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownModule' in the type store
                # Getting the type of 'stypy_return_type' (line 346)
                stypy_return_type_207179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207179)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownModule'
                return stypy_return_type_207179

        
        # Assigning a type to the variable 'Module' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'Module', Module)
        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 350)
        unittest_207180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 350)
        TestCase_207181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 19), unittest_207180, 'TestCase')

        class Test(TestCase_207181, ):

            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 351, 12, False)
                # Assigning a type to the variable 'self' (line 352)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 351)
                stypy_return_type_207182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207182)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_207182


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 353, 12, False)
                # Assigning a type to the variable 'self' (line 354)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 353)
                stypy_return_type_207183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207183)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_207183

        
        # Assigning a type to the variable 'Test' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'Test', Test)
        
        # Assigning a Str to a Attribute (line 355):
        
        # Assigning a Str to a Attribute (line 355):
        str_207184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 26), 'str', 'Module')
        # Getting the type of 'Test' (line 355)
        Test_207185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'Test')
        # Setting the type of the member '__module__' of a type (line 355)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), Test_207185, '__module__', str_207184)
        
        # Assigning a Name to a Subscript (line 356):
        
        # Assigning a Name to a Subscript (line 356):
        # Getting the type of 'Module' (line 356)
        Module_207186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 32), 'Module')
        # Getting the type of 'sys' (line 356)
        sys_207187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 356)
        modules_207188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), sys_207187, 'modules')
        str_207189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 20), 'str', 'Module')
        # Storing an element on a container (line 356)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 8), modules_207188, (str_207189, Module_207186))
        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to runTests(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'Test' (line 358)
        Test_207192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 31), 'Test', False)
        # Processing the call keyword arguments (line 358)
        kwargs_207193 = {}
        # Getting the type of 'self' (line 358)
        self_207190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 358)
        runTests_207191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 17), self_207190, 'runTests')
        # Calling runTests(args, kwargs) (line 358)
        runTests_call_result_207194 = invoke(stypy.reporting.localization.Localization(__file__, 358, 17), runTests_207191, *[Test_207192], **kwargs_207193)
        
        # Assigning a type to the variable 'result' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'result', runTests_call_result_207194)
        
        # Call to assertEqual(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'Module' (line 359)
        Module_207197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 25), 'Module', False)
        # Obtaining the member 'moduleTornDown' of a type (line 359)
        moduleTornDown_207198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 25), Module_207197, 'moduleTornDown')
        int_207199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 48), 'int')
        # Processing the call keyword arguments (line 359)
        kwargs_207200 = {}
        # Getting the type of 'self' (line 359)
        self_207195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 359)
        assertEqual_207196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), self_207195, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 359)
        assertEqual_call_result_207201 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), assertEqual_207196, *[moduleTornDown_207198, int_207199], **kwargs_207200)
        
        
        # Call to assertEqual(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'result' (line 360)
        result_207204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 360)
        testsRun_207205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 25), result_207204, 'testsRun')
        int_207206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 42), 'int')
        # Processing the call keyword arguments (line 360)
        kwargs_207207 = {}
        # Getting the type of 'self' (line 360)
        self_207202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 360)
        assertEqual_207203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), self_207202, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 360)
        assertEqual_call_result_207208 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), assertEqual_207203, *[testsRun_207205, int_207206], **kwargs_207207)
        
        
        # Call to assertEqual(...): (line 361)
        # Processing the call arguments (line 361)
        
        # Call to len(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'result' (line 361)
        result_207212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 361)
        errors_207213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 29), result_207212, 'errors')
        # Processing the call keyword arguments (line 361)
        kwargs_207214 = {}
        # Getting the type of 'len' (line 361)
        len_207211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 25), 'len', False)
        # Calling len(args, kwargs) (line 361)
        len_call_result_207215 = invoke(stypy.reporting.localization.Localization(__file__, 361, 25), len_207211, *[errors_207213], **kwargs_207214)
        
        int_207216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 45), 'int')
        # Processing the call keyword arguments (line 361)
        kwargs_207217 = {}
        # Getting the type of 'self' (line 361)
        self_207209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 361)
        assertEqual_207210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), self_207209, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 361)
        assertEqual_call_result_207218 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), assertEqual_207210, *[len_call_result_207215, int_207216], **kwargs_207217)
        
        
        # ################# End of 'test_teardown_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_teardown_module' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_207219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_teardown_module'
        return stypy_return_type_207219


    @norecursion
    def test_error_in_teardown_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_error_in_teardown_module'
        module_type_store = module_type_store.open_function_context('test_error_in_teardown_module', 363, 4, False)
        # Assigning a type to the variable 'self' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_error_in_teardown_module')
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_error_in_teardown_module.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_error_in_teardown_module', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_error_in_teardown_module', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_error_in_teardown_module(...)' code ##################

        # Declaration of the 'Module' class

        class Module(object, ):
            
            # Assigning a Num to a Name (line 365):
            
            # Assigning a Num to a Name (line 365):
            int_207220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 29), 'int')
            # Assigning a type to the variable 'moduleTornDown' (line 365)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'moduleTornDown', int_207220)

            @staticmethod
            @norecursion
            def tearDownModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownModule'
                module_type_store = module_type_store.open_function_context('tearDownModule', 366, 12, False)
                
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

                
                # Getting the type of 'Module' (line 368)
                Module_207221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'Module')
                # Obtaining the member 'moduleTornDown' of a type (line 368)
                moduleTornDown_207222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 16), Module_207221, 'moduleTornDown')
                int_207223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 41), 'int')
                # Applying the binary operator '+=' (line 368)
                result_iadd_207224 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 16), '+=', moduleTornDown_207222, int_207223)
                # Getting the type of 'Module' (line 368)
                Module_207225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'Module')
                # Setting the type of the member 'moduleTornDown' of a type (line 368)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 16), Module_207225, 'moduleTornDown', result_iadd_207224)
                
                
                # Call to TypeError(...): (line 369)
                # Processing the call arguments (line 369)
                str_207227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 32), 'str', 'foo')
                # Processing the call keyword arguments (line 369)
                kwargs_207228 = {}
                # Getting the type of 'TypeError' (line 369)
                TypeError_207226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 369)
                TypeError_call_result_207229 = invoke(stypy.reporting.localization.Localization(__file__, 369, 22), TypeError_207226, *[str_207227], **kwargs_207228)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 369, 16), TypeError_call_result_207229, 'raise parameter', BaseException)
                
                # ################# End of 'tearDownModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownModule' in the type store
                # Getting the type of 'stypy_return_type' (line 366)
                stypy_return_type_207230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207230)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownModule'
                return stypy_return_type_207230

        
        # Assigning a type to the variable 'Module' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'Module', Module)
        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 371)
        unittest_207231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 371)
        TestCase_207232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 19), unittest_207231, 'TestCase')

        class Test(TestCase_207232, ):
            
            # Assigning a Name to a Name (line 372):
            
            # Assigning a Name to a Name (line 372):
            # Getting the type of 'False' (line 372)
            False_207233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 25), 'False')
            # Assigning a type to the variable 'classSetUp' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'classSetUp', False_207233)
            
            # Assigning a Name to a Name (line 373):
            
            # Assigning a Name to a Name (line 373):
            # Getting the type of 'False' (line 373)
            False_207234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 28), 'False')
            # Assigning a type to the variable 'classTornDown' (line 373)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'classTornDown', False_207234)

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 374, 12, False)
                # Assigning a type to the variable 'self' (line 375)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test.setUpClass')
                Test.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Assigning a Name to a Attribute (line 376):
                
                # Assigning a Name to a Attribute (line 376):
                # Getting the type of 'True' (line 376)
                True_207235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 34), 'True')
                # Getting the type of 'Test' (line 376)
                Test_207236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'Test')
                # Setting the type of the member 'classSetUp' of a type (line 376)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 16), Test_207236, 'classSetUp', True_207235)
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 374)
                stypy_return_type_207237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207237)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_207237


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 377, 12, False)
                # Assigning a type to the variable 'self' (line 378)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test.tearDownClass')
                Test.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Assigning a Name to a Attribute (line 379):
                
                # Assigning a Name to a Attribute (line 379):
                # Getting the type of 'True' (line 379)
                True_207238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 37), 'True')
                # Getting the type of 'Test' (line 379)
                Test_207239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 16), 'Test')
                # Setting the type of the member 'classTornDown' of a type (line 379)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 16), Test_207239, 'classTornDown', True_207238)
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 377)
                stypy_return_type_207240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207240)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_207240


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 380, 12, False)
                # Assigning a type to the variable 'self' (line 381)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 380)
                stypy_return_type_207241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207241)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_207241


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 382, 12, False)
                # Assigning a type to the variable 'self' (line 383)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 382)
                stypy_return_type_207242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207242)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_207242

        
        # Assigning a type to the variable 'Test' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'Test', Test)
        # Declaration of the 'Test2' class
        # Getting the type of 'unittest' (line 385)
        unittest_207243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 385)
        TestCase_207244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 20), unittest_207243, 'TestCase')

        class Test2(TestCase_207244, ):

            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 386, 12, False)
                # Assigning a type to the variable 'self' (line 387)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test2.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.test_one.__dict__.__setitem__('stypy_function_name', 'Test2.test_one')
                Test2.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 386)
                stypy_return_type_207245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207245)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_207245


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 388, 12, False)
                # Assigning a type to the variable 'self' (line 389)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test2.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test2.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test2.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test2.test_two.__dict__.__setitem__('stypy_function_name', 'Test2.test_two')
                Test2.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test2.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test2.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test2.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test2.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test2.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test2.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test2.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 388)
                stypy_return_type_207246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207246)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_207246

        
        # Assigning a type to the variable 'Test2' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'Test2', Test2)
        
        # Assigning a Str to a Attribute (line 390):
        
        # Assigning a Str to a Attribute (line 390):
        str_207247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 26), 'str', 'Module')
        # Getting the type of 'Test' (line 390)
        Test_207248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'Test')
        # Setting the type of the member '__module__' of a type (line 390)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), Test_207248, '__module__', str_207247)
        
        # Assigning a Str to a Attribute (line 391):
        
        # Assigning a Str to a Attribute (line 391):
        str_207249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 27), 'str', 'Module')
        # Getting the type of 'Test2' (line 391)
        Test2_207250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'Test2')
        # Setting the type of the member '__module__' of a type (line 391)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), Test2_207250, '__module__', str_207249)
        
        # Assigning a Name to a Subscript (line 392):
        
        # Assigning a Name to a Subscript (line 392):
        # Getting the type of 'Module' (line 392)
        Module_207251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 32), 'Module')
        # Getting the type of 'sys' (line 392)
        sys_207252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 392)
        modules_207253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), sys_207252, 'modules')
        str_207254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 20), 'str', 'Module')
        # Storing an element on a container (line 392)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 8), modules_207253, (str_207254, Module_207251))
        
        # Assigning a Call to a Name (line 394):
        
        # Assigning a Call to a Name (line 394):
        
        # Call to runTests(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'Test' (line 394)
        Test_207257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 31), 'Test', False)
        # Getting the type of 'Test2' (line 394)
        Test2_207258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 37), 'Test2', False)
        # Processing the call keyword arguments (line 394)
        kwargs_207259 = {}
        # Getting the type of 'self' (line 394)
        self_207255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 394)
        runTests_207256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 17), self_207255, 'runTests')
        # Calling runTests(args, kwargs) (line 394)
        runTests_call_result_207260 = invoke(stypy.reporting.localization.Localization(__file__, 394, 17), runTests_207256, *[Test_207257, Test2_207258], **kwargs_207259)
        
        # Assigning a type to the variable 'result' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'result', runTests_call_result_207260)
        
        # Call to assertEqual(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'Module' (line 395)
        Module_207263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 25), 'Module', False)
        # Obtaining the member 'moduleTornDown' of a type (line 395)
        moduleTornDown_207264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 25), Module_207263, 'moduleTornDown')
        int_207265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 48), 'int')
        # Processing the call keyword arguments (line 395)
        kwargs_207266 = {}
        # Getting the type of 'self' (line 395)
        self_207261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 395)
        assertEqual_207262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), self_207261, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 395)
        assertEqual_call_result_207267 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), assertEqual_207262, *[moduleTornDown_207264, int_207265], **kwargs_207266)
        
        
        # Call to assertEqual(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'result' (line 396)
        result_207270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 396)
        testsRun_207271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 25), result_207270, 'testsRun')
        int_207272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 42), 'int')
        # Processing the call keyword arguments (line 396)
        kwargs_207273 = {}
        # Getting the type of 'self' (line 396)
        self_207268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 396)
        assertEqual_207269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_207268, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 396)
        assertEqual_call_result_207274 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), assertEqual_207269, *[testsRun_207271, int_207272], **kwargs_207273)
        
        
        # Call to assertTrue(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'Test' (line 397)
        Test_207277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'Test', False)
        # Obtaining the member 'classSetUp' of a type (line 397)
        classSetUp_207278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 24), Test_207277, 'classSetUp')
        # Processing the call keyword arguments (line 397)
        kwargs_207279 = {}
        # Getting the type of 'self' (line 397)
        self_207275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 397)
        assertTrue_207276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_207275, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 397)
        assertTrue_call_result_207280 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), assertTrue_207276, *[classSetUp_207278], **kwargs_207279)
        
        
        # Call to assertTrue(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'Test' (line 398)
        Test_207283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 24), 'Test', False)
        # Obtaining the member 'classTornDown' of a type (line 398)
        classTornDown_207284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 24), Test_207283, 'classTornDown')
        # Processing the call keyword arguments (line 398)
        kwargs_207285 = {}
        # Getting the type of 'self' (line 398)
        self_207281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 398)
        assertTrue_207282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), self_207281, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 398)
        assertTrue_call_result_207286 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), assertTrue_207282, *[classTornDown_207284], **kwargs_207285)
        
        
        # Call to assertEqual(...): (line 399)
        # Processing the call arguments (line 399)
        
        # Call to len(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'result' (line 399)
        result_207290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 399)
        errors_207291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 29), result_207290, 'errors')
        # Processing the call keyword arguments (line 399)
        kwargs_207292 = {}
        # Getting the type of 'len' (line 399)
        len_207289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 25), 'len', False)
        # Calling len(args, kwargs) (line 399)
        len_call_result_207293 = invoke(stypy.reporting.localization.Localization(__file__, 399, 25), len_207289, *[errors_207291], **kwargs_207292)
        
        int_207294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 45), 'int')
        # Processing the call keyword arguments (line 399)
        kwargs_207295 = {}
        # Getting the type of 'self' (line 399)
        self_207287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 399)
        assertEqual_207288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_207287, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 399)
        assertEqual_call_result_207296 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), assertEqual_207288, *[len_call_result_207293, int_207294], **kwargs_207295)
        
        
        # Assigning a Subscript to a Tuple (line 400):
        
        # Assigning a Subscript to a Name (line 400):
        
        # Obtaining the type of the subscript
        int_207297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 8), 'int')
        
        # Obtaining the type of the subscript
        int_207298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 33), 'int')
        # Getting the type of 'result' (line 400)
        result_207299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 19), 'result')
        # Obtaining the member 'errors' of a type (line 400)
        errors_207300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 19), result_207299, 'errors')
        # Obtaining the member '__getitem__' of a type (line 400)
        getitem___207301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 19), errors_207300, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 400)
        subscript_call_result_207302 = invoke(stypy.reporting.localization.Localization(__file__, 400, 19), getitem___207301, int_207298)
        
        # Obtaining the member '__getitem__' of a type (line 400)
        getitem___207303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 8), subscript_call_result_207302, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 400)
        subscript_call_result_207304 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), getitem___207303, int_207297)
        
        # Assigning a type to the variable 'tuple_var_assignment_206295' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'tuple_var_assignment_206295', subscript_call_result_207304)
        
        # Assigning a Subscript to a Name (line 400):
        
        # Obtaining the type of the subscript
        int_207305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 8), 'int')
        
        # Obtaining the type of the subscript
        int_207306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 33), 'int')
        # Getting the type of 'result' (line 400)
        result_207307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 19), 'result')
        # Obtaining the member 'errors' of a type (line 400)
        errors_207308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 19), result_207307, 'errors')
        # Obtaining the member '__getitem__' of a type (line 400)
        getitem___207309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 19), errors_207308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 400)
        subscript_call_result_207310 = invoke(stypy.reporting.localization.Localization(__file__, 400, 19), getitem___207309, int_207306)
        
        # Obtaining the member '__getitem__' of a type (line 400)
        getitem___207311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 8), subscript_call_result_207310, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 400)
        subscript_call_result_207312 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), getitem___207311, int_207305)
        
        # Assigning a type to the variable 'tuple_var_assignment_206296' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'tuple_var_assignment_206296', subscript_call_result_207312)
        
        # Assigning a Name to a Name (line 400):
        # Getting the type of 'tuple_var_assignment_206295' (line 400)
        tuple_var_assignment_206295_207313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'tuple_var_assignment_206295')
        # Assigning a type to the variable 'error' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'error', tuple_var_assignment_206295_207313)
        
        # Assigning a Name to a Name (line 400):
        # Getting the type of 'tuple_var_assignment_206296' (line 400)
        tuple_var_assignment_206296_207314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'tuple_var_assignment_206296')
        # Assigning a type to the variable '_' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), '_', tuple_var_assignment_206296_207314)
        
        # Call to assertEqual(...): (line 401)
        # Processing the call arguments (line 401)
        
        # Call to str(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'error' (line 401)
        error_207318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 29), 'error', False)
        # Processing the call keyword arguments (line 401)
        kwargs_207319 = {}
        # Getting the type of 'str' (line 401)
        str_207317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 25), 'str', False)
        # Calling str(args, kwargs) (line 401)
        str_call_result_207320 = invoke(stypy.reporting.localization.Localization(__file__, 401, 25), str_207317, *[error_207318], **kwargs_207319)
        
        str_207321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 37), 'str', 'tearDownModule (Module)')
        # Processing the call keyword arguments (line 401)
        kwargs_207322 = {}
        # Getting the type of 'self' (line 401)
        self_207315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 401)
        assertEqual_207316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), self_207315, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 401)
        assertEqual_call_result_207323 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), assertEqual_207316, *[str_call_result_207320, str_207321], **kwargs_207322)
        
        
        # ################# End of 'test_error_in_teardown_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_error_in_teardown_module' in the type store
        # Getting the type of 'stypy_return_type' (line 363)
        stypy_return_type_207324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207324)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_error_in_teardown_module'
        return stypy_return_type_207324


    @norecursion
    def test_skiptest_in_setupclass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_skiptest_in_setupclass'
        module_type_store = module_type_store.open_function_context('test_skiptest_in_setupclass', 403, 4, False)
        # Assigning a type to the variable 'self' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_skiptest_in_setupclass')
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_skiptest_in_setupclass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_skiptest_in_setupclass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_skiptest_in_setupclass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_skiptest_in_setupclass(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 404)
        unittest_207325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 404)
        TestCase_207326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 19), unittest_207325, 'TestCase')

        class Test(TestCase_207326, ):

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 405, 12, False)
                # Assigning a type to the variable 'self' (line 406)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test.setUpClass')
                Test.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to SkipTest(...): (line 407)
                # Processing the call arguments (line 407)
                str_207329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 40), 'str', 'foo')
                # Processing the call keyword arguments (line 407)
                kwargs_207330 = {}
                # Getting the type of 'unittest' (line 407)
                unittest_207327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 22), 'unittest', False)
                # Obtaining the member 'SkipTest' of a type (line 407)
                SkipTest_207328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 22), unittest_207327, 'SkipTest')
                # Calling SkipTest(args, kwargs) (line 407)
                SkipTest_call_result_207331 = invoke(stypy.reporting.localization.Localization(__file__, 407, 22), SkipTest_207328, *[str_207329], **kwargs_207330)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 407, 16), SkipTest_call_result_207331, 'raise parameter', BaseException)
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 405)
                stypy_return_type_207332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207332)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_207332


            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 408, 12, False)
                # Assigning a type to the variable 'self' (line 409)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 408)
                stypy_return_type_207333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207333)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_207333


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 410, 12, False)
                # Assigning a type to the variable 'self' (line 411)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 410)
                stypy_return_type_207334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207334)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_207334

        
        # Assigning a type to the variable 'Test' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'Test', Test)
        
        # Assigning a Call to a Name (line 413):
        
        # Assigning a Call to a Name (line 413):
        
        # Call to runTests(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'Test' (line 413)
        Test_207337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 31), 'Test', False)
        # Processing the call keyword arguments (line 413)
        kwargs_207338 = {}
        # Getting the type of 'self' (line 413)
        self_207335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 413)
        runTests_207336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 17), self_207335, 'runTests')
        # Calling runTests(args, kwargs) (line 413)
        runTests_call_result_207339 = invoke(stypy.reporting.localization.Localization(__file__, 413, 17), runTests_207336, *[Test_207337], **kwargs_207338)
        
        # Assigning a type to the variable 'result' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'result', runTests_call_result_207339)
        
        # Call to assertEqual(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'result' (line 414)
        result_207342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 414)
        testsRun_207343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 25), result_207342, 'testsRun')
        int_207344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 42), 'int')
        # Processing the call keyword arguments (line 414)
        kwargs_207345 = {}
        # Getting the type of 'self' (line 414)
        self_207340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 414)
        assertEqual_207341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), self_207340, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 414)
        assertEqual_call_result_207346 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), assertEqual_207341, *[testsRun_207343, int_207344], **kwargs_207345)
        
        
        # Call to assertEqual(...): (line 415)
        # Processing the call arguments (line 415)
        
        # Call to len(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'result' (line 415)
        result_207350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 415)
        errors_207351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 29), result_207350, 'errors')
        # Processing the call keyword arguments (line 415)
        kwargs_207352 = {}
        # Getting the type of 'len' (line 415)
        len_207349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 25), 'len', False)
        # Calling len(args, kwargs) (line 415)
        len_call_result_207353 = invoke(stypy.reporting.localization.Localization(__file__, 415, 25), len_207349, *[errors_207351], **kwargs_207352)
        
        int_207354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 45), 'int')
        # Processing the call keyword arguments (line 415)
        kwargs_207355 = {}
        # Getting the type of 'self' (line 415)
        self_207347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 415)
        assertEqual_207348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), self_207347, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 415)
        assertEqual_call_result_207356 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), assertEqual_207348, *[len_call_result_207353, int_207354], **kwargs_207355)
        
        
        # Call to assertEqual(...): (line 416)
        # Processing the call arguments (line 416)
        
        # Call to len(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'result' (line 416)
        result_207360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 29), 'result', False)
        # Obtaining the member 'skipped' of a type (line 416)
        skipped_207361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 29), result_207360, 'skipped')
        # Processing the call keyword arguments (line 416)
        kwargs_207362 = {}
        # Getting the type of 'len' (line 416)
        len_207359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 25), 'len', False)
        # Calling len(args, kwargs) (line 416)
        len_call_result_207363 = invoke(stypy.reporting.localization.Localization(__file__, 416, 25), len_207359, *[skipped_207361], **kwargs_207362)
        
        int_207364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 46), 'int')
        # Processing the call keyword arguments (line 416)
        kwargs_207365 = {}
        # Getting the type of 'self' (line 416)
        self_207357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 416)
        assertEqual_207358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), self_207357, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 416)
        assertEqual_call_result_207366 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), assertEqual_207358, *[len_call_result_207363, int_207364], **kwargs_207365)
        
        
        # Assigning a Subscript to a Name (line 417):
        
        # Assigning a Subscript to a Name (line 417):
        
        # Obtaining the type of the subscript
        int_207367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 36), 'int')
        
        # Obtaining the type of the subscript
        int_207368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 33), 'int')
        # Getting the type of 'result' (line 417)
        result_207369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'result')
        # Obtaining the member 'skipped' of a type (line 417)
        skipped_207370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 18), result_207369, 'skipped')
        # Obtaining the member '__getitem__' of a type (line 417)
        getitem___207371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 18), skipped_207370, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 417)
        subscript_call_result_207372 = invoke(stypy.reporting.localization.Localization(__file__, 417, 18), getitem___207371, int_207368)
        
        # Obtaining the member '__getitem__' of a type (line 417)
        getitem___207373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 18), subscript_call_result_207372, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 417)
        subscript_call_result_207374 = invoke(stypy.reporting.localization.Localization(__file__, 417, 18), getitem___207373, int_207367)
        
        # Assigning a type to the variable 'skipped' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'skipped', subscript_call_result_207374)
        
        # Call to assertEqual(...): (line 418)
        # Processing the call arguments (line 418)
        
        # Call to str(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'skipped' (line 418)
        skipped_207378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 29), 'skipped', False)
        # Processing the call keyword arguments (line 418)
        kwargs_207379 = {}
        # Getting the type of 'str' (line 418)
        str_207377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 25), 'str', False)
        # Calling str(args, kwargs) (line 418)
        str_call_result_207380 = invoke(stypy.reporting.localization.Localization(__file__, 418, 25), str_207377, *[skipped_207378], **kwargs_207379)
        
        str_207381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 39), 'str', 'setUpClass (%s.Test)')
        # Getting the type of '__name__' (line 418)
        name___207382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 64), '__name__', False)
        # Applying the binary operator '%' (line 418)
        result_mod_207383 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 39), '%', str_207381, name___207382)
        
        # Processing the call keyword arguments (line 418)
        kwargs_207384 = {}
        # Getting the type of 'self' (line 418)
        self_207375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 418)
        assertEqual_207376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_207375, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 418)
        assertEqual_call_result_207385 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), assertEqual_207376, *[str_call_result_207380, result_mod_207383], **kwargs_207384)
        
        
        # ################# End of 'test_skiptest_in_setupclass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_skiptest_in_setupclass' in the type store
        # Getting the type of 'stypy_return_type' (line 403)
        stypy_return_type_207386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_skiptest_in_setupclass'
        return stypy_return_type_207386


    @norecursion
    def test_skiptest_in_setupmodule(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_skiptest_in_setupmodule'
        module_type_store = module_type_store.open_function_context('test_skiptest_in_setupmodule', 420, 4, False)
        # Assigning a type to the variable 'self' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_skiptest_in_setupmodule')
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_skiptest_in_setupmodule.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_skiptest_in_setupmodule', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_skiptest_in_setupmodule', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_skiptest_in_setupmodule(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 421)
        unittest_207387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 421)
        TestCase_207388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 19), unittest_207387, 'TestCase')

        class Test(TestCase_207388, ):

            @norecursion
            def test_one(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_one'
                module_type_store = module_type_store.open_function_context('test_one', 422, 12, False)
                # Assigning a type to the variable 'self' (line 423)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_one.__dict__.__setitem__('stypy_localization', localization)
                Test.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_one.__dict__.__setitem__('stypy_function_name', 'Test.test_one')
                Test.test_one.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_one', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_one', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_one(...)' code ##################

                pass
                
                # ################# End of 'test_one(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_one' in the type store
                # Getting the type of 'stypy_return_type' (line 422)
                stypy_return_type_207389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207389)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_one'
                return stypy_return_type_207389


            @norecursion
            def test_two(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_two'
                module_type_store = module_type_store.open_function_context('test_two', 424, 12, False)
                # Assigning a type to the variable 'self' (line 425)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_two.__dict__.__setitem__('stypy_localization', localization)
                Test.test_two.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_two.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_two.__dict__.__setitem__('stypy_function_name', 'Test.test_two')
                Test.test_two.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_two.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_two.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_two.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_two.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_two.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_two', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_two', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_two(...)' code ##################

                pass
                
                # ################# End of 'test_two(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_two' in the type store
                # Getting the type of 'stypy_return_type' (line 424)
                stypy_return_type_207390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207390)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_two'
                return stypy_return_type_207390

        
        # Assigning a type to the variable 'Test' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'Test', Test)
        # Declaration of the 'Module' class

        class Module(object, ):

            @staticmethod
            @norecursion
            def setUpModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpModule'
                module_type_store = module_type_store.open_function_context('setUpModule', 428, 12, False)
                
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

                
                # Call to SkipTest(...): (line 430)
                # Processing the call arguments (line 430)
                str_207393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 40), 'str', 'foo')
                # Processing the call keyword arguments (line 430)
                kwargs_207394 = {}
                # Getting the type of 'unittest' (line 430)
                unittest_207391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 22), 'unittest', False)
                # Obtaining the member 'SkipTest' of a type (line 430)
                SkipTest_207392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 22), unittest_207391, 'SkipTest')
                # Calling SkipTest(args, kwargs) (line 430)
                SkipTest_call_result_207395 = invoke(stypy.reporting.localization.Localization(__file__, 430, 22), SkipTest_207392, *[str_207393], **kwargs_207394)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 430, 16), SkipTest_call_result_207395, 'raise parameter', BaseException)
                
                # ################# End of 'setUpModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpModule' in the type store
                # Getting the type of 'stypy_return_type' (line 428)
                stypy_return_type_207396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207396)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpModule'
                return stypy_return_type_207396

        
        # Assigning a type to the variable 'Module' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'Module', Module)
        
        # Assigning a Str to a Attribute (line 432):
        
        # Assigning a Str to a Attribute (line 432):
        str_207397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 26), 'str', 'Module')
        # Getting the type of 'Test' (line 432)
        Test_207398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'Test')
        # Setting the type of the member '__module__' of a type (line 432)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), Test_207398, '__module__', str_207397)
        
        # Assigning a Name to a Subscript (line 433):
        
        # Assigning a Name to a Subscript (line 433):
        # Getting the type of 'Module' (line 433)
        Module_207399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 32), 'Module')
        # Getting the type of 'sys' (line 433)
        sys_207400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 433)
        modules_207401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), sys_207400, 'modules')
        str_207402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 20), 'str', 'Module')
        # Storing an element on a container (line 433)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 8), modules_207401, (str_207402, Module_207399))
        
        # Assigning a Call to a Name (line 435):
        
        # Assigning a Call to a Name (line 435):
        
        # Call to runTests(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'Test' (line 435)
        Test_207405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 31), 'Test', False)
        # Processing the call keyword arguments (line 435)
        kwargs_207406 = {}
        # Getting the type of 'self' (line 435)
        self_207403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 17), 'self', False)
        # Obtaining the member 'runTests' of a type (line 435)
        runTests_207404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 17), self_207403, 'runTests')
        # Calling runTests(args, kwargs) (line 435)
        runTests_call_result_207407 = invoke(stypy.reporting.localization.Localization(__file__, 435, 17), runTests_207404, *[Test_207405], **kwargs_207406)
        
        # Assigning a type to the variable 'result' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'result', runTests_call_result_207407)
        
        # Call to assertEqual(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'result' (line 436)
        result_207410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 436)
        testsRun_207411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 25), result_207410, 'testsRun')
        int_207412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 42), 'int')
        # Processing the call keyword arguments (line 436)
        kwargs_207413 = {}
        # Getting the type of 'self' (line 436)
        self_207408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 436)
        assertEqual_207409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), self_207408, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 436)
        assertEqual_call_result_207414 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), assertEqual_207409, *[testsRun_207411, int_207412], **kwargs_207413)
        
        
        # Call to assertEqual(...): (line 437)
        # Processing the call arguments (line 437)
        
        # Call to len(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'result' (line 437)
        result_207418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 437)
        errors_207419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 29), result_207418, 'errors')
        # Processing the call keyword arguments (line 437)
        kwargs_207420 = {}
        # Getting the type of 'len' (line 437)
        len_207417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 25), 'len', False)
        # Calling len(args, kwargs) (line 437)
        len_call_result_207421 = invoke(stypy.reporting.localization.Localization(__file__, 437, 25), len_207417, *[errors_207419], **kwargs_207420)
        
        int_207422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 45), 'int')
        # Processing the call keyword arguments (line 437)
        kwargs_207423 = {}
        # Getting the type of 'self' (line 437)
        self_207415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 437)
        assertEqual_207416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), self_207415, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 437)
        assertEqual_call_result_207424 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), assertEqual_207416, *[len_call_result_207421, int_207422], **kwargs_207423)
        
        
        # Call to assertEqual(...): (line 438)
        # Processing the call arguments (line 438)
        
        # Call to len(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'result' (line 438)
        result_207428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 29), 'result', False)
        # Obtaining the member 'skipped' of a type (line 438)
        skipped_207429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 29), result_207428, 'skipped')
        # Processing the call keyword arguments (line 438)
        kwargs_207430 = {}
        # Getting the type of 'len' (line 438)
        len_207427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 25), 'len', False)
        # Calling len(args, kwargs) (line 438)
        len_call_result_207431 = invoke(stypy.reporting.localization.Localization(__file__, 438, 25), len_207427, *[skipped_207429], **kwargs_207430)
        
        int_207432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 46), 'int')
        # Processing the call keyword arguments (line 438)
        kwargs_207433 = {}
        # Getting the type of 'self' (line 438)
        self_207425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 438)
        assertEqual_207426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), self_207425, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 438)
        assertEqual_call_result_207434 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), assertEqual_207426, *[len_call_result_207431, int_207432], **kwargs_207433)
        
        
        # Assigning a Subscript to a Name (line 439):
        
        # Assigning a Subscript to a Name (line 439):
        
        # Obtaining the type of the subscript
        int_207435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 36), 'int')
        
        # Obtaining the type of the subscript
        int_207436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 33), 'int')
        # Getting the type of 'result' (line 439)
        result_207437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 18), 'result')
        # Obtaining the member 'skipped' of a type (line 439)
        skipped_207438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 18), result_207437, 'skipped')
        # Obtaining the member '__getitem__' of a type (line 439)
        getitem___207439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 18), skipped_207438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 439)
        subscript_call_result_207440 = invoke(stypy.reporting.localization.Localization(__file__, 439, 18), getitem___207439, int_207436)
        
        # Obtaining the member '__getitem__' of a type (line 439)
        getitem___207441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 18), subscript_call_result_207440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 439)
        subscript_call_result_207442 = invoke(stypy.reporting.localization.Localization(__file__, 439, 18), getitem___207441, int_207435)
        
        # Assigning a type to the variable 'skipped' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'skipped', subscript_call_result_207442)
        
        # Call to assertEqual(...): (line 440)
        # Processing the call arguments (line 440)
        
        # Call to str(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'skipped' (line 440)
        skipped_207446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 29), 'skipped', False)
        # Processing the call keyword arguments (line 440)
        kwargs_207447 = {}
        # Getting the type of 'str' (line 440)
        str_207445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'str', False)
        # Calling str(args, kwargs) (line 440)
        str_call_result_207448 = invoke(stypy.reporting.localization.Localization(__file__, 440, 25), str_207445, *[skipped_207446], **kwargs_207447)
        
        str_207449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 39), 'str', 'setUpModule (Module)')
        # Processing the call keyword arguments (line 440)
        kwargs_207450 = {}
        # Getting the type of 'self' (line 440)
        self_207443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 440)
        assertEqual_207444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), self_207443, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 440)
        assertEqual_call_result_207451 = invoke(stypy.reporting.localization.Localization(__file__, 440, 8), assertEqual_207444, *[str_call_result_207448, str_207449], **kwargs_207450)
        
        
        # ################# End of 'test_skiptest_in_setupmodule(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_skiptest_in_setupmodule' in the type store
        # Getting the type of 'stypy_return_type' (line 420)
        stypy_return_type_207452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207452)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_skiptest_in_setupmodule'
        return stypy_return_type_207452


    @norecursion
    def test_suite_debug_executes_setups_and_teardowns(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_suite_debug_executes_setups_and_teardowns'
        module_type_store = module_type_store.open_function_context('test_suite_debug_executes_setups_and_teardowns', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_suite_debug_executes_setups_and_teardowns')
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_suite_debug_executes_setups_and_teardowns.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_suite_debug_executes_setups_and_teardowns', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_suite_debug_executes_setups_and_teardowns', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_suite_debug_executes_setups_and_teardowns(...)' code ##################

        
        # Assigning a List to a Name (line 443):
        
        # Assigning a List to a Name (line 443):
        
        # Obtaining an instance of the builtin type 'list' (line 443)
        list_207453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 443)
        
        # Assigning a type to the variable 'ordering' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'ordering', list_207453)
        # Declaration of the 'Module' class

        class Module(object, ):

            @staticmethod
            @norecursion
            def setUpModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpModule'
                module_type_store = module_type_store.open_function_context('setUpModule', 446, 12, False)
                
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

                
                # Call to append(...): (line 448)
                # Processing the call arguments (line 448)
                str_207456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 32), 'str', 'setUpModule')
                # Processing the call keyword arguments (line 448)
                kwargs_207457 = {}
                # Getting the type of 'ordering' (line 448)
                ordering_207454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 448)
                append_207455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 16), ordering_207454, 'append')
                # Calling append(args, kwargs) (line 448)
                append_call_result_207458 = invoke(stypy.reporting.localization.Localization(__file__, 448, 16), append_207455, *[str_207456], **kwargs_207457)
                
                
                # ################# End of 'setUpModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpModule' in the type store
                # Getting the type of 'stypy_return_type' (line 446)
                stypy_return_type_207459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207459)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpModule'
                return stypy_return_type_207459


            @staticmethod
            @norecursion
            def tearDownModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownModule'
                module_type_store = module_type_store.open_function_context('tearDownModule', 449, 12, False)
                
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

                
                # Call to append(...): (line 451)
                # Processing the call arguments (line 451)
                str_207462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 32), 'str', 'tearDownModule')
                # Processing the call keyword arguments (line 451)
                kwargs_207463 = {}
                # Getting the type of 'ordering' (line 451)
                ordering_207460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 451)
                append_207461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 16), ordering_207460, 'append')
                # Calling append(args, kwargs) (line 451)
                append_call_result_207464 = invoke(stypy.reporting.localization.Localization(__file__, 451, 16), append_207461, *[str_207462], **kwargs_207463)
                
                
                # ################# End of 'tearDownModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownModule' in the type store
                # Getting the type of 'stypy_return_type' (line 449)
                stypy_return_type_207465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207465)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownModule'
                return stypy_return_type_207465

        
        # Assigning a type to the variable 'Module' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'Module', Module)
        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 453)
        unittest_207466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 453)
        TestCase_207467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 19), unittest_207466, 'TestCase')

        class Test(TestCase_207467, ):

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 454, 12, False)
                # Assigning a type to the variable 'self' (line 455)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test.setUpClass')
                Test.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 456)
                # Processing the call arguments (line 456)
                str_207470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 32), 'str', 'setUpClass')
                # Processing the call keyword arguments (line 456)
                kwargs_207471 = {}
                # Getting the type of 'ordering' (line 456)
                ordering_207468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 456)
                append_207469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 16), ordering_207468, 'append')
                # Calling append(args, kwargs) (line 456)
                append_call_result_207472 = invoke(stypy.reporting.localization.Localization(__file__, 456, 16), append_207469, *[str_207470], **kwargs_207471)
                
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 454)
                stypy_return_type_207473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207473)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_207473


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 457, 12, False)
                # Assigning a type to the variable 'self' (line 458)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test.tearDownClass')
                Test.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 459)
                # Processing the call arguments (line 459)
                str_207476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 32), 'str', 'tearDownClass')
                # Processing the call keyword arguments (line 459)
                kwargs_207477 = {}
                # Getting the type of 'ordering' (line 459)
                ordering_207474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 459)
                append_207475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 16), ordering_207474, 'append')
                # Calling append(args, kwargs) (line 459)
                append_call_result_207478 = invoke(stypy.reporting.localization.Localization(__file__, 459, 16), append_207475, *[str_207476], **kwargs_207477)
                
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 457)
                stypy_return_type_207479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207479)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_207479


            @norecursion
            def test_something(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_something'
                module_type_store = module_type_store.open_function_context('test_something', 460, 12, False)
                # Assigning a type to the variable 'self' (line 461)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_something.__dict__.__setitem__('stypy_localization', localization)
                Test.test_something.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_something.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_something.__dict__.__setitem__('stypy_function_name', 'Test.test_something')
                Test.test_something.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_something.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_something.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_something.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_something.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_something.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_something.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_something', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_something', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_something(...)' code ##################

                
                # Call to append(...): (line 461)
                # Processing the call arguments (line 461)
                str_207482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 32), 'str', 'test_something')
                # Processing the call keyword arguments (line 461)
                kwargs_207483 = {}
                # Getting the type of 'ordering' (line 461)
                ordering_207480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 461)
                append_207481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 16), ordering_207480, 'append')
                # Calling append(args, kwargs) (line 461)
                append_call_result_207484 = invoke(stypy.reporting.localization.Localization(__file__, 461, 16), append_207481, *[str_207482], **kwargs_207483)
                
                
                # ################# End of 'test_something(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_something' in the type store
                # Getting the type of 'stypy_return_type' (line 460)
                stypy_return_type_207485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207485)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_something'
                return stypy_return_type_207485

        
        # Assigning a type to the variable 'Test' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'Test', Test)
        
        # Assigning a Str to a Attribute (line 463):
        
        # Assigning a Str to a Attribute (line 463):
        str_207486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 26), 'str', 'Module')
        # Getting the type of 'Test' (line 463)
        Test_207487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'Test')
        # Setting the type of the member '__module__' of a type (line 463)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), Test_207487, '__module__', str_207486)
        
        # Assigning a Name to a Subscript (line 464):
        
        # Assigning a Name to a Subscript (line 464):
        # Getting the type of 'Module' (line 464)
        Module_207488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 32), 'Module')
        # Getting the type of 'sys' (line 464)
        sys_207489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 464)
        modules_207490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 8), sys_207489, 'modules')
        str_207491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 20), 'str', 'Module')
        # Storing an element on a container (line 464)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 8), modules_207490, (str_207491, Module_207488))
        
        # Assigning a Call to a Name (line 466):
        
        # Assigning a Call to a Name (line 466):
        
        # Call to loadTestsFromTestCase(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'Test' (line 466)
        Test_207495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 65), 'Test', False)
        # Processing the call keyword arguments (line 466)
        kwargs_207496 = {}
        # Getting the type of 'unittest' (line 466)
        unittest_207492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'unittest', False)
        # Obtaining the member 'defaultTestLoader' of a type (line 466)
        defaultTestLoader_207493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 16), unittest_207492, 'defaultTestLoader')
        # Obtaining the member 'loadTestsFromTestCase' of a type (line 466)
        loadTestsFromTestCase_207494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 16), defaultTestLoader_207493, 'loadTestsFromTestCase')
        # Calling loadTestsFromTestCase(args, kwargs) (line 466)
        loadTestsFromTestCase_call_result_207497 = invoke(stypy.reporting.localization.Localization(__file__, 466, 16), loadTestsFromTestCase_207494, *[Test_207495], **kwargs_207496)
        
        # Assigning a type to the variable 'suite' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'suite', loadTestsFromTestCase_call_result_207497)
        
        # Call to debug(...): (line 467)
        # Processing the call keyword arguments (line 467)
        kwargs_207500 = {}
        # Getting the type of 'suite' (line 467)
        suite_207498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'suite', False)
        # Obtaining the member 'debug' of a type (line 467)
        debug_207499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 8), suite_207498, 'debug')
        # Calling debug(args, kwargs) (line 467)
        debug_call_result_207501 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), debug_207499, *[], **kwargs_207500)
        
        
        # Assigning a List to a Name (line 468):
        
        # Assigning a List to a Name (line 468):
        
        # Obtaining an instance of the builtin type 'list' (line 468)
        list_207502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 468)
        # Adding element type (line 468)
        str_207503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 25), 'str', 'setUpModule')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 24), list_207502, str_207503)
        # Adding element type (line 468)
        str_207504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 40), 'str', 'setUpClass')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 24), list_207502, str_207504)
        # Adding element type (line 468)
        str_207505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 54), 'str', 'test_something')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 24), list_207502, str_207505)
        # Adding element type (line 468)
        str_207506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 72), 'str', 'tearDownClass')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 24), list_207502, str_207506)
        # Adding element type (line 468)
        str_207507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 89), 'str', 'tearDownModule')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 24), list_207502, str_207507)
        
        # Assigning a type to the variable 'expectedOrder' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'expectedOrder', list_207502)
        
        # Call to assertEqual(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'ordering' (line 469)
        ordering_207510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 25), 'ordering', False)
        # Getting the type of 'expectedOrder' (line 469)
        expectedOrder_207511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 35), 'expectedOrder', False)
        # Processing the call keyword arguments (line 469)
        kwargs_207512 = {}
        # Getting the type of 'self' (line 469)
        self_207508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 469)
        assertEqual_207509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), self_207508, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 469)
        assertEqual_call_result_207513 = invoke(stypy.reporting.localization.Localization(__file__, 469, 8), assertEqual_207509, *[ordering_207510, expectedOrder_207511], **kwargs_207512)
        
        
        # ################# End of 'test_suite_debug_executes_setups_and_teardowns(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_suite_debug_executes_setups_and_teardowns' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_207514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_suite_debug_executes_setups_and_teardowns'
        return stypy_return_type_207514


    @norecursion
    def test_suite_debug_propagates_exceptions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_suite_debug_propagates_exceptions'
        module_type_store = module_type_store.open_function_context('test_suite_debug_propagates_exceptions', 471, 4, False)
        # Assigning a type to the variable 'self' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_localization', localization)
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_function_name', 'TestSetups.test_suite_debug_propagates_exceptions')
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_param_names_list', [])
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSetups.test_suite_debug_propagates_exceptions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.test_suite_debug_propagates_exceptions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_suite_debug_propagates_exceptions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_suite_debug_propagates_exceptions(...)' code ##################

        # Declaration of the 'Module' class

        class Module(object, ):

            @staticmethod
            @norecursion
            def setUpModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpModule'
                module_type_store = module_type_store.open_function_context('setUpModule', 473, 12, False)
                
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

                
                
                # Getting the type of 'phase' (line 475)
                phase_207515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'phase')
                int_207516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 28), 'int')
                # Applying the binary operator '==' (line 475)
                result_eq_207517 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 19), '==', phase_207515, int_207516)
                
                # Testing the type of an if condition (line 475)
                if_condition_207518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 475, 16), result_eq_207517)
                # Assigning a type to the variable 'if_condition_207518' (line 475)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 16), 'if_condition_207518', if_condition_207518)
                # SSA begins for if statement (line 475)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to Exception(...): (line 476)
                # Processing the call arguments (line 476)
                str_207520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 36), 'str', 'setUpModule')
                # Processing the call keyword arguments (line 476)
                kwargs_207521 = {}
                # Getting the type of 'Exception' (line 476)
                Exception_207519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 26), 'Exception', False)
                # Calling Exception(args, kwargs) (line 476)
                Exception_call_result_207522 = invoke(stypy.reporting.localization.Localization(__file__, 476, 26), Exception_207519, *[str_207520], **kwargs_207521)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 476, 20), Exception_call_result_207522, 'raise parameter', BaseException)
                # SSA join for if statement (line 475)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # ################# End of 'setUpModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpModule' in the type store
                # Getting the type of 'stypy_return_type' (line 473)
                stypy_return_type_207523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207523)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpModule'
                return stypy_return_type_207523


            @staticmethod
            @norecursion
            def tearDownModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownModule'
                module_type_store = module_type_store.open_function_context('tearDownModule', 477, 12, False)
                
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

                
                
                # Getting the type of 'phase' (line 479)
                phase_207524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 19), 'phase')
                int_207525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 28), 'int')
                # Applying the binary operator '==' (line 479)
                result_eq_207526 = python_operator(stypy.reporting.localization.Localization(__file__, 479, 19), '==', phase_207524, int_207525)
                
                # Testing the type of an if condition (line 479)
                if_condition_207527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 16), result_eq_207526)
                # Assigning a type to the variable 'if_condition_207527' (line 479)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'if_condition_207527', if_condition_207527)
                # SSA begins for if statement (line 479)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to Exception(...): (line 480)
                # Processing the call arguments (line 480)
                str_207529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 36), 'str', 'tearDownModule')
                # Processing the call keyword arguments (line 480)
                kwargs_207530 = {}
                # Getting the type of 'Exception' (line 480)
                Exception_207528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 26), 'Exception', False)
                # Calling Exception(args, kwargs) (line 480)
                Exception_call_result_207531 = invoke(stypy.reporting.localization.Localization(__file__, 480, 26), Exception_207528, *[str_207529], **kwargs_207530)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 480, 20), Exception_call_result_207531, 'raise parameter', BaseException)
                # SSA join for if statement (line 479)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # ################# End of 'tearDownModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownModule' in the type store
                # Getting the type of 'stypy_return_type' (line 477)
                stypy_return_type_207532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207532)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownModule'
                return stypy_return_type_207532

        
        # Assigning a type to the variable 'Module' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'Module', Module)
        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 482)
        unittest_207533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 482)
        TestCase_207534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 19), unittest_207533, 'TestCase')

        class Test(TestCase_207534, ):

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 483, 12, False)
                # Assigning a type to the variable 'self' (line 484)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.setUpClass.__dict__.__setitem__('stypy_localization', localization)
                Test.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.setUpClass.__dict__.__setitem__('stypy_function_name', 'Test.setUpClass')
                Test.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.setUpClass', [], None, None, defaults, varargs, kwargs)

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

                
                
                # Getting the type of 'phase' (line 485)
                phase_207535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 19), 'phase')
                int_207536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 28), 'int')
                # Applying the binary operator '==' (line 485)
                result_eq_207537 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 19), '==', phase_207535, int_207536)
                
                # Testing the type of an if condition (line 485)
                if_condition_207538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 485, 16), result_eq_207537)
                # Assigning a type to the variable 'if_condition_207538' (line 485)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'if_condition_207538', if_condition_207538)
                # SSA begins for if statement (line 485)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to Exception(...): (line 486)
                # Processing the call arguments (line 486)
                str_207540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 36), 'str', 'setUpClass')
                # Processing the call keyword arguments (line 486)
                kwargs_207541 = {}
                # Getting the type of 'Exception' (line 486)
                Exception_207539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 26), 'Exception', False)
                # Calling Exception(args, kwargs) (line 486)
                Exception_call_result_207542 = invoke(stypy.reporting.localization.Localization(__file__, 486, 26), Exception_207539, *[str_207540], **kwargs_207541)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 486, 20), Exception_call_result_207542, 'raise parameter', BaseException)
                # SSA join for if statement (line 485)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 483)
                stypy_return_type_207543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207543)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_207543


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 487, 12, False)
                # Assigning a type to the variable 'self' (line 488)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.tearDownClass.__dict__.__setitem__('stypy_function_name', 'Test.tearDownClass')
                Test.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

                
                
                # Getting the type of 'phase' (line 489)
                phase_207544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 19), 'phase')
                int_207545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 28), 'int')
                # Applying the binary operator '==' (line 489)
                result_eq_207546 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 19), '==', phase_207544, int_207545)
                
                # Testing the type of an if condition (line 489)
                if_condition_207547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 489, 16), result_eq_207546)
                # Assigning a type to the variable 'if_condition_207547' (line 489)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 16), 'if_condition_207547', if_condition_207547)
                # SSA begins for if statement (line 489)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to Exception(...): (line 490)
                # Processing the call arguments (line 490)
                str_207549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 36), 'str', 'tearDownClass')
                # Processing the call keyword arguments (line 490)
                kwargs_207550 = {}
                # Getting the type of 'Exception' (line 490)
                Exception_207548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 26), 'Exception', False)
                # Calling Exception(args, kwargs) (line 490)
                Exception_call_result_207551 = invoke(stypy.reporting.localization.Localization(__file__, 490, 26), Exception_207548, *[str_207549], **kwargs_207550)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 490, 20), Exception_call_result_207551, 'raise parameter', BaseException)
                # SSA join for if statement (line 489)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 487)
                stypy_return_type_207552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207552)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_207552


            @norecursion
            def test_something(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_something'
                module_type_store = module_type_store.open_function_context('test_something', 491, 12, False)
                # Assigning a type to the variable 'self' (line 492)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.test_something.__dict__.__setitem__('stypy_localization', localization)
                Test.test_something.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.test_something.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.test_something.__dict__.__setitem__('stypy_function_name', 'Test.test_something')
                Test.test_something.__dict__.__setitem__('stypy_param_names_list', [])
                Test.test_something.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.test_something.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.test_something.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.test_something.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.test_something.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.test_something.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.test_something', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_something', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_something(...)' code ##################

                
                
                # Getting the type of 'phase' (line 492)
                phase_207553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 19), 'phase')
                int_207554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 28), 'int')
                # Applying the binary operator '==' (line 492)
                result_eq_207555 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 19), '==', phase_207553, int_207554)
                
                # Testing the type of an if condition (line 492)
                if_condition_207556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 492, 16), result_eq_207555)
                # Assigning a type to the variable 'if_condition_207556' (line 492)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 16), 'if_condition_207556', if_condition_207556)
                # SSA begins for if statement (line 492)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to Exception(...): (line 493)
                # Processing the call arguments (line 493)
                str_207558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 36), 'str', 'test_something')
                # Processing the call keyword arguments (line 493)
                kwargs_207559 = {}
                # Getting the type of 'Exception' (line 493)
                Exception_207557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 26), 'Exception', False)
                # Calling Exception(args, kwargs) (line 493)
                Exception_call_result_207560 = invoke(stypy.reporting.localization.Localization(__file__, 493, 26), Exception_207557, *[str_207558], **kwargs_207559)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 493, 20), Exception_call_result_207560, 'raise parameter', BaseException)
                # SSA join for if statement (line 492)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # ################# End of 'test_something(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_something' in the type store
                # Getting the type of 'stypy_return_type' (line 491)
                stypy_return_type_207561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_207561)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_something'
                return stypy_return_type_207561

        
        # Assigning a type to the variable 'Test' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'Test', Test)
        
        # Assigning a Str to a Attribute (line 495):
        
        # Assigning a Str to a Attribute (line 495):
        str_207562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 26), 'str', 'Module')
        # Getting the type of 'Test' (line 495)
        Test_207563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'Test')
        # Setting the type of the member '__module__' of a type (line 495)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 8), Test_207563, '__module__', str_207562)
        
        # Assigning a Name to a Subscript (line 496):
        
        # Assigning a Name to a Subscript (line 496):
        # Getting the type of 'Module' (line 496)
        Module_207564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 32), 'Module')
        # Getting the type of 'sys' (line 496)
        sys_207565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 496)
        modules_207566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 8), sys_207565, 'modules')
        str_207567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 20), 'str', 'Module')
        # Storing an element on a container (line 496)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 8), modules_207566, (str_207567, Module_207564))
        
        # Assigning a Call to a Name (line 498):
        
        # Assigning a Call to a Name (line 498):
        
        # Call to loadTestsFromTestCase(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'Test' (line 498)
        Test_207571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 66), 'Test', False)
        # Processing the call keyword arguments (line 498)
        kwargs_207572 = {}
        # Getting the type of 'unittest' (line 498)
        unittest_207568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'unittest', False)
        # Obtaining the member 'defaultTestLoader' of a type (line 498)
        defaultTestLoader_207569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 17), unittest_207568, 'defaultTestLoader')
        # Obtaining the member 'loadTestsFromTestCase' of a type (line 498)
        loadTestsFromTestCase_207570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 17), defaultTestLoader_207569, 'loadTestsFromTestCase')
        # Calling loadTestsFromTestCase(args, kwargs) (line 498)
        loadTestsFromTestCase_call_result_207573 = invoke(stypy.reporting.localization.Localization(__file__, 498, 17), loadTestsFromTestCase_207570, *[Test_207571], **kwargs_207572)
        
        # Assigning a type to the variable '_suite' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), '_suite', loadTestsFromTestCase_call_result_207573)
        
        # Assigning a Call to a Name (line 499):
        
        # Assigning a Call to a Name (line 499):
        
        # Call to TestSuite(...): (line 499)
        # Processing the call keyword arguments (line 499)
        kwargs_207576 = {}
        # Getting the type of 'unittest' (line 499)
        unittest_207574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 499)
        TestSuite_207575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 16), unittest_207574, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 499)
        TestSuite_call_result_207577 = invoke(stypy.reporting.localization.Localization(__file__, 499, 16), TestSuite_207575, *[], **kwargs_207576)
        
        # Assigning a type to the variable 'suite' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'suite', TestSuite_call_result_207577)
        
        # Call to addTest(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of '_suite' (line 500)
        _suite_207580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 22), '_suite', False)
        # Processing the call keyword arguments (line 500)
        kwargs_207581 = {}
        # Getting the type of 'suite' (line 500)
        suite_207578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'suite', False)
        # Obtaining the member 'addTest' of a type (line 500)
        addTest_207579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), suite_207578, 'addTest')
        # Calling addTest(args, kwargs) (line 500)
        addTest_call_result_207582 = invoke(stypy.reporting.localization.Localization(__file__, 500, 8), addTest_207579, *[_suite_207580], **kwargs_207581)
        
        
        # Assigning a Tuple to a Name (line 502):
        
        # Assigning a Tuple to a Name (line 502):
        
        # Obtaining an instance of the builtin type 'tuple' (line 502)
        tuple_207583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 502)
        # Adding element type (line 502)
        str_207584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 20), 'str', 'setUpModule')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 20), tuple_207583, str_207584)
        # Adding element type (line 502)
        str_207585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 35), 'str', 'tearDownModule')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 20), tuple_207583, str_207585)
        # Adding element type (line 502)
        str_207586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 53), 'str', 'setUpClass')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 20), tuple_207583, str_207586)
        # Adding element type (line 502)
        str_207587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 67), 'str', 'tearDownClass')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 20), tuple_207583, str_207587)
        # Adding element type (line 502)
        str_207588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 84), 'str', 'test_something')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 20), tuple_207583, str_207588)
        
        # Assigning a type to the variable 'messages' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'messages', tuple_207583)
        
        
        # Call to enumerate(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'messages' (line 503)
        messages_207590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 36), 'messages', False)
        # Processing the call keyword arguments (line 503)
        kwargs_207591 = {}
        # Getting the type of 'enumerate' (line 503)
        enumerate_207589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 503)
        enumerate_call_result_207592 = invoke(stypy.reporting.localization.Localization(__file__, 503, 26), enumerate_207589, *[messages_207590], **kwargs_207591)
        
        # Testing the type of a for loop iterable (line 503)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 503, 8), enumerate_call_result_207592)
        # Getting the type of the for loop variable (line 503)
        for_loop_var_207593 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 503, 8), enumerate_call_result_207592)
        # Assigning a type to the variable 'phase' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'phase', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 8), for_loop_var_207593))
        # Assigning a type to the variable 'msg' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'msg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 8), for_loop_var_207593))
        # SSA begins for a for statement (line 503)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertRaisesRegexp(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'Exception' (line 504)
        Exception_207596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 41), 'Exception', False)
        # Getting the type of 'msg' (line 504)
        msg_207597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 52), 'msg', False)
        # Processing the call keyword arguments (line 504)
        kwargs_207598 = {}
        # Getting the type of 'self' (line 504)
        self_207594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 17), 'self', False)
        # Obtaining the member 'assertRaisesRegexp' of a type (line 504)
        assertRaisesRegexp_207595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 17), self_207594, 'assertRaisesRegexp')
        # Calling assertRaisesRegexp(args, kwargs) (line 504)
        assertRaisesRegexp_call_result_207599 = invoke(stypy.reporting.localization.Localization(__file__, 504, 17), assertRaisesRegexp_207595, *[Exception_207596, msg_207597], **kwargs_207598)
        
        with_207600 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 504, 17), assertRaisesRegexp_call_result_207599, 'with parameter', '__enter__', '__exit__')

        if with_207600:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 504)
            enter___207601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 17), assertRaisesRegexp_call_result_207599, '__enter__')
            with_enter_207602 = invoke(stypy.reporting.localization.Localization(__file__, 504, 17), enter___207601)
            
            # Call to debug(...): (line 505)
            # Processing the call keyword arguments (line 505)
            kwargs_207605 = {}
            # Getting the type of 'suite' (line 505)
            suite_207603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 16), 'suite', False)
            # Obtaining the member 'debug' of a type (line 505)
            debug_207604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 16), suite_207603, 'debug')
            # Calling debug(args, kwargs) (line 505)
            debug_call_result_207606 = invoke(stypy.reporting.localization.Localization(__file__, 505, 16), debug_207604, *[], **kwargs_207605)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 504)
            exit___207607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 17), assertRaisesRegexp_call_result_207599, '__exit__')
            with_exit_207608 = invoke(stypy.reporting.localization.Localization(__file__, 504, 17), exit___207607, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_suite_debug_propagates_exceptions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_suite_debug_propagates_exceptions' in the type store
        # Getting the type of 'stypy_return_type' (line 471)
        stypy_return_type_207609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_207609)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_suite_debug_propagates_exceptions'
        return stypy_return_type_207609


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSetups.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSetups' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TestSetups', TestSetups)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 508)
    # Processing the call keyword arguments (line 508)
    kwargs_207612 = {}
    # Getting the type of 'unittest' (line 508)
    unittest_207610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 508)
    main_207611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 4), unittest_207610, 'main')
    # Calling main(args, kwargs) (line 508)
    main_call_result_207613 = invoke(stypy.reporting.localization.Localization(__file__, 508, 4), main_207611, *[], **kwargs_207612)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
