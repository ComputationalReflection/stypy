
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import unittest
2: 
3: import sys
4: from unittest.test.support import LoggingResult, TestEquality
5: 
6: 
7: ### Support code for Test_TestSuite
8: ################################################################
9: 
10: class Test(object):
11:     class Foo(unittest.TestCase):
12:         def test_1(self): pass
13:         def test_2(self): pass
14:         def test_3(self): pass
15:         def runTest(self): pass
16: 
17: def _mk_TestSuite(*names):
18:     return unittest.TestSuite(Test.Foo(n) for n in names)
19: 
20: ################################################################
21: 
22: 
23: class Test_TestSuite(unittest.TestCase, TestEquality):
24: 
25:     ### Set up attributes needed by inherited tests
26:     ################################################################
27: 
28:     # Used by TestEquality.test_eq
29:     eq_pairs = [(unittest.TestSuite(), unittest.TestSuite()),
30:                 (unittest.TestSuite(), unittest.TestSuite([])),
31:                (_mk_TestSuite('test_1'), _mk_TestSuite('test_1'))]
32: 
33:     # Used by TestEquality.test_ne
34:     ne_pairs = [(unittest.TestSuite(), _mk_TestSuite('test_1')),
35:                 (unittest.TestSuite([]), _mk_TestSuite('test_1')),
36:                 (_mk_TestSuite('test_1', 'test_2'), _mk_TestSuite('test_1', 'test_3')),
37:                 (_mk_TestSuite('test_1'), _mk_TestSuite('test_2'))]
38: 
39:     ################################################################
40:     ### /Set up attributes needed by inherited tests
41: 
42:     ### Tests for TestSuite.__init__
43:     ################################################################
44: 
45:     # "class TestSuite([tests])"
46:     #
47:     # The tests iterable should be optional
48:     def test_init__tests_optional(self):
49:         suite = unittest.TestSuite()
50: 
51:         self.assertEqual(suite.countTestCases(), 0)
52: 
53:     # "class TestSuite([tests])"
54:     # ...
55:     # "If tests is given, it must be an iterable of individual test cases
56:     # or other test suites that will be used to build the suite initially"
57:     #
58:     # TestSuite should deal with empty tests iterables by allowing the
59:     # creation of an empty suite
60:     def test_init__empty_tests(self):
61:         suite = unittest.TestSuite([])
62: 
63:         self.assertEqual(suite.countTestCases(), 0)
64: 
65:     # "class TestSuite([tests])"
66:     # ...
67:     # "If tests is given, it must be an iterable of individual test cases
68:     # or other test suites that will be used to build the suite initially"
69:     #
70:     # TestSuite should allow any iterable to provide tests
71:     def test_init__tests_from_any_iterable(self):
72:         def tests():
73:             yield unittest.FunctionTestCase(lambda: None)
74:             yield unittest.FunctionTestCase(lambda: None)
75: 
76:         suite_1 = unittest.TestSuite(tests())
77:         self.assertEqual(suite_1.countTestCases(), 2)
78: 
79:         suite_2 = unittest.TestSuite(suite_1)
80:         self.assertEqual(suite_2.countTestCases(), 2)
81: 
82:         suite_3 = unittest.TestSuite(set(suite_1))
83:         self.assertEqual(suite_3.countTestCases(), 2)
84: 
85:     # "class TestSuite([tests])"
86:     # ...
87:     # "If tests is given, it must be an iterable of individual test cases
88:     # or other test suites that will be used to build the suite initially"
89:     #
90:     # Does TestSuite() also allow other TestSuite() instances to be present
91:     # in the tests iterable?
92:     def test_init__TestSuite_instances_in_tests(self):
93:         def tests():
94:             ftc = unittest.FunctionTestCase(lambda: None)
95:             yield unittest.TestSuite([ftc])
96:             yield unittest.FunctionTestCase(lambda: None)
97: 
98:         suite = unittest.TestSuite(tests())
99:         self.assertEqual(suite.countTestCases(), 2)
100: 
101:     ################################################################
102:     ### /Tests for TestSuite.__init__
103: 
104:     # Container types should support the iter protocol
105:     def test_iter(self):
106:         test1 = unittest.FunctionTestCase(lambda: None)
107:         test2 = unittest.FunctionTestCase(lambda: None)
108:         suite = unittest.TestSuite((test1, test2))
109: 
110:         self.assertEqual(list(suite), [test1, test2])
111: 
112:     # "Return the number of tests represented by the this test object.
113:     # ...this method is also implemented by the TestSuite class, which can
114:     # return larger [greater than 1] values"
115:     #
116:     # Presumably an empty TestSuite returns 0?
117:     def test_countTestCases_zero_simple(self):
118:         suite = unittest.TestSuite()
119: 
120:         self.assertEqual(suite.countTestCases(), 0)
121: 
122:     # "Return the number of tests represented by the this test object.
123:     # ...this method is also implemented by the TestSuite class, which can
124:     # return larger [greater than 1] values"
125:     #
126:     # Presumably an empty TestSuite (even if it contains other empty
127:     # TestSuite instances) returns 0?
128:     def test_countTestCases_zero_nested(self):
129:         class Test1(unittest.TestCase):
130:             def test(self):
131:                 pass
132: 
133:         suite = unittest.TestSuite([unittest.TestSuite()])
134: 
135:         self.assertEqual(suite.countTestCases(), 0)
136: 
137:     # "Return the number of tests represented by the this test object.
138:     # ...this method is also implemented by the TestSuite class, which can
139:     # return larger [greater than 1] values"
140:     def test_countTestCases_simple(self):
141:         test1 = unittest.FunctionTestCase(lambda: None)
142:         test2 = unittest.FunctionTestCase(lambda: None)
143:         suite = unittest.TestSuite((test1, test2))
144: 
145:         self.assertEqual(suite.countTestCases(), 2)
146: 
147:     # "Return the number of tests represented by the this test object.
148:     # ...this method is also implemented by the TestSuite class, which can
149:     # return larger [greater than 1] values"
150:     #
151:     # Make sure this holds for nested TestSuite instances, too
152:     def test_countTestCases_nested(self):
153:         class Test1(unittest.TestCase):
154:             def test1(self): pass
155:             def test2(self): pass
156: 
157:         test2 = unittest.FunctionTestCase(lambda: None)
158:         test3 = unittest.FunctionTestCase(lambda: None)
159:         child = unittest.TestSuite((Test1('test2'), test2))
160:         parent = unittest.TestSuite((test3, child, Test1('test1')))
161: 
162:         self.assertEqual(parent.countTestCases(), 4)
163: 
164:     # "Run the tests associated with this suite, collecting the result into
165:     # the test result object passed as result."
166:     #
167:     # And if there are no tests? What then?
168:     def test_run__empty_suite(self):
169:         events = []
170:         result = LoggingResult(events)
171: 
172:         suite = unittest.TestSuite()
173: 
174:         suite.run(result)
175: 
176:         self.assertEqual(events, [])
177: 
178:     # "Note that unlike TestCase.run(), TestSuite.run() requires the
179:     # "result object to be passed in."
180:     def test_run__requires_result(self):
181:         suite = unittest.TestSuite()
182: 
183:         try:
184:             suite.run()
185:         except TypeError:
186:             pass
187:         else:
188:             self.fail("Failed to raise TypeError")
189: 
190:     # "Run the tests associated with this suite, collecting the result into
191:     # the test result object passed as result."
192:     def test_run(self):
193:         events = []
194:         result = LoggingResult(events)
195: 
196:         class LoggingCase(unittest.TestCase):
197:             def run(self, result):
198:                 events.append('run %s' % self._testMethodName)
199: 
200:             def test1(self): pass
201:             def test2(self): pass
202: 
203:         tests = [LoggingCase('test1'), LoggingCase('test2')]
204: 
205:         unittest.TestSuite(tests).run(result)
206: 
207:         self.assertEqual(events, ['run test1', 'run test2'])
208: 
209:     # "Add a TestCase ... to the suite"
210:     def test_addTest__TestCase(self):
211:         class Foo(unittest.TestCase):
212:             def test(self): pass
213: 
214:         test = Foo('test')
215:         suite = unittest.TestSuite()
216: 
217:         suite.addTest(test)
218: 
219:         self.assertEqual(suite.countTestCases(), 1)
220:         self.assertEqual(list(suite), [test])
221: 
222:     # "Add a ... TestSuite to the suite"
223:     def test_addTest__TestSuite(self):
224:         class Foo(unittest.TestCase):
225:             def test(self): pass
226: 
227:         suite_2 = unittest.TestSuite([Foo('test')])
228: 
229:         suite = unittest.TestSuite()
230:         suite.addTest(suite_2)
231: 
232:         self.assertEqual(suite.countTestCases(), 1)
233:         self.assertEqual(list(suite), [suite_2])
234: 
235:     # "Add all the tests from an iterable of TestCase and TestSuite
236:     # instances to this test suite."
237:     #
238:     # "This is equivalent to iterating over tests, calling addTest() for
239:     # each element"
240:     def test_addTests(self):
241:         class Foo(unittest.TestCase):
242:             def test_1(self): pass
243:             def test_2(self): pass
244: 
245:         test_1 = Foo('test_1')
246:         test_2 = Foo('test_2')
247:         inner_suite = unittest.TestSuite([test_2])
248: 
249:         def gen():
250:             yield test_1
251:             yield test_2
252:             yield inner_suite
253: 
254:         suite_1 = unittest.TestSuite()
255:         suite_1.addTests(gen())
256: 
257:         self.assertEqual(list(suite_1), list(gen()))
258: 
259:         # "This is equivalent to iterating over tests, calling addTest() for
260:         # each element"
261:         suite_2 = unittest.TestSuite()
262:         for t in gen():
263:             suite_2.addTest(t)
264: 
265:         self.assertEqual(suite_1, suite_2)
266: 
267:     # "Add all the tests from an iterable of TestCase and TestSuite
268:     # instances to this test suite."
269:     #
270:     # What happens if it doesn't get an iterable?
271:     def test_addTest__noniterable(self):
272:         suite = unittest.TestSuite()
273: 
274:         try:
275:             suite.addTests(5)
276:         except TypeError:
277:             pass
278:         else:
279:             self.fail("Failed to raise TypeError")
280: 
281:     def test_addTest__noncallable(self):
282:         suite = unittest.TestSuite()
283:         self.assertRaises(TypeError, suite.addTest, 5)
284: 
285:     def test_addTest__casesuiteclass(self):
286:         suite = unittest.TestSuite()
287:         self.assertRaises(TypeError, suite.addTest, Test_TestSuite)
288:         self.assertRaises(TypeError, suite.addTest, unittest.TestSuite)
289: 
290:     def test_addTests__string(self):
291:         suite = unittest.TestSuite()
292:         self.assertRaises(TypeError, suite.addTests, "foo")
293: 
294:     def test_function_in_suite(self):
295:         def f(_):
296:             pass
297:         suite = unittest.TestSuite()
298:         suite.addTest(f)
299: 
300:         # when the bug is fixed this line will not crash
301:         suite.run(unittest.TestResult())
302: 
303: 
304: 
305:     def test_basetestsuite(self):
306:         class Test(unittest.TestCase):
307:             wasSetUp = False
308:             wasTornDown = False
309:             @classmethod
310:             def setUpClass(cls):
311:                 cls.wasSetUp = True
312:             @classmethod
313:             def tearDownClass(cls):
314:                 cls.wasTornDown = True
315:             def testPass(self):
316:                 pass
317:             def testFail(self):
318:                 fail
319:         class Module(object):
320:             wasSetUp = False
321:             wasTornDown = False
322:             @staticmethod
323:             def setUpModule():
324:                 Module.wasSetUp = True
325:             @staticmethod
326:             def tearDownModule():
327:                 Module.wasTornDown = True
328: 
329:         Test.__module__ = 'Module'
330:         sys.modules['Module'] = Module
331:         self.addCleanup(sys.modules.pop, 'Module')
332: 
333:         suite = unittest.BaseTestSuite()
334:         suite.addTests([Test('testPass'), Test('testFail')])
335:         self.assertEqual(suite.countTestCases(), 2)
336: 
337:         result = unittest.TestResult()
338:         suite.run(result)
339:         self.assertFalse(Module.wasSetUp)
340:         self.assertFalse(Module.wasTornDown)
341:         self.assertFalse(Test.wasSetUp)
342:         self.assertFalse(Test.wasTornDown)
343:         self.assertEqual(len(result.errors), 1)
344:         self.assertEqual(len(result.failures), 0)
345:         self.assertEqual(result.testsRun, 2)
346: 
347: 
348:     def test_overriding_call(self):
349:         class MySuite(unittest.TestSuite):
350:             called = False
351:             def __call__(self, *args, **kw):
352:                 self.called = True
353:                 unittest.TestSuite.__call__(self, *args, **kw)
354: 
355:         suite = MySuite()
356:         result = unittest.TestResult()
357:         wrapper = unittest.TestSuite()
358:         wrapper.addTest(suite)
359:         wrapper(result)
360:         self.assertTrue(suite.called)
361: 
362:         # reusing results should be permitted even if abominable
363:         self.assertFalse(result._testRunEntered)
364: 
365: 
366: if __name__ == '__main__':
367:     unittest.main()
368: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import unittest' statement (line 1)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from unittest.test.support import LoggingResult, TestEquality' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/unittest/test/')
import_208136 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest.test.support')

if (type(import_208136) is not StypyTypeError):

    if (import_208136 != 'pyd_module'):
        __import__(import_208136)
        sys_modules_208137 = sys.modules[import_208136]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest.test.support', sys_modules_208137.module_type_store, module_type_store, ['LoggingResult', 'TestEquality'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_208137, sys_modules_208137.module_type_store, module_type_store)
    else:
        from unittest.test.support import LoggingResult, TestEquality

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest.test.support', None, module_type_store, ['LoggingResult', 'TestEquality'], [LoggingResult, TestEquality])

else:
    # Assigning a type to the variable 'unittest.test.support' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'unittest.test.support', import_208136)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/test/')

# Declaration of the 'Test' class

class Test(object, ):
    # Declaration of the 'Foo' class
    # Getting the type of 'unittest' (line 11)
    unittest_208138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'unittest')
    # Obtaining the member 'TestCase' of a type (line 11)
    TestCase_208139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 14), unittest_208138, 'TestCase')

    class Foo(TestCase_208139, ):

        @norecursion
        def test_1(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test_1'
            module_type_store = module_type_store.open_function_context('test_1', 12, 8, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self', type_of_self)
            
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
            # Getting the type of 'stypy_return_type' (line 12)
            stypy_return_type_208140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208140)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test_1'
            return stypy_return_type_208140


        @norecursion
        def test_2(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test_2'
            module_type_store = module_type_store.open_function_context('test_2', 13, 8, False)
            # Assigning a type to the variable 'self' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Foo.test_2.__dict__.__setitem__('stypy_localization', localization)
            Foo.test_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Foo.test_2.__dict__.__setitem__('stypy_type_store', module_type_store)
            Foo.test_2.__dict__.__setitem__('stypy_function_name', 'Foo.test_2')
            Foo.test_2.__dict__.__setitem__('stypy_param_names_list', [])
            Foo.test_2.__dict__.__setitem__('stypy_varargs_param_name', None)
            Foo.test_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Foo.test_2.__dict__.__setitem__('stypy_call_defaults', defaults)
            Foo.test_2.__dict__.__setitem__('stypy_call_varargs', varargs)
            Foo.test_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Foo.test_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_2', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'test_2', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'test_2(...)' code ##################

            pass
            
            # ################# End of 'test_2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test_2' in the type store
            # Getting the type of 'stypy_return_type' (line 13)
            stypy_return_type_208141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208141)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test_2'
            return stypy_return_type_208141


        @norecursion
        def test_3(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test_3'
            module_type_store = module_type_store.open_function_context('test_3', 14, 8, False)
            # Assigning a type to the variable 'self' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Foo.test_3.__dict__.__setitem__('stypy_localization', localization)
            Foo.test_3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Foo.test_3.__dict__.__setitem__('stypy_type_store', module_type_store)
            Foo.test_3.__dict__.__setitem__('stypy_function_name', 'Foo.test_3')
            Foo.test_3.__dict__.__setitem__('stypy_param_names_list', [])
            Foo.test_3.__dict__.__setitem__('stypy_varargs_param_name', None)
            Foo.test_3.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Foo.test_3.__dict__.__setitem__('stypy_call_defaults', defaults)
            Foo.test_3.__dict__.__setitem__('stypy_call_varargs', varargs)
            Foo.test_3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Foo.test_3.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_3', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'test_3', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'test_3(...)' code ##################

            pass
            
            # ################# End of 'test_3(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test_3' in the type store
            # Getting the type of 'stypy_return_type' (line 14)
            stypy_return_type_208142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208142)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test_3'
            return stypy_return_type_208142


        @norecursion
        def runTest(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'runTest'
            module_type_store = module_type_store.open_function_context('runTest', 15, 8, False)
            # Assigning a type to the variable 'self' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Foo.runTest.__dict__.__setitem__('stypy_localization', localization)
            Foo.runTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Foo.runTest.__dict__.__setitem__('stypy_type_store', module_type_store)
            Foo.runTest.__dict__.__setitem__('stypy_function_name', 'Foo.runTest')
            Foo.runTest.__dict__.__setitem__('stypy_param_names_list', [])
            Foo.runTest.__dict__.__setitem__('stypy_varargs_param_name', None)
            Foo.runTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Foo.runTest.__dict__.__setitem__('stypy_call_defaults', defaults)
            Foo.runTest.__dict__.__setitem__('stypy_call_varargs', varargs)
            Foo.runTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Foo.runTest.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.runTest', [], None, None, defaults, varargs, kwargs)

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

            pass
            
            # ################# End of 'runTest(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'runTest' in the type store
            # Getting the type of 'stypy_return_type' (line 15)
            stypy_return_type_208143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208143)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'runTest'
            return stypy_return_type_208143

    
    # Assigning a type to the variable 'Foo' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'Foo', Foo)

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Test', Test)

@norecursion
def _mk_TestSuite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_mk_TestSuite'
    module_type_store = module_type_store.open_function_context('_mk_TestSuite', 17, 0, False)
    
    # Passed parameters checking function
    _mk_TestSuite.stypy_localization = localization
    _mk_TestSuite.stypy_type_of_self = None
    _mk_TestSuite.stypy_type_store = module_type_store
    _mk_TestSuite.stypy_function_name = '_mk_TestSuite'
    _mk_TestSuite.stypy_param_names_list = []
    _mk_TestSuite.stypy_varargs_param_name = 'names'
    _mk_TestSuite.stypy_kwargs_param_name = None
    _mk_TestSuite.stypy_call_defaults = defaults
    _mk_TestSuite.stypy_call_varargs = varargs
    _mk_TestSuite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_mk_TestSuite', [], 'names', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_mk_TestSuite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_mk_TestSuite(...)' code ##################

    
    # Call to TestSuite(...): (line 18)
    # Processing the call arguments (line 18)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 18, 30, True)
    # Calculating comprehension expression
    # Getting the type of 'names' (line 18)
    names_208151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 51), 'names', False)
    comprehension_208152 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 30), names_208151)
    # Assigning a type to the variable 'n' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 30), 'n', comprehension_208152)
    
    # Call to Foo(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'n' (line 18)
    n_208148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 39), 'n', False)
    # Processing the call keyword arguments (line 18)
    kwargs_208149 = {}
    # Getting the type of 'Test' (line 18)
    Test_208146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 30), 'Test', False)
    # Obtaining the member 'Foo' of a type (line 18)
    Foo_208147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 30), Test_208146, 'Foo')
    # Calling Foo(args, kwargs) (line 18)
    Foo_call_result_208150 = invoke(stypy.reporting.localization.Localization(__file__, 18, 30), Foo_208147, *[n_208148], **kwargs_208149)
    
    list_208153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 30), list_208153, Foo_call_result_208150)
    # Processing the call keyword arguments (line 18)
    kwargs_208154 = {}
    # Getting the type of 'unittest' (line 18)
    unittest_208144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'unittest', False)
    # Obtaining the member 'TestSuite' of a type (line 18)
    TestSuite_208145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), unittest_208144, 'TestSuite')
    # Calling TestSuite(args, kwargs) (line 18)
    TestSuite_call_result_208155 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), TestSuite_208145, *[list_208153], **kwargs_208154)
    
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', TestSuite_call_result_208155)
    
    # ################# End of '_mk_TestSuite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_mk_TestSuite' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_208156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_208156)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_mk_TestSuite'
    return stypy_return_type_208156

# Assigning a type to the variable '_mk_TestSuite' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '_mk_TestSuite', _mk_TestSuite)
# Declaration of the 'Test_TestSuite' class
# Getting the type of 'unittest' (line 23)
unittest_208157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'unittest')
# Obtaining the member 'TestCase' of a type (line 23)
TestCase_208158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 21), unittest_208157, 'TestCase')
# Getting the type of 'TestEquality' (line 23)
TestEquality_208159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 40), 'TestEquality')

class Test_TestSuite(TestCase_208158, TestEquality_208159, ):

    @norecursion
    def test_init__tests_optional(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_init__tests_optional'
        module_type_store = module_type_store.open_function_context('test_init__tests_optional', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_init__tests_optional')
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_init__tests_optional.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_init__tests_optional', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_init__tests_optional', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_init__tests_optional(...)' code ##################

        
        # Assigning a Call to a Name (line 49):
        
        # Call to TestSuite(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_208162 = {}
        # Getting the type of 'unittest' (line 49)
        unittest_208160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 49)
        TestSuite_208161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), unittest_208160, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 49)
        TestSuite_call_result_208163 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), TestSuite_208161, *[], **kwargs_208162)
        
        # Assigning a type to the variable 'suite' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'suite', TestSuite_call_result_208163)
        
        # Call to assertEqual(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to countTestCases(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_208168 = {}
        # Getting the type of 'suite' (line 51)
        suite_208166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 51)
        countTestCases_208167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), suite_208166, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 51)
        countTestCases_call_result_208169 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), countTestCases_208167, *[], **kwargs_208168)
        
        int_208170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 49), 'int')
        # Processing the call keyword arguments (line 51)
        kwargs_208171 = {}
        # Getting the type of 'self' (line 51)
        self_208164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 51)
        assertEqual_208165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_208164, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 51)
        assertEqual_call_result_208172 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assertEqual_208165, *[countTestCases_call_result_208169, int_208170], **kwargs_208171)
        
        
        # ################# End of 'test_init__tests_optional(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_init__tests_optional' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_208173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_init__tests_optional'
        return stypy_return_type_208173


    @norecursion
    def test_init__empty_tests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_init__empty_tests'
        module_type_store = module_type_store.open_function_context('test_init__empty_tests', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_init__empty_tests')
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_init__empty_tests.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_init__empty_tests', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_init__empty_tests', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_init__empty_tests(...)' code ##################

        
        # Assigning a Call to a Name (line 61):
        
        # Call to TestSuite(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_208176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        
        # Processing the call keyword arguments (line 61)
        kwargs_208177 = {}
        # Getting the type of 'unittest' (line 61)
        unittest_208174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 61)
        TestSuite_208175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), unittest_208174, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 61)
        TestSuite_call_result_208178 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), TestSuite_208175, *[list_208176], **kwargs_208177)
        
        # Assigning a type to the variable 'suite' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'suite', TestSuite_call_result_208178)
        
        # Call to assertEqual(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to countTestCases(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_208183 = {}
        # Getting the type of 'suite' (line 63)
        suite_208181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 63)
        countTestCases_208182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 25), suite_208181, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 63)
        countTestCases_call_result_208184 = invoke(stypy.reporting.localization.Localization(__file__, 63, 25), countTestCases_208182, *[], **kwargs_208183)
        
        int_208185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 49), 'int')
        # Processing the call keyword arguments (line 63)
        kwargs_208186 = {}
        # Getting the type of 'self' (line 63)
        self_208179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 63)
        assertEqual_208180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_208179, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 63)
        assertEqual_call_result_208187 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), assertEqual_208180, *[countTestCases_call_result_208184, int_208185], **kwargs_208186)
        
        
        # ################# End of 'test_init__empty_tests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_init__empty_tests' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_208188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208188)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_init__empty_tests'
        return stypy_return_type_208188


    @norecursion
    def test_init__tests_from_any_iterable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_init__tests_from_any_iterable'
        module_type_store = module_type_store.open_function_context('test_init__tests_from_any_iterable', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_init__tests_from_any_iterable')
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_init__tests_from_any_iterable.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_init__tests_from_any_iterable', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_init__tests_from_any_iterable', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_init__tests_from_any_iterable(...)' code ##################


        @norecursion
        def tests(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'tests'
            module_type_store = module_type_store.open_function_context('tests', 72, 8, False)
            
            # Passed parameters checking function
            tests.stypy_localization = localization
            tests.stypy_type_of_self = None
            tests.stypy_type_store = module_type_store
            tests.stypy_function_name = 'tests'
            tests.stypy_param_names_list = []
            tests.stypy_varargs_param_name = None
            tests.stypy_kwargs_param_name = None
            tests.stypy_call_defaults = defaults
            tests.stypy_call_varargs = varargs
            tests.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'tests', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'tests', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'tests(...)' code ##################

            # Creating a generator
            
            # Call to FunctionTestCase(...): (line 73)
            # Processing the call arguments (line 73)

            @norecursion
            def _stypy_temp_lambda_96(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_96'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_96', 73, 44, True)
                # Passed parameters checking function
                _stypy_temp_lambda_96.stypy_localization = localization
                _stypy_temp_lambda_96.stypy_type_of_self = None
                _stypy_temp_lambda_96.stypy_type_store = module_type_store
                _stypy_temp_lambda_96.stypy_function_name = '_stypy_temp_lambda_96'
                _stypy_temp_lambda_96.stypy_param_names_list = []
                _stypy_temp_lambda_96.stypy_varargs_param_name = None
                _stypy_temp_lambda_96.stypy_kwargs_param_name = None
                _stypy_temp_lambda_96.stypy_call_defaults = defaults
                _stypy_temp_lambda_96.stypy_call_varargs = varargs
                _stypy_temp_lambda_96.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_96', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_96', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                # Getting the type of 'None' (line 73)
                None_208191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 52), 'None', False)
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'stypy_return_type', None_208191)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_96' in the type store
                # Getting the type of 'stypy_return_type' (line 73)
                stypy_return_type_208192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208192)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_96'
                return stypy_return_type_208192

            # Assigning a type to the variable '_stypy_temp_lambda_96' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), '_stypy_temp_lambda_96', _stypy_temp_lambda_96)
            # Getting the type of '_stypy_temp_lambda_96' (line 73)
            _stypy_temp_lambda_96_208193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), '_stypy_temp_lambda_96')
            # Processing the call keyword arguments (line 73)
            kwargs_208194 = {}
            # Getting the type of 'unittest' (line 73)
            unittest_208189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'unittest', False)
            # Obtaining the member 'FunctionTestCase' of a type (line 73)
            FunctionTestCase_208190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), unittest_208189, 'FunctionTestCase')
            # Calling FunctionTestCase(args, kwargs) (line 73)
            FunctionTestCase_call_result_208195 = invoke(stypy.reporting.localization.Localization(__file__, 73, 18), FunctionTestCase_208190, *[_stypy_temp_lambda_96_208193], **kwargs_208194)
            
            GeneratorType_208196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), GeneratorType_208196, FunctionTestCase_call_result_208195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', GeneratorType_208196)
            # Creating a generator
            
            # Call to FunctionTestCase(...): (line 74)
            # Processing the call arguments (line 74)

            @norecursion
            def _stypy_temp_lambda_97(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_97'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_97', 74, 44, True)
                # Passed parameters checking function
                _stypy_temp_lambda_97.stypy_localization = localization
                _stypy_temp_lambda_97.stypy_type_of_self = None
                _stypy_temp_lambda_97.stypy_type_store = module_type_store
                _stypy_temp_lambda_97.stypy_function_name = '_stypy_temp_lambda_97'
                _stypy_temp_lambda_97.stypy_param_names_list = []
                _stypy_temp_lambda_97.stypy_varargs_param_name = None
                _stypy_temp_lambda_97.stypy_kwargs_param_name = None
                _stypy_temp_lambda_97.stypy_call_defaults = defaults
                _stypy_temp_lambda_97.stypy_call_varargs = varargs
                _stypy_temp_lambda_97.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_97', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_97', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                # Getting the type of 'None' (line 74)
                None_208199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 52), 'None', False)
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 44), 'stypy_return_type', None_208199)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_97' in the type store
                # Getting the type of 'stypy_return_type' (line 74)
                stypy_return_type_208200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 44), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208200)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_97'
                return stypy_return_type_208200

            # Assigning a type to the variable '_stypy_temp_lambda_97' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 44), '_stypy_temp_lambda_97', _stypy_temp_lambda_97)
            # Getting the type of '_stypy_temp_lambda_97' (line 74)
            _stypy_temp_lambda_97_208201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 44), '_stypy_temp_lambda_97')
            # Processing the call keyword arguments (line 74)
            kwargs_208202 = {}
            # Getting the type of 'unittest' (line 74)
            unittest_208197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'unittest', False)
            # Obtaining the member 'FunctionTestCase' of a type (line 74)
            FunctionTestCase_208198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 18), unittest_208197, 'FunctionTestCase')
            # Calling FunctionTestCase(args, kwargs) (line 74)
            FunctionTestCase_call_result_208203 = invoke(stypy.reporting.localization.Localization(__file__, 74, 18), FunctionTestCase_208198, *[_stypy_temp_lambda_97_208201], **kwargs_208202)
            
            GeneratorType_208204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), GeneratorType_208204, FunctionTestCase_call_result_208203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'stypy_return_type', GeneratorType_208204)
            
            # ################# End of 'tests(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'tests' in the type store
            # Getting the type of 'stypy_return_type' (line 72)
            stypy_return_type_208205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208205)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'tests'
            return stypy_return_type_208205

        # Assigning a type to the variable 'tests' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'tests', tests)
        
        # Assigning a Call to a Name (line 76):
        
        # Call to TestSuite(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to tests(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_208209 = {}
        # Getting the type of 'tests' (line 76)
        tests_208208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 37), 'tests', False)
        # Calling tests(args, kwargs) (line 76)
        tests_call_result_208210 = invoke(stypy.reporting.localization.Localization(__file__, 76, 37), tests_208208, *[], **kwargs_208209)
        
        # Processing the call keyword arguments (line 76)
        kwargs_208211 = {}
        # Getting the type of 'unittest' (line 76)
        unittest_208206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 76)
        TestSuite_208207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 18), unittest_208206, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 76)
        TestSuite_call_result_208212 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), TestSuite_208207, *[tests_call_result_208210], **kwargs_208211)
        
        # Assigning a type to the variable 'suite_1' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'suite_1', TestSuite_call_result_208212)
        
        # Call to assertEqual(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to countTestCases(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_208217 = {}
        # Getting the type of 'suite_1' (line 77)
        suite_1_208215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'suite_1', False)
        # Obtaining the member 'countTestCases' of a type (line 77)
        countTestCases_208216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), suite_1_208215, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 77)
        countTestCases_call_result_208218 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), countTestCases_208216, *[], **kwargs_208217)
        
        int_208219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 51), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_208220 = {}
        # Getting the type of 'self' (line 77)
        self_208213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 77)
        assertEqual_208214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_208213, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 77)
        assertEqual_call_result_208221 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), assertEqual_208214, *[countTestCases_call_result_208218, int_208219], **kwargs_208220)
        
        
        # Assigning a Call to a Name (line 79):
        
        # Call to TestSuite(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'suite_1' (line 79)
        suite_1_208224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 'suite_1', False)
        # Processing the call keyword arguments (line 79)
        kwargs_208225 = {}
        # Getting the type of 'unittest' (line 79)
        unittest_208222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 79)
        TestSuite_208223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 18), unittest_208222, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 79)
        TestSuite_call_result_208226 = invoke(stypy.reporting.localization.Localization(__file__, 79, 18), TestSuite_208223, *[suite_1_208224], **kwargs_208225)
        
        # Assigning a type to the variable 'suite_2' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'suite_2', TestSuite_call_result_208226)
        
        # Call to assertEqual(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to countTestCases(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_208231 = {}
        # Getting the type of 'suite_2' (line 80)
        suite_2_208229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'suite_2', False)
        # Obtaining the member 'countTestCases' of a type (line 80)
        countTestCases_208230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 25), suite_2_208229, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 80)
        countTestCases_call_result_208232 = invoke(stypy.reporting.localization.Localization(__file__, 80, 25), countTestCases_208230, *[], **kwargs_208231)
        
        int_208233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 51), 'int')
        # Processing the call keyword arguments (line 80)
        kwargs_208234 = {}
        # Getting the type of 'self' (line 80)
        self_208227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 80)
        assertEqual_208228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_208227, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 80)
        assertEqual_call_result_208235 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assertEqual_208228, *[countTestCases_call_result_208232, int_208233], **kwargs_208234)
        
        
        # Assigning a Call to a Name (line 82):
        
        # Call to TestSuite(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Call to set(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'suite_1' (line 82)
        suite_1_208239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'suite_1', False)
        # Processing the call keyword arguments (line 82)
        kwargs_208240 = {}
        # Getting the type of 'set' (line 82)
        set_208238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'set', False)
        # Calling set(args, kwargs) (line 82)
        set_call_result_208241 = invoke(stypy.reporting.localization.Localization(__file__, 82, 37), set_208238, *[suite_1_208239], **kwargs_208240)
        
        # Processing the call keyword arguments (line 82)
        kwargs_208242 = {}
        # Getting the type of 'unittest' (line 82)
        unittest_208236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 82)
        TestSuite_208237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 18), unittest_208236, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 82)
        TestSuite_call_result_208243 = invoke(stypy.reporting.localization.Localization(__file__, 82, 18), TestSuite_208237, *[set_call_result_208241], **kwargs_208242)
        
        # Assigning a type to the variable 'suite_3' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'suite_3', TestSuite_call_result_208243)
        
        # Call to assertEqual(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to countTestCases(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_208248 = {}
        # Getting the type of 'suite_3' (line 83)
        suite_3_208246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 25), 'suite_3', False)
        # Obtaining the member 'countTestCases' of a type (line 83)
        countTestCases_208247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 25), suite_3_208246, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 83)
        countTestCases_call_result_208249 = invoke(stypy.reporting.localization.Localization(__file__, 83, 25), countTestCases_208247, *[], **kwargs_208248)
        
        int_208250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 51), 'int')
        # Processing the call keyword arguments (line 83)
        kwargs_208251 = {}
        # Getting the type of 'self' (line 83)
        self_208244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 83)
        assertEqual_208245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_208244, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 83)
        assertEqual_call_result_208252 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assertEqual_208245, *[countTestCases_call_result_208249, int_208250], **kwargs_208251)
        
        
        # ################# End of 'test_init__tests_from_any_iterable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_init__tests_from_any_iterable' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_208253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208253)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_init__tests_from_any_iterable'
        return stypy_return_type_208253


    @norecursion
    def test_init__TestSuite_instances_in_tests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_init__TestSuite_instances_in_tests'
        module_type_store = module_type_store.open_function_context('test_init__TestSuite_instances_in_tests', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_init__TestSuite_instances_in_tests')
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_init__TestSuite_instances_in_tests.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_init__TestSuite_instances_in_tests', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_init__TestSuite_instances_in_tests', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_init__TestSuite_instances_in_tests(...)' code ##################


        @norecursion
        def tests(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'tests'
            module_type_store = module_type_store.open_function_context('tests', 93, 8, False)
            
            # Passed parameters checking function
            tests.stypy_localization = localization
            tests.stypy_type_of_self = None
            tests.stypy_type_store = module_type_store
            tests.stypy_function_name = 'tests'
            tests.stypy_param_names_list = []
            tests.stypy_varargs_param_name = None
            tests.stypy_kwargs_param_name = None
            tests.stypy_call_defaults = defaults
            tests.stypy_call_varargs = varargs
            tests.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'tests', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'tests', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'tests(...)' code ##################

            
            # Assigning a Call to a Name (line 94):
            
            # Call to FunctionTestCase(...): (line 94)
            # Processing the call arguments (line 94)

            @norecursion
            def _stypy_temp_lambda_98(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_98'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_98', 94, 44, True)
                # Passed parameters checking function
                _stypy_temp_lambda_98.stypy_localization = localization
                _stypy_temp_lambda_98.stypy_type_of_self = None
                _stypy_temp_lambda_98.stypy_type_store = module_type_store
                _stypy_temp_lambda_98.stypy_function_name = '_stypy_temp_lambda_98'
                _stypy_temp_lambda_98.stypy_param_names_list = []
                _stypy_temp_lambda_98.stypy_varargs_param_name = None
                _stypy_temp_lambda_98.stypy_kwargs_param_name = None
                _stypy_temp_lambda_98.stypy_call_defaults = defaults
                _stypy_temp_lambda_98.stypy_call_varargs = varargs
                _stypy_temp_lambda_98.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_98', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_98', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                # Getting the type of 'None' (line 94)
                None_208256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 52), 'None', False)
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 94)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), 'stypy_return_type', None_208256)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_98' in the type store
                # Getting the type of 'stypy_return_type' (line 94)
                stypy_return_type_208257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208257)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_98'
                return stypy_return_type_208257

            # Assigning a type to the variable '_stypy_temp_lambda_98' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), '_stypy_temp_lambda_98', _stypy_temp_lambda_98)
            # Getting the type of '_stypy_temp_lambda_98' (line 94)
            _stypy_temp_lambda_98_208258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), '_stypy_temp_lambda_98')
            # Processing the call keyword arguments (line 94)
            kwargs_208259 = {}
            # Getting the type of 'unittest' (line 94)
            unittest_208254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'unittest', False)
            # Obtaining the member 'FunctionTestCase' of a type (line 94)
            FunctionTestCase_208255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 18), unittest_208254, 'FunctionTestCase')
            # Calling FunctionTestCase(args, kwargs) (line 94)
            FunctionTestCase_call_result_208260 = invoke(stypy.reporting.localization.Localization(__file__, 94, 18), FunctionTestCase_208255, *[_stypy_temp_lambda_98_208258], **kwargs_208259)
            
            # Assigning a type to the variable 'ftc' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'ftc', FunctionTestCase_call_result_208260)
            # Creating a generator
            
            # Call to TestSuite(...): (line 95)
            # Processing the call arguments (line 95)
            
            # Obtaining an instance of the builtin type 'list' (line 95)
            list_208263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 37), 'list')
            # Adding type elements to the builtin type 'list' instance (line 95)
            # Adding element type (line 95)
            # Getting the type of 'ftc' (line 95)
            ftc_208264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'ftc', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 37), list_208263, ftc_208264)
            
            # Processing the call keyword arguments (line 95)
            kwargs_208265 = {}
            # Getting the type of 'unittest' (line 95)
            unittest_208261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'unittest', False)
            # Obtaining the member 'TestSuite' of a type (line 95)
            TestSuite_208262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 18), unittest_208261, 'TestSuite')
            # Calling TestSuite(args, kwargs) (line 95)
            TestSuite_call_result_208266 = invoke(stypy.reporting.localization.Localization(__file__, 95, 18), TestSuite_208262, *[list_208263], **kwargs_208265)
            
            GeneratorType_208267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 12), GeneratorType_208267, TestSuite_call_result_208266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'stypy_return_type', GeneratorType_208267)
            # Creating a generator
            
            # Call to FunctionTestCase(...): (line 96)
            # Processing the call arguments (line 96)

            @norecursion
            def _stypy_temp_lambda_99(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_99'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_99', 96, 44, True)
                # Passed parameters checking function
                _stypy_temp_lambda_99.stypy_localization = localization
                _stypy_temp_lambda_99.stypy_type_of_self = None
                _stypy_temp_lambda_99.stypy_type_store = module_type_store
                _stypy_temp_lambda_99.stypy_function_name = '_stypy_temp_lambda_99'
                _stypy_temp_lambda_99.stypy_param_names_list = []
                _stypy_temp_lambda_99.stypy_varargs_param_name = None
                _stypy_temp_lambda_99.stypy_kwargs_param_name = None
                _stypy_temp_lambda_99.stypy_call_defaults = defaults
                _stypy_temp_lambda_99.stypy_call_varargs = varargs
                _stypy_temp_lambda_99.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_99', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_99', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                # Getting the type of 'None' (line 96)
                None_208270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 52), 'None', False)
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 44), 'stypy_return_type', None_208270)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_99' in the type store
                # Getting the type of 'stypy_return_type' (line 96)
                stypy_return_type_208271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 44), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208271)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_99'
                return stypy_return_type_208271

            # Assigning a type to the variable '_stypy_temp_lambda_99' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 44), '_stypy_temp_lambda_99', _stypy_temp_lambda_99)
            # Getting the type of '_stypy_temp_lambda_99' (line 96)
            _stypy_temp_lambda_99_208272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 44), '_stypy_temp_lambda_99')
            # Processing the call keyword arguments (line 96)
            kwargs_208273 = {}
            # Getting the type of 'unittest' (line 96)
            unittest_208268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'unittest', False)
            # Obtaining the member 'FunctionTestCase' of a type (line 96)
            FunctionTestCase_208269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 18), unittest_208268, 'FunctionTestCase')
            # Calling FunctionTestCase(args, kwargs) (line 96)
            FunctionTestCase_call_result_208274 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), FunctionTestCase_208269, *[_stypy_temp_lambda_99_208272], **kwargs_208273)
            
            GeneratorType_208275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 12), GeneratorType_208275, FunctionTestCase_call_result_208274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'stypy_return_type', GeneratorType_208275)
            
            # ################# End of 'tests(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'tests' in the type store
            # Getting the type of 'stypy_return_type' (line 93)
            stypy_return_type_208276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208276)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'tests'
            return stypy_return_type_208276

        # Assigning a type to the variable 'tests' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tests', tests)
        
        # Assigning a Call to a Name (line 98):
        
        # Call to TestSuite(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to tests(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_208280 = {}
        # Getting the type of 'tests' (line 98)
        tests_208279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'tests', False)
        # Calling tests(args, kwargs) (line 98)
        tests_call_result_208281 = invoke(stypy.reporting.localization.Localization(__file__, 98, 35), tests_208279, *[], **kwargs_208280)
        
        # Processing the call keyword arguments (line 98)
        kwargs_208282 = {}
        # Getting the type of 'unittest' (line 98)
        unittest_208277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 98)
        TestSuite_208278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), unittest_208277, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 98)
        TestSuite_call_result_208283 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), TestSuite_208278, *[tests_call_result_208281], **kwargs_208282)
        
        # Assigning a type to the variable 'suite' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'suite', TestSuite_call_result_208283)
        
        # Call to assertEqual(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to countTestCases(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_208288 = {}
        # Getting the type of 'suite' (line 99)
        suite_208286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 99)
        countTestCases_208287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 25), suite_208286, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 99)
        countTestCases_call_result_208289 = invoke(stypy.reporting.localization.Localization(__file__, 99, 25), countTestCases_208287, *[], **kwargs_208288)
        
        int_208290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 49), 'int')
        # Processing the call keyword arguments (line 99)
        kwargs_208291 = {}
        # Getting the type of 'self' (line 99)
        self_208284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 99)
        assertEqual_208285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_208284, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 99)
        assertEqual_call_result_208292 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), assertEqual_208285, *[countTestCases_call_result_208289, int_208290], **kwargs_208291)
        
        
        # ################# End of 'test_init__TestSuite_instances_in_tests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_init__TestSuite_instances_in_tests' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_208293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_init__TestSuite_instances_in_tests'
        return stypy_return_type_208293


    @norecursion
    def test_iter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_iter'
        module_type_store = module_type_store.open_function_context('test_iter', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_iter')
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_iter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_iter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_iter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_iter(...)' code ##################

        
        # Assigning a Call to a Name (line 106):
        
        # Call to FunctionTestCase(...): (line 106)
        # Processing the call arguments (line 106)

        @norecursion
        def _stypy_temp_lambda_100(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_100'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_100', 106, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_100.stypy_localization = localization
            _stypy_temp_lambda_100.stypy_type_of_self = None
            _stypy_temp_lambda_100.stypy_type_store = module_type_store
            _stypy_temp_lambda_100.stypy_function_name = '_stypy_temp_lambda_100'
            _stypy_temp_lambda_100.stypy_param_names_list = []
            _stypy_temp_lambda_100.stypy_varargs_param_name = None
            _stypy_temp_lambda_100.stypy_kwargs_param_name = None
            _stypy_temp_lambda_100.stypy_call_defaults = defaults
            _stypy_temp_lambda_100.stypy_call_varargs = varargs
            _stypy_temp_lambda_100.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_100', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_100', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 106)
            None_208296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 50), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), 'stypy_return_type', None_208296)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_100' in the type store
            # Getting the type of 'stypy_return_type' (line 106)
            stypy_return_type_208297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208297)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_100'
            return stypy_return_type_208297

        # Assigning a type to the variable '_stypy_temp_lambda_100' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), '_stypy_temp_lambda_100', _stypy_temp_lambda_100)
        # Getting the type of '_stypy_temp_lambda_100' (line 106)
        _stypy_temp_lambda_100_208298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), '_stypy_temp_lambda_100')
        # Processing the call keyword arguments (line 106)
        kwargs_208299 = {}
        # Getting the type of 'unittest' (line 106)
        unittest_208294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 106)
        FunctionTestCase_208295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), unittest_208294, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 106)
        FunctionTestCase_call_result_208300 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), FunctionTestCase_208295, *[_stypy_temp_lambda_100_208298], **kwargs_208299)
        
        # Assigning a type to the variable 'test1' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'test1', FunctionTestCase_call_result_208300)
        
        # Assigning a Call to a Name (line 107):
        
        # Call to FunctionTestCase(...): (line 107)
        # Processing the call arguments (line 107)

        @norecursion
        def _stypy_temp_lambda_101(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_101'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_101', 107, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_101.stypy_localization = localization
            _stypy_temp_lambda_101.stypy_type_of_self = None
            _stypy_temp_lambda_101.stypy_type_store = module_type_store
            _stypy_temp_lambda_101.stypy_function_name = '_stypy_temp_lambda_101'
            _stypy_temp_lambda_101.stypy_param_names_list = []
            _stypy_temp_lambda_101.stypy_varargs_param_name = None
            _stypy_temp_lambda_101.stypy_kwargs_param_name = None
            _stypy_temp_lambda_101.stypy_call_defaults = defaults
            _stypy_temp_lambda_101.stypy_call_varargs = varargs
            _stypy_temp_lambda_101.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_101', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_101', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 107)
            None_208303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 50), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 42), 'stypy_return_type', None_208303)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_101' in the type store
            # Getting the type of 'stypy_return_type' (line 107)
            stypy_return_type_208304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208304)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_101'
            return stypy_return_type_208304

        # Assigning a type to the variable '_stypy_temp_lambda_101' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 42), '_stypy_temp_lambda_101', _stypy_temp_lambda_101)
        # Getting the type of '_stypy_temp_lambda_101' (line 107)
        _stypy_temp_lambda_101_208305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 42), '_stypy_temp_lambda_101')
        # Processing the call keyword arguments (line 107)
        kwargs_208306 = {}
        # Getting the type of 'unittest' (line 107)
        unittest_208301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 107)
        FunctionTestCase_208302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), unittest_208301, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 107)
        FunctionTestCase_call_result_208307 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), FunctionTestCase_208302, *[_stypy_temp_lambda_101_208305], **kwargs_208306)
        
        # Assigning a type to the variable 'test2' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'test2', FunctionTestCase_call_result_208307)
        
        # Assigning a Call to a Name (line 108):
        
        # Call to TestSuite(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_208310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'test1' (line 108)
        test1_208311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 36), 'test1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 36), tuple_208310, test1_208311)
        # Adding element type (line 108)
        # Getting the type of 'test2' (line 108)
        test2_208312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 43), 'test2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 36), tuple_208310, test2_208312)
        
        # Processing the call keyword arguments (line 108)
        kwargs_208313 = {}
        # Getting the type of 'unittest' (line 108)
        unittest_208308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 108)
        TestSuite_208309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), unittest_208308, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 108)
        TestSuite_call_result_208314 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), TestSuite_208309, *[tuple_208310], **kwargs_208313)
        
        # Assigning a type to the variable 'suite' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'suite', TestSuite_call_result_208314)
        
        # Call to assertEqual(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to list(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'suite' (line 110)
        suite_208318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 30), 'suite', False)
        # Processing the call keyword arguments (line 110)
        kwargs_208319 = {}
        # Getting the type of 'list' (line 110)
        list_208317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'list', False)
        # Calling list(args, kwargs) (line 110)
        list_call_result_208320 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), list_208317, *[suite_208318], **kwargs_208319)
        
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_208321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        # Getting the type of 'test1' (line 110)
        test1_208322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 39), 'test1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 38), list_208321, test1_208322)
        # Adding element type (line 110)
        # Getting the type of 'test2' (line 110)
        test2_208323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 46), 'test2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 38), list_208321, test2_208323)
        
        # Processing the call keyword arguments (line 110)
        kwargs_208324 = {}
        # Getting the type of 'self' (line 110)
        self_208315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 110)
        assertEqual_208316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_208315, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 110)
        assertEqual_call_result_208325 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assertEqual_208316, *[list_call_result_208320, list_208321], **kwargs_208324)
        
        
        # ################# End of 'test_iter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_iter' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_208326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208326)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_iter'
        return stypy_return_type_208326


    @norecursion
    def test_countTestCases_zero_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_countTestCases_zero_simple'
        module_type_store = module_type_store.open_function_context('test_countTestCases_zero_simple', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_countTestCases_zero_simple')
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_countTestCases_zero_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_countTestCases_zero_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_countTestCases_zero_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_countTestCases_zero_simple(...)' code ##################

        
        # Assigning a Call to a Name (line 118):
        
        # Call to TestSuite(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_208329 = {}
        # Getting the type of 'unittest' (line 118)
        unittest_208327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 118)
        TestSuite_208328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), unittest_208327, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 118)
        TestSuite_call_result_208330 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), TestSuite_208328, *[], **kwargs_208329)
        
        # Assigning a type to the variable 'suite' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'suite', TestSuite_call_result_208330)
        
        # Call to assertEqual(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to countTestCases(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_208335 = {}
        # Getting the type of 'suite' (line 120)
        suite_208333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 120)
        countTestCases_208334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 25), suite_208333, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 120)
        countTestCases_call_result_208336 = invoke(stypy.reporting.localization.Localization(__file__, 120, 25), countTestCases_208334, *[], **kwargs_208335)
        
        int_208337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 49), 'int')
        # Processing the call keyword arguments (line 120)
        kwargs_208338 = {}
        # Getting the type of 'self' (line 120)
        self_208331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 120)
        assertEqual_208332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_208331, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 120)
        assertEqual_call_result_208339 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), assertEqual_208332, *[countTestCases_call_result_208336, int_208337], **kwargs_208338)
        
        
        # ################# End of 'test_countTestCases_zero_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_countTestCases_zero_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_208340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208340)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_countTestCases_zero_simple'
        return stypy_return_type_208340


    @norecursion
    def test_countTestCases_zero_nested(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_countTestCases_zero_nested'
        module_type_store = module_type_store.open_function_context('test_countTestCases_zero_nested', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_countTestCases_zero_nested')
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_countTestCases_zero_nested.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_countTestCases_zero_nested', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_countTestCases_zero_nested', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_countTestCases_zero_nested(...)' code ##################

        # Declaration of the 'Test1' class
        # Getting the type of 'unittest' (line 129)
        unittest_208341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 129)
        TestCase_208342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 20), unittest_208341, 'TestCase')

        class Test1(TestCase_208342, ):

            @norecursion
            def test(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test'
                module_type_store = module_type_store.open_function_context('test', 130, 12, False)
                # Assigning a type to the variable 'self' (line 131)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test1.test.__dict__.__setitem__('stypy_localization', localization)
                Test1.test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test1.test.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test1.test.__dict__.__setitem__('stypy_function_name', 'Test1.test')
                Test1.test.__dict__.__setitem__('stypy_param_names_list', [])
                Test1.test.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test1.test.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test1.test.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test1.test.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test1.test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test1.test.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test1.test', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test(...)' code ##################

                pass
                
                # ################# End of 'test(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test' in the type store
                # Getting the type of 'stypy_return_type' (line 130)
                stypy_return_type_208343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208343)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test'
                return stypy_return_type_208343

        
        # Assigning a type to the variable 'Test1' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'Test1', Test1)
        
        # Assigning a Call to a Name (line 133):
        
        # Call to TestSuite(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_208346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        
        # Call to TestSuite(...): (line 133)
        # Processing the call keyword arguments (line 133)
        kwargs_208349 = {}
        # Getting the type of 'unittest' (line 133)
        unittest_208347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 133)
        TestSuite_208348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 36), unittest_208347, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 133)
        TestSuite_call_result_208350 = invoke(stypy.reporting.localization.Localization(__file__, 133, 36), TestSuite_208348, *[], **kwargs_208349)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 35), list_208346, TestSuite_call_result_208350)
        
        # Processing the call keyword arguments (line 133)
        kwargs_208351 = {}
        # Getting the type of 'unittest' (line 133)
        unittest_208344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 133)
        TestSuite_208345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), unittest_208344, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 133)
        TestSuite_call_result_208352 = invoke(stypy.reporting.localization.Localization(__file__, 133, 16), TestSuite_208345, *[list_208346], **kwargs_208351)
        
        # Assigning a type to the variable 'suite' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'suite', TestSuite_call_result_208352)
        
        # Call to assertEqual(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Call to countTestCases(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_208357 = {}
        # Getting the type of 'suite' (line 135)
        suite_208355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 135)
        countTestCases_208356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 25), suite_208355, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 135)
        countTestCases_call_result_208358 = invoke(stypy.reporting.localization.Localization(__file__, 135, 25), countTestCases_208356, *[], **kwargs_208357)
        
        int_208359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 49), 'int')
        # Processing the call keyword arguments (line 135)
        kwargs_208360 = {}
        # Getting the type of 'self' (line 135)
        self_208353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 135)
        assertEqual_208354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), self_208353, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 135)
        assertEqual_call_result_208361 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), assertEqual_208354, *[countTestCases_call_result_208358, int_208359], **kwargs_208360)
        
        
        # ################# End of 'test_countTestCases_zero_nested(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_countTestCases_zero_nested' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_208362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208362)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_countTestCases_zero_nested'
        return stypy_return_type_208362


    @norecursion
    def test_countTestCases_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_countTestCases_simple'
        module_type_store = module_type_store.open_function_context('test_countTestCases_simple', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_countTestCases_simple')
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_countTestCases_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_countTestCases_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_countTestCases_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_countTestCases_simple(...)' code ##################

        
        # Assigning a Call to a Name (line 141):
        
        # Call to FunctionTestCase(...): (line 141)
        # Processing the call arguments (line 141)

        @norecursion
        def _stypy_temp_lambda_102(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_102'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_102', 141, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_102.stypy_localization = localization
            _stypy_temp_lambda_102.stypy_type_of_self = None
            _stypy_temp_lambda_102.stypy_type_store = module_type_store
            _stypy_temp_lambda_102.stypy_function_name = '_stypy_temp_lambda_102'
            _stypy_temp_lambda_102.stypy_param_names_list = []
            _stypy_temp_lambda_102.stypy_varargs_param_name = None
            _stypy_temp_lambda_102.stypy_kwargs_param_name = None
            _stypy_temp_lambda_102.stypy_call_defaults = defaults
            _stypy_temp_lambda_102.stypy_call_varargs = varargs
            _stypy_temp_lambda_102.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_102', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_102', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 141)
            None_208365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 50), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), 'stypy_return_type', None_208365)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_102' in the type store
            # Getting the type of 'stypy_return_type' (line 141)
            stypy_return_type_208366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208366)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_102'
            return stypy_return_type_208366

        # Assigning a type to the variable '_stypy_temp_lambda_102' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), '_stypy_temp_lambda_102', _stypy_temp_lambda_102)
        # Getting the type of '_stypy_temp_lambda_102' (line 141)
        _stypy_temp_lambda_102_208367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), '_stypy_temp_lambda_102')
        # Processing the call keyword arguments (line 141)
        kwargs_208368 = {}
        # Getting the type of 'unittest' (line 141)
        unittest_208363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 141)
        FunctionTestCase_208364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), unittest_208363, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 141)
        FunctionTestCase_call_result_208369 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), FunctionTestCase_208364, *[_stypy_temp_lambda_102_208367], **kwargs_208368)
        
        # Assigning a type to the variable 'test1' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'test1', FunctionTestCase_call_result_208369)
        
        # Assigning a Call to a Name (line 142):
        
        # Call to FunctionTestCase(...): (line 142)
        # Processing the call arguments (line 142)

        @norecursion
        def _stypy_temp_lambda_103(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_103'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_103', 142, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_103.stypy_localization = localization
            _stypy_temp_lambda_103.stypy_type_of_self = None
            _stypy_temp_lambda_103.stypy_type_store = module_type_store
            _stypy_temp_lambda_103.stypy_function_name = '_stypy_temp_lambda_103'
            _stypy_temp_lambda_103.stypy_param_names_list = []
            _stypy_temp_lambda_103.stypy_varargs_param_name = None
            _stypy_temp_lambda_103.stypy_kwargs_param_name = None
            _stypy_temp_lambda_103.stypy_call_defaults = defaults
            _stypy_temp_lambda_103.stypy_call_varargs = varargs
            _stypy_temp_lambda_103.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_103', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_103', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 142)
            None_208372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 50), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'stypy_return_type', None_208372)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_103' in the type store
            # Getting the type of 'stypy_return_type' (line 142)
            stypy_return_type_208373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208373)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_103'
            return stypy_return_type_208373

        # Assigning a type to the variable '_stypy_temp_lambda_103' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), '_stypy_temp_lambda_103', _stypy_temp_lambda_103)
        # Getting the type of '_stypy_temp_lambda_103' (line 142)
        _stypy_temp_lambda_103_208374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), '_stypy_temp_lambda_103')
        # Processing the call keyword arguments (line 142)
        kwargs_208375 = {}
        # Getting the type of 'unittest' (line 142)
        unittest_208370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 142)
        FunctionTestCase_208371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 16), unittest_208370, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 142)
        FunctionTestCase_call_result_208376 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), FunctionTestCase_208371, *[_stypy_temp_lambda_103_208374], **kwargs_208375)
        
        # Assigning a type to the variable 'test2' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'test2', FunctionTestCase_call_result_208376)
        
        # Assigning a Call to a Name (line 143):
        
        # Call to TestSuite(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining an instance of the builtin type 'tuple' (line 143)
        tuple_208379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 143)
        # Adding element type (line 143)
        # Getting the type of 'test1' (line 143)
        test1_208380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 36), 'test1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 36), tuple_208379, test1_208380)
        # Adding element type (line 143)
        # Getting the type of 'test2' (line 143)
        test2_208381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 43), 'test2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 36), tuple_208379, test2_208381)
        
        # Processing the call keyword arguments (line 143)
        kwargs_208382 = {}
        # Getting the type of 'unittest' (line 143)
        unittest_208377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 143)
        TestSuite_208378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), unittest_208377, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 143)
        TestSuite_call_result_208383 = invoke(stypy.reporting.localization.Localization(__file__, 143, 16), TestSuite_208378, *[tuple_208379], **kwargs_208382)
        
        # Assigning a type to the variable 'suite' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'suite', TestSuite_call_result_208383)
        
        # Call to assertEqual(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Call to countTestCases(...): (line 145)
        # Processing the call keyword arguments (line 145)
        kwargs_208388 = {}
        # Getting the type of 'suite' (line 145)
        suite_208386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 145)
        countTestCases_208387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), suite_208386, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 145)
        countTestCases_call_result_208389 = invoke(stypy.reporting.localization.Localization(__file__, 145, 25), countTestCases_208387, *[], **kwargs_208388)
        
        int_208390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 49), 'int')
        # Processing the call keyword arguments (line 145)
        kwargs_208391 = {}
        # Getting the type of 'self' (line 145)
        self_208384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 145)
        assertEqual_208385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_208384, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 145)
        assertEqual_call_result_208392 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), assertEqual_208385, *[countTestCases_call_result_208389, int_208390], **kwargs_208391)
        
        
        # ################# End of 'test_countTestCases_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_countTestCases_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_208393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208393)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_countTestCases_simple'
        return stypy_return_type_208393


    @norecursion
    def test_countTestCases_nested(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_countTestCases_nested'
        module_type_store = module_type_store.open_function_context('test_countTestCases_nested', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_countTestCases_nested')
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_countTestCases_nested.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_countTestCases_nested', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_countTestCases_nested', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_countTestCases_nested(...)' code ##################

        # Declaration of the 'Test1' class
        # Getting the type of 'unittest' (line 153)
        unittest_208394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 153)
        TestCase_208395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 20), unittest_208394, 'TestCase')

        class Test1(TestCase_208395, ):

            @norecursion
            def test1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test1'
                module_type_store = module_type_store.open_function_context('test1', 154, 12, False)
                # Assigning a type to the variable 'self' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test1.test1.__dict__.__setitem__('stypy_localization', localization)
                Test1.test1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test1.test1.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test1.test1.__dict__.__setitem__('stypy_function_name', 'Test1.test1')
                Test1.test1.__dict__.__setitem__('stypy_param_names_list', [])
                Test1.test1.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test1.test1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test1.test1.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test1.test1.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test1.test1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test1.test1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test1.test1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test1(...)' code ##################

                pass
                
                # ################# End of 'test1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test1' in the type store
                # Getting the type of 'stypy_return_type' (line 154)
                stypy_return_type_208396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208396)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test1'
                return stypy_return_type_208396


            @norecursion
            def test2(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test2'
                module_type_store = module_type_store.open_function_context('test2', 155, 12, False)
                # Assigning a type to the variable 'self' (line 156)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test1.test2.__dict__.__setitem__('stypy_localization', localization)
                Test1.test2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test1.test2.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test1.test2.__dict__.__setitem__('stypy_function_name', 'Test1.test2')
                Test1.test2.__dict__.__setitem__('stypy_param_names_list', [])
                Test1.test2.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test1.test2.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test1.test2.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test1.test2.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test1.test2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test1.test2.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test1.test2', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test2', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test2(...)' code ##################

                pass
                
                # ################# End of 'test2(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test2' in the type store
                # Getting the type of 'stypy_return_type' (line 155)
                stypy_return_type_208397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208397)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test2'
                return stypy_return_type_208397

        
        # Assigning a type to the variable 'Test1' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'Test1', Test1)
        
        # Assigning a Call to a Name (line 157):
        
        # Call to FunctionTestCase(...): (line 157)
        # Processing the call arguments (line 157)

        @norecursion
        def _stypy_temp_lambda_104(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_104'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_104', 157, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_104.stypy_localization = localization
            _stypy_temp_lambda_104.stypy_type_of_self = None
            _stypy_temp_lambda_104.stypy_type_store = module_type_store
            _stypy_temp_lambda_104.stypy_function_name = '_stypy_temp_lambda_104'
            _stypy_temp_lambda_104.stypy_param_names_list = []
            _stypy_temp_lambda_104.stypy_varargs_param_name = None
            _stypy_temp_lambda_104.stypy_kwargs_param_name = None
            _stypy_temp_lambda_104.stypy_call_defaults = defaults
            _stypy_temp_lambda_104.stypy_call_varargs = varargs
            _stypy_temp_lambda_104.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_104', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_104', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 157)
            None_208400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 50), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 42), 'stypy_return_type', None_208400)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_104' in the type store
            # Getting the type of 'stypy_return_type' (line 157)
            stypy_return_type_208401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208401)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_104'
            return stypy_return_type_208401

        # Assigning a type to the variable '_stypy_temp_lambda_104' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 42), '_stypy_temp_lambda_104', _stypy_temp_lambda_104)
        # Getting the type of '_stypy_temp_lambda_104' (line 157)
        _stypy_temp_lambda_104_208402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 42), '_stypy_temp_lambda_104')
        # Processing the call keyword arguments (line 157)
        kwargs_208403 = {}
        # Getting the type of 'unittest' (line 157)
        unittest_208398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 157)
        FunctionTestCase_208399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), unittest_208398, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 157)
        FunctionTestCase_call_result_208404 = invoke(stypy.reporting.localization.Localization(__file__, 157, 16), FunctionTestCase_208399, *[_stypy_temp_lambda_104_208402], **kwargs_208403)
        
        # Assigning a type to the variable 'test2' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'test2', FunctionTestCase_call_result_208404)
        
        # Assigning a Call to a Name (line 158):
        
        # Call to FunctionTestCase(...): (line 158)
        # Processing the call arguments (line 158)

        @norecursion
        def _stypy_temp_lambda_105(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_105'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_105', 158, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_105.stypy_localization = localization
            _stypy_temp_lambda_105.stypy_type_of_self = None
            _stypy_temp_lambda_105.stypy_type_store = module_type_store
            _stypy_temp_lambda_105.stypy_function_name = '_stypy_temp_lambda_105'
            _stypy_temp_lambda_105.stypy_param_names_list = []
            _stypy_temp_lambda_105.stypy_varargs_param_name = None
            _stypy_temp_lambda_105.stypy_kwargs_param_name = None
            _stypy_temp_lambda_105.stypy_call_defaults = defaults
            _stypy_temp_lambda_105.stypy_call_varargs = varargs
            _stypy_temp_lambda_105.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_105', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_105', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 158)
            None_208407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 50), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 42), 'stypy_return_type', None_208407)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_105' in the type store
            # Getting the type of 'stypy_return_type' (line 158)
            stypy_return_type_208408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208408)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_105'
            return stypy_return_type_208408

        # Assigning a type to the variable '_stypy_temp_lambda_105' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 42), '_stypy_temp_lambda_105', _stypy_temp_lambda_105)
        # Getting the type of '_stypy_temp_lambda_105' (line 158)
        _stypy_temp_lambda_105_208409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 42), '_stypy_temp_lambda_105')
        # Processing the call keyword arguments (line 158)
        kwargs_208410 = {}
        # Getting the type of 'unittest' (line 158)
        unittest_208405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'unittest', False)
        # Obtaining the member 'FunctionTestCase' of a type (line 158)
        FunctionTestCase_208406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), unittest_208405, 'FunctionTestCase')
        # Calling FunctionTestCase(args, kwargs) (line 158)
        FunctionTestCase_call_result_208411 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), FunctionTestCase_208406, *[_stypy_temp_lambda_105_208409], **kwargs_208410)
        
        # Assigning a type to the variable 'test3' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'test3', FunctionTestCase_call_result_208411)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to TestSuite(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Obtaining an instance of the builtin type 'tuple' (line 159)
        tuple_208414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 159)
        # Adding element type (line 159)
        
        # Call to Test1(...): (line 159)
        # Processing the call arguments (line 159)
        str_208416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 42), 'str', 'test2')
        # Processing the call keyword arguments (line 159)
        kwargs_208417 = {}
        # Getting the type of 'Test1' (line 159)
        Test1_208415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 36), 'Test1', False)
        # Calling Test1(args, kwargs) (line 159)
        Test1_call_result_208418 = invoke(stypy.reporting.localization.Localization(__file__, 159, 36), Test1_208415, *[str_208416], **kwargs_208417)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 36), tuple_208414, Test1_call_result_208418)
        # Adding element type (line 159)
        # Getting the type of 'test2' (line 159)
        test2_208419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 52), 'test2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 36), tuple_208414, test2_208419)
        
        # Processing the call keyword arguments (line 159)
        kwargs_208420 = {}
        # Getting the type of 'unittest' (line 159)
        unittest_208412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 159)
        TestSuite_208413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 16), unittest_208412, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 159)
        TestSuite_call_result_208421 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), TestSuite_208413, *[tuple_208414], **kwargs_208420)
        
        # Assigning a type to the variable 'child' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'child', TestSuite_call_result_208421)
        
        # Assigning a Call to a Name (line 160):
        
        # Call to TestSuite(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Obtaining an instance of the builtin type 'tuple' (line 160)
        tuple_208424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 160)
        # Adding element type (line 160)
        # Getting the type of 'test3' (line 160)
        test3_208425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'test3', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 37), tuple_208424, test3_208425)
        # Adding element type (line 160)
        # Getting the type of 'child' (line 160)
        child_208426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 44), 'child', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 37), tuple_208424, child_208426)
        # Adding element type (line 160)
        
        # Call to Test1(...): (line 160)
        # Processing the call arguments (line 160)
        str_208428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 57), 'str', 'test1')
        # Processing the call keyword arguments (line 160)
        kwargs_208429 = {}
        # Getting the type of 'Test1' (line 160)
        Test1_208427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 51), 'Test1', False)
        # Calling Test1(args, kwargs) (line 160)
        Test1_call_result_208430 = invoke(stypy.reporting.localization.Localization(__file__, 160, 51), Test1_208427, *[str_208428], **kwargs_208429)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 37), tuple_208424, Test1_call_result_208430)
        
        # Processing the call keyword arguments (line 160)
        kwargs_208431 = {}
        # Getting the type of 'unittest' (line 160)
        unittest_208422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 160)
        TestSuite_208423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 17), unittest_208422, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 160)
        TestSuite_call_result_208432 = invoke(stypy.reporting.localization.Localization(__file__, 160, 17), TestSuite_208423, *[tuple_208424], **kwargs_208431)
        
        # Assigning a type to the variable 'parent' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'parent', TestSuite_call_result_208432)
        
        # Call to assertEqual(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Call to countTestCases(...): (line 162)
        # Processing the call keyword arguments (line 162)
        kwargs_208437 = {}
        # Getting the type of 'parent' (line 162)
        parent_208435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'parent', False)
        # Obtaining the member 'countTestCases' of a type (line 162)
        countTestCases_208436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 25), parent_208435, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 162)
        countTestCases_call_result_208438 = invoke(stypy.reporting.localization.Localization(__file__, 162, 25), countTestCases_208436, *[], **kwargs_208437)
        
        int_208439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 50), 'int')
        # Processing the call keyword arguments (line 162)
        kwargs_208440 = {}
        # Getting the type of 'self' (line 162)
        self_208433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 162)
        assertEqual_208434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_208433, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 162)
        assertEqual_call_result_208441 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), assertEqual_208434, *[countTestCases_call_result_208438, int_208439], **kwargs_208440)
        
        
        # ################# End of 'test_countTestCases_nested(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_countTestCases_nested' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_208442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208442)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_countTestCases_nested'
        return stypy_return_type_208442


    @norecursion
    def test_run__empty_suite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run__empty_suite'
        module_type_store = module_type_store.open_function_context('test_run__empty_suite', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_run__empty_suite')
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_run__empty_suite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_run__empty_suite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run__empty_suite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run__empty_suite(...)' code ##################

        
        # Assigning a List to a Name (line 169):
        
        # Obtaining an instance of the builtin type 'list' (line 169)
        list_208443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 169)
        
        # Assigning a type to the variable 'events' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'events', list_208443)
        
        # Assigning a Call to a Name (line 170):
        
        # Call to LoggingResult(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'events' (line 170)
        events_208445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'events', False)
        # Processing the call keyword arguments (line 170)
        kwargs_208446 = {}
        # Getting the type of 'LoggingResult' (line 170)
        LoggingResult_208444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 170)
        LoggingResult_call_result_208447 = invoke(stypy.reporting.localization.Localization(__file__, 170, 17), LoggingResult_208444, *[events_208445], **kwargs_208446)
        
        # Assigning a type to the variable 'result' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'result', LoggingResult_call_result_208447)
        
        # Assigning a Call to a Name (line 172):
        
        # Call to TestSuite(...): (line 172)
        # Processing the call keyword arguments (line 172)
        kwargs_208450 = {}
        # Getting the type of 'unittest' (line 172)
        unittest_208448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 172)
        TestSuite_208449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), unittest_208448, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 172)
        TestSuite_call_result_208451 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), TestSuite_208449, *[], **kwargs_208450)
        
        # Assigning a type to the variable 'suite' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'suite', TestSuite_call_result_208451)
        
        # Call to run(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'result' (line 174)
        result_208454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'result', False)
        # Processing the call keyword arguments (line 174)
        kwargs_208455 = {}
        # Getting the type of 'suite' (line 174)
        suite_208452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'suite', False)
        # Obtaining the member 'run' of a type (line 174)
        run_208453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), suite_208452, 'run')
        # Calling run(args, kwargs) (line 174)
        run_call_result_208456 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), run_208453, *[result_208454], **kwargs_208455)
        
        
        # Call to assertEqual(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'events' (line 176)
        events_208459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'events', False)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_208460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        
        # Processing the call keyword arguments (line 176)
        kwargs_208461 = {}
        # Getting the type of 'self' (line 176)
        self_208457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 176)
        assertEqual_208458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_208457, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 176)
        assertEqual_call_result_208462 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assertEqual_208458, *[events_208459, list_208460], **kwargs_208461)
        
        
        # ################# End of 'test_run__empty_suite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run__empty_suite' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_208463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208463)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run__empty_suite'
        return stypy_return_type_208463


    @norecursion
    def test_run__requires_result(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run__requires_result'
        module_type_store = module_type_store.open_function_context('test_run__requires_result', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_run__requires_result')
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_run__requires_result.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_run__requires_result', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run__requires_result', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run__requires_result(...)' code ##################

        
        # Assigning a Call to a Name (line 181):
        
        # Call to TestSuite(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_208466 = {}
        # Getting the type of 'unittest' (line 181)
        unittest_208464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 181)
        TestSuite_208465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), unittest_208464, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 181)
        TestSuite_call_result_208467 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), TestSuite_208465, *[], **kwargs_208466)
        
        # Assigning a type to the variable 'suite' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'suite', TestSuite_call_result_208467)
        
        
        # SSA begins for try-except statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to run(...): (line 184)
        # Processing the call keyword arguments (line 184)
        kwargs_208470 = {}
        # Getting the type of 'suite' (line 184)
        suite_208468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'suite', False)
        # Obtaining the member 'run' of a type (line 184)
        run_208469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 12), suite_208468, 'run')
        # Calling run(args, kwargs) (line 184)
        run_call_result_208471 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), run_208469, *[], **kwargs_208470)
        
        # SSA branch for the except part of a try statement (line 183)
        # SSA branch for the except 'TypeError' branch of a try statement (line 183)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 183)
        module_type_store.open_ssa_branch('except else')
        
        # Call to fail(...): (line 188)
        # Processing the call arguments (line 188)
        str_208474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 22), 'str', 'Failed to raise TypeError')
        # Processing the call keyword arguments (line 188)
        kwargs_208475 = {}
        # Getting the type of 'self' (line 188)
        self_208472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 188)
        fail_208473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), self_208472, 'fail')
        # Calling fail(args, kwargs) (line 188)
        fail_call_result_208476 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), fail_208473, *[str_208474], **kwargs_208475)
        
        # SSA join for try-except statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_run__requires_result(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run__requires_result' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_208477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208477)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run__requires_result'
        return stypy_return_type_208477


    @norecursion
    def test_run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_run'
        module_type_store = module_type_store.open_function_context('test_run', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_run')
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_run(...)' code ##################

        
        # Assigning a List to a Name (line 193):
        
        # Obtaining an instance of the builtin type 'list' (line 193)
        list_208478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 193)
        
        # Assigning a type to the variable 'events' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'events', list_208478)
        
        # Assigning a Call to a Name (line 194):
        
        # Call to LoggingResult(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'events' (line 194)
        events_208480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 31), 'events', False)
        # Processing the call keyword arguments (line 194)
        kwargs_208481 = {}
        # Getting the type of 'LoggingResult' (line 194)
        LoggingResult_208479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'LoggingResult', False)
        # Calling LoggingResult(args, kwargs) (line 194)
        LoggingResult_call_result_208482 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), LoggingResult_208479, *[events_208480], **kwargs_208481)
        
        # Assigning a type to the variable 'result' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'result', LoggingResult_call_result_208482)
        # Declaration of the 'LoggingCase' class
        # Getting the type of 'unittest' (line 196)
        unittest_208483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 26), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 196)
        TestCase_208484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 26), unittest_208483, 'TestCase')

        class LoggingCase(TestCase_208484, ):

            @norecursion
            def run(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'run'
                module_type_store = module_type_store.open_function_context('run', 197, 12, False)
                # Assigning a type to the variable 'self' (line 198)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                LoggingCase.run.__dict__.__setitem__('stypy_localization', localization)
                LoggingCase.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                LoggingCase.run.__dict__.__setitem__('stypy_type_store', module_type_store)
                LoggingCase.run.__dict__.__setitem__('stypy_function_name', 'LoggingCase.run')
                LoggingCase.run.__dict__.__setitem__('stypy_param_names_list', ['result'])
                LoggingCase.run.__dict__.__setitem__('stypy_varargs_param_name', None)
                LoggingCase.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
                LoggingCase.run.__dict__.__setitem__('stypy_call_defaults', defaults)
                LoggingCase.run.__dict__.__setitem__('stypy_call_varargs', varargs)
                LoggingCase.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                LoggingCase.run.__dict__.__setitem__('stypy_declared_arg_number', 2)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingCase.run', ['result'], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 198)
                # Processing the call arguments (line 198)
                str_208487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 30), 'str', 'run %s')
                # Getting the type of 'self' (line 198)
                self_208488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 41), 'self', False)
                # Obtaining the member '_testMethodName' of a type (line 198)
                _testMethodName_208489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 41), self_208488, '_testMethodName')
                # Applying the binary operator '%' (line 198)
                result_mod_208490 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 30), '%', str_208487, _testMethodName_208489)
                
                # Processing the call keyword arguments (line 198)
                kwargs_208491 = {}
                # Getting the type of 'events' (line 198)
                events_208485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'events', False)
                # Obtaining the member 'append' of a type (line 198)
                append_208486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 16), events_208485, 'append')
                # Calling append(args, kwargs) (line 198)
                append_call_result_208492 = invoke(stypy.reporting.localization.Localization(__file__, 198, 16), append_208486, *[result_mod_208490], **kwargs_208491)
                
                
                # ################# End of 'run(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'run' in the type store
                # Getting the type of 'stypy_return_type' (line 197)
                stypy_return_type_208493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208493)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'run'
                return stypy_return_type_208493


            @norecursion
            def test1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test1'
                module_type_store = module_type_store.open_function_context('test1', 200, 12, False)
                # Assigning a type to the variable 'self' (line 201)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                LoggingCase.test1.__dict__.__setitem__('stypy_localization', localization)
                LoggingCase.test1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                LoggingCase.test1.__dict__.__setitem__('stypy_type_store', module_type_store)
                LoggingCase.test1.__dict__.__setitem__('stypy_function_name', 'LoggingCase.test1')
                LoggingCase.test1.__dict__.__setitem__('stypy_param_names_list', [])
                LoggingCase.test1.__dict__.__setitem__('stypy_varargs_param_name', None)
                LoggingCase.test1.__dict__.__setitem__('stypy_kwargs_param_name', None)
                LoggingCase.test1.__dict__.__setitem__('stypy_call_defaults', defaults)
                LoggingCase.test1.__dict__.__setitem__('stypy_call_varargs', varargs)
                LoggingCase.test1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                LoggingCase.test1.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingCase.test1', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test1', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test1(...)' code ##################

                pass
                
                # ################# End of 'test1(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test1' in the type store
                # Getting the type of 'stypy_return_type' (line 200)
                stypy_return_type_208494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208494)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test1'
                return stypy_return_type_208494


            @norecursion
            def test2(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test2'
                module_type_store = module_type_store.open_function_context('test2', 201, 12, False)
                # Assigning a type to the variable 'self' (line 202)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                LoggingCase.test2.__dict__.__setitem__('stypy_localization', localization)
                LoggingCase.test2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                LoggingCase.test2.__dict__.__setitem__('stypy_type_store', module_type_store)
                LoggingCase.test2.__dict__.__setitem__('stypy_function_name', 'LoggingCase.test2')
                LoggingCase.test2.__dict__.__setitem__('stypy_param_names_list', [])
                LoggingCase.test2.__dict__.__setitem__('stypy_varargs_param_name', None)
                LoggingCase.test2.__dict__.__setitem__('stypy_kwargs_param_name', None)
                LoggingCase.test2.__dict__.__setitem__('stypy_call_defaults', defaults)
                LoggingCase.test2.__dict__.__setitem__('stypy_call_varargs', varargs)
                LoggingCase.test2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                LoggingCase.test2.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingCase.test2', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test2', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test2(...)' code ##################

                pass
                
                # ################# End of 'test2(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test2' in the type store
                # Getting the type of 'stypy_return_type' (line 201)
                stypy_return_type_208495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208495)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test2'
                return stypy_return_type_208495

        
        # Assigning a type to the variable 'LoggingCase' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'LoggingCase', LoggingCase)
        
        # Assigning a List to a Name (line 203):
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_208496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        
        # Call to LoggingCase(...): (line 203)
        # Processing the call arguments (line 203)
        str_208498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 29), 'str', 'test1')
        # Processing the call keyword arguments (line 203)
        kwargs_208499 = {}
        # Getting the type of 'LoggingCase' (line 203)
        LoggingCase_208497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 17), 'LoggingCase', False)
        # Calling LoggingCase(args, kwargs) (line 203)
        LoggingCase_call_result_208500 = invoke(stypy.reporting.localization.Localization(__file__, 203, 17), LoggingCase_208497, *[str_208498], **kwargs_208499)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 16), list_208496, LoggingCase_call_result_208500)
        # Adding element type (line 203)
        
        # Call to LoggingCase(...): (line 203)
        # Processing the call arguments (line 203)
        str_208502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 51), 'str', 'test2')
        # Processing the call keyword arguments (line 203)
        kwargs_208503 = {}
        # Getting the type of 'LoggingCase' (line 203)
        LoggingCase_208501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 39), 'LoggingCase', False)
        # Calling LoggingCase(args, kwargs) (line 203)
        LoggingCase_call_result_208504 = invoke(stypy.reporting.localization.Localization(__file__, 203, 39), LoggingCase_208501, *[str_208502], **kwargs_208503)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 16), list_208496, LoggingCase_call_result_208504)
        
        # Assigning a type to the variable 'tests' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tests', list_208496)
        
        # Call to run(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'result' (line 205)
        result_208511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 38), 'result', False)
        # Processing the call keyword arguments (line 205)
        kwargs_208512 = {}
        
        # Call to TestSuite(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'tests' (line 205)
        tests_208507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'tests', False)
        # Processing the call keyword arguments (line 205)
        kwargs_208508 = {}
        # Getting the type of 'unittest' (line 205)
        unittest_208505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 205)
        TestSuite_208506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), unittest_208505, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 205)
        TestSuite_call_result_208509 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), TestSuite_208506, *[tests_208507], **kwargs_208508)
        
        # Obtaining the member 'run' of a type (line 205)
        run_208510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), TestSuite_call_result_208509, 'run')
        # Calling run(args, kwargs) (line 205)
        run_call_result_208513 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), run_208510, *[result_208511], **kwargs_208512)
        
        
        # Call to assertEqual(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'events' (line 207)
        events_208516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 25), 'events', False)
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_208517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        str_208518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 34), 'str', 'run test1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 33), list_208517, str_208518)
        # Adding element type (line 207)
        str_208519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 47), 'str', 'run test2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 33), list_208517, str_208519)
        
        # Processing the call keyword arguments (line 207)
        kwargs_208520 = {}
        # Getting the type of 'self' (line 207)
        self_208514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 207)
        assertEqual_208515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_208514, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 207)
        assertEqual_call_result_208521 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), assertEqual_208515, *[events_208516, list_208517], **kwargs_208520)
        
        
        # ################# End of 'test_run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_run' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_208522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208522)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_run'
        return stypy_return_type_208522


    @norecursion
    def test_addTest__TestCase(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addTest__TestCase'
        module_type_store = module_type_store.open_function_context('test_addTest__TestCase', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_addTest__TestCase')
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_addTest__TestCase.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_addTest__TestCase', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addTest__TestCase', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addTest__TestCase(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 211)
        unittest_208523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 211)
        TestCase_208524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 18), unittest_208523, 'TestCase')

        class Foo(TestCase_208524, ):

            @norecursion
            def test(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test'
                module_type_store = module_type_store.open_function_context('test', 212, 12, False)
                # Assigning a type to the variable 'self' (line 213)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test.__dict__.__setitem__('stypy_localization', localization)
                Foo.test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test.__dict__.__setitem__('stypy_function_name', 'Foo.test')
                Foo.test.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test(...)' code ##################

                pass
                
                # ################# End of 'test(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test' in the type store
                # Getting the type of 'stypy_return_type' (line 212)
                stypy_return_type_208525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208525)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test'
                return stypy_return_type_208525

        
        # Assigning a type to the variable 'Foo' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 214):
        
        # Call to Foo(...): (line 214)
        # Processing the call arguments (line 214)
        str_208527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 19), 'str', 'test')
        # Processing the call keyword arguments (line 214)
        kwargs_208528 = {}
        # Getting the type of 'Foo' (line 214)
        Foo_208526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'Foo', False)
        # Calling Foo(args, kwargs) (line 214)
        Foo_call_result_208529 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), Foo_208526, *[str_208527], **kwargs_208528)
        
        # Assigning a type to the variable 'test' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'test', Foo_call_result_208529)
        
        # Assigning a Call to a Name (line 215):
        
        # Call to TestSuite(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_208532 = {}
        # Getting the type of 'unittest' (line 215)
        unittest_208530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 215)
        TestSuite_208531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 16), unittest_208530, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 215)
        TestSuite_call_result_208533 = invoke(stypy.reporting.localization.Localization(__file__, 215, 16), TestSuite_208531, *[], **kwargs_208532)
        
        # Assigning a type to the variable 'suite' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'suite', TestSuite_call_result_208533)
        
        # Call to addTest(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'test' (line 217)
        test_208536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 22), 'test', False)
        # Processing the call keyword arguments (line 217)
        kwargs_208537 = {}
        # Getting the type of 'suite' (line 217)
        suite_208534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'suite', False)
        # Obtaining the member 'addTest' of a type (line 217)
        addTest_208535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), suite_208534, 'addTest')
        # Calling addTest(args, kwargs) (line 217)
        addTest_call_result_208538 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), addTest_208535, *[test_208536], **kwargs_208537)
        
        
        # Call to assertEqual(...): (line 219)
        # Processing the call arguments (line 219)
        
        # Call to countTestCases(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_208543 = {}
        # Getting the type of 'suite' (line 219)
        suite_208541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 219)
        countTestCases_208542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 25), suite_208541, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 219)
        countTestCases_call_result_208544 = invoke(stypy.reporting.localization.Localization(__file__, 219, 25), countTestCases_208542, *[], **kwargs_208543)
        
        int_208545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 49), 'int')
        # Processing the call keyword arguments (line 219)
        kwargs_208546 = {}
        # Getting the type of 'self' (line 219)
        self_208539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 219)
        assertEqual_208540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_208539, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 219)
        assertEqual_call_result_208547 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), assertEqual_208540, *[countTestCases_call_result_208544, int_208545], **kwargs_208546)
        
        
        # Call to assertEqual(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to list(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'suite' (line 220)
        suite_208551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'suite', False)
        # Processing the call keyword arguments (line 220)
        kwargs_208552 = {}
        # Getting the type of 'list' (line 220)
        list_208550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'list', False)
        # Calling list(args, kwargs) (line 220)
        list_call_result_208553 = invoke(stypy.reporting.localization.Localization(__file__, 220, 25), list_208550, *[suite_208551], **kwargs_208552)
        
        
        # Obtaining an instance of the builtin type 'list' (line 220)
        list_208554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 220)
        # Adding element type (line 220)
        # Getting the type of 'test' (line 220)
        test_208555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 39), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 38), list_208554, test_208555)
        
        # Processing the call keyword arguments (line 220)
        kwargs_208556 = {}
        # Getting the type of 'self' (line 220)
        self_208548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 220)
        assertEqual_208549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_208548, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 220)
        assertEqual_call_result_208557 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), assertEqual_208549, *[list_call_result_208553, list_208554], **kwargs_208556)
        
        
        # ################# End of 'test_addTest__TestCase(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addTest__TestCase' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_208558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208558)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addTest__TestCase'
        return stypy_return_type_208558


    @norecursion
    def test_addTest__TestSuite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addTest__TestSuite'
        module_type_store = module_type_store.open_function_context('test_addTest__TestSuite', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_addTest__TestSuite')
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_addTest__TestSuite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_addTest__TestSuite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addTest__TestSuite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addTest__TestSuite(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 224)
        unittest_208559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 224)
        TestCase_208560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 18), unittest_208559, 'TestCase')

        class Foo(TestCase_208560, ):

            @norecursion
            def test(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test'
                module_type_store = module_type_store.open_function_context('test', 225, 12, False)
                # Assigning a type to the variable 'self' (line 226)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test.__dict__.__setitem__('stypy_localization', localization)
                Foo.test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test.__dict__.__setitem__('stypy_function_name', 'Foo.test')
                Foo.test.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test(...)' code ##################

                pass
                
                # ################# End of 'test(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test' in the type store
                # Getting the type of 'stypy_return_type' (line 225)
                stypy_return_type_208561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208561)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test'
                return stypy_return_type_208561

        
        # Assigning a type to the variable 'Foo' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 227):
        
        # Call to TestSuite(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_208564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        
        # Call to Foo(...): (line 227)
        # Processing the call arguments (line 227)
        str_208566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 42), 'str', 'test')
        # Processing the call keyword arguments (line 227)
        kwargs_208567 = {}
        # Getting the type of 'Foo' (line 227)
        Foo_208565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 38), 'Foo', False)
        # Calling Foo(args, kwargs) (line 227)
        Foo_call_result_208568 = invoke(stypy.reporting.localization.Localization(__file__, 227, 38), Foo_208565, *[str_208566], **kwargs_208567)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 37), list_208564, Foo_call_result_208568)
        
        # Processing the call keyword arguments (line 227)
        kwargs_208569 = {}
        # Getting the type of 'unittest' (line 227)
        unittest_208562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 18), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 227)
        TestSuite_208563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 18), unittest_208562, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 227)
        TestSuite_call_result_208570 = invoke(stypy.reporting.localization.Localization(__file__, 227, 18), TestSuite_208563, *[list_208564], **kwargs_208569)
        
        # Assigning a type to the variable 'suite_2' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'suite_2', TestSuite_call_result_208570)
        
        # Assigning a Call to a Name (line 229):
        
        # Call to TestSuite(...): (line 229)
        # Processing the call keyword arguments (line 229)
        kwargs_208573 = {}
        # Getting the type of 'unittest' (line 229)
        unittest_208571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 229)
        TestSuite_208572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), unittest_208571, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 229)
        TestSuite_call_result_208574 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), TestSuite_208572, *[], **kwargs_208573)
        
        # Assigning a type to the variable 'suite' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'suite', TestSuite_call_result_208574)
        
        # Call to addTest(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'suite_2' (line 230)
        suite_2_208577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'suite_2', False)
        # Processing the call keyword arguments (line 230)
        kwargs_208578 = {}
        # Getting the type of 'suite' (line 230)
        suite_208575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'suite', False)
        # Obtaining the member 'addTest' of a type (line 230)
        addTest_208576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), suite_208575, 'addTest')
        # Calling addTest(args, kwargs) (line 230)
        addTest_call_result_208579 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), addTest_208576, *[suite_2_208577], **kwargs_208578)
        
        
        # Call to assertEqual(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to countTestCases(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_208584 = {}
        # Getting the type of 'suite' (line 232)
        suite_208582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 232)
        countTestCases_208583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 25), suite_208582, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 232)
        countTestCases_call_result_208585 = invoke(stypy.reporting.localization.Localization(__file__, 232, 25), countTestCases_208583, *[], **kwargs_208584)
        
        int_208586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 49), 'int')
        # Processing the call keyword arguments (line 232)
        kwargs_208587 = {}
        # Getting the type of 'self' (line 232)
        self_208580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 232)
        assertEqual_208581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_208580, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 232)
        assertEqual_call_result_208588 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), assertEqual_208581, *[countTestCases_call_result_208585, int_208586], **kwargs_208587)
        
        
        # Call to assertEqual(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Call to list(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'suite' (line 233)
        suite_208592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 30), 'suite', False)
        # Processing the call keyword arguments (line 233)
        kwargs_208593 = {}
        # Getting the type of 'list' (line 233)
        list_208591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 25), 'list', False)
        # Calling list(args, kwargs) (line 233)
        list_call_result_208594 = invoke(stypy.reporting.localization.Localization(__file__, 233, 25), list_208591, *[suite_208592], **kwargs_208593)
        
        
        # Obtaining an instance of the builtin type 'list' (line 233)
        list_208595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 233)
        # Adding element type (line 233)
        # Getting the type of 'suite_2' (line 233)
        suite_2_208596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 39), 'suite_2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 38), list_208595, suite_2_208596)
        
        # Processing the call keyword arguments (line 233)
        kwargs_208597 = {}
        # Getting the type of 'self' (line 233)
        self_208589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 233)
        assertEqual_208590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), self_208589, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 233)
        assertEqual_call_result_208598 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), assertEqual_208590, *[list_call_result_208594, list_208595], **kwargs_208597)
        
        
        # ################# End of 'test_addTest__TestSuite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addTest__TestSuite' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_208599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208599)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addTest__TestSuite'
        return stypy_return_type_208599


    @norecursion
    def test_addTests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addTests'
        module_type_store = module_type_store.open_function_context('test_addTests', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_addTests')
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_addTests.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_addTests', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addTests', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addTests(...)' code ##################

        # Declaration of the 'Foo' class
        # Getting the type of 'unittest' (line 241)
        unittest_208600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 241)
        TestCase_208601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 18), unittest_208600, 'TestCase')

        class Foo(TestCase_208601, ):

            @norecursion
            def test_1(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_1'
                module_type_store = module_type_store.open_function_context('test_1', 242, 12, False)
                # Assigning a type to the variable 'self' (line 243)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self', type_of_self)
                
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
                # Getting the type of 'stypy_return_type' (line 242)
                stypy_return_type_208602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208602)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_1'
                return stypy_return_type_208602


            @norecursion
            def test_2(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'test_2'
                module_type_store = module_type_store.open_function_context('test_2', 243, 12, False)
                # Assigning a type to the variable 'self' (line 244)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Foo.test_2.__dict__.__setitem__('stypy_localization', localization)
                Foo.test_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Foo.test_2.__dict__.__setitem__('stypy_type_store', module_type_store)
                Foo.test_2.__dict__.__setitem__('stypy_function_name', 'Foo.test_2')
                Foo.test_2.__dict__.__setitem__('stypy_param_names_list', [])
                Foo.test_2.__dict__.__setitem__('stypy_varargs_param_name', None)
                Foo.test_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Foo.test_2.__dict__.__setitem__('stypy_call_defaults', defaults)
                Foo.test_2.__dict__.__setitem__('stypy_call_varargs', varargs)
                Foo.test_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Foo.test_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.test_2', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'test_2', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'test_2(...)' code ##################

                pass
                
                # ################# End of 'test_2(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'test_2' in the type store
                # Getting the type of 'stypy_return_type' (line 243)
                stypy_return_type_208603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208603)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'test_2'
                return stypy_return_type_208603

        
        # Assigning a type to the variable 'Foo' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'Foo', Foo)
        
        # Assigning a Call to a Name (line 245):
        
        # Call to Foo(...): (line 245)
        # Processing the call arguments (line 245)
        str_208605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 21), 'str', 'test_1')
        # Processing the call keyword arguments (line 245)
        kwargs_208606 = {}
        # Getting the type of 'Foo' (line 245)
        Foo_208604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'Foo', False)
        # Calling Foo(args, kwargs) (line 245)
        Foo_call_result_208607 = invoke(stypy.reporting.localization.Localization(__file__, 245, 17), Foo_208604, *[str_208605], **kwargs_208606)
        
        # Assigning a type to the variable 'test_1' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'test_1', Foo_call_result_208607)
        
        # Assigning a Call to a Name (line 246):
        
        # Call to Foo(...): (line 246)
        # Processing the call arguments (line 246)
        str_208609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 21), 'str', 'test_2')
        # Processing the call keyword arguments (line 246)
        kwargs_208610 = {}
        # Getting the type of 'Foo' (line 246)
        Foo_208608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 17), 'Foo', False)
        # Calling Foo(args, kwargs) (line 246)
        Foo_call_result_208611 = invoke(stypy.reporting.localization.Localization(__file__, 246, 17), Foo_208608, *[str_208609], **kwargs_208610)
        
        # Assigning a type to the variable 'test_2' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'test_2', Foo_call_result_208611)
        
        # Assigning a Call to a Name (line 247):
        
        # Call to TestSuite(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_208614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        # Getting the type of 'test_2' (line 247)
        test_2_208615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 42), 'test_2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 41), list_208614, test_2_208615)
        
        # Processing the call keyword arguments (line 247)
        kwargs_208616 = {}
        # Getting the type of 'unittest' (line 247)
        unittest_208612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 247)
        TestSuite_208613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 22), unittest_208612, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 247)
        TestSuite_call_result_208617 = invoke(stypy.reporting.localization.Localization(__file__, 247, 22), TestSuite_208613, *[list_208614], **kwargs_208616)
        
        # Assigning a type to the variable 'inner_suite' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'inner_suite', TestSuite_call_result_208617)

        @norecursion
        def gen(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'gen'
            module_type_store = module_type_store.open_function_context('gen', 249, 8, False)
            
            # Passed parameters checking function
            gen.stypy_localization = localization
            gen.stypy_type_of_self = None
            gen.stypy_type_store = module_type_store
            gen.stypy_function_name = 'gen'
            gen.stypy_param_names_list = []
            gen.stypy_varargs_param_name = None
            gen.stypy_kwargs_param_name = None
            gen.stypy_call_defaults = defaults
            gen.stypy_call_varargs = varargs
            gen.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'gen', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'gen', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'gen(...)' code ##################

            # Creating a generator
            # Getting the type of 'test_1' (line 250)
            test_1_208618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 18), 'test_1')
            GeneratorType_208619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 12), GeneratorType_208619, test_1_208618)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'stypy_return_type', GeneratorType_208619)
            # Creating a generator
            # Getting the type of 'test_2' (line 251)
            test_2_208620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 18), 'test_2')
            GeneratorType_208621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 12), GeneratorType_208621, test_2_208620)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'stypy_return_type', GeneratorType_208621)
            # Creating a generator
            # Getting the type of 'inner_suite' (line 252)
            inner_suite_208622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'inner_suite')
            GeneratorType_208623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 12), GeneratorType_208623, inner_suite_208622)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'stypy_return_type', GeneratorType_208623)
            
            # ################# End of 'gen(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'gen' in the type store
            # Getting the type of 'stypy_return_type' (line 249)
            stypy_return_type_208624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208624)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'gen'
            return stypy_return_type_208624

        # Assigning a type to the variable 'gen' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'gen', gen)
        
        # Assigning a Call to a Name (line 254):
        
        # Call to TestSuite(...): (line 254)
        # Processing the call keyword arguments (line 254)
        kwargs_208627 = {}
        # Getting the type of 'unittest' (line 254)
        unittest_208625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 18), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 254)
        TestSuite_208626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 18), unittest_208625, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 254)
        TestSuite_call_result_208628 = invoke(stypy.reporting.localization.Localization(__file__, 254, 18), TestSuite_208626, *[], **kwargs_208627)
        
        # Assigning a type to the variable 'suite_1' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'suite_1', TestSuite_call_result_208628)
        
        # Call to addTests(...): (line 255)
        # Processing the call arguments (line 255)
        
        # Call to gen(...): (line 255)
        # Processing the call keyword arguments (line 255)
        kwargs_208632 = {}
        # Getting the type of 'gen' (line 255)
        gen_208631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 25), 'gen', False)
        # Calling gen(args, kwargs) (line 255)
        gen_call_result_208633 = invoke(stypy.reporting.localization.Localization(__file__, 255, 25), gen_208631, *[], **kwargs_208632)
        
        # Processing the call keyword arguments (line 255)
        kwargs_208634 = {}
        # Getting the type of 'suite_1' (line 255)
        suite_1_208629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'suite_1', False)
        # Obtaining the member 'addTests' of a type (line 255)
        addTests_208630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), suite_1_208629, 'addTests')
        # Calling addTests(args, kwargs) (line 255)
        addTests_call_result_208635 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), addTests_208630, *[gen_call_result_208633], **kwargs_208634)
        
        
        # Call to assertEqual(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Call to list(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'suite_1' (line 257)
        suite_1_208639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 30), 'suite_1', False)
        # Processing the call keyword arguments (line 257)
        kwargs_208640 = {}
        # Getting the type of 'list' (line 257)
        list_208638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'list', False)
        # Calling list(args, kwargs) (line 257)
        list_call_result_208641 = invoke(stypy.reporting.localization.Localization(__file__, 257, 25), list_208638, *[suite_1_208639], **kwargs_208640)
        
        
        # Call to list(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Call to gen(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_208644 = {}
        # Getting the type of 'gen' (line 257)
        gen_208643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 45), 'gen', False)
        # Calling gen(args, kwargs) (line 257)
        gen_call_result_208645 = invoke(stypy.reporting.localization.Localization(__file__, 257, 45), gen_208643, *[], **kwargs_208644)
        
        # Processing the call keyword arguments (line 257)
        kwargs_208646 = {}
        # Getting the type of 'list' (line 257)
        list_208642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 40), 'list', False)
        # Calling list(args, kwargs) (line 257)
        list_call_result_208647 = invoke(stypy.reporting.localization.Localization(__file__, 257, 40), list_208642, *[gen_call_result_208645], **kwargs_208646)
        
        # Processing the call keyword arguments (line 257)
        kwargs_208648 = {}
        # Getting the type of 'self' (line 257)
        self_208636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 257)
        assertEqual_208637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_208636, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 257)
        assertEqual_call_result_208649 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), assertEqual_208637, *[list_call_result_208641, list_call_result_208647], **kwargs_208648)
        
        
        # Assigning a Call to a Name (line 261):
        
        # Call to TestSuite(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_208652 = {}
        # Getting the type of 'unittest' (line 261)
        unittest_208650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 18), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 261)
        TestSuite_208651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 18), unittest_208650, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 261)
        TestSuite_call_result_208653 = invoke(stypy.reporting.localization.Localization(__file__, 261, 18), TestSuite_208651, *[], **kwargs_208652)
        
        # Assigning a type to the variable 'suite_2' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'suite_2', TestSuite_call_result_208653)
        
        
        # Call to gen(...): (line 262)
        # Processing the call keyword arguments (line 262)
        kwargs_208655 = {}
        # Getting the type of 'gen' (line 262)
        gen_208654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 17), 'gen', False)
        # Calling gen(args, kwargs) (line 262)
        gen_call_result_208656 = invoke(stypy.reporting.localization.Localization(__file__, 262, 17), gen_208654, *[], **kwargs_208655)
        
        # Testing the type of a for loop iterable (line 262)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 262, 8), gen_call_result_208656)
        # Getting the type of the for loop variable (line 262)
        for_loop_var_208657 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 262, 8), gen_call_result_208656)
        # Assigning a type to the variable 't' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 't', for_loop_var_208657)
        # SSA begins for a for statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to addTest(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 't' (line 263)
        t_208660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 28), 't', False)
        # Processing the call keyword arguments (line 263)
        kwargs_208661 = {}
        # Getting the type of 'suite_2' (line 263)
        suite_2_208658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'suite_2', False)
        # Obtaining the member 'addTest' of a type (line 263)
        addTest_208659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), suite_2_208658, 'addTest')
        # Calling addTest(args, kwargs) (line 263)
        addTest_call_result_208662 = invoke(stypy.reporting.localization.Localization(__file__, 263, 12), addTest_208659, *[t_208660], **kwargs_208661)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertEqual(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'suite_1' (line 265)
        suite_1_208665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'suite_1', False)
        # Getting the type of 'suite_2' (line 265)
        suite_2_208666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 34), 'suite_2', False)
        # Processing the call keyword arguments (line 265)
        kwargs_208667 = {}
        # Getting the type of 'self' (line 265)
        self_208663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 265)
        assertEqual_208664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), self_208663, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 265)
        assertEqual_call_result_208668 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), assertEqual_208664, *[suite_1_208665, suite_2_208666], **kwargs_208667)
        
        
        # ################# End of 'test_addTests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addTests' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_208669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208669)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addTests'
        return stypy_return_type_208669


    @norecursion
    def test_addTest__noniterable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addTest__noniterable'
        module_type_store = module_type_store.open_function_context('test_addTest__noniterable', 271, 4, False)
        # Assigning a type to the variable 'self' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_addTest__noniterable')
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_addTest__noniterable.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_addTest__noniterable', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addTest__noniterable', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addTest__noniterable(...)' code ##################

        
        # Assigning a Call to a Name (line 272):
        
        # Call to TestSuite(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_208672 = {}
        # Getting the type of 'unittest' (line 272)
        unittest_208670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 272)
        TestSuite_208671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), unittest_208670, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 272)
        TestSuite_call_result_208673 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), TestSuite_208671, *[], **kwargs_208672)
        
        # Assigning a type to the variable 'suite' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'suite', TestSuite_call_result_208673)
        
        
        # SSA begins for try-except statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to addTests(...): (line 275)
        # Processing the call arguments (line 275)
        int_208676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 27), 'int')
        # Processing the call keyword arguments (line 275)
        kwargs_208677 = {}
        # Getting the type of 'suite' (line 275)
        suite_208674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'suite', False)
        # Obtaining the member 'addTests' of a type (line 275)
        addTests_208675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), suite_208674, 'addTests')
        # Calling addTests(args, kwargs) (line 275)
        addTests_call_result_208678 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), addTests_208675, *[int_208676], **kwargs_208677)
        
        # SSA branch for the except part of a try statement (line 274)
        # SSA branch for the except 'TypeError' branch of a try statement (line 274)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 274)
        module_type_store.open_ssa_branch('except else')
        
        # Call to fail(...): (line 279)
        # Processing the call arguments (line 279)
        str_208681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 22), 'str', 'Failed to raise TypeError')
        # Processing the call keyword arguments (line 279)
        kwargs_208682 = {}
        # Getting the type of 'self' (line 279)
        self_208679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 279)
        fail_208680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), self_208679, 'fail')
        # Calling fail(args, kwargs) (line 279)
        fail_call_result_208683 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), fail_208680, *[str_208681], **kwargs_208682)
        
        # SSA join for try-except statement (line 274)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_addTest__noniterable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addTest__noniterable' in the type store
        # Getting the type of 'stypy_return_type' (line 271)
        stypy_return_type_208684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208684)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addTest__noniterable'
        return stypy_return_type_208684


    @norecursion
    def test_addTest__noncallable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addTest__noncallable'
        module_type_store = module_type_store.open_function_context('test_addTest__noncallable', 281, 4, False)
        # Assigning a type to the variable 'self' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_addTest__noncallable')
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_addTest__noncallable.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_addTest__noncallable', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addTest__noncallable', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addTest__noncallable(...)' code ##################

        
        # Assigning a Call to a Name (line 282):
        
        # Call to TestSuite(...): (line 282)
        # Processing the call keyword arguments (line 282)
        kwargs_208687 = {}
        # Getting the type of 'unittest' (line 282)
        unittest_208685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 282)
        TestSuite_208686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 16), unittest_208685, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 282)
        TestSuite_call_result_208688 = invoke(stypy.reporting.localization.Localization(__file__, 282, 16), TestSuite_208686, *[], **kwargs_208687)
        
        # Assigning a type to the variable 'suite' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'suite', TestSuite_call_result_208688)
        
        # Call to assertRaises(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'TypeError' (line 283)
        TypeError_208691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 'TypeError', False)
        # Getting the type of 'suite' (line 283)
        suite_208692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 37), 'suite', False)
        # Obtaining the member 'addTest' of a type (line 283)
        addTest_208693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 37), suite_208692, 'addTest')
        int_208694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 52), 'int')
        # Processing the call keyword arguments (line 283)
        kwargs_208695 = {}
        # Getting the type of 'self' (line 283)
        self_208689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 283)
        assertRaises_208690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_208689, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 283)
        assertRaises_call_result_208696 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assertRaises_208690, *[TypeError_208691, addTest_208693, int_208694], **kwargs_208695)
        
        
        # ################# End of 'test_addTest__noncallable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addTest__noncallable' in the type store
        # Getting the type of 'stypy_return_type' (line 281)
        stypy_return_type_208697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208697)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addTest__noncallable'
        return stypy_return_type_208697


    @norecursion
    def test_addTest__casesuiteclass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addTest__casesuiteclass'
        module_type_store = module_type_store.open_function_context('test_addTest__casesuiteclass', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_addTest__casesuiteclass')
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_addTest__casesuiteclass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_addTest__casesuiteclass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addTest__casesuiteclass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addTest__casesuiteclass(...)' code ##################

        
        # Assigning a Call to a Name (line 286):
        
        # Call to TestSuite(...): (line 286)
        # Processing the call keyword arguments (line 286)
        kwargs_208700 = {}
        # Getting the type of 'unittest' (line 286)
        unittest_208698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 286)
        TestSuite_208699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 16), unittest_208698, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 286)
        TestSuite_call_result_208701 = invoke(stypy.reporting.localization.Localization(__file__, 286, 16), TestSuite_208699, *[], **kwargs_208700)
        
        # Assigning a type to the variable 'suite' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'suite', TestSuite_call_result_208701)
        
        # Call to assertRaises(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'TypeError' (line 287)
        TypeError_208704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'TypeError', False)
        # Getting the type of 'suite' (line 287)
        suite_208705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 37), 'suite', False)
        # Obtaining the member 'addTest' of a type (line 287)
        addTest_208706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 37), suite_208705, 'addTest')
        # Getting the type of 'Test_TestSuite' (line 287)
        Test_TestSuite_208707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 52), 'Test_TestSuite', False)
        # Processing the call keyword arguments (line 287)
        kwargs_208708 = {}
        # Getting the type of 'self' (line 287)
        self_208702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 287)
        assertRaises_208703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_208702, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 287)
        assertRaises_call_result_208709 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), assertRaises_208703, *[TypeError_208704, addTest_208706, Test_TestSuite_208707], **kwargs_208708)
        
        
        # Call to assertRaises(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'TypeError' (line 288)
        TypeError_208712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 26), 'TypeError', False)
        # Getting the type of 'suite' (line 288)
        suite_208713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'suite', False)
        # Obtaining the member 'addTest' of a type (line 288)
        addTest_208714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 37), suite_208713, 'addTest')
        # Getting the type of 'unittest' (line 288)
        unittest_208715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 52), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 288)
        TestSuite_208716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 52), unittest_208715, 'TestSuite')
        # Processing the call keyword arguments (line 288)
        kwargs_208717 = {}
        # Getting the type of 'self' (line 288)
        self_208710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 288)
        assertRaises_208711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), self_208710, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 288)
        assertRaises_call_result_208718 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), assertRaises_208711, *[TypeError_208712, addTest_208714, TestSuite_208716], **kwargs_208717)
        
        
        # ################# End of 'test_addTest__casesuiteclass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addTest__casesuiteclass' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_208719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208719)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addTest__casesuiteclass'
        return stypy_return_type_208719


    @norecursion
    def test_addTests__string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_addTests__string'
        module_type_store = module_type_store.open_function_context('test_addTests__string', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_addTests__string')
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_addTests__string.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_addTests__string', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_addTests__string', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_addTests__string(...)' code ##################

        
        # Assigning a Call to a Name (line 291):
        
        # Call to TestSuite(...): (line 291)
        # Processing the call keyword arguments (line 291)
        kwargs_208722 = {}
        # Getting the type of 'unittest' (line 291)
        unittest_208720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 291)
        TestSuite_208721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), unittest_208720, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 291)
        TestSuite_call_result_208723 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), TestSuite_208721, *[], **kwargs_208722)
        
        # Assigning a type to the variable 'suite' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'suite', TestSuite_call_result_208723)
        
        # Call to assertRaises(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'TypeError' (line 292)
        TypeError_208726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'TypeError', False)
        # Getting the type of 'suite' (line 292)
        suite_208727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 37), 'suite', False)
        # Obtaining the member 'addTests' of a type (line 292)
        addTests_208728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 37), suite_208727, 'addTests')
        str_208729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 53), 'str', 'foo')
        # Processing the call keyword arguments (line 292)
        kwargs_208730 = {}
        # Getting the type of 'self' (line 292)
        self_208724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 292)
        assertRaises_208725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), self_208724, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 292)
        assertRaises_call_result_208731 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), assertRaises_208725, *[TypeError_208726, addTests_208728, str_208729], **kwargs_208730)
        
        
        # ################# End of 'test_addTests__string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_addTests__string' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_208732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_addTests__string'
        return stypy_return_type_208732


    @norecursion
    def test_function_in_suite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_function_in_suite'
        module_type_store = module_type_store.open_function_context('test_function_in_suite', 294, 4, False)
        # Assigning a type to the variable 'self' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_function_in_suite')
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_function_in_suite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_function_in_suite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_function_in_suite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_function_in_suite(...)' code ##################


        @norecursion
        def f(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'f'
            module_type_store = module_type_store.open_function_context('f', 295, 8, False)
            
            # Passed parameters checking function
            f.stypy_localization = localization
            f.stypy_type_of_self = None
            f.stypy_type_store = module_type_store
            f.stypy_function_name = 'f'
            f.stypy_param_names_list = ['_']
            f.stypy_varargs_param_name = None
            f.stypy_kwargs_param_name = None
            f.stypy_call_defaults = defaults
            f.stypy_call_varargs = varargs
            f.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'f', ['_'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'f', localization, ['_'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'f(...)' code ##################

            pass
            
            # ################# End of 'f(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'f' in the type store
            # Getting the type of 'stypy_return_type' (line 295)
            stypy_return_type_208733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_208733)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'f'
            return stypy_return_type_208733

        # Assigning a type to the variable 'f' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'f', f)
        
        # Assigning a Call to a Name (line 297):
        
        # Call to TestSuite(...): (line 297)
        # Processing the call keyword arguments (line 297)
        kwargs_208736 = {}
        # Getting the type of 'unittest' (line 297)
        unittest_208734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 297)
        TestSuite_208735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 16), unittest_208734, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 297)
        TestSuite_call_result_208737 = invoke(stypy.reporting.localization.Localization(__file__, 297, 16), TestSuite_208735, *[], **kwargs_208736)
        
        # Assigning a type to the variable 'suite' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'suite', TestSuite_call_result_208737)
        
        # Call to addTest(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'f' (line 298)
        f_208740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 22), 'f', False)
        # Processing the call keyword arguments (line 298)
        kwargs_208741 = {}
        # Getting the type of 'suite' (line 298)
        suite_208738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'suite', False)
        # Obtaining the member 'addTest' of a type (line 298)
        addTest_208739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), suite_208738, 'addTest')
        # Calling addTest(args, kwargs) (line 298)
        addTest_call_result_208742 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), addTest_208739, *[f_208740], **kwargs_208741)
        
        
        # Call to run(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Call to TestResult(...): (line 301)
        # Processing the call keyword arguments (line 301)
        kwargs_208747 = {}
        # Getting the type of 'unittest' (line 301)
        unittest_208745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 18), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 301)
        TestResult_208746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 18), unittest_208745, 'TestResult')
        # Calling TestResult(args, kwargs) (line 301)
        TestResult_call_result_208748 = invoke(stypy.reporting.localization.Localization(__file__, 301, 18), TestResult_208746, *[], **kwargs_208747)
        
        # Processing the call keyword arguments (line 301)
        kwargs_208749 = {}
        # Getting the type of 'suite' (line 301)
        suite_208743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'suite', False)
        # Obtaining the member 'run' of a type (line 301)
        run_208744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), suite_208743, 'run')
        # Calling run(args, kwargs) (line 301)
        run_call_result_208750 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), run_208744, *[TestResult_call_result_208748], **kwargs_208749)
        
        
        # ################# End of 'test_function_in_suite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_function_in_suite' in the type store
        # Getting the type of 'stypy_return_type' (line 294)
        stypy_return_type_208751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_function_in_suite'
        return stypy_return_type_208751


    @norecursion
    def test_basetestsuite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basetestsuite'
        module_type_store = module_type_store.open_function_context('test_basetestsuite', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_basetestsuite')
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_basetestsuite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_basetestsuite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basetestsuite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basetestsuite(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 306)
        unittest_208752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 306)
        TestCase_208753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), unittest_208752, 'TestCase')

        class Test(TestCase_208753, ):
            
            # Assigning a Name to a Name (line 307):
            # Getting the type of 'False' (line 307)
            False_208754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 23), 'False')
            # Assigning a type to the variable 'wasSetUp' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'wasSetUp', False_208754)
            
            # Assigning a Name to a Name (line 308):
            # Getting the type of 'False' (line 308)
            False_208755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 26), 'False')
            # Assigning a type to the variable 'wasTornDown' (line 308)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'wasTornDown', False_208755)

            @norecursion
            def setUpClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpClass'
                module_type_store = module_type_store.open_function_context('setUpClass', 309, 12, False)
                # Assigning a type to the variable 'self' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'self', type_of_self)
                
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

                
                # Assigning a Name to a Attribute (line 311):
                # Getting the type of 'True' (line 311)
                True_208756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 31), 'True')
                # Getting the type of 'cls' (line 311)
                cls_208757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'cls')
                # Setting the type of the member 'wasSetUp' of a type (line 311)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 16), cls_208757, 'wasSetUp', True_208756)
                
                # ################# End of 'setUpClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpClass' in the type store
                # Getting the type of 'stypy_return_type' (line 309)
                stypy_return_type_208758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208758)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpClass'
                return stypy_return_type_208758


            @norecursion
            def tearDownClass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownClass'
                module_type_store = module_type_store.open_function_context('tearDownClass', 312, 12, False)
                # Assigning a type to the variable 'self' (line 313)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'self', type_of_self)
                
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

                
                # Assigning a Name to a Attribute (line 314):
                # Getting the type of 'True' (line 314)
                True_208759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 34), 'True')
                # Getting the type of 'cls' (line 314)
                cls_208760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'cls')
                # Setting the type of the member 'wasTornDown' of a type (line 314)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), cls_208760, 'wasTornDown', True_208759)
                
                # ################# End of 'tearDownClass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownClass' in the type store
                # Getting the type of 'stypy_return_type' (line 312)
                stypy_return_type_208761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208761)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownClass'
                return stypy_return_type_208761


            @norecursion
            def testPass(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testPass'
                module_type_store = module_type_store.open_function_context('testPass', 315, 12, False)
                # Assigning a type to the variable 'self' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.testPass.__dict__.__setitem__('stypy_localization', localization)
                Test.testPass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.testPass.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.testPass.__dict__.__setitem__('stypy_function_name', 'Test.testPass')
                Test.testPass.__dict__.__setitem__('stypy_param_names_list', [])
                Test.testPass.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.testPass.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.testPass.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.testPass.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.testPass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.testPass.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.testPass', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testPass', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testPass(...)' code ##################

                pass
                
                # ################# End of 'testPass(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testPass' in the type store
                # Getting the type of 'stypy_return_type' (line 315)
                stypy_return_type_208762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208762)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testPass'
                return stypy_return_type_208762


            @norecursion
            def testFail(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testFail'
                module_type_store = module_type_store.open_function_context('testFail', 317, 12, False)
                # Assigning a type to the variable 'self' (line 318)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Test.testFail.__dict__.__setitem__('stypy_localization', localization)
                Test.testFail.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Test.testFail.__dict__.__setitem__('stypy_type_store', module_type_store)
                Test.testFail.__dict__.__setitem__('stypy_function_name', 'Test.testFail')
                Test.testFail.__dict__.__setitem__('stypy_param_names_list', [])
                Test.testFail.__dict__.__setitem__('stypy_varargs_param_name', None)
                Test.testFail.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Test.testFail.__dict__.__setitem__('stypy_call_defaults', defaults)
                Test.testFail.__dict__.__setitem__('stypy_call_varargs', varargs)
                Test.testFail.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Test.testFail.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.testFail', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testFail', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testFail(...)' code ##################

                # Getting the type of 'fail' (line 318)
                fail_208763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'fail')
                
                # ################# End of 'testFail(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testFail' in the type store
                # Getting the type of 'stypy_return_type' (line 317)
                stypy_return_type_208764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208764)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testFail'
                return stypy_return_type_208764

        
        # Assigning a type to the variable 'Test' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'Test', Test)
        # Declaration of the 'Module' class

        class Module(object, ):
            
            # Assigning a Name to a Name (line 320):
            # Getting the type of 'False' (line 320)
            False_208765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'False')
            # Assigning a type to the variable 'wasSetUp' (line 320)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'wasSetUp', False_208765)
            
            # Assigning a Name to a Name (line 321):
            # Getting the type of 'False' (line 321)
            False_208766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 26), 'False')
            # Assigning a type to the variable 'wasTornDown' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'wasTornDown', False_208766)

            @staticmethod
            @norecursion
            def setUpModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUpModule'
                module_type_store = module_type_store.open_function_context('setUpModule', 322, 12, False)
                
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

                
                # Assigning a Name to a Attribute (line 324):
                # Getting the type of 'True' (line 324)
                True_208767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'True')
                # Getting the type of 'Module' (line 324)
                Module_208768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'Module')
                # Setting the type of the member 'wasSetUp' of a type (line 324)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 16), Module_208768, 'wasSetUp', True_208767)
                
                # ################# End of 'setUpModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUpModule' in the type store
                # Getting the type of 'stypy_return_type' (line 322)
                stypy_return_type_208769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208769)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUpModule'
                return stypy_return_type_208769


            @staticmethod
            @norecursion
            def tearDownModule(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDownModule'
                module_type_store = module_type_store.open_function_context('tearDownModule', 325, 12, False)
                
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

                
                # Assigning a Name to a Attribute (line 327):
                # Getting the type of 'True' (line 327)
                True_208770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 37), 'True')
                # Getting the type of 'Module' (line 327)
                Module_208771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'Module')
                # Setting the type of the member 'wasTornDown' of a type (line 327)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), Module_208771, 'wasTornDown', True_208770)
                
                # ################# End of 'tearDownModule(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDownModule' in the type store
                # Getting the type of 'stypy_return_type' (line 325)
                stypy_return_type_208772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208772)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDownModule'
                return stypy_return_type_208772

        
        # Assigning a type to the variable 'Module' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'Module', Module)
        
        # Assigning a Str to a Attribute (line 329):
        str_208773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 26), 'str', 'Module')
        # Getting the type of 'Test' (line 329)
        Test_208774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'Test')
        # Setting the type of the member '__module__' of a type (line 329)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), Test_208774, '__module__', str_208773)
        
        # Assigning a Name to a Subscript (line 330):
        # Getting the type of 'Module' (line 330)
        Module_208775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 32), 'Module')
        # Getting the type of 'sys' (line 330)
        sys_208776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 330)
        modules_208777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), sys_208776, 'modules')
        str_208778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 20), 'str', 'Module')
        # Storing an element on a container (line 330)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 8), modules_208777, (str_208778, Module_208775))
        
        # Call to addCleanup(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'sys' (line 331)
        sys_208781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'sys', False)
        # Obtaining the member 'modules' of a type (line 331)
        modules_208782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 24), sys_208781, 'modules')
        # Obtaining the member 'pop' of a type (line 331)
        pop_208783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 24), modules_208782, 'pop')
        str_208784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 41), 'str', 'Module')
        # Processing the call keyword arguments (line 331)
        kwargs_208785 = {}
        # Getting the type of 'self' (line 331)
        self_208779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 331)
        addCleanup_208780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), self_208779, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 331)
        addCleanup_call_result_208786 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), addCleanup_208780, *[pop_208783, str_208784], **kwargs_208785)
        
        
        # Assigning a Call to a Name (line 333):
        
        # Call to BaseTestSuite(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_208789 = {}
        # Getting the type of 'unittest' (line 333)
        unittest_208787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'unittest', False)
        # Obtaining the member 'BaseTestSuite' of a type (line 333)
        BaseTestSuite_208788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), unittest_208787, 'BaseTestSuite')
        # Calling BaseTestSuite(args, kwargs) (line 333)
        BaseTestSuite_call_result_208790 = invoke(stypy.reporting.localization.Localization(__file__, 333, 16), BaseTestSuite_208788, *[], **kwargs_208789)
        
        # Assigning a type to the variable 'suite' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'suite', BaseTestSuite_call_result_208790)
        
        # Call to addTests(...): (line 334)
        # Processing the call arguments (line 334)
        
        # Obtaining an instance of the builtin type 'list' (line 334)
        list_208793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 334)
        # Adding element type (line 334)
        
        # Call to Test(...): (line 334)
        # Processing the call arguments (line 334)
        str_208795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 29), 'str', 'testPass')
        # Processing the call keyword arguments (line 334)
        kwargs_208796 = {}
        # Getting the type of 'Test' (line 334)
        Test_208794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 24), 'Test', False)
        # Calling Test(args, kwargs) (line 334)
        Test_call_result_208797 = invoke(stypy.reporting.localization.Localization(__file__, 334, 24), Test_208794, *[str_208795], **kwargs_208796)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 23), list_208793, Test_call_result_208797)
        # Adding element type (line 334)
        
        # Call to Test(...): (line 334)
        # Processing the call arguments (line 334)
        str_208799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 47), 'str', 'testFail')
        # Processing the call keyword arguments (line 334)
        kwargs_208800 = {}
        # Getting the type of 'Test' (line 334)
        Test_208798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 42), 'Test', False)
        # Calling Test(args, kwargs) (line 334)
        Test_call_result_208801 = invoke(stypy.reporting.localization.Localization(__file__, 334, 42), Test_208798, *[str_208799], **kwargs_208800)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 23), list_208793, Test_call_result_208801)
        
        # Processing the call keyword arguments (line 334)
        kwargs_208802 = {}
        # Getting the type of 'suite' (line 334)
        suite_208791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'suite', False)
        # Obtaining the member 'addTests' of a type (line 334)
        addTests_208792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), suite_208791, 'addTests')
        # Calling addTests(args, kwargs) (line 334)
        addTests_call_result_208803 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), addTests_208792, *[list_208793], **kwargs_208802)
        
        
        # Call to assertEqual(...): (line 335)
        # Processing the call arguments (line 335)
        
        # Call to countTestCases(...): (line 335)
        # Processing the call keyword arguments (line 335)
        kwargs_208808 = {}
        # Getting the type of 'suite' (line 335)
        suite_208806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 335)
        countTestCases_208807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 25), suite_208806, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 335)
        countTestCases_call_result_208809 = invoke(stypy.reporting.localization.Localization(__file__, 335, 25), countTestCases_208807, *[], **kwargs_208808)
        
        int_208810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 49), 'int')
        # Processing the call keyword arguments (line 335)
        kwargs_208811 = {}
        # Getting the type of 'self' (line 335)
        self_208804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 335)
        assertEqual_208805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), self_208804, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 335)
        assertEqual_call_result_208812 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), assertEqual_208805, *[countTestCases_call_result_208809, int_208810], **kwargs_208811)
        
        
        # Assigning a Call to a Name (line 337):
        
        # Call to TestResult(...): (line 337)
        # Processing the call keyword arguments (line 337)
        kwargs_208815 = {}
        # Getting the type of 'unittest' (line 337)
        unittest_208813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 337)
        TestResult_208814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 17), unittest_208813, 'TestResult')
        # Calling TestResult(args, kwargs) (line 337)
        TestResult_call_result_208816 = invoke(stypy.reporting.localization.Localization(__file__, 337, 17), TestResult_208814, *[], **kwargs_208815)
        
        # Assigning a type to the variable 'result' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'result', TestResult_call_result_208816)
        
        # Call to run(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'result' (line 338)
        result_208819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'result', False)
        # Processing the call keyword arguments (line 338)
        kwargs_208820 = {}
        # Getting the type of 'suite' (line 338)
        suite_208817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'suite', False)
        # Obtaining the member 'run' of a type (line 338)
        run_208818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), suite_208817, 'run')
        # Calling run(args, kwargs) (line 338)
        run_call_result_208821 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), run_208818, *[result_208819], **kwargs_208820)
        
        
        # Call to assertFalse(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'Module' (line 339)
        Module_208824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 25), 'Module', False)
        # Obtaining the member 'wasSetUp' of a type (line 339)
        wasSetUp_208825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 25), Module_208824, 'wasSetUp')
        # Processing the call keyword arguments (line 339)
        kwargs_208826 = {}
        # Getting the type of 'self' (line 339)
        self_208822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 339)
        assertFalse_208823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), self_208822, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 339)
        assertFalse_call_result_208827 = invoke(stypy.reporting.localization.Localization(__file__, 339, 8), assertFalse_208823, *[wasSetUp_208825], **kwargs_208826)
        
        
        # Call to assertFalse(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'Module' (line 340)
        Module_208830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 25), 'Module', False)
        # Obtaining the member 'wasTornDown' of a type (line 340)
        wasTornDown_208831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 25), Module_208830, 'wasTornDown')
        # Processing the call keyword arguments (line 340)
        kwargs_208832 = {}
        # Getting the type of 'self' (line 340)
        self_208828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 340)
        assertFalse_208829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_208828, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 340)
        assertFalse_call_result_208833 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), assertFalse_208829, *[wasTornDown_208831], **kwargs_208832)
        
        
        # Call to assertFalse(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'Test' (line 341)
        Test_208836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'Test', False)
        # Obtaining the member 'wasSetUp' of a type (line 341)
        wasSetUp_208837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 25), Test_208836, 'wasSetUp')
        # Processing the call keyword arguments (line 341)
        kwargs_208838 = {}
        # Getting the type of 'self' (line 341)
        self_208834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 341)
        assertFalse_208835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_208834, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 341)
        assertFalse_call_result_208839 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), assertFalse_208835, *[wasSetUp_208837], **kwargs_208838)
        
        
        # Call to assertFalse(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'Test' (line 342)
        Test_208842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 25), 'Test', False)
        # Obtaining the member 'wasTornDown' of a type (line 342)
        wasTornDown_208843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 25), Test_208842, 'wasTornDown')
        # Processing the call keyword arguments (line 342)
        kwargs_208844 = {}
        # Getting the type of 'self' (line 342)
        self_208840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 342)
        assertFalse_208841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_208840, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 342)
        assertFalse_call_result_208845 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), assertFalse_208841, *[wasTornDown_208843], **kwargs_208844)
        
        
        # Call to assertEqual(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Call to len(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'result' (line 343)
        result_208849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 29), 'result', False)
        # Obtaining the member 'errors' of a type (line 343)
        errors_208850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 29), result_208849, 'errors')
        # Processing the call keyword arguments (line 343)
        kwargs_208851 = {}
        # Getting the type of 'len' (line 343)
        len_208848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'len', False)
        # Calling len(args, kwargs) (line 343)
        len_call_result_208852 = invoke(stypy.reporting.localization.Localization(__file__, 343, 25), len_208848, *[errors_208850], **kwargs_208851)
        
        int_208853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 45), 'int')
        # Processing the call keyword arguments (line 343)
        kwargs_208854 = {}
        # Getting the type of 'self' (line 343)
        self_208846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 343)
        assertEqual_208847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_208846, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 343)
        assertEqual_call_result_208855 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), assertEqual_208847, *[len_call_result_208852, int_208853], **kwargs_208854)
        
        
        # Call to assertEqual(...): (line 344)
        # Processing the call arguments (line 344)
        
        # Call to len(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'result' (line 344)
        result_208859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 29), 'result', False)
        # Obtaining the member 'failures' of a type (line 344)
        failures_208860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 29), result_208859, 'failures')
        # Processing the call keyword arguments (line 344)
        kwargs_208861 = {}
        # Getting the type of 'len' (line 344)
        len_208858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 25), 'len', False)
        # Calling len(args, kwargs) (line 344)
        len_call_result_208862 = invoke(stypy.reporting.localization.Localization(__file__, 344, 25), len_208858, *[failures_208860], **kwargs_208861)
        
        int_208863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 47), 'int')
        # Processing the call keyword arguments (line 344)
        kwargs_208864 = {}
        # Getting the type of 'self' (line 344)
        self_208856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 344)
        assertEqual_208857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), self_208856, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 344)
        assertEqual_call_result_208865 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), assertEqual_208857, *[len_call_result_208862, int_208863], **kwargs_208864)
        
        
        # Call to assertEqual(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'result' (line 345)
        result_208868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 25), 'result', False)
        # Obtaining the member 'testsRun' of a type (line 345)
        testsRun_208869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 25), result_208868, 'testsRun')
        int_208870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 42), 'int')
        # Processing the call keyword arguments (line 345)
        kwargs_208871 = {}
        # Getting the type of 'self' (line 345)
        self_208866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 345)
        assertEqual_208867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), self_208866, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 345)
        assertEqual_call_result_208872 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assertEqual_208867, *[testsRun_208869, int_208870], **kwargs_208871)
        
        
        # ################# End of 'test_basetestsuite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basetestsuite' in the type store
        # Getting the type of 'stypy_return_type' (line 305)
        stypy_return_type_208873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208873)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basetestsuite'
        return stypy_return_type_208873


    @norecursion
    def test_overriding_call(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_overriding_call'
        module_type_store = module_type_store.open_function_context('test_overriding_call', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_localization', localization)
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_function_name', 'Test_TestSuite.test_overriding_call')
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestSuite.test_overriding_call.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.test_overriding_call', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_overriding_call', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_overriding_call(...)' code ##################

        # Declaration of the 'MySuite' class
        # Getting the type of 'unittest' (line 349)
        unittest_208874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 22), 'unittest')
        # Obtaining the member 'TestSuite' of a type (line 349)
        TestSuite_208875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 22), unittest_208874, 'TestSuite')

        class MySuite(TestSuite_208875, ):
            
            # Assigning a Name to a Name (line 350):
            # Getting the type of 'False' (line 350)
            False_208876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 21), 'False')
            # Assigning a type to the variable 'called' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'called', False_208876)

            @norecursion
            def __call__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__call__'
                module_type_store = module_type_store.open_function_context('__call__', 351, 12, False)
                # Assigning a type to the variable 'self' (line 352)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                MySuite.__call__.__dict__.__setitem__('stypy_localization', localization)
                MySuite.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                MySuite.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
                MySuite.__call__.__dict__.__setitem__('stypy_function_name', 'MySuite.__call__')
                MySuite.__call__.__dict__.__setitem__('stypy_param_names_list', [])
                MySuite.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
                MySuite.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
                MySuite.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
                MySuite.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
                MySuite.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                MySuite.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'MySuite.__call__', [], 'args', 'kw', defaults, varargs, kwargs)

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

                
                # Assigning a Name to a Attribute (line 352):
                # Getting the type of 'True' (line 352)
                True_208877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 30), 'True')
                # Getting the type of 'self' (line 352)
                self_208878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'self')
                # Setting the type of the member 'called' of a type (line 352)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 16), self_208878, 'called', True_208877)
                
                # Call to __call__(...): (line 353)
                # Processing the call arguments (line 353)
                # Getting the type of 'self' (line 353)
                self_208882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 44), 'self', False)
                # Getting the type of 'args' (line 353)
                args_208883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 51), 'args', False)
                # Processing the call keyword arguments (line 353)
                # Getting the type of 'kw' (line 353)
                kw_208884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 59), 'kw', False)
                kwargs_208885 = {'kw_208884': kw_208884}
                # Getting the type of 'unittest' (line 353)
                unittest_208879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'unittest', False)
                # Obtaining the member 'TestSuite' of a type (line 353)
                TestSuite_208880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), unittest_208879, 'TestSuite')
                # Obtaining the member '__call__' of a type (line 353)
                call___208881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), TestSuite_208880, '__call__')
                # Calling __call__(args, kwargs) (line 353)
                call___call_result_208886 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), call___208881, *[self_208882, args_208883], **kwargs_208885)
                
                
                # ################# End of '__call__(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function '__call__' in the type store
                # Getting the type of 'stypy_return_type' (line 351)
                stypy_return_type_208887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_208887)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '__call__'
                return stypy_return_type_208887

        
        # Assigning a type to the variable 'MySuite' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'MySuite', MySuite)
        
        # Assigning a Call to a Name (line 355):
        
        # Call to MySuite(...): (line 355)
        # Processing the call keyword arguments (line 355)
        kwargs_208889 = {}
        # Getting the type of 'MySuite' (line 355)
        MySuite_208888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'MySuite', False)
        # Calling MySuite(args, kwargs) (line 355)
        MySuite_call_result_208890 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), MySuite_208888, *[], **kwargs_208889)
        
        # Assigning a type to the variable 'suite' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'suite', MySuite_call_result_208890)
        
        # Assigning a Call to a Name (line 356):
        
        # Call to TestResult(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_208893 = {}
        # Getting the type of 'unittest' (line 356)
        unittest_208891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 356)
        TestResult_208892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 17), unittest_208891, 'TestResult')
        # Calling TestResult(args, kwargs) (line 356)
        TestResult_call_result_208894 = invoke(stypy.reporting.localization.Localization(__file__, 356, 17), TestResult_208892, *[], **kwargs_208893)
        
        # Assigning a type to the variable 'result' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'result', TestResult_call_result_208894)
        
        # Assigning a Call to a Name (line 357):
        
        # Call to TestSuite(...): (line 357)
        # Processing the call keyword arguments (line 357)
        kwargs_208897 = {}
        # Getting the type of 'unittest' (line 357)
        unittest_208895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 18), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 357)
        TestSuite_208896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 18), unittest_208895, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 357)
        TestSuite_call_result_208898 = invoke(stypy.reporting.localization.Localization(__file__, 357, 18), TestSuite_208896, *[], **kwargs_208897)
        
        # Assigning a type to the variable 'wrapper' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'wrapper', TestSuite_call_result_208898)
        
        # Call to addTest(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'suite' (line 358)
        suite_208901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'suite', False)
        # Processing the call keyword arguments (line 358)
        kwargs_208902 = {}
        # Getting the type of 'wrapper' (line 358)
        wrapper_208899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'wrapper', False)
        # Obtaining the member 'addTest' of a type (line 358)
        addTest_208900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), wrapper_208899, 'addTest')
        # Calling addTest(args, kwargs) (line 358)
        addTest_call_result_208903 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), addTest_208900, *[suite_208901], **kwargs_208902)
        
        
        # Call to wrapper(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'result' (line 359)
        result_208905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'result', False)
        # Processing the call keyword arguments (line 359)
        kwargs_208906 = {}
        # Getting the type of 'wrapper' (line 359)
        wrapper_208904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'wrapper', False)
        # Calling wrapper(args, kwargs) (line 359)
        wrapper_call_result_208907 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), wrapper_208904, *[result_208905], **kwargs_208906)
        
        
        # Call to assertTrue(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'suite' (line 360)
        suite_208910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 24), 'suite', False)
        # Obtaining the member 'called' of a type (line 360)
        called_208911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 24), suite_208910, 'called')
        # Processing the call keyword arguments (line 360)
        kwargs_208912 = {}
        # Getting the type of 'self' (line 360)
        self_208908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 360)
        assertTrue_208909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), self_208908, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 360)
        assertTrue_call_result_208913 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), assertTrue_208909, *[called_208911], **kwargs_208912)
        
        
        # Call to assertFalse(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'result' (line 363)
        result_208916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'result', False)
        # Obtaining the member '_testRunEntered' of a type (line 363)
        _testRunEntered_208917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 25), result_208916, '_testRunEntered')
        # Processing the call keyword arguments (line 363)
        kwargs_208918 = {}
        # Getting the type of 'self' (line 363)
        self_208914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 363)
        assertFalse_208915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), self_208914, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 363)
        assertFalse_call_result_208919 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), assertFalse_208915, *[_testRunEntered_208917], **kwargs_208918)
        
        
        # ################# End of 'test_overriding_call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_overriding_call' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_208920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208920)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_overriding_call'
        return stypy_return_type_208920


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 0, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestSuite.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_TestSuite' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'Test_TestSuite', Test_TestSuite)

# Assigning a List to a Name (line 29):

# Obtaining an instance of the builtin type 'list' (line 29)
list_208921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 29)
tuple_208922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 29)
# Adding element type (line 29)

# Call to TestSuite(...): (line 29)
# Processing the call keyword arguments (line 29)
kwargs_208925 = {}
# Getting the type of 'unittest' (line 29)
unittest_208923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'unittest', False)
# Obtaining the member 'TestSuite' of a type (line 29)
TestSuite_208924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 17), unittest_208923, 'TestSuite')
# Calling TestSuite(args, kwargs) (line 29)
TestSuite_call_result_208926 = invoke(stypy.reporting.localization.Localization(__file__, 29, 17), TestSuite_208924, *[], **kwargs_208925)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), tuple_208922, TestSuite_call_result_208926)
# Adding element type (line 29)

# Call to TestSuite(...): (line 29)
# Processing the call keyword arguments (line 29)
kwargs_208929 = {}
# Getting the type of 'unittest' (line 29)
unittest_208927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 39), 'unittest', False)
# Obtaining the member 'TestSuite' of a type (line 29)
TestSuite_208928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 39), unittest_208927, 'TestSuite')
# Calling TestSuite(args, kwargs) (line 29)
TestSuite_call_result_208930 = invoke(stypy.reporting.localization.Localization(__file__, 29, 39), TestSuite_208928, *[], **kwargs_208929)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 17), tuple_208922, TestSuite_call_result_208930)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), list_208921, tuple_208922)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_208931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)

# Call to TestSuite(...): (line 30)
# Processing the call keyword arguments (line 30)
kwargs_208934 = {}
# Getting the type of 'unittest' (line 30)
unittest_208932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'unittest', False)
# Obtaining the member 'TestSuite' of a type (line 30)
TestSuite_208933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 17), unittest_208932, 'TestSuite')
# Calling TestSuite(args, kwargs) (line 30)
TestSuite_call_result_208935 = invoke(stypy.reporting.localization.Localization(__file__, 30, 17), TestSuite_208933, *[], **kwargs_208934)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 17), tuple_208931, TestSuite_call_result_208935)
# Adding element type (line 30)

# Call to TestSuite(...): (line 30)
# Processing the call arguments (line 30)

# Obtaining an instance of the builtin type 'list' (line 30)
list_208938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 58), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)

# Processing the call keyword arguments (line 30)
kwargs_208939 = {}
# Getting the type of 'unittest' (line 30)
unittest_208936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'unittest', False)
# Obtaining the member 'TestSuite' of a type (line 30)
TestSuite_208937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 39), unittest_208936, 'TestSuite')
# Calling TestSuite(args, kwargs) (line 30)
TestSuite_call_result_208940 = invoke(stypy.reporting.localization.Localization(__file__, 30, 39), TestSuite_208937, *[list_208938], **kwargs_208939)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 17), tuple_208931, TestSuite_call_result_208940)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), list_208921, tuple_208931)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 31)
tuple_208941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 31)
# Adding element type (line 31)

# Call to _mk_TestSuite(...): (line 31)
# Processing the call arguments (line 31)
str_208943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 30), 'str', 'test_1')
# Processing the call keyword arguments (line 31)
kwargs_208944 = {}
# Getting the type of '_mk_TestSuite' (line 31)
_mk_TestSuite_208942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), '_mk_TestSuite', False)
# Calling _mk_TestSuite(args, kwargs) (line 31)
_mk_TestSuite_call_result_208945 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), _mk_TestSuite_208942, *[str_208943], **kwargs_208944)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), tuple_208941, _mk_TestSuite_call_result_208945)
# Adding element type (line 31)

# Call to _mk_TestSuite(...): (line 31)
# Processing the call arguments (line 31)
str_208947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 55), 'str', 'test_1')
# Processing the call keyword arguments (line 31)
kwargs_208948 = {}
# Getting the type of '_mk_TestSuite' (line 31)
_mk_TestSuite_208946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 41), '_mk_TestSuite', False)
# Calling _mk_TestSuite(args, kwargs) (line 31)
_mk_TestSuite_call_result_208949 = invoke(stypy.reporting.localization.Localization(__file__, 31, 41), _mk_TestSuite_208946, *[str_208947], **kwargs_208948)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), tuple_208941, _mk_TestSuite_call_result_208949)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), list_208921, tuple_208941)

# Getting the type of 'Test_TestSuite'
Test_TestSuite_208950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_TestSuite')
# Setting the type of the member 'eq_pairs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_TestSuite_208950, 'eq_pairs', list_208921)

# Assigning a List to a Name (line 34):

# Obtaining an instance of the builtin type 'list' (line 34)
list_208951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 34)
tuple_208952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 34)
# Adding element type (line 34)

# Call to TestSuite(...): (line 34)
# Processing the call keyword arguments (line 34)
kwargs_208955 = {}
# Getting the type of 'unittest' (line 34)
unittest_208953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'unittest', False)
# Obtaining the member 'TestSuite' of a type (line 34)
TestSuite_208954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), unittest_208953, 'TestSuite')
# Calling TestSuite(args, kwargs) (line 34)
TestSuite_call_result_208956 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), TestSuite_208954, *[], **kwargs_208955)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), tuple_208952, TestSuite_call_result_208956)
# Adding element type (line 34)

# Call to _mk_TestSuite(...): (line 34)
# Processing the call arguments (line 34)
str_208958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 53), 'str', 'test_1')
# Processing the call keyword arguments (line 34)
kwargs_208959 = {}
# Getting the type of '_mk_TestSuite' (line 34)
_mk_TestSuite_208957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 39), '_mk_TestSuite', False)
# Calling _mk_TestSuite(args, kwargs) (line 34)
_mk_TestSuite_call_result_208960 = invoke(stypy.reporting.localization.Localization(__file__, 34, 39), _mk_TestSuite_208957, *[str_208958], **kwargs_208959)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 17), tuple_208952, _mk_TestSuite_call_result_208960)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 15), list_208951, tuple_208952)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_208961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)

# Call to TestSuite(...): (line 35)
# Processing the call arguments (line 35)

# Obtaining an instance of the builtin type 'list' (line 35)
list_208964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 35)

# Processing the call keyword arguments (line 35)
kwargs_208965 = {}
# Getting the type of 'unittest' (line 35)
unittest_208962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'unittest', False)
# Obtaining the member 'TestSuite' of a type (line 35)
TestSuite_208963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 17), unittest_208962, 'TestSuite')
# Calling TestSuite(args, kwargs) (line 35)
TestSuite_call_result_208966 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), TestSuite_208963, *[list_208964], **kwargs_208965)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), tuple_208961, TestSuite_call_result_208966)
# Adding element type (line 35)

# Call to _mk_TestSuite(...): (line 35)
# Processing the call arguments (line 35)
str_208968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 55), 'str', 'test_1')
# Processing the call keyword arguments (line 35)
kwargs_208969 = {}
# Getting the type of '_mk_TestSuite' (line 35)
_mk_TestSuite_208967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 41), '_mk_TestSuite', False)
# Calling _mk_TestSuite(args, kwargs) (line 35)
_mk_TestSuite_call_result_208970 = invoke(stypy.reporting.localization.Localization(__file__, 35, 41), _mk_TestSuite_208967, *[str_208968], **kwargs_208969)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), tuple_208961, _mk_TestSuite_call_result_208970)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 15), list_208951, tuple_208961)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_208971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)

# Call to _mk_TestSuite(...): (line 36)
# Processing the call arguments (line 36)
str_208973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'str', 'test_1')
str_208974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 41), 'str', 'test_2')
# Processing the call keyword arguments (line 36)
kwargs_208975 = {}
# Getting the type of '_mk_TestSuite' (line 36)
_mk_TestSuite_208972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), '_mk_TestSuite', False)
# Calling _mk_TestSuite(args, kwargs) (line 36)
_mk_TestSuite_call_result_208976 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), _mk_TestSuite_208972, *[str_208973, str_208974], **kwargs_208975)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), tuple_208971, _mk_TestSuite_call_result_208976)
# Adding element type (line 36)

# Call to _mk_TestSuite(...): (line 36)
# Processing the call arguments (line 36)
str_208978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 66), 'str', 'test_1')
str_208979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 76), 'str', 'test_3')
# Processing the call keyword arguments (line 36)
kwargs_208980 = {}
# Getting the type of '_mk_TestSuite' (line 36)
_mk_TestSuite_208977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 52), '_mk_TestSuite', False)
# Calling _mk_TestSuite(args, kwargs) (line 36)
_mk_TestSuite_call_result_208981 = invoke(stypy.reporting.localization.Localization(__file__, 36, 52), _mk_TestSuite_208977, *[str_208978, str_208979], **kwargs_208980)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), tuple_208971, _mk_TestSuite_call_result_208981)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 15), list_208951, tuple_208971)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_208982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)

# Call to _mk_TestSuite(...): (line 37)
# Processing the call arguments (line 37)
str_208984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 31), 'str', 'test_1')
# Processing the call keyword arguments (line 37)
kwargs_208985 = {}
# Getting the type of '_mk_TestSuite' (line 37)
_mk_TestSuite_208983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), '_mk_TestSuite', False)
# Calling _mk_TestSuite(args, kwargs) (line 37)
_mk_TestSuite_call_result_208986 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), _mk_TestSuite_208983, *[str_208984], **kwargs_208985)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), tuple_208982, _mk_TestSuite_call_result_208986)
# Adding element type (line 37)

# Call to _mk_TestSuite(...): (line 37)
# Processing the call arguments (line 37)
str_208988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 56), 'str', 'test_2')
# Processing the call keyword arguments (line 37)
kwargs_208989 = {}
# Getting the type of '_mk_TestSuite' (line 37)
_mk_TestSuite_208987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 42), '_mk_TestSuite', False)
# Calling _mk_TestSuite(args, kwargs) (line 37)
_mk_TestSuite_call_result_208990 = invoke(stypy.reporting.localization.Localization(__file__, 37, 42), _mk_TestSuite_208987, *[str_208988], **kwargs_208989)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), tuple_208982, _mk_TestSuite_call_result_208990)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 15), list_208951, tuple_208982)

# Getting the type of 'Test_TestSuite'
Test_TestSuite_208991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test_TestSuite')
# Setting the type of the member 'ne_pairs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_TestSuite_208991, 'ne_pairs', list_208951)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 367)
    # Processing the call keyword arguments (line 367)
    kwargs_208994 = {}
    # Getting the type of 'unittest' (line 367)
    unittest_208992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 367)
    main_208993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 4), unittest_208992, 'main')
    # Calling main(args, kwargs) (line 367)
    main_call_result_208995 = invoke(stypy.reporting.localization.Localization(__file__, 367, 4), main_208993, *[], **kwargs_208994)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
