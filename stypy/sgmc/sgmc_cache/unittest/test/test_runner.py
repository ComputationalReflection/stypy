
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import unittest
2: 
3: from cStringIO import StringIO
4: import pickle
5: 
6: from unittest.test.support import (LoggingResult,
7:                                    ResultWithNoStartTestRunStopTestRun)
8: 
9: 
10: class TestCleanUp(unittest.TestCase):
11: 
12:     def testCleanUp(self):
13:         class TestableTest(unittest.TestCase):
14:             def testNothing(self):
15:                 pass
16: 
17:         test = TestableTest('testNothing')
18:         self.assertEqual(test._cleanups, [])
19: 
20:         cleanups = []
21: 
22:         def cleanup1(*args, **kwargs):
23:             cleanups.append((1, args, kwargs))
24: 
25:         def cleanup2(*args, **kwargs):
26:             cleanups.append((2, args, kwargs))
27: 
28:         test.addCleanup(cleanup1, 1, 2, 3, four='hello', five='goodbye')
29:         test.addCleanup(cleanup2)
30: 
31:         self.assertEqual(test._cleanups,
32:                          [(cleanup1, (1, 2, 3), dict(four='hello', five='goodbye')),
33:                           (cleanup2, (), {})])
34: 
35:         result = test.doCleanups()
36:         self.assertTrue(result)
37: 
38:         self.assertEqual(cleanups, [(2, (), {}), (1, (1, 2, 3),
39:                                     dict(four='hello', five='goodbye'))])
40: 
41:     def testCleanUpWithErrors(self):
42:         class TestableTest(unittest.TestCase):
43:             def testNothing(self):
44:                 pass
45: 
46:         class MockResult(object):
47:             errors = []
48:             def addError(self, test, exc_info):
49:                 self.errors.append((test, exc_info))
50: 
51:         result = MockResult()
52:         test = TestableTest('testNothing')
53:         test._resultForDoCleanups = result
54: 
55:         exc1 = Exception('foo')
56:         exc2 = Exception('bar')
57:         def cleanup1():
58:             raise exc1
59: 
60:         def cleanup2():
61:             raise exc2
62: 
63:         test.addCleanup(cleanup1)
64:         test.addCleanup(cleanup2)
65: 
66:         self.assertFalse(test.doCleanups())
67: 
68:         (test1, (Type1, instance1, _)), (test2, (Type2, instance2, _)) = reversed(MockResult.errors)
69:         self.assertEqual((test1, Type1, instance1), (test, Exception, exc1))
70:         self.assertEqual((test2, Type2, instance2), (test, Exception, exc2))
71: 
72:     def testCleanupInRun(self):
73:         blowUp = False
74:         ordering = []
75: 
76:         class TestableTest(unittest.TestCase):
77:             def setUp(self):
78:                 ordering.append('setUp')
79:                 if blowUp:
80:                     raise Exception('foo')
81: 
82:             def testNothing(self):
83:                 ordering.append('test')
84: 
85:             def tearDown(self):
86:                 ordering.append('tearDown')
87: 
88:         test = TestableTest('testNothing')
89: 
90:         def cleanup1():
91:             ordering.append('cleanup1')
92:         def cleanup2():
93:             ordering.append('cleanup2')
94:         test.addCleanup(cleanup1)
95:         test.addCleanup(cleanup2)
96: 
97:         def success(some_test):
98:             self.assertEqual(some_test, test)
99:             ordering.append('success')
100: 
101:         result = unittest.TestResult()
102:         result.addSuccess = success
103: 
104:         test.run(result)
105:         self.assertEqual(ordering, ['setUp', 'test', 'tearDown',
106:                                     'cleanup2', 'cleanup1', 'success'])
107: 
108:         blowUp = True
109:         ordering = []
110:         test = TestableTest('testNothing')
111:         test.addCleanup(cleanup1)
112:         test.run(result)
113:         self.assertEqual(ordering, ['setUp', 'cleanup1'])
114: 
115:     def testTestCaseDebugExecutesCleanups(self):
116:         ordering = []
117: 
118:         class TestableTest(unittest.TestCase):
119:             def setUp(self):
120:                 ordering.append('setUp')
121:                 self.addCleanup(cleanup1)
122: 
123:             def testNothing(self):
124:                 ordering.append('test')
125: 
126:             def tearDown(self):
127:                 ordering.append('tearDown')
128: 
129:         test = TestableTest('testNothing')
130: 
131:         def cleanup1():
132:             ordering.append('cleanup1')
133:             test.addCleanup(cleanup2)
134:         def cleanup2():
135:             ordering.append('cleanup2')
136: 
137:         test.debug()
138:         self.assertEqual(ordering, ['setUp', 'test', 'tearDown', 'cleanup1', 'cleanup2'])
139: 
140: 
141: class Test_TextTestRunner(unittest.TestCase):
142:     '''Tests for TextTestRunner.'''
143: 
144:     def test_init(self):
145:         runner = unittest.TextTestRunner()
146:         self.assertFalse(runner.failfast)
147:         self.assertFalse(runner.buffer)
148:         self.assertEqual(runner.verbosity, 1)
149:         self.assertTrue(runner.descriptions)
150:         self.assertEqual(runner.resultclass, unittest.TextTestResult)
151: 
152: 
153:     def test_multiple_inheritance(self):
154:         class AResult(unittest.TestResult):
155:             def __init__(self, stream, descriptions, verbosity):
156:                 super(AResult, self).__init__(stream, descriptions, verbosity)
157: 
158:         class ATextResult(unittest.TextTestResult, AResult):
159:             pass
160: 
161:         # This used to raise an exception due to TextTestResult not passing
162:         # on arguments in its __init__ super call
163:         ATextResult(None, None, 1)
164: 
165: 
166:     def testBufferAndFailfast(self):
167:         class Test(unittest.TestCase):
168:             def testFoo(self):
169:                 pass
170:         result = unittest.TestResult()
171:         runner = unittest.TextTestRunner(stream=StringIO(), failfast=True,
172:                                            buffer=True)
173:         # Use our result object
174:         runner._makeResult = lambda: result
175:         runner.run(Test('testFoo'))
176: 
177:         self.assertTrue(result.failfast)
178:         self.assertTrue(result.buffer)
179: 
180:     def testRunnerRegistersResult(self):
181:         class Test(unittest.TestCase):
182:             def testFoo(self):
183:                 pass
184:         originalRegisterResult = unittest.runner.registerResult
185:         def cleanup():
186:             unittest.runner.registerResult = originalRegisterResult
187:         self.addCleanup(cleanup)
188: 
189:         result = unittest.TestResult()
190:         runner = unittest.TextTestRunner(stream=StringIO())
191:         # Use our result object
192:         runner._makeResult = lambda: result
193: 
194:         self.wasRegistered = 0
195:         def fakeRegisterResult(thisResult):
196:             self.wasRegistered += 1
197:             self.assertEqual(thisResult, result)
198:         unittest.runner.registerResult = fakeRegisterResult
199: 
200:         runner.run(unittest.TestSuite())
201:         self.assertEqual(self.wasRegistered, 1)
202: 
203:     def test_works_with_result_without_startTestRun_stopTestRun(self):
204:         class OldTextResult(ResultWithNoStartTestRunStopTestRun):
205:             separator2 = ''
206:             def printErrors(self):
207:                 pass
208: 
209:         class Runner(unittest.TextTestRunner):
210:             def __init__(self):
211:                 super(Runner, self).__init__(StringIO())
212: 
213:             def _makeResult(self):
214:                 return OldTextResult()
215: 
216:         runner = Runner()
217:         runner.run(unittest.TestSuite())
218: 
219:     def test_startTestRun_stopTestRun_called(self):
220:         class LoggingTextResult(LoggingResult):
221:             separator2 = ''
222:             def printErrors(self):
223:                 pass
224: 
225:         class LoggingRunner(unittest.TextTestRunner):
226:             def __init__(self, events):
227:                 super(LoggingRunner, self).__init__(StringIO())
228:                 self._events = events
229: 
230:             def _makeResult(self):
231:                 return LoggingTextResult(self._events)
232: 
233:         events = []
234:         runner = LoggingRunner(events)
235:         runner.run(unittest.TestSuite())
236:         expected = ['startTestRun', 'stopTestRun']
237:         self.assertEqual(events, expected)
238: 
239:     def test_pickle_unpickle(self):
240:         # Issue #7197: a TextTestRunner should be (un)pickleable. This is
241:         # required by test_multiprocessing under Windows (in verbose mode).
242:         from StringIO import StringIO as PickleableIO
243:         # cStringIO objects are not pickleable, but StringIO objects are.
244:         stream = PickleableIO("foo")
245:         runner = unittest.TextTestRunner(stream)
246:         for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
247:             s = pickle.dumps(runner, protocol=protocol)
248:             obj = pickle.loads(s)
249:             # StringIO objects never compare equal, a cheap test instead.
250:             self.assertEqual(obj.stream.getvalue(), stream.getvalue())
251: 
252:     def test_resultclass(self):
253:         def MockResultClass(*args):
254:             return args
255:         STREAM = object()
256:         DESCRIPTIONS = object()
257:         VERBOSITY = object()
258:         runner = unittest.TextTestRunner(STREAM, DESCRIPTIONS, VERBOSITY,
259:                                          resultclass=MockResultClass)
260:         self.assertEqual(runner.resultclass, MockResultClass)
261: 
262:         expectedresult = (runner.stream, DESCRIPTIONS, VERBOSITY)
263:         self.assertEqual(runner._makeResult(), expectedresult)
264: 
265: 
266: if __name__ == '__main__':
267:     unittest.main()
268: 

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

# 'from cStringIO import StringIO' statement (line 3)
from cStringIO import StringIO

import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import pickle' statement (line 4)
import pickle

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pickle', pickle, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from unittest.test.support import LoggingResult, ResultWithNoStartTestRunStopTestRun' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/unittest/test/')
import_205521 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test.support')

if (type(import_205521) is not StypyTypeError):

    if (import_205521 != 'pyd_module'):
        __import__(import_205521)
        sys_modules_205522 = sys.modules[import_205521]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test.support', sys_modules_205522.module_type_store, module_type_store, ['LoggingResult', 'ResultWithNoStartTestRunStopTestRun'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_205522, sys_modules_205522.module_type_store, module_type_store)
    else:
        from unittest.test.support import LoggingResult, ResultWithNoStartTestRunStopTestRun

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test.support', None, module_type_store, ['LoggingResult', 'ResultWithNoStartTestRunStopTestRun'], [LoggingResult, ResultWithNoStartTestRunStopTestRun])

else:
    # Assigning a type to the variable 'unittest.test.support' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test.support', import_205521)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/test/')

# Declaration of the 'TestCleanUp' class
# Getting the type of 'unittest' (line 10)
unittest_205523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'unittest')
# Obtaining the member 'TestCase' of a type (line 10)
TestCase_205524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 18), unittest_205523, 'TestCase')

class TestCleanUp(TestCase_205524, ):

    @norecursion
    def testCleanUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testCleanUp'
        module_type_store = module_type_store.open_function_context('testCleanUp', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_localization', localization)
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_function_name', 'TestCleanUp.testCleanUp')
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_param_names_list', [])
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCleanUp.testCleanUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCleanUp.testCleanUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testCleanUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testCleanUp(...)' code ##################

        # Declaration of the 'TestableTest' class
        # Getting the type of 'unittest' (line 13)
        unittest_205525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 27), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 13)
        TestCase_205526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 27), unittest_205525, 'TestCase')

        class TestableTest(TestCase_205526, ):

            @norecursion
            def testNothing(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testNothing'
                module_type_store = module_type_store.open_function_context('testNothing', 14, 12, False)
                # Assigning a type to the variable 'self' (line 15)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTest.testNothing.__dict__.__setitem__('stypy_localization', localization)
                TestableTest.testNothing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTest.testNothing.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTest.testNothing.__dict__.__setitem__('stypy_function_name', 'TestableTest.testNothing')
                TestableTest.testNothing.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTest.testNothing.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTest.testNothing.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTest.testNothing.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTest.testNothing', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testNothing', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testNothing(...)' code ##################

                pass
                
                # ################# End of 'testNothing(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testNothing' in the type store
                # Getting the type of 'stypy_return_type' (line 14)
                stypy_return_type_205527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205527)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testNothing'
                return stypy_return_type_205527

        
        # Assigning a type to the variable 'TestableTest' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'TestableTest', TestableTest)
        
        # Assigning a Call to a Name (line 17):
        
        # Assigning a Call to a Name (line 17):
        
        # Assigning a Call to a Name (line 17):
        
        # Assigning a Call to a Name (line 17):
        
        # Call to TestableTest(...): (line 17)
        # Processing the call arguments (line 17)
        str_205529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'str', 'testNothing')
        # Processing the call keyword arguments (line 17)
        kwargs_205530 = {}
        # Getting the type of 'TestableTest' (line 17)
        TestableTest_205528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'TestableTest', False)
        # Calling TestableTest(args, kwargs) (line 17)
        TestableTest_call_result_205531 = invoke(stypy.reporting.localization.Localization(__file__, 17, 15), TestableTest_205528, *[str_205529], **kwargs_205530)
        
        # Assigning a type to the variable 'test' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'test', TestableTest_call_result_205531)
        
        # Call to assertEqual(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'test' (line 18)
        test_205534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'test', False)
        # Obtaining the member '_cleanups' of a type (line 18)
        _cleanups_205535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), test_205534, '_cleanups')
        
        # Obtaining an instance of the builtin type 'list' (line 18)
        list_205536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 18)
        
        # Processing the call keyword arguments (line 18)
        kwargs_205537 = {}
        # Getting the type of 'self' (line 18)
        self_205532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 18)
        assertEqual_205533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_205532, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 18)
        assertEqual_call_result_205538 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), assertEqual_205533, *[_cleanups_205535, list_205536], **kwargs_205537)
        
        
        # Assigning a List to a Name (line 20):
        
        # Assigning a List to a Name (line 20):
        
        # Assigning a List to a Name (line 20):
        
        # Assigning a List to a Name (line 20):
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_205539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        
        # Assigning a type to the variable 'cleanups' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'cleanups', list_205539)

        @norecursion
        def cleanup1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup1'
            module_type_store = module_type_store.open_function_context('cleanup1', 22, 8, False)
            
            # Passed parameters checking function
            cleanup1.stypy_localization = localization
            cleanup1.stypy_type_of_self = None
            cleanup1.stypy_type_store = module_type_store
            cleanup1.stypy_function_name = 'cleanup1'
            cleanup1.stypy_param_names_list = []
            cleanup1.stypy_varargs_param_name = 'args'
            cleanup1.stypy_kwargs_param_name = 'kwargs'
            cleanup1.stypy_call_defaults = defaults
            cleanup1.stypy_call_varargs = varargs
            cleanup1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup1', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup1', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup1(...)' code ##################

            
            # Call to append(...): (line 23)
            # Processing the call arguments (line 23)
            
            # Obtaining an instance of the builtin type 'tuple' (line 23)
            tuple_205542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 23)
            # Adding element type (line 23)
            int_205543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 29), tuple_205542, int_205543)
            # Adding element type (line 23)
            # Getting the type of 'args' (line 23)
            args_205544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 32), 'args', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 29), tuple_205542, args_205544)
            # Adding element type (line 23)
            # Getting the type of 'kwargs' (line 23)
            kwargs_205545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 38), 'kwargs', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 29), tuple_205542, kwargs_205545)
            
            # Processing the call keyword arguments (line 23)
            kwargs_205546 = {}
            # Getting the type of 'cleanups' (line 23)
            cleanups_205540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'cleanups', False)
            # Obtaining the member 'append' of a type (line 23)
            append_205541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), cleanups_205540, 'append')
            # Calling append(args, kwargs) (line 23)
            append_call_result_205547 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), append_205541, *[tuple_205542], **kwargs_205546)
            
            
            # ################# End of 'cleanup1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup1' in the type store
            # Getting the type of 'stypy_return_type' (line 22)
            stypy_return_type_205548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205548)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup1'
            return stypy_return_type_205548

        # Assigning a type to the variable 'cleanup1' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'cleanup1', cleanup1)

        @norecursion
        def cleanup2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup2'
            module_type_store = module_type_store.open_function_context('cleanup2', 25, 8, False)
            
            # Passed parameters checking function
            cleanup2.stypy_localization = localization
            cleanup2.stypy_type_of_self = None
            cleanup2.stypy_type_store = module_type_store
            cleanup2.stypy_function_name = 'cleanup2'
            cleanup2.stypy_param_names_list = []
            cleanup2.stypy_varargs_param_name = 'args'
            cleanup2.stypy_kwargs_param_name = 'kwargs'
            cleanup2.stypy_call_defaults = defaults
            cleanup2.stypy_call_varargs = varargs
            cleanup2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup2', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup2', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup2(...)' code ##################

            
            # Call to append(...): (line 26)
            # Processing the call arguments (line 26)
            
            # Obtaining an instance of the builtin type 'tuple' (line 26)
            tuple_205551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 26)
            # Adding element type (line 26)
            int_205552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 29), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 29), tuple_205551, int_205552)
            # Adding element type (line 26)
            # Getting the type of 'args' (line 26)
            args_205553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 32), 'args', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 29), tuple_205551, args_205553)
            # Adding element type (line 26)
            # Getting the type of 'kwargs' (line 26)
            kwargs_205554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'kwargs', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 29), tuple_205551, kwargs_205554)
            
            # Processing the call keyword arguments (line 26)
            kwargs_205555 = {}
            # Getting the type of 'cleanups' (line 26)
            cleanups_205549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'cleanups', False)
            # Obtaining the member 'append' of a type (line 26)
            append_205550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), cleanups_205549, 'append')
            # Calling append(args, kwargs) (line 26)
            append_call_result_205556 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), append_205550, *[tuple_205551], **kwargs_205555)
            
            
            # ################# End of 'cleanup2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup2' in the type store
            # Getting the type of 'stypy_return_type' (line 25)
            stypy_return_type_205557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205557)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup2'
            return stypy_return_type_205557

        # Assigning a type to the variable 'cleanup2' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'cleanup2', cleanup2)
        
        # Call to addCleanup(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'cleanup1' (line 28)
        cleanup1_205560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'cleanup1', False)
        int_205561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'int')
        int_205562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'int')
        int_205563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 40), 'int')
        # Processing the call keyword arguments (line 28)
        str_205564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 48), 'str', 'hello')
        keyword_205565 = str_205564
        str_205566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 62), 'str', 'goodbye')
        keyword_205567 = str_205566
        kwargs_205568 = {'four': keyword_205565, 'five': keyword_205567}
        # Getting the type of 'test' (line 28)
        test_205558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'test', False)
        # Obtaining the member 'addCleanup' of a type (line 28)
        addCleanup_205559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), test_205558, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 28)
        addCleanup_call_result_205569 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), addCleanup_205559, *[cleanup1_205560, int_205561, int_205562, int_205563], **kwargs_205568)
        
        
        # Call to addCleanup(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'cleanup2' (line 29)
        cleanup2_205572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'cleanup2', False)
        # Processing the call keyword arguments (line 29)
        kwargs_205573 = {}
        # Getting the type of 'test' (line 29)
        test_205570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'test', False)
        # Obtaining the member 'addCleanup' of a type (line 29)
        addCleanup_205571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), test_205570, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 29)
        addCleanup_call_result_205574 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), addCleanup_205571, *[cleanup2_205572], **kwargs_205573)
        
        
        # Call to assertEqual(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'test' (line 31)
        test_205577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'test', False)
        # Obtaining the member '_cleanups' of a type (line 31)
        _cleanups_205578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 25), test_205577, '_cleanups')
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_205579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        
        # Obtaining an instance of the builtin type 'tuple' (line 32)
        tuple_205580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 32)
        # Adding element type (line 32)
        # Getting the type of 'cleanup1' (line 32)
        cleanup1_205581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'cleanup1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 27), tuple_205580, cleanup1_205581)
        # Adding element type (line 32)
        
        # Obtaining an instance of the builtin type 'tuple' (line 32)
        tuple_205582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 32)
        # Adding element type (line 32)
        int_205583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 38), tuple_205582, int_205583)
        # Adding element type (line 32)
        int_205584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 38), tuple_205582, int_205584)
        # Adding element type (line 32)
        int_205585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 38), tuple_205582, int_205585)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 27), tuple_205580, tuple_205582)
        # Adding element type (line 32)
        
        # Call to dict(...): (line 32)
        # Processing the call keyword arguments (line 32)
        str_205587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 58), 'str', 'hello')
        keyword_205588 = str_205587
        str_205589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 72), 'str', 'goodbye')
        keyword_205590 = str_205589
        kwargs_205591 = {'four': keyword_205588, 'five': keyword_205590}
        # Getting the type of 'dict' (line 32)
        dict_205586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 48), 'dict', False)
        # Calling dict(args, kwargs) (line 32)
        dict_call_result_205592 = invoke(stypy.reporting.localization.Localization(__file__, 32, 48), dict_205586, *[], **kwargs_205591)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 27), tuple_205580, dict_call_result_205592)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 25), list_205579, tuple_205580)
        # Adding element type (line 32)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_205593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        # Getting the type of 'cleanup2' (line 33)
        cleanup2_205594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'cleanup2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 27), tuple_205593, cleanup2_205594)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_205595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 27), tuple_205593, tuple_205595)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'dict' (line 33)
        dict_205596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 41), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 33)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 27), tuple_205593, dict_205596)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 25), list_205579, tuple_205593)
        
        # Processing the call keyword arguments (line 31)
        kwargs_205597 = {}
        # Getting the type of 'self' (line 31)
        self_205575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 31)
        assertEqual_205576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_205575, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 31)
        assertEqual_call_result_205598 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assertEqual_205576, *[_cleanups_205578, list_205579], **kwargs_205597)
        
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to doCleanups(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_205601 = {}
        # Getting the type of 'test' (line 35)
        test_205599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'test', False)
        # Obtaining the member 'doCleanups' of a type (line 35)
        doCleanups_205600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 17), test_205599, 'doCleanups')
        # Calling doCleanups(args, kwargs) (line 35)
        doCleanups_call_result_205602 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), doCleanups_205600, *[], **kwargs_205601)
        
        # Assigning a type to the variable 'result' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'result', doCleanups_call_result_205602)
        
        # Call to assertTrue(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'result' (line 36)
        result_205605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'result', False)
        # Processing the call keyword arguments (line 36)
        kwargs_205606 = {}
        # Getting the type of 'self' (line 36)
        self_205603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 36)
        assertTrue_205604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_205603, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 36)
        assertTrue_call_result_205607 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assertTrue_205604, *[result_205605], **kwargs_205606)
        
        
        # Call to assertEqual(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'cleanups' (line 38)
        cleanups_205610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'cleanups', False)
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_205611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_205612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        int_205613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 37), tuple_205612, int_205613)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_205614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 37), tuple_205612, tuple_205614)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'dict' (line 38)
        dict_205615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 44), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 38)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 37), tuple_205612, dict_205615)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 35), list_205611, tuple_205612)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_205616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        int_205617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 50), tuple_205616, int_205617)
        # Adding element type (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_205618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        int_205619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 54), tuple_205618, int_205619)
        # Adding element type (line 38)
        int_205620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 54), tuple_205618, int_205620)
        # Adding element type (line 38)
        int_205621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 54), tuple_205618, int_205621)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 50), tuple_205616, tuple_205618)
        # Adding element type (line 38)
        
        # Call to dict(...): (line 39)
        # Processing the call keyword arguments (line 39)
        str_205623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'str', 'hello')
        keyword_205624 = str_205623
        str_205625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 60), 'str', 'goodbye')
        keyword_205626 = str_205625
        kwargs_205627 = {'four': keyword_205624, 'five': keyword_205626}
        # Getting the type of 'dict' (line 39)
        dict_205622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 36), 'dict', False)
        # Calling dict(args, kwargs) (line 39)
        dict_call_result_205628 = invoke(stypy.reporting.localization.Localization(__file__, 39, 36), dict_205622, *[], **kwargs_205627)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 50), tuple_205616, dict_call_result_205628)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 35), list_205611, tuple_205616)
        
        # Processing the call keyword arguments (line 38)
        kwargs_205629 = {}
        # Getting the type of 'self' (line 38)
        self_205608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 38)
        assertEqual_205609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_205608, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 38)
        assertEqual_call_result_205630 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assertEqual_205609, *[cleanups_205610, list_205611], **kwargs_205629)
        
        
        # ################# End of 'testCleanUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testCleanUp' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_205631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testCleanUp'
        return stypy_return_type_205631


    @norecursion
    def testCleanUpWithErrors(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testCleanUpWithErrors'
        module_type_store = module_type_store.open_function_context('testCleanUpWithErrors', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_localization', localization)
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_function_name', 'TestCleanUp.testCleanUpWithErrors')
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_param_names_list', [])
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCleanUp.testCleanUpWithErrors.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCleanUp.testCleanUpWithErrors', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testCleanUpWithErrors', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testCleanUpWithErrors(...)' code ##################

        # Declaration of the 'TestableTest' class
        # Getting the type of 'unittest' (line 42)
        unittest_205632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 42)
        TestCase_205633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), unittest_205632, 'TestCase')

        class TestableTest(TestCase_205633, ):

            @norecursion
            def testNothing(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testNothing'
                module_type_store = module_type_store.open_function_context('testNothing', 43, 12, False)
                # Assigning a type to the variable 'self' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTest.testNothing.__dict__.__setitem__('stypy_localization', localization)
                TestableTest.testNothing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTest.testNothing.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTest.testNothing.__dict__.__setitem__('stypy_function_name', 'TestableTest.testNothing')
                TestableTest.testNothing.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTest.testNothing.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTest.testNothing.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTest.testNothing.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTest.testNothing', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testNothing', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testNothing(...)' code ##################

                pass
                
                # ################# End of 'testNothing(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testNothing' in the type store
                # Getting the type of 'stypy_return_type' (line 43)
                stypy_return_type_205634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205634)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testNothing'
                return stypy_return_type_205634

        
        # Assigning a type to the variable 'TestableTest' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'TestableTest', TestableTest)
        # Declaration of the 'MockResult' class

        class MockResult(object, ):
            
            # Assigning a List to a Name (line 47):
            
            # Assigning a List to a Name (line 47):
            
            # Assigning a List to a Name (line 47):
            
            # Assigning a List to a Name (line 47):
            
            # Obtaining an instance of the builtin type 'list' (line 47)
            list_205635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'list')
            # Adding type elements to the builtin type 'list' instance (line 47)
            
            # Assigning a type to the variable 'errors' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'errors', list_205635)

            @norecursion
            def addError(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'addError'
                module_type_store = module_type_store.open_function_context('addError', 48, 12, False)
                # Assigning a type to the variable 'self' (line 49)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                MockResult.addError.__dict__.__setitem__('stypy_localization', localization)
                MockResult.addError.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                MockResult.addError.__dict__.__setitem__('stypy_type_store', module_type_store)
                MockResult.addError.__dict__.__setitem__('stypy_function_name', 'MockResult.addError')
                MockResult.addError.__dict__.__setitem__('stypy_param_names_list', ['test', 'exc_info'])
                MockResult.addError.__dict__.__setitem__('stypy_varargs_param_name', None)
                MockResult.addError.__dict__.__setitem__('stypy_kwargs_param_name', None)
                MockResult.addError.__dict__.__setitem__('stypy_call_defaults', defaults)
                MockResult.addError.__dict__.__setitem__('stypy_call_varargs', varargs)
                MockResult.addError.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                MockResult.addError.__dict__.__setitem__('stypy_declared_arg_number', 3)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'MockResult.addError', ['test', 'exc_info'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'addError', localization, ['test', 'exc_info'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'addError(...)' code ##################

                
                # Call to append(...): (line 49)
                # Processing the call arguments (line 49)
                
                # Obtaining an instance of the builtin type 'tuple' (line 49)
                tuple_205639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 49)
                # Adding element type (line 49)
                # Getting the type of 'test' (line 49)
                test_205640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'test', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 36), tuple_205639, test_205640)
                # Adding element type (line 49)
                # Getting the type of 'exc_info' (line 49)
                exc_info_205641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 42), 'exc_info', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 36), tuple_205639, exc_info_205641)
                
                # Processing the call keyword arguments (line 49)
                kwargs_205642 = {}
                # Getting the type of 'self' (line 49)
                self_205636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'self', False)
                # Obtaining the member 'errors' of a type (line 49)
                errors_205637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), self_205636, 'errors')
                # Obtaining the member 'append' of a type (line 49)
                append_205638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), errors_205637, 'append')
                # Calling append(args, kwargs) (line 49)
                append_call_result_205643 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), append_205638, *[tuple_205639], **kwargs_205642)
                
                
                # ################# End of 'addError(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'addError' in the type store
                # Getting the type of 'stypy_return_type' (line 48)
                stypy_return_type_205644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205644)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'addError'
                return stypy_return_type_205644

        
        # Assigning a type to the variable 'MockResult' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'MockResult', MockResult)
        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Call to MockResult(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_205646 = {}
        # Getting the type of 'MockResult' (line 51)
        MockResult_205645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'MockResult', False)
        # Calling MockResult(args, kwargs) (line 51)
        MockResult_call_result_205647 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), MockResult_205645, *[], **kwargs_205646)
        
        # Assigning a type to the variable 'result' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'result', MockResult_call_result_205647)
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to TestableTest(...): (line 52)
        # Processing the call arguments (line 52)
        str_205649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'str', 'testNothing')
        # Processing the call keyword arguments (line 52)
        kwargs_205650 = {}
        # Getting the type of 'TestableTest' (line 52)
        TestableTest_205648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'TestableTest', False)
        # Calling TestableTest(args, kwargs) (line 52)
        TestableTest_call_result_205651 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), TestableTest_205648, *[str_205649], **kwargs_205650)
        
        # Assigning a type to the variable 'test' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'test', TestableTest_call_result_205651)
        
        # Assigning a Name to a Attribute (line 53):
        
        # Assigning a Name to a Attribute (line 53):
        
        # Assigning a Name to a Attribute (line 53):
        
        # Assigning a Name to a Attribute (line 53):
        # Getting the type of 'result' (line 53)
        result_205652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 36), 'result')
        # Getting the type of 'test' (line 53)
        test_205653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'test')
        # Setting the type of the member '_resultForDoCleanups' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), test_205653, '_resultForDoCleanups', result_205652)
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to Exception(...): (line 55)
        # Processing the call arguments (line 55)
        str_205655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 25), 'str', 'foo')
        # Processing the call keyword arguments (line 55)
        kwargs_205656 = {}
        # Getting the type of 'Exception' (line 55)
        Exception_205654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 'Exception', False)
        # Calling Exception(args, kwargs) (line 55)
        Exception_call_result_205657 = invoke(stypy.reporting.localization.Localization(__file__, 55, 15), Exception_205654, *[str_205655], **kwargs_205656)
        
        # Assigning a type to the variable 'exc1' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'exc1', Exception_call_result_205657)
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to Exception(...): (line 56)
        # Processing the call arguments (line 56)
        str_205659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'str', 'bar')
        # Processing the call keyword arguments (line 56)
        kwargs_205660 = {}
        # Getting the type of 'Exception' (line 56)
        Exception_205658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'Exception', False)
        # Calling Exception(args, kwargs) (line 56)
        Exception_call_result_205661 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), Exception_205658, *[str_205659], **kwargs_205660)
        
        # Assigning a type to the variable 'exc2' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'exc2', Exception_call_result_205661)

        @norecursion
        def cleanup1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup1'
            module_type_store = module_type_store.open_function_context('cleanup1', 57, 8, False)
            
            # Passed parameters checking function
            cleanup1.stypy_localization = localization
            cleanup1.stypy_type_of_self = None
            cleanup1.stypy_type_store = module_type_store
            cleanup1.stypy_function_name = 'cleanup1'
            cleanup1.stypy_param_names_list = []
            cleanup1.stypy_varargs_param_name = None
            cleanup1.stypy_kwargs_param_name = None
            cleanup1.stypy_call_defaults = defaults
            cleanup1.stypy_call_varargs = varargs
            cleanup1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup1', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup1', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup1(...)' code ##################

            # Getting the type of 'exc1' (line 58)
            exc1_205662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'exc1')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 58, 12), exc1_205662, 'raise parameter', BaseException)
            
            # ################# End of 'cleanup1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup1' in the type store
            # Getting the type of 'stypy_return_type' (line 57)
            stypy_return_type_205663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205663)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup1'
            return stypy_return_type_205663

        # Assigning a type to the variable 'cleanup1' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'cleanup1', cleanup1)

        @norecursion
        def cleanup2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup2'
            module_type_store = module_type_store.open_function_context('cleanup2', 60, 8, False)
            
            # Passed parameters checking function
            cleanup2.stypy_localization = localization
            cleanup2.stypy_type_of_self = None
            cleanup2.stypy_type_store = module_type_store
            cleanup2.stypy_function_name = 'cleanup2'
            cleanup2.stypy_param_names_list = []
            cleanup2.stypy_varargs_param_name = None
            cleanup2.stypy_kwargs_param_name = None
            cleanup2.stypy_call_defaults = defaults
            cleanup2.stypy_call_varargs = varargs
            cleanup2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup2', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup2', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup2(...)' code ##################

            # Getting the type of 'exc2' (line 61)
            exc2_205664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'exc2')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 61, 12), exc2_205664, 'raise parameter', BaseException)
            
            # ################# End of 'cleanup2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup2' in the type store
            # Getting the type of 'stypy_return_type' (line 60)
            stypy_return_type_205665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205665)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup2'
            return stypy_return_type_205665

        # Assigning a type to the variable 'cleanup2' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'cleanup2', cleanup2)
        
        # Call to addCleanup(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'cleanup1' (line 63)
        cleanup1_205668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'cleanup1', False)
        # Processing the call keyword arguments (line 63)
        kwargs_205669 = {}
        # Getting the type of 'test' (line 63)
        test_205666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'test', False)
        # Obtaining the member 'addCleanup' of a type (line 63)
        addCleanup_205667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), test_205666, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 63)
        addCleanup_call_result_205670 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), addCleanup_205667, *[cleanup1_205668], **kwargs_205669)
        
        
        # Call to addCleanup(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'cleanup2' (line 64)
        cleanup2_205673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'cleanup2', False)
        # Processing the call keyword arguments (line 64)
        kwargs_205674 = {}
        # Getting the type of 'test' (line 64)
        test_205671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'test', False)
        # Obtaining the member 'addCleanup' of a type (line 64)
        addCleanup_205672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), test_205671, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 64)
        addCleanup_call_result_205675 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), addCleanup_205672, *[cleanup2_205673], **kwargs_205674)
        
        
        # Call to assertFalse(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to doCleanups(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_205680 = {}
        # Getting the type of 'test' (line 66)
        test_205678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'test', False)
        # Obtaining the member 'doCleanups' of a type (line 66)
        doCleanups_205679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), test_205678, 'doCleanups')
        # Calling doCleanups(args, kwargs) (line 66)
        doCleanups_call_result_205681 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), doCleanups_205679, *[], **kwargs_205680)
        
        # Processing the call keyword arguments (line 66)
        kwargs_205682 = {}
        # Getting the type of 'self' (line 66)
        self_205676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 66)
        assertFalse_205677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_205676, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 66)
        assertFalse_call_result_205683 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assertFalse_205677, *[doCleanups_call_result_205681], **kwargs_205682)
        
        
        # Assigning a Call to a Tuple (line 68):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to reversed(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'MockResult' (line 68)
        MockResult_205685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 82), 'MockResult', False)
        # Obtaining the member 'errors' of a type (line 68)
        errors_205686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 82), MockResult_205685, 'errors')
        # Processing the call keyword arguments (line 68)
        kwargs_205687 = {}
        # Getting the type of 'reversed' (line 68)
        reversed_205684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 73), 'reversed', False)
        # Calling reversed(args, kwargs) (line 68)
        reversed_call_result_205688 = invoke(stypy.reporting.localization.Localization(__file__, 68, 73), reversed_205684, *[errors_205686], **kwargs_205687)
        
        # Assigning a type to the variable 'call_assignment_205508' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call_assignment_205508', reversed_call_result_205688)
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_205691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Processing the call keyword arguments
        kwargs_205692 = {}
        # Getting the type of 'call_assignment_205508' (line 68)
        call_assignment_205508_205689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call_assignment_205508', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), call_assignment_205508_205689, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_205693 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___205690, *[int_205691], **kwargs_205692)
        
        # Assigning a type to the variable 'call_assignment_205509' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call_assignment_205509', getitem___call_result_205693)
        
        # Assigning a Name to a Tuple (line 68):
        
        # Assigning a Subscript to a Name (line 68):
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'call_assignment_205509' (line 68)
        call_assignment_205509_205695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call_assignment_205509')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), call_assignment_205509_205695, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205697 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205696, int_205694)
        
        # Assigning a type to the variable 'tuple_var_assignment_205511' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205511', subscript_call_result_205697)
        
        # Assigning a Subscript to a Name (line 68):
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'call_assignment_205509' (line 68)
        call_assignment_205509_205699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call_assignment_205509')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), call_assignment_205509_205699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205701 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205700, int_205698)
        
        # Assigning a type to the variable 'tuple_var_assignment_205512' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205512', subscript_call_result_205701)
        
        # Assigning a Name to a Name (line 68):
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_var_assignment_205511' (line 68)
        tuple_var_assignment_205511_205702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205511')
        # Assigning a type to the variable 'test1' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 9), 'test1', tuple_var_assignment_205511_205702)
        
        # Assigning a Name to a Tuple (line 68):
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'tuple_var_assignment_205512' (line 68)
        tuple_var_assignment_205512_205704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205512')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), tuple_var_assignment_205512_205704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205706 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205705, int_205703)
        
        # Assigning a type to the variable 'tuple_var_assignment_205515' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205515', subscript_call_result_205706)
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'tuple_var_assignment_205512' (line 68)
        tuple_var_assignment_205512_205708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205512')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), tuple_var_assignment_205512_205708, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205710 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205709, int_205707)
        
        # Assigning a type to the variable 'tuple_var_assignment_205516' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205516', subscript_call_result_205710)
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'tuple_var_assignment_205512' (line 68)
        tuple_var_assignment_205512_205712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205512')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), tuple_var_assignment_205512_205712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205714 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205713, int_205711)
        
        # Assigning a type to the variable 'tuple_var_assignment_205517' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205517', subscript_call_result_205714)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_var_assignment_205515' (line 68)
        tuple_var_assignment_205515_205715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205515')
        # Assigning a type to the variable 'Type1' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'Type1', tuple_var_assignment_205515_205715)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_var_assignment_205516' (line 68)
        tuple_var_assignment_205516_205716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205516')
        # Assigning a type to the variable 'instance1' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'instance1', tuple_var_assignment_205516_205716)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_var_assignment_205517' (line 68)
        tuple_var_assignment_205517_205717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205517')
        # Assigning a type to the variable '_' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), '_', tuple_var_assignment_205517_205717)
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_205720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Processing the call keyword arguments
        kwargs_205721 = {}
        # Getting the type of 'call_assignment_205508' (line 68)
        call_assignment_205508_205718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call_assignment_205508', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), call_assignment_205508_205718, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_205722 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___205719, *[int_205720], **kwargs_205721)
        
        # Assigning a type to the variable 'call_assignment_205510' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call_assignment_205510', getitem___call_result_205722)
        
        # Assigning a Name to a Tuple (line 68):
        
        # Assigning a Subscript to a Name (line 68):
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'call_assignment_205510' (line 68)
        call_assignment_205510_205724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call_assignment_205510')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), call_assignment_205510_205724, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205726 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205725, int_205723)
        
        # Assigning a type to the variable 'tuple_var_assignment_205513' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205513', subscript_call_result_205726)
        
        # Assigning a Subscript to a Name (line 68):
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'call_assignment_205510' (line 68)
        call_assignment_205510_205728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call_assignment_205510')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), call_assignment_205510_205728, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205730 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205729, int_205727)
        
        # Assigning a type to the variable 'tuple_var_assignment_205514' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205514', subscript_call_result_205730)
        
        # Assigning a Name to a Name (line 68):
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_var_assignment_205513' (line 68)
        tuple_var_assignment_205513_205731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205513')
        # Assigning a type to the variable 'test2' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 41), 'test2', tuple_var_assignment_205513_205731)
        
        # Assigning a Name to a Tuple (line 68):
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'tuple_var_assignment_205514' (line 68)
        tuple_var_assignment_205514_205733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205514')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), tuple_var_assignment_205514_205733, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205735 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205734, int_205732)
        
        # Assigning a type to the variable 'tuple_var_assignment_205518' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205518', subscript_call_result_205735)
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'tuple_var_assignment_205514' (line 68)
        tuple_var_assignment_205514_205737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205514')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), tuple_var_assignment_205514_205737, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205739 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205738, int_205736)
        
        # Assigning a type to the variable 'tuple_var_assignment_205519' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205519', subscript_call_result_205739)
        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        int_205740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
        # Getting the type of 'tuple_var_assignment_205514' (line 68)
        tuple_var_assignment_205514_205741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205514')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___205742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), tuple_var_assignment_205514_205741, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_205743 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), getitem___205742, int_205740)
        
        # Assigning a type to the variable 'tuple_var_assignment_205520' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205520', subscript_call_result_205743)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_var_assignment_205518' (line 68)
        tuple_var_assignment_205518_205744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205518')
        # Assigning a type to the variable 'Type2' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 49), 'Type2', tuple_var_assignment_205518_205744)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_var_assignment_205519' (line 68)
        tuple_var_assignment_205519_205745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205519')
        # Assigning a type to the variable 'instance2' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 56), 'instance2', tuple_var_assignment_205519_205745)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_var_assignment_205520' (line 68)
        tuple_var_assignment_205520_205746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_var_assignment_205520')
        # Assigning a type to the variable '_' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 67), '_', tuple_var_assignment_205520_205746)
        
        # Call to assertEqual(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining an instance of the builtin type 'tuple' (line 69)
        tuple_205749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 69)
        # Adding element type (line 69)
        # Getting the type of 'test1' (line 69)
        test1_205750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'test1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 26), tuple_205749, test1_205750)
        # Adding element type (line 69)
        # Getting the type of 'Type1' (line 69)
        Type1_205751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 33), 'Type1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 26), tuple_205749, Type1_205751)
        # Adding element type (line 69)
        # Getting the type of 'instance1' (line 69)
        instance1_205752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 40), 'instance1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 26), tuple_205749, instance1_205752)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 69)
        tuple_205753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 69)
        # Adding element type (line 69)
        # Getting the type of 'test' (line 69)
        test_205754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 53), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 53), tuple_205753, test_205754)
        # Adding element type (line 69)
        # Getting the type of 'Exception' (line 69)
        Exception_205755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 59), 'Exception', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 53), tuple_205753, Exception_205755)
        # Adding element type (line 69)
        # Getting the type of 'exc1' (line 69)
        exc1_205756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 70), 'exc1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 53), tuple_205753, exc1_205756)
        
        # Processing the call keyword arguments (line 69)
        kwargs_205757 = {}
        # Getting the type of 'self' (line 69)
        self_205747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 69)
        assertEqual_205748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_205747, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 69)
        assertEqual_call_result_205758 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), assertEqual_205748, *[tuple_205749, tuple_205753], **kwargs_205757)
        
        
        # Call to assertEqual(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Obtaining an instance of the builtin type 'tuple' (line 70)
        tuple_205761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 70)
        # Adding element type (line 70)
        # Getting the type of 'test2' (line 70)
        test2_205762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'test2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 26), tuple_205761, test2_205762)
        # Adding element type (line 70)
        # Getting the type of 'Type2' (line 70)
        Type2_205763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 33), 'Type2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 26), tuple_205761, Type2_205763)
        # Adding element type (line 70)
        # Getting the type of 'instance2' (line 70)
        instance2_205764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'instance2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 26), tuple_205761, instance2_205764)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 70)
        tuple_205765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 70)
        # Adding element type (line 70)
        # Getting the type of 'test' (line 70)
        test_205766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 53), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 53), tuple_205765, test_205766)
        # Adding element type (line 70)
        # Getting the type of 'Exception' (line 70)
        Exception_205767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 59), 'Exception', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 53), tuple_205765, Exception_205767)
        # Adding element type (line 70)
        # Getting the type of 'exc2' (line 70)
        exc2_205768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 70), 'exc2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 53), tuple_205765, exc2_205768)
        
        # Processing the call keyword arguments (line 70)
        kwargs_205769 = {}
        # Getting the type of 'self' (line 70)
        self_205759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 70)
        assertEqual_205760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_205759, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 70)
        assertEqual_call_result_205770 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), assertEqual_205760, *[tuple_205761, tuple_205765], **kwargs_205769)
        
        
        # ################# End of 'testCleanUpWithErrors(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testCleanUpWithErrors' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_205771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205771)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testCleanUpWithErrors'
        return stypy_return_type_205771


    @norecursion
    def testCleanupInRun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testCleanupInRun'
        module_type_store = module_type_store.open_function_context('testCleanupInRun', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_localization', localization)
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_function_name', 'TestCleanUp.testCleanupInRun')
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_param_names_list', [])
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCleanUp.testCleanupInRun.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCleanUp.testCleanupInRun', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testCleanupInRun', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testCleanupInRun(...)' code ##################

        
        # Assigning a Name to a Name (line 73):
        
        # Assigning a Name to a Name (line 73):
        
        # Assigning a Name to a Name (line 73):
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'False' (line 73)
        False_205772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'False')
        # Assigning a type to the variable 'blowUp' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'blowUp', False_205772)
        
        # Assigning a List to a Name (line 74):
        
        # Assigning a List to a Name (line 74):
        
        # Assigning a List to a Name (line 74):
        
        # Assigning a List to a Name (line 74):
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_205773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        
        # Assigning a type to the variable 'ordering' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'ordering', list_205773)
        # Declaration of the 'TestableTest' class
        # Getting the type of 'unittest' (line 76)
        unittest_205774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 76)
        TestCase_205775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 27), unittest_205774, 'TestCase')

        class TestableTest(TestCase_205775, ):

            @norecursion
            def setUp(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUp'
                module_type_store = module_type_store.open_function_context('setUp', 77, 12, False)
                # Assigning a type to the variable 'self' (line 78)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTest.setUp.__dict__.__setitem__('stypy_localization', localization)
                TestableTest.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTest.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTest.setUp.__dict__.__setitem__('stypy_function_name', 'TestableTest.setUp')
                TestableTest.setUp.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTest.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTest.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTest.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTest.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTest.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTest.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTest.setUp', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 78)
                # Processing the call arguments (line 78)
                str_205778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'str', 'setUp')
                # Processing the call keyword arguments (line 78)
                kwargs_205779 = {}
                # Getting the type of 'ordering' (line 78)
                ordering_205776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 78)
                append_205777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), ordering_205776, 'append')
                # Calling append(args, kwargs) (line 78)
                append_call_result_205780 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), append_205777, *[str_205778], **kwargs_205779)
                
                
                # Getting the type of 'blowUp' (line 79)
                blowUp_205781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'blowUp')
                # Testing the type of an if condition (line 79)
                if_condition_205782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 16), blowUp_205781)
                # Assigning a type to the variable 'if_condition_205782' (line 79)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'if_condition_205782', if_condition_205782)
                # SSA begins for if statement (line 79)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to Exception(...): (line 80)
                # Processing the call arguments (line 80)
                str_205784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'str', 'foo')
                # Processing the call keyword arguments (line 80)
                kwargs_205785 = {}
                # Getting the type of 'Exception' (line 80)
                Exception_205783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'Exception', False)
                # Calling Exception(args, kwargs) (line 80)
                Exception_call_result_205786 = invoke(stypy.reporting.localization.Localization(__file__, 80, 26), Exception_205783, *[str_205784], **kwargs_205785)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 80, 20), Exception_call_result_205786, 'raise parameter', BaseException)
                # SSA join for if statement (line 79)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # ################# End of 'setUp(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUp' in the type store
                # Getting the type of 'stypy_return_type' (line 77)
                stypy_return_type_205787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205787)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUp'
                return stypy_return_type_205787


            @norecursion
            def testNothing(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testNothing'
                module_type_store = module_type_store.open_function_context('testNothing', 82, 12, False)
                # Assigning a type to the variable 'self' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTest.testNothing.__dict__.__setitem__('stypy_localization', localization)
                TestableTest.testNothing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTest.testNothing.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTest.testNothing.__dict__.__setitem__('stypy_function_name', 'TestableTest.testNothing')
                TestableTest.testNothing.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTest.testNothing.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTest.testNothing.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTest.testNothing.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTest.testNothing', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testNothing', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testNothing(...)' code ##################

                
                # Call to append(...): (line 83)
                # Processing the call arguments (line 83)
                str_205790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 32), 'str', 'test')
                # Processing the call keyword arguments (line 83)
                kwargs_205791 = {}
                # Getting the type of 'ordering' (line 83)
                ordering_205788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 83)
                append_205789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), ordering_205788, 'append')
                # Calling append(args, kwargs) (line 83)
                append_call_result_205792 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), append_205789, *[str_205790], **kwargs_205791)
                
                
                # ################# End of 'testNothing(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testNothing' in the type store
                # Getting the type of 'stypy_return_type' (line 82)
                stypy_return_type_205793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205793)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testNothing'
                return stypy_return_type_205793


            @norecursion
            def tearDown(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDown'
                module_type_store = module_type_store.open_function_context('tearDown', 85, 12, False)
                # Assigning a type to the variable 'self' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTest.tearDown.__dict__.__setitem__('stypy_localization', localization)
                TestableTest.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTest.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTest.tearDown.__dict__.__setitem__('stypy_function_name', 'TestableTest.tearDown')
                TestableTest.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTest.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTest.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTest.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTest.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTest.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTest.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTest.tearDown', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 86)
                # Processing the call arguments (line 86)
                str_205796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 32), 'str', 'tearDown')
                # Processing the call keyword arguments (line 86)
                kwargs_205797 = {}
                # Getting the type of 'ordering' (line 86)
                ordering_205794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 86)
                append_205795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), ordering_205794, 'append')
                # Calling append(args, kwargs) (line 86)
                append_call_result_205798 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), append_205795, *[str_205796], **kwargs_205797)
                
                
                # ################# End of 'tearDown(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDown' in the type store
                # Getting the type of 'stypy_return_type' (line 85)
                stypy_return_type_205799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205799)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDown'
                return stypy_return_type_205799

        
        # Assigning a type to the variable 'TestableTest' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'TestableTest', TestableTest)
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to TestableTest(...): (line 88)
        # Processing the call arguments (line 88)
        str_205801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'str', 'testNothing')
        # Processing the call keyword arguments (line 88)
        kwargs_205802 = {}
        # Getting the type of 'TestableTest' (line 88)
        TestableTest_205800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'TestableTest', False)
        # Calling TestableTest(args, kwargs) (line 88)
        TestableTest_call_result_205803 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), TestableTest_205800, *[str_205801], **kwargs_205802)
        
        # Assigning a type to the variable 'test' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'test', TestableTest_call_result_205803)

        @norecursion
        def cleanup1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup1'
            module_type_store = module_type_store.open_function_context('cleanup1', 90, 8, False)
            
            # Passed parameters checking function
            cleanup1.stypy_localization = localization
            cleanup1.stypy_type_of_self = None
            cleanup1.stypy_type_store = module_type_store
            cleanup1.stypy_function_name = 'cleanup1'
            cleanup1.stypy_param_names_list = []
            cleanup1.stypy_varargs_param_name = None
            cleanup1.stypy_kwargs_param_name = None
            cleanup1.stypy_call_defaults = defaults
            cleanup1.stypy_call_varargs = varargs
            cleanup1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup1', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup1', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup1(...)' code ##################

            
            # Call to append(...): (line 91)
            # Processing the call arguments (line 91)
            str_205806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 28), 'str', 'cleanup1')
            # Processing the call keyword arguments (line 91)
            kwargs_205807 = {}
            # Getting the type of 'ordering' (line 91)
            ordering_205804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'ordering', False)
            # Obtaining the member 'append' of a type (line 91)
            append_205805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), ordering_205804, 'append')
            # Calling append(args, kwargs) (line 91)
            append_call_result_205808 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), append_205805, *[str_205806], **kwargs_205807)
            
            
            # ################# End of 'cleanup1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup1' in the type store
            # Getting the type of 'stypy_return_type' (line 90)
            stypy_return_type_205809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205809)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup1'
            return stypy_return_type_205809

        # Assigning a type to the variable 'cleanup1' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'cleanup1', cleanup1)

        @norecursion
        def cleanup2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup2'
            module_type_store = module_type_store.open_function_context('cleanup2', 92, 8, False)
            
            # Passed parameters checking function
            cleanup2.stypy_localization = localization
            cleanup2.stypy_type_of_self = None
            cleanup2.stypy_type_store = module_type_store
            cleanup2.stypy_function_name = 'cleanup2'
            cleanup2.stypy_param_names_list = []
            cleanup2.stypy_varargs_param_name = None
            cleanup2.stypy_kwargs_param_name = None
            cleanup2.stypy_call_defaults = defaults
            cleanup2.stypy_call_varargs = varargs
            cleanup2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup2', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup2', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup2(...)' code ##################

            
            # Call to append(...): (line 93)
            # Processing the call arguments (line 93)
            str_205812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'str', 'cleanup2')
            # Processing the call keyword arguments (line 93)
            kwargs_205813 = {}
            # Getting the type of 'ordering' (line 93)
            ordering_205810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'ordering', False)
            # Obtaining the member 'append' of a type (line 93)
            append_205811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), ordering_205810, 'append')
            # Calling append(args, kwargs) (line 93)
            append_call_result_205814 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), append_205811, *[str_205812], **kwargs_205813)
            
            
            # ################# End of 'cleanup2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup2' in the type store
            # Getting the type of 'stypy_return_type' (line 92)
            stypy_return_type_205815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205815)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup2'
            return stypy_return_type_205815

        # Assigning a type to the variable 'cleanup2' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'cleanup2', cleanup2)
        
        # Call to addCleanup(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'cleanup1' (line 94)
        cleanup1_205818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'cleanup1', False)
        # Processing the call keyword arguments (line 94)
        kwargs_205819 = {}
        # Getting the type of 'test' (line 94)
        test_205816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'test', False)
        # Obtaining the member 'addCleanup' of a type (line 94)
        addCleanup_205817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), test_205816, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 94)
        addCleanup_call_result_205820 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), addCleanup_205817, *[cleanup1_205818], **kwargs_205819)
        
        
        # Call to addCleanup(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'cleanup2' (line 95)
        cleanup2_205823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'cleanup2', False)
        # Processing the call keyword arguments (line 95)
        kwargs_205824 = {}
        # Getting the type of 'test' (line 95)
        test_205821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'test', False)
        # Obtaining the member 'addCleanup' of a type (line 95)
        addCleanup_205822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), test_205821, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 95)
        addCleanup_call_result_205825 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), addCleanup_205822, *[cleanup2_205823], **kwargs_205824)
        

        @norecursion
        def success(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'success'
            module_type_store = module_type_store.open_function_context('success', 97, 8, False)
            
            # Passed parameters checking function
            success.stypy_localization = localization
            success.stypy_type_of_self = None
            success.stypy_type_store = module_type_store
            success.stypy_function_name = 'success'
            success.stypy_param_names_list = ['some_test']
            success.stypy_varargs_param_name = None
            success.stypy_kwargs_param_name = None
            success.stypy_call_defaults = defaults
            success.stypy_call_varargs = varargs
            success.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'success', ['some_test'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'success', localization, ['some_test'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'success(...)' code ##################

            
            # Call to assertEqual(...): (line 98)
            # Processing the call arguments (line 98)
            # Getting the type of 'some_test' (line 98)
            some_test_205828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 29), 'some_test', False)
            # Getting the type of 'test' (line 98)
            test_205829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'test', False)
            # Processing the call keyword arguments (line 98)
            kwargs_205830 = {}
            # Getting the type of 'self' (line 98)
            self_205826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 98)
            assertEqual_205827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_205826, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 98)
            assertEqual_call_result_205831 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), assertEqual_205827, *[some_test_205828, test_205829], **kwargs_205830)
            
            
            # Call to append(...): (line 99)
            # Processing the call arguments (line 99)
            str_205834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'str', 'success')
            # Processing the call keyword arguments (line 99)
            kwargs_205835 = {}
            # Getting the type of 'ordering' (line 99)
            ordering_205832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'ordering', False)
            # Obtaining the member 'append' of a type (line 99)
            append_205833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), ordering_205832, 'append')
            # Calling append(args, kwargs) (line 99)
            append_call_result_205836 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), append_205833, *[str_205834], **kwargs_205835)
            
            
            # ################# End of 'success(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'success' in the type store
            # Getting the type of 'stypy_return_type' (line 97)
            stypy_return_type_205837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205837)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'success'
            return stypy_return_type_205837

        # Assigning a type to the variable 'success' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'success', success)
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to TestResult(...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_205840 = {}
        # Getting the type of 'unittest' (line 101)
        unittest_205838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 101)
        TestResult_205839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), unittest_205838, 'TestResult')
        # Calling TestResult(args, kwargs) (line 101)
        TestResult_call_result_205841 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), TestResult_205839, *[], **kwargs_205840)
        
        # Assigning a type to the variable 'result' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'result', TestResult_call_result_205841)
        
        # Assigning a Name to a Attribute (line 102):
        
        # Assigning a Name to a Attribute (line 102):
        
        # Assigning a Name to a Attribute (line 102):
        
        # Assigning a Name to a Attribute (line 102):
        # Getting the type of 'success' (line 102)
        success_205842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'success')
        # Getting the type of 'result' (line 102)
        result_205843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'result')
        # Setting the type of the member 'addSuccess' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), result_205843, 'addSuccess', success_205842)
        
        # Call to run(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'result' (line 104)
        result_205846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'result', False)
        # Processing the call keyword arguments (line 104)
        kwargs_205847 = {}
        # Getting the type of 'test' (line 104)
        test_205844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'test', False)
        # Obtaining the member 'run' of a type (line 104)
        run_205845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), test_205844, 'run')
        # Calling run(args, kwargs) (line 104)
        run_call_result_205848 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), run_205845, *[result_205846], **kwargs_205847)
        
        
        # Call to assertEqual(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'ordering' (line 105)
        ordering_205851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'ordering', False)
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_205852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        str_205853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'str', 'setUp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 35), list_205852, str_205853)
        # Adding element type (line 105)
        str_205854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 45), 'str', 'test')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 35), list_205852, str_205854)
        # Adding element type (line 105)
        str_205855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 53), 'str', 'tearDown')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 35), list_205852, str_205855)
        # Adding element type (line 105)
        str_205856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 36), 'str', 'cleanup2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 35), list_205852, str_205856)
        # Adding element type (line 105)
        str_205857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 48), 'str', 'cleanup1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 35), list_205852, str_205857)
        # Adding element type (line 105)
        str_205858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 60), 'str', 'success')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 35), list_205852, str_205858)
        
        # Processing the call keyword arguments (line 105)
        kwargs_205859 = {}
        # Getting the type of 'self' (line 105)
        self_205849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 105)
        assertEqual_205850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_205849, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 105)
        assertEqual_call_result_205860 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assertEqual_205850, *[ordering_205851, list_205852], **kwargs_205859)
        
        
        # Assigning a Name to a Name (line 108):
        
        # Assigning a Name to a Name (line 108):
        
        # Assigning a Name to a Name (line 108):
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'True' (line 108)
        True_205861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'True')
        # Assigning a type to the variable 'blowUp' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'blowUp', True_205861)
        
        # Assigning a List to a Name (line 109):
        
        # Assigning a List to a Name (line 109):
        
        # Assigning a List to a Name (line 109):
        
        # Assigning a List to a Name (line 109):
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_205862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        
        # Assigning a type to the variable 'ordering' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'ordering', list_205862)
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to TestableTest(...): (line 110)
        # Processing the call arguments (line 110)
        str_205864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 28), 'str', 'testNothing')
        # Processing the call keyword arguments (line 110)
        kwargs_205865 = {}
        # Getting the type of 'TestableTest' (line 110)
        TestableTest_205863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'TestableTest', False)
        # Calling TestableTest(args, kwargs) (line 110)
        TestableTest_call_result_205866 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), TestableTest_205863, *[str_205864], **kwargs_205865)
        
        # Assigning a type to the variable 'test' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'test', TestableTest_call_result_205866)
        
        # Call to addCleanup(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'cleanup1' (line 111)
        cleanup1_205869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'cleanup1', False)
        # Processing the call keyword arguments (line 111)
        kwargs_205870 = {}
        # Getting the type of 'test' (line 111)
        test_205867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'test', False)
        # Obtaining the member 'addCleanup' of a type (line 111)
        addCleanup_205868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), test_205867, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 111)
        addCleanup_call_result_205871 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), addCleanup_205868, *[cleanup1_205869], **kwargs_205870)
        
        
        # Call to run(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'result' (line 112)
        result_205874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'result', False)
        # Processing the call keyword arguments (line 112)
        kwargs_205875 = {}
        # Getting the type of 'test' (line 112)
        test_205872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'test', False)
        # Obtaining the member 'run' of a type (line 112)
        run_205873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), test_205872, 'run')
        # Calling run(args, kwargs) (line 112)
        run_call_result_205876 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), run_205873, *[result_205874], **kwargs_205875)
        
        
        # Call to assertEqual(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'ordering' (line 113)
        ordering_205879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'ordering', False)
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_205880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        str_205881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 36), 'str', 'setUp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 35), list_205880, str_205881)
        # Adding element type (line 113)
        str_205882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 45), 'str', 'cleanup1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 35), list_205880, str_205882)
        
        # Processing the call keyword arguments (line 113)
        kwargs_205883 = {}
        # Getting the type of 'self' (line 113)
        self_205877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 113)
        assertEqual_205878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_205877, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 113)
        assertEqual_call_result_205884 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assertEqual_205878, *[ordering_205879, list_205880], **kwargs_205883)
        
        
        # ################# End of 'testCleanupInRun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testCleanupInRun' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_205885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205885)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testCleanupInRun'
        return stypy_return_type_205885


    @norecursion
    def testTestCaseDebugExecutesCleanups(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testTestCaseDebugExecutesCleanups'
        module_type_store = module_type_store.open_function_context('testTestCaseDebugExecutesCleanups', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_localization', localization)
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_function_name', 'TestCleanUp.testTestCaseDebugExecutesCleanups')
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_param_names_list', [])
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCleanUp.testTestCaseDebugExecutesCleanups.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCleanUp.testTestCaseDebugExecutesCleanups', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testTestCaseDebugExecutesCleanups', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testTestCaseDebugExecutesCleanups(...)' code ##################

        
        # Assigning a List to a Name (line 116):
        
        # Assigning a List to a Name (line 116):
        
        # Assigning a List to a Name (line 116):
        
        # Assigning a List to a Name (line 116):
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_205886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        
        # Assigning a type to the variable 'ordering' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'ordering', list_205886)
        # Declaration of the 'TestableTest' class
        # Getting the type of 'unittest' (line 118)
        unittest_205887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 118)
        TestCase_205888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 27), unittest_205887, 'TestCase')

        class TestableTest(TestCase_205888, ):

            @norecursion
            def setUp(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'setUp'
                module_type_store = module_type_store.open_function_context('setUp', 119, 12, False)
                # Assigning a type to the variable 'self' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTest.setUp.__dict__.__setitem__('stypy_localization', localization)
                TestableTest.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTest.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTest.setUp.__dict__.__setitem__('stypy_function_name', 'TestableTest.setUp')
                TestableTest.setUp.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTest.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTest.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTest.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTest.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTest.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTest.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTest.setUp', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 120)
                # Processing the call arguments (line 120)
                str_205891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 32), 'str', 'setUp')
                # Processing the call keyword arguments (line 120)
                kwargs_205892 = {}
                # Getting the type of 'ordering' (line 120)
                ordering_205889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 120)
                append_205890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), ordering_205889, 'append')
                # Calling append(args, kwargs) (line 120)
                append_call_result_205893 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), append_205890, *[str_205891], **kwargs_205892)
                
                
                # Call to addCleanup(...): (line 121)
                # Processing the call arguments (line 121)
                # Getting the type of 'cleanup1' (line 121)
                cleanup1_205896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'cleanup1', False)
                # Processing the call keyword arguments (line 121)
                kwargs_205897 = {}
                # Getting the type of 'self' (line 121)
                self_205894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'self', False)
                # Obtaining the member 'addCleanup' of a type (line 121)
                addCleanup_205895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), self_205894, 'addCleanup')
                # Calling addCleanup(args, kwargs) (line 121)
                addCleanup_call_result_205898 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), addCleanup_205895, *[cleanup1_205896], **kwargs_205897)
                
                
                # ################# End of 'setUp(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'setUp' in the type store
                # Getting the type of 'stypy_return_type' (line 119)
                stypy_return_type_205899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205899)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'setUp'
                return stypy_return_type_205899


            @norecursion
            def testNothing(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testNothing'
                module_type_store = module_type_store.open_function_context('testNothing', 123, 12, False)
                # Assigning a type to the variable 'self' (line 124)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTest.testNothing.__dict__.__setitem__('stypy_localization', localization)
                TestableTest.testNothing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTest.testNothing.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTest.testNothing.__dict__.__setitem__('stypy_function_name', 'TestableTest.testNothing')
                TestableTest.testNothing.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTest.testNothing.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTest.testNothing.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTest.testNothing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTest.testNothing.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTest.testNothing', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testNothing', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testNothing(...)' code ##################

                
                # Call to append(...): (line 124)
                # Processing the call arguments (line 124)
                str_205902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 32), 'str', 'test')
                # Processing the call keyword arguments (line 124)
                kwargs_205903 = {}
                # Getting the type of 'ordering' (line 124)
                ordering_205900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 124)
                append_205901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), ordering_205900, 'append')
                # Calling append(args, kwargs) (line 124)
                append_call_result_205904 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), append_205901, *[str_205902], **kwargs_205903)
                
                
                # ################# End of 'testNothing(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testNothing' in the type store
                # Getting the type of 'stypy_return_type' (line 123)
                stypy_return_type_205905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205905)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testNothing'
                return stypy_return_type_205905


            @norecursion
            def tearDown(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'tearDown'
                module_type_store = module_type_store.open_function_context('tearDown', 126, 12, False)
                # Assigning a type to the variable 'self' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTest.tearDown.__dict__.__setitem__('stypy_localization', localization)
                TestableTest.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTest.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTest.tearDown.__dict__.__setitem__('stypy_function_name', 'TestableTest.tearDown')
                TestableTest.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTest.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTest.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTest.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTest.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTest.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTest.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTest.tearDown', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 127)
                # Processing the call arguments (line 127)
                str_205908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 32), 'str', 'tearDown')
                # Processing the call keyword arguments (line 127)
                kwargs_205909 = {}
                # Getting the type of 'ordering' (line 127)
                ordering_205906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'ordering', False)
                # Obtaining the member 'append' of a type (line 127)
                append_205907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), ordering_205906, 'append')
                # Calling append(args, kwargs) (line 127)
                append_call_result_205910 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), append_205907, *[str_205908], **kwargs_205909)
                
                
                # ################# End of 'tearDown(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'tearDown' in the type store
                # Getting the type of 'stypy_return_type' (line 126)
                stypy_return_type_205911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_205911)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'tearDown'
                return stypy_return_type_205911

        
        # Assigning a type to the variable 'TestableTest' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'TestableTest', TestableTest)
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to TestableTest(...): (line 129)
        # Processing the call arguments (line 129)
        str_205913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'str', 'testNothing')
        # Processing the call keyword arguments (line 129)
        kwargs_205914 = {}
        # Getting the type of 'TestableTest' (line 129)
        TestableTest_205912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'TestableTest', False)
        # Calling TestableTest(args, kwargs) (line 129)
        TestableTest_call_result_205915 = invoke(stypy.reporting.localization.Localization(__file__, 129, 15), TestableTest_205912, *[str_205913], **kwargs_205914)
        
        # Assigning a type to the variable 'test' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'test', TestableTest_call_result_205915)

        @norecursion
        def cleanup1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup1'
            module_type_store = module_type_store.open_function_context('cleanup1', 131, 8, False)
            
            # Passed parameters checking function
            cleanup1.stypy_localization = localization
            cleanup1.stypy_type_of_self = None
            cleanup1.stypy_type_store = module_type_store
            cleanup1.stypy_function_name = 'cleanup1'
            cleanup1.stypy_param_names_list = []
            cleanup1.stypy_varargs_param_name = None
            cleanup1.stypy_kwargs_param_name = None
            cleanup1.stypy_call_defaults = defaults
            cleanup1.stypy_call_varargs = varargs
            cleanup1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup1', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup1', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup1(...)' code ##################

            
            # Call to append(...): (line 132)
            # Processing the call arguments (line 132)
            str_205918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 28), 'str', 'cleanup1')
            # Processing the call keyword arguments (line 132)
            kwargs_205919 = {}
            # Getting the type of 'ordering' (line 132)
            ordering_205916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'ordering', False)
            # Obtaining the member 'append' of a type (line 132)
            append_205917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), ordering_205916, 'append')
            # Calling append(args, kwargs) (line 132)
            append_call_result_205920 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), append_205917, *[str_205918], **kwargs_205919)
            
            
            # Call to addCleanup(...): (line 133)
            # Processing the call arguments (line 133)
            # Getting the type of 'cleanup2' (line 133)
            cleanup2_205923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'cleanup2', False)
            # Processing the call keyword arguments (line 133)
            kwargs_205924 = {}
            # Getting the type of 'test' (line 133)
            test_205921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'test', False)
            # Obtaining the member 'addCleanup' of a type (line 133)
            addCleanup_205922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), test_205921, 'addCleanup')
            # Calling addCleanup(args, kwargs) (line 133)
            addCleanup_call_result_205925 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), addCleanup_205922, *[cleanup2_205923], **kwargs_205924)
            
            
            # ################# End of 'cleanup1(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup1' in the type store
            # Getting the type of 'stypy_return_type' (line 131)
            stypy_return_type_205926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205926)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup1'
            return stypy_return_type_205926

        # Assigning a type to the variable 'cleanup1' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'cleanup1', cleanup1)

        @norecursion
        def cleanup2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup2'
            module_type_store = module_type_store.open_function_context('cleanup2', 134, 8, False)
            
            # Passed parameters checking function
            cleanup2.stypy_localization = localization
            cleanup2.stypy_type_of_self = None
            cleanup2.stypy_type_store = module_type_store
            cleanup2.stypy_function_name = 'cleanup2'
            cleanup2.stypy_param_names_list = []
            cleanup2.stypy_varargs_param_name = None
            cleanup2.stypy_kwargs_param_name = None
            cleanup2.stypy_call_defaults = defaults
            cleanup2.stypy_call_varargs = varargs
            cleanup2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup2', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup2', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup2(...)' code ##################

            
            # Call to append(...): (line 135)
            # Processing the call arguments (line 135)
            str_205929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 28), 'str', 'cleanup2')
            # Processing the call keyword arguments (line 135)
            kwargs_205930 = {}
            # Getting the type of 'ordering' (line 135)
            ordering_205927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'ordering', False)
            # Obtaining the member 'append' of a type (line 135)
            append_205928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), ordering_205927, 'append')
            # Calling append(args, kwargs) (line 135)
            append_call_result_205931 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), append_205928, *[str_205929], **kwargs_205930)
            
            
            # ################# End of 'cleanup2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup2' in the type store
            # Getting the type of 'stypy_return_type' (line 134)
            stypy_return_type_205932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_205932)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup2'
            return stypy_return_type_205932

        # Assigning a type to the variable 'cleanup2' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'cleanup2', cleanup2)
        
        # Call to debug(...): (line 137)
        # Processing the call keyword arguments (line 137)
        kwargs_205935 = {}
        # Getting the type of 'test' (line 137)
        test_205933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'test', False)
        # Obtaining the member 'debug' of a type (line 137)
        debug_205934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), test_205933, 'debug')
        # Calling debug(args, kwargs) (line 137)
        debug_call_result_205936 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), debug_205934, *[], **kwargs_205935)
        
        
        # Call to assertEqual(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'ordering' (line 138)
        ordering_205939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'ordering', False)
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_205940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        str_205941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 36), 'str', 'setUp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), list_205940, str_205941)
        # Adding element type (line 138)
        str_205942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 45), 'str', 'test')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), list_205940, str_205942)
        # Adding element type (line 138)
        str_205943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 53), 'str', 'tearDown')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), list_205940, str_205943)
        # Adding element type (line 138)
        str_205944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 65), 'str', 'cleanup1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), list_205940, str_205944)
        # Adding element type (line 138)
        str_205945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 77), 'str', 'cleanup2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), list_205940, str_205945)
        
        # Processing the call keyword arguments (line 138)
        kwargs_205946 = {}
        # Getting the type of 'self' (line 138)
        self_205937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 138)
        assertEqual_205938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_205937, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 138)
        assertEqual_call_result_205947 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), assertEqual_205938, *[ordering_205939, list_205940], **kwargs_205946)
        
        
        # ################# End of 'testTestCaseDebugExecutesCleanups(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testTestCaseDebugExecutesCleanups' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_205948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testTestCaseDebugExecutesCleanups'
        return stypy_return_type_205948


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCleanUp.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCleanUp' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'TestCleanUp', TestCleanUp)
# Declaration of the 'Test_TextTestRunner' class
# Getting the type of 'unittest' (line 141)
unittest_205949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'unittest')
# Obtaining the member 'TestCase' of a type (line 141)
TestCase_205950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 26), unittest_205949, 'TestCase')

class Test_TextTestRunner(TestCase_205950, ):
    str_205951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'str', 'Tests for TextTestRunner.')

    @norecursion
    def test_init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_init'
        module_type_store = module_type_store.open_function_context('test_init', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_localization', localization)
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_function_name', 'Test_TextTestRunner.test_init')
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TextTestRunner.test_init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TextTestRunner.test_init', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to TextTestRunner(...): (line 145)
        # Processing the call keyword arguments (line 145)
        kwargs_205954 = {}
        # Getting the type of 'unittest' (line 145)
        unittest_205952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 145)
        TextTestRunner_205953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 17), unittest_205952, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 145)
        TextTestRunner_call_result_205955 = invoke(stypy.reporting.localization.Localization(__file__, 145, 17), TextTestRunner_205953, *[], **kwargs_205954)
        
        # Assigning a type to the variable 'runner' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'runner', TextTestRunner_call_result_205955)
        
        # Call to assertFalse(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'runner' (line 146)
        runner_205958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'runner', False)
        # Obtaining the member 'failfast' of a type (line 146)
        failfast_205959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 25), runner_205958, 'failfast')
        # Processing the call keyword arguments (line 146)
        kwargs_205960 = {}
        # Getting the type of 'self' (line 146)
        self_205956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 146)
        assertFalse_205957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_205956, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 146)
        assertFalse_call_result_205961 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), assertFalse_205957, *[failfast_205959], **kwargs_205960)
        
        
        # Call to assertFalse(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'runner' (line 147)
        runner_205964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'runner', False)
        # Obtaining the member 'buffer' of a type (line 147)
        buffer_205965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), runner_205964, 'buffer')
        # Processing the call keyword arguments (line 147)
        kwargs_205966 = {}
        # Getting the type of 'self' (line 147)
        self_205962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 147)
        assertFalse_205963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_205962, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 147)
        assertFalse_call_result_205967 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assertFalse_205963, *[buffer_205965], **kwargs_205966)
        
        
        # Call to assertEqual(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'runner' (line 148)
        runner_205970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'runner', False)
        # Obtaining the member 'verbosity' of a type (line 148)
        verbosity_205971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 25), runner_205970, 'verbosity')
        int_205972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 43), 'int')
        # Processing the call keyword arguments (line 148)
        kwargs_205973 = {}
        # Getting the type of 'self' (line 148)
        self_205968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 148)
        assertEqual_205969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_205968, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 148)
        assertEqual_call_result_205974 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), assertEqual_205969, *[verbosity_205971, int_205972], **kwargs_205973)
        
        
        # Call to assertTrue(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'runner' (line 149)
        runner_205977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'runner', False)
        # Obtaining the member 'descriptions' of a type (line 149)
        descriptions_205978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), runner_205977, 'descriptions')
        # Processing the call keyword arguments (line 149)
        kwargs_205979 = {}
        # Getting the type of 'self' (line 149)
        self_205975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 149)
        assertTrue_205976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_205975, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 149)
        assertTrue_call_result_205980 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assertTrue_205976, *[descriptions_205978], **kwargs_205979)
        
        
        # Call to assertEqual(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'runner' (line 150)
        runner_205983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 25), 'runner', False)
        # Obtaining the member 'resultclass' of a type (line 150)
        resultclass_205984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 25), runner_205983, 'resultclass')
        # Getting the type of 'unittest' (line 150)
        unittest_205985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 45), 'unittest', False)
        # Obtaining the member 'TextTestResult' of a type (line 150)
        TextTestResult_205986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 45), unittest_205985, 'TextTestResult')
        # Processing the call keyword arguments (line 150)
        kwargs_205987 = {}
        # Getting the type of 'self' (line 150)
        self_205981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 150)
        assertEqual_205982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_205981, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 150)
        assertEqual_call_result_205988 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), assertEqual_205982, *[resultclass_205984, TextTestResult_205986], **kwargs_205987)
        
        
        # ################# End of 'test_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_init' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_205989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205989)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_init'
        return stypy_return_type_205989


    @norecursion
    def test_multiple_inheritance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_multiple_inheritance'
        module_type_store = module_type_store.open_function_context('test_multiple_inheritance', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_localization', localization)
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_function_name', 'Test_TextTestRunner.test_multiple_inheritance')
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TextTestRunner.test_multiple_inheritance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TextTestRunner.test_multiple_inheritance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_multiple_inheritance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_multiple_inheritance(...)' code ##################

        # Declaration of the 'AResult' class
        # Getting the type of 'unittest' (line 154)
        unittest_205990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'unittest')
        # Obtaining the member 'TestResult' of a type (line 154)
        TestResult_205991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 22), unittest_205990, 'TestResult')

        class AResult(TestResult_205991, ):

            @norecursion
            def __init__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__init__'
                module_type_store = module_type_store.open_function_context('__init__', 155, 12, False)
                # Assigning a type to the variable 'self' (line 156)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'AResult.__init__', ['stream', 'descriptions', 'verbosity'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return

                # Initialize method data
                init_call_information(module_type_store, '__init__', localization, ['stream', 'descriptions', 'verbosity'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of '__init__(...)' code ##################

                
                # Call to __init__(...): (line 156)
                # Processing the call arguments (line 156)
                # Getting the type of 'stream' (line 156)
                stream_205998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 46), 'stream', False)
                # Getting the type of 'descriptions' (line 156)
                descriptions_205999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 54), 'descriptions', False)
                # Getting the type of 'verbosity' (line 156)
                verbosity_206000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 68), 'verbosity', False)
                # Processing the call keyword arguments (line 156)
                kwargs_206001 = {}
                
                # Call to super(...): (line 156)
                # Processing the call arguments (line 156)
                # Getting the type of 'AResult' (line 156)
                AResult_205993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'AResult', False)
                # Getting the type of 'self' (line 156)
                self_205994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'self', False)
                # Processing the call keyword arguments (line 156)
                kwargs_205995 = {}
                # Getting the type of 'super' (line 156)
                super_205992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'super', False)
                # Calling super(args, kwargs) (line 156)
                super_call_result_205996 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), super_205992, *[AResult_205993, self_205994], **kwargs_205995)
                
                # Obtaining the member '__init__' of a type (line 156)
                init___205997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), super_call_result_205996, '__init__')
                # Calling __init__(args, kwargs) (line 156)
                init___call_result_206002 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), init___205997, *[stream_205998, descriptions_205999, verbosity_206000], **kwargs_206001)
                
                
                # ################# End of '__init__(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()

        
        # Assigning a type to the variable 'AResult' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'AResult', AResult)
        # Declaration of the 'ATextResult' class
        # Getting the type of 'unittest' (line 158)
        unittest_206003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'unittest')
        # Obtaining the member 'TextTestResult' of a type (line 158)
        TextTestResult_206004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 26), unittest_206003, 'TextTestResult')
        # Getting the type of 'AResult' (line 158)
        AResult_206005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 51), 'AResult')

        class ATextResult(TextTestResult_206004, AResult_206005, ):
            pass
        
        # Assigning a type to the variable 'ATextResult' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'ATextResult', ATextResult)
        
        # Call to ATextResult(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'None' (line 163)
        None_206007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'None', False)
        # Getting the type of 'None' (line 163)
        None_206008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'None', False)
        int_206009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 32), 'int')
        # Processing the call keyword arguments (line 163)
        kwargs_206010 = {}
        # Getting the type of 'ATextResult' (line 163)
        ATextResult_206006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'ATextResult', False)
        # Calling ATextResult(args, kwargs) (line 163)
        ATextResult_call_result_206011 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), ATextResult_206006, *[None_206007, None_206008, int_206009], **kwargs_206010)
        
        
        # ################# End of 'test_multiple_inheritance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_multiple_inheritance' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_206012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206012)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_multiple_inheritance'
        return stypy_return_type_206012


    @norecursion
    def testBufferAndFailfast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBufferAndFailfast'
        module_type_store = module_type_store.open_function_context('testBufferAndFailfast', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_localization', localization)
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_function_name', 'Test_TextTestRunner.testBufferAndFailfast')
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TextTestRunner.testBufferAndFailfast.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TextTestRunner.testBufferAndFailfast', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBufferAndFailfast', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBufferAndFailfast(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 167)
        unittest_206013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 167)
        TestCase_206014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 19), unittest_206013, 'TestCase')

        class Test(TestCase_206014, ):

            @norecursion
            def testFoo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testFoo'
                module_type_store = module_type_store.open_function_context('testFoo', 168, 12, False)
                # Assigning a type to the variable 'self' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'self', type_of_self)
                
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
                # Getting the type of 'stypy_return_type' (line 168)
                stypy_return_type_206015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206015)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testFoo'
                return stypy_return_type_206015

        
        # Assigning a type to the variable 'Test' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'Test', Test)
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Call to TestResult(...): (line 170)
        # Processing the call keyword arguments (line 170)
        kwargs_206018 = {}
        # Getting the type of 'unittest' (line 170)
        unittest_206016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 170)
        TestResult_206017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 17), unittest_206016, 'TestResult')
        # Calling TestResult(args, kwargs) (line 170)
        TestResult_call_result_206019 = invoke(stypy.reporting.localization.Localization(__file__, 170, 17), TestResult_206017, *[], **kwargs_206018)
        
        # Assigning a type to the variable 'result' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'result', TestResult_call_result_206019)
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to TextTestRunner(...): (line 171)
        # Processing the call keyword arguments (line 171)
        
        # Call to StringIO(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_206023 = {}
        # Getting the type of 'StringIO' (line 171)
        StringIO_206022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 48), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 171)
        StringIO_call_result_206024 = invoke(stypy.reporting.localization.Localization(__file__, 171, 48), StringIO_206022, *[], **kwargs_206023)
        
        keyword_206025 = StringIO_call_result_206024
        # Getting the type of 'True' (line 171)
        True_206026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 69), 'True', False)
        keyword_206027 = True_206026
        # Getting the type of 'True' (line 172)
        True_206028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 50), 'True', False)
        keyword_206029 = True_206028
        kwargs_206030 = {'buffer': keyword_206029, 'failfast': keyword_206027, 'stream': keyword_206025}
        # Getting the type of 'unittest' (line 171)
        unittest_206020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 171)
        TextTestRunner_206021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), unittest_206020, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 171)
        TextTestRunner_call_result_206031 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), TextTestRunner_206021, *[], **kwargs_206030)
        
        # Assigning a type to the variable 'runner' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'runner', TextTestRunner_call_result_206031)
        
        # Assigning a Lambda to a Attribute (line 174):
        
        # Assigning a Lambda to a Attribute (line 174):
        
        # Assigning a Lambda to a Attribute (line 174):
        
        # Assigning a Lambda to a Attribute (line 174):

        @norecursion
        def _stypy_temp_lambda_94(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_94'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_94', 174, 29, True)
            # Passed parameters checking function
            _stypy_temp_lambda_94.stypy_localization = localization
            _stypy_temp_lambda_94.stypy_type_of_self = None
            _stypy_temp_lambda_94.stypy_type_store = module_type_store
            _stypy_temp_lambda_94.stypy_function_name = '_stypy_temp_lambda_94'
            _stypy_temp_lambda_94.stypy_param_names_list = []
            _stypy_temp_lambda_94.stypy_varargs_param_name = None
            _stypy_temp_lambda_94.stypy_kwargs_param_name = None
            _stypy_temp_lambda_94.stypy_call_defaults = defaults
            _stypy_temp_lambda_94.stypy_call_varargs = varargs
            _stypy_temp_lambda_94.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_94', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_94', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'result' (line 174)
            result_206032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 37), 'result')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'stypy_return_type', result_206032)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_94' in the type store
            # Getting the type of 'stypy_return_type' (line 174)
            stypy_return_type_206033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_206033)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_94'
            return stypy_return_type_206033

        # Assigning a type to the variable '_stypy_temp_lambda_94' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), '_stypy_temp_lambda_94', _stypy_temp_lambda_94)
        # Getting the type of '_stypy_temp_lambda_94' (line 174)
        _stypy_temp_lambda_94_206034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), '_stypy_temp_lambda_94')
        # Getting the type of 'runner' (line 174)
        runner_206035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'runner')
        # Setting the type of the member '_makeResult' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), runner_206035, '_makeResult', _stypy_temp_lambda_94_206034)
        
        # Call to run(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Call to Test(...): (line 175)
        # Processing the call arguments (line 175)
        str_206039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'str', 'testFoo')
        # Processing the call keyword arguments (line 175)
        kwargs_206040 = {}
        # Getting the type of 'Test' (line 175)
        Test_206038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 19), 'Test', False)
        # Calling Test(args, kwargs) (line 175)
        Test_call_result_206041 = invoke(stypy.reporting.localization.Localization(__file__, 175, 19), Test_206038, *[str_206039], **kwargs_206040)
        
        # Processing the call keyword arguments (line 175)
        kwargs_206042 = {}
        # Getting the type of 'runner' (line 175)
        runner_206036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'runner', False)
        # Obtaining the member 'run' of a type (line 175)
        run_206037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), runner_206036, 'run')
        # Calling run(args, kwargs) (line 175)
        run_call_result_206043 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), run_206037, *[Test_call_result_206041], **kwargs_206042)
        
        
        # Call to assertTrue(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'result' (line 177)
        result_206046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'result', False)
        # Obtaining the member 'failfast' of a type (line 177)
        failfast_206047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 24), result_206046, 'failfast')
        # Processing the call keyword arguments (line 177)
        kwargs_206048 = {}
        # Getting the type of 'self' (line 177)
        self_206044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 177)
        assertTrue_206045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_206044, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 177)
        assertTrue_call_result_206049 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assertTrue_206045, *[failfast_206047], **kwargs_206048)
        
        
        # Call to assertTrue(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'result' (line 178)
        result_206052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'result', False)
        # Obtaining the member 'buffer' of a type (line 178)
        buffer_206053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 24), result_206052, 'buffer')
        # Processing the call keyword arguments (line 178)
        kwargs_206054 = {}
        # Getting the type of 'self' (line 178)
        self_206050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 178)
        assertTrue_206051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_206050, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 178)
        assertTrue_call_result_206055 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), assertTrue_206051, *[buffer_206053], **kwargs_206054)
        
        
        # ################# End of 'testBufferAndFailfast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBufferAndFailfast' in the type store
        # Getting the type of 'stypy_return_type' (line 166)
        stypy_return_type_206056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206056)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBufferAndFailfast'
        return stypy_return_type_206056


    @norecursion
    def testRunnerRegistersResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testRunnerRegistersResult'
        module_type_store = module_type_store.open_function_context('testRunnerRegistersResult', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_localization', localization)
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_function_name', 'Test_TextTestRunner.testRunnerRegistersResult')
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TextTestRunner.testRunnerRegistersResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TextTestRunner.testRunnerRegistersResult', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testRunnerRegistersResult', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testRunnerRegistersResult(...)' code ##################

        # Declaration of the 'Test' class
        # Getting the type of 'unittest' (line 181)
        unittest_206057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 181)
        TestCase_206058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 19), unittest_206057, 'TestCase')

        class Test(TestCase_206058, ):

            @norecursion
            def testFoo(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testFoo'
                module_type_store = module_type_store.open_function_context('testFoo', 182, 12, False)
                # Assigning a type to the variable 'self' (line 183)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'self', type_of_self)
                
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
                # Getting the type of 'stypy_return_type' (line 182)
                stypy_return_type_206059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206059)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testFoo'
                return stypy_return_type_206059

        
        # Assigning a type to the variable 'Test' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'Test', Test)
        
        # Assigning a Attribute to a Name (line 184):
        
        # Assigning a Attribute to a Name (line 184):
        
        # Assigning a Attribute to a Name (line 184):
        
        # Assigning a Attribute to a Name (line 184):
        # Getting the type of 'unittest' (line 184)
        unittest_206060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'unittest')
        # Obtaining the member 'runner' of a type (line 184)
        runner_206061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 33), unittest_206060, 'runner')
        # Obtaining the member 'registerResult' of a type (line 184)
        registerResult_206062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 33), runner_206061, 'registerResult')
        # Assigning a type to the variable 'originalRegisterResult' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'originalRegisterResult', registerResult_206062)

        @norecursion
        def cleanup(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup'
            module_type_store = module_type_store.open_function_context('cleanup', 185, 8, False)
            
            # Passed parameters checking function
            cleanup.stypy_localization = localization
            cleanup.stypy_type_of_self = None
            cleanup.stypy_type_store = module_type_store
            cleanup.stypy_function_name = 'cleanup'
            cleanup.stypy_param_names_list = []
            cleanup.stypy_varargs_param_name = None
            cleanup.stypy_kwargs_param_name = None
            cleanup.stypy_call_defaults = defaults
            cleanup.stypy_call_varargs = varargs
            cleanup.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup(...)' code ##################

            
            # Assigning a Name to a Attribute (line 186):
            
            # Assigning a Name to a Attribute (line 186):
            
            # Assigning a Name to a Attribute (line 186):
            
            # Assigning a Name to a Attribute (line 186):
            # Getting the type of 'originalRegisterResult' (line 186)
            originalRegisterResult_206063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 45), 'originalRegisterResult')
            # Getting the type of 'unittest' (line 186)
            unittest_206064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'unittest')
            # Obtaining the member 'runner' of a type (line 186)
            runner_206065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), unittest_206064, 'runner')
            # Setting the type of the member 'registerResult' of a type (line 186)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), runner_206065, 'registerResult', originalRegisterResult_206063)
            
            # ################# End of 'cleanup(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup' in the type store
            # Getting the type of 'stypy_return_type' (line 185)
            stypy_return_type_206066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_206066)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup'
            return stypy_return_type_206066

        # Assigning a type to the variable 'cleanup' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'cleanup', cleanup)
        
        # Call to addCleanup(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'cleanup' (line 187)
        cleanup_206069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'cleanup', False)
        # Processing the call keyword arguments (line 187)
        kwargs_206070 = {}
        # Getting the type of 'self' (line 187)
        self_206067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 187)
        addCleanup_206068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_206067, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 187)
        addCleanup_call_result_206071 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), addCleanup_206068, *[cleanup_206069], **kwargs_206070)
        
        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Call to TestResult(...): (line 189)
        # Processing the call keyword arguments (line 189)
        kwargs_206074 = {}
        # Getting the type of 'unittest' (line 189)
        unittest_206072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 189)
        TestResult_206073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 17), unittest_206072, 'TestResult')
        # Calling TestResult(args, kwargs) (line 189)
        TestResult_call_result_206075 = invoke(stypy.reporting.localization.Localization(__file__, 189, 17), TestResult_206073, *[], **kwargs_206074)
        
        # Assigning a type to the variable 'result' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'result', TestResult_call_result_206075)
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to TextTestRunner(...): (line 190)
        # Processing the call keyword arguments (line 190)
        
        # Call to StringIO(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_206079 = {}
        # Getting the type of 'StringIO' (line 190)
        StringIO_206078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 48), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 190)
        StringIO_call_result_206080 = invoke(stypy.reporting.localization.Localization(__file__, 190, 48), StringIO_206078, *[], **kwargs_206079)
        
        keyword_206081 = StringIO_call_result_206080
        kwargs_206082 = {'stream': keyword_206081}
        # Getting the type of 'unittest' (line 190)
        unittest_206076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 190)
        TextTestRunner_206077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 17), unittest_206076, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 190)
        TextTestRunner_call_result_206083 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), TextTestRunner_206077, *[], **kwargs_206082)
        
        # Assigning a type to the variable 'runner' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'runner', TextTestRunner_call_result_206083)
        
        # Assigning a Lambda to a Attribute (line 192):
        
        # Assigning a Lambda to a Attribute (line 192):
        
        # Assigning a Lambda to a Attribute (line 192):
        
        # Assigning a Lambda to a Attribute (line 192):

        @norecursion
        def _stypy_temp_lambda_95(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_95'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_95', 192, 29, True)
            # Passed parameters checking function
            _stypy_temp_lambda_95.stypy_localization = localization
            _stypy_temp_lambda_95.stypy_type_of_self = None
            _stypy_temp_lambda_95.stypy_type_store = module_type_store
            _stypy_temp_lambda_95.stypy_function_name = '_stypy_temp_lambda_95'
            _stypy_temp_lambda_95.stypy_param_names_list = []
            _stypy_temp_lambda_95.stypy_varargs_param_name = None
            _stypy_temp_lambda_95.stypy_kwargs_param_name = None
            _stypy_temp_lambda_95.stypy_call_defaults = defaults
            _stypy_temp_lambda_95.stypy_call_varargs = varargs
            _stypy_temp_lambda_95.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_95', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_95', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'result' (line 192)
            result_206084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 37), 'result')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 29), 'stypy_return_type', result_206084)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_95' in the type store
            # Getting the type of 'stypy_return_type' (line 192)
            stypy_return_type_206085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 29), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_206085)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_95'
            return stypy_return_type_206085

        # Assigning a type to the variable '_stypy_temp_lambda_95' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 29), '_stypy_temp_lambda_95', _stypy_temp_lambda_95)
        # Getting the type of '_stypy_temp_lambda_95' (line 192)
        _stypy_temp_lambda_95_206086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 29), '_stypy_temp_lambda_95')
        # Getting the type of 'runner' (line 192)
        runner_206087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'runner')
        # Setting the type of the member '_makeResult' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), runner_206087, '_makeResult', _stypy_temp_lambda_95_206086)
        
        # Assigning a Num to a Attribute (line 194):
        
        # Assigning a Num to a Attribute (line 194):
        
        # Assigning a Num to a Attribute (line 194):
        
        # Assigning a Num to a Attribute (line 194):
        int_206088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 29), 'int')
        # Getting the type of 'self' (line 194)
        self_206089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self')
        # Setting the type of the member 'wasRegistered' of a type (line 194)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_206089, 'wasRegistered', int_206088)

        @norecursion
        def fakeRegisterResult(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fakeRegisterResult'
            module_type_store = module_type_store.open_function_context('fakeRegisterResult', 195, 8, False)
            
            # Passed parameters checking function
            fakeRegisterResult.stypy_localization = localization
            fakeRegisterResult.stypy_type_of_self = None
            fakeRegisterResult.stypy_type_store = module_type_store
            fakeRegisterResult.stypy_function_name = 'fakeRegisterResult'
            fakeRegisterResult.stypy_param_names_list = ['thisResult']
            fakeRegisterResult.stypy_varargs_param_name = None
            fakeRegisterResult.stypy_kwargs_param_name = None
            fakeRegisterResult.stypy_call_defaults = defaults
            fakeRegisterResult.stypy_call_varargs = varargs
            fakeRegisterResult.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fakeRegisterResult', ['thisResult'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fakeRegisterResult', localization, ['thisResult'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fakeRegisterResult(...)' code ##################

            
            # Getting the type of 'self' (line 196)
            self_206090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'self')
            # Obtaining the member 'wasRegistered' of a type (line 196)
            wasRegistered_206091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), self_206090, 'wasRegistered')
            int_206092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 34), 'int')
            # Applying the binary operator '+=' (line 196)
            result_iadd_206093 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 12), '+=', wasRegistered_206091, int_206092)
            # Getting the type of 'self' (line 196)
            self_206094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'self')
            # Setting the type of the member 'wasRegistered' of a type (line 196)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), self_206094, 'wasRegistered', result_iadd_206093)
            
            
            # Call to assertEqual(...): (line 197)
            # Processing the call arguments (line 197)
            # Getting the type of 'thisResult' (line 197)
            thisResult_206097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 29), 'thisResult', False)
            # Getting the type of 'result' (line 197)
            result_206098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 41), 'result', False)
            # Processing the call keyword arguments (line 197)
            kwargs_206099 = {}
            # Getting the type of 'self' (line 197)
            self_206095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 197)
            assertEqual_206096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), self_206095, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 197)
            assertEqual_call_result_206100 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), assertEqual_206096, *[thisResult_206097, result_206098], **kwargs_206099)
            
            
            # ################# End of 'fakeRegisterResult(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fakeRegisterResult' in the type store
            # Getting the type of 'stypy_return_type' (line 195)
            stypy_return_type_206101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_206101)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fakeRegisterResult'
            return stypy_return_type_206101

        # Assigning a type to the variable 'fakeRegisterResult' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'fakeRegisterResult', fakeRegisterResult)
        
        # Assigning a Name to a Attribute (line 198):
        
        # Assigning a Name to a Attribute (line 198):
        
        # Assigning a Name to a Attribute (line 198):
        
        # Assigning a Name to a Attribute (line 198):
        # Getting the type of 'fakeRegisterResult' (line 198)
        fakeRegisterResult_206102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 41), 'fakeRegisterResult')
        # Getting the type of 'unittest' (line 198)
        unittest_206103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'unittest')
        # Obtaining the member 'runner' of a type (line 198)
        runner_206104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), unittest_206103, 'runner')
        # Setting the type of the member 'registerResult' of a type (line 198)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), runner_206104, 'registerResult', fakeRegisterResult_206102)
        
        # Call to run(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Call to TestSuite(...): (line 200)
        # Processing the call keyword arguments (line 200)
        kwargs_206109 = {}
        # Getting the type of 'unittest' (line 200)
        unittest_206107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 200)
        TestSuite_206108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 19), unittest_206107, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 200)
        TestSuite_call_result_206110 = invoke(stypy.reporting.localization.Localization(__file__, 200, 19), TestSuite_206108, *[], **kwargs_206109)
        
        # Processing the call keyword arguments (line 200)
        kwargs_206111 = {}
        # Getting the type of 'runner' (line 200)
        runner_206105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'runner', False)
        # Obtaining the member 'run' of a type (line 200)
        run_206106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), runner_206105, 'run')
        # Calling run(args, kwargs) (line 200)
        run_call_result_206112 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), run_206106, *[TestSuite_call_result_206110], **kwargs_206111)
        
        
        # Call to assertEqual(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'self' (line 201)
        self_206115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'self', False)
        # Obtaining the member 'wasRegistered' of a type (line 201)
        wasRegistered_206116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 25), self_206115, 'wasRegistered')
        int_206117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 45), 'int')
        # Processing the call keyword arguments (line 201)
        kwargs_206118 = {}
        # Getting the type of 'self' (line 201)
        self_206113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 201)
        assertEqual_206114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), self_206113, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 201)
        assertEqual_call_result_206119 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), assertEqual_206114, *[wasRegistered_206116, int_206117], **kwargs_206118)
        
        
        # ################# End of 'testRunnerRegistersResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testRunnerRegistersResult' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_206120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206120)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testRunnerRegistersResult'
        return stypy_return_type_206120


    @norecursion
    def test_works_with_result_without_startTestRun_stopTestRun(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_works_with_result_without_startTestRun_stopTestRun'
        module_type_store = module_type_store.open_function_context('test_works_with_result_without_startTestRun_stopTestRun', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_localization', localization)
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_function_name', 'Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun')
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TextTestRunner.test_works_with_result_without_startTestRun_stopTestRun', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_works_with_result_without_startTestRun_stopTestRun', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_works_with_result_without_startTestRun_stopTestRun(...)' code ##################

        # Declaration of the 'OldTextResult' class
        # Getting the type of 'ResultWithNoStartTestRunStopTestRun' (line 204)
        ResultWithNoStartTestRunStopTestRun_206121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 28), 'ResultWithNoStartTestRunStopTestRun')

        class OldTextResult(ResultWithNoStartTestRunStopTestRun_206121, ):
            
            # Assigning a Str to a Name (line 205):
            
            # Assigning a Str to a Name (line 205):
            
            # Assigning a Str to a Name (line 205):
            
            # Assigning a Str to a Name (line 205):
            str_206122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 25), 'str', '')
            # Assigning a type to the variable 'separator2' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'separator2', str_206122)

            @norecursion
            def printErrors(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'printErrors'
                module_type_store = module_type_store.open_function_context('printErrors', 206, 12, False)
                # Assigning a type to the variable 'self' (line 207)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                OldTextResult.printErrors.__dict__.__setitem__('stypy_localization', localization)
                OldTextResult.printErrors.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                OldTextResult.printErrors.__dict__.__setitem__('stypy_type_store', module_type_store)
                OldTextResult.printErrors.__dict__.__setitem__('stypy_function_name', 'OldTextResult.printErrors')
                OldTextResult.printErrors.__dict__.__setitem__('stypy_param_names_list', [])
                OldTextResult.printErrors.__dict__.__setitem__('stypy_varargs_param_name', None)
                OldTextResult.printErrors.__dict__.__setitem__('stypy_kwargs_param_name', None)
                OldTextResult.printErrors.__dict__.__setitem__('stypy_call_defaults', defaults)
                OldTextResult.printErrors.__dict__.__setitem__('stypy_call_varargs', varargs)
                OldTextResult.printErrors.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                OldTextResult.printErrors.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'OldTextResult.printErrors', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'printErrors', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'printErrors(...)' code ##################

                pass
                
                # ################# End of 'printErrors(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'printErrors' in the type store
                # Getting the type of 'stypy_return_type' (line 206)
                stypy_return_type_206123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206123)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'printErrors'
                return stypy_return_type_206123

        
        # Assigning a type to the variable 'OldTextResult' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'OldTextResult', OldTextResult)
        # Declaration of the 'Runner' class
        # Getting the type of 'unittest' (line 209)
        unittest_206124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'unittest')
        # Obtaining the member 'TextTestRunner' of a type (line 209)
        TextTestRunner_206125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 21), unittest_206124, 'TextTestRunner')

        class Runner(TextTestRunner_206125, ):

            @norecursion
            def __init__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__init__'
                module_type_store = module_type_store.open_function_context('__init__', 210, 12, False)
                # Assigning a type to the variable 'self' (line 211)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Runner.__init__', [], None, None, defaults, varargs, kwargs)

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

                
                # Call to __init__(...): (line 211)
                # Processing the call arguments (line 211)
                
                # Call to StringIO(...): (line 211)
                # Processing the call keyword arguments (line 211)
                kwargs_206133 = {}
                # Getting the type of 'StringIO' (line 211)
                StringIO_206132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 45), 'StringIO', False)
                # Calling StringIO(args, kwargs) (line 211)
                StringIO_call_result_206134 = invoke(stypy.reporting.localization.Localization(__file__, 211, 45), StringIO_206132, *[], **kwargs_206133)
                
                # Processing the call keyword arguments (line 211)
                kwargs_206135 = {}
                
                # Call to super(...): (line 211)
                # Processing the call arguments (line 211)
                # Getting the type of 'Runner' (line 211)
                Runner_206127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'Runner', False)
                # Getting the type of 'self' (line 211)
                self_206128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'self', False)
                # Processing the call keyword arguments (line 211)
                kwargs_206129 = {}
                # Getting the type of 'super' (line 211)
                super_206126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'super', False)
                # Calling super(args, kwargs) (line 211)
                super_call_result_206130 = invoke(stypy.reporting.localization.Localization(__file__, 211, 16), super_206126, *[Runner_206127, self_206128], **kwargs_206129)
                
                # Obtaining the member '__init__' of a type (line 211)
                init___206131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 16), super_call_result_206130, '__init__')
                # Calling __init__(args, kwargs) (line 211)
                init___call_result_206136 = invoke(stypy.reporting.localization.Localization(__file__, 211, 16), init___206131, *[StringIO_call_result_206134], **kwargs_206135)
                
                
                # ################# End of '__init__(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()


            @norecursion
            def _makeResult(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_makeResult'
                module_type_store = module_type_store.open_function_context('_makeResult', 213, 12, False)
                # Assigning a type to the variable 'self' (line 214)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Runner._makeResult.__dict__.__setitem__('stypy_localization', localization)
                Runner._makeResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Runner._makeResult.__dict__.__setitem__('stypy_type_store', module_type_store)
                Runner._makeResult.__dict__.__setitem__('stypy_function_name', 'Runner._makeResult')
                Runner._makeResult.__dict__.__setitem__('stypy_param_names_list', [])
                Runner._makeResult.__dict__.__setitem__('stypy_varargs_param_name', None)
                Runner._makeResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Runner._makeResult.__dict__.__setitem__('stypy_call_defaults', defaults)
                Runner._makeResult.__dict__.__setitem__('stypy_call_varargs', varargs)
                Runner._makeResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Runner._makeResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Runner._makeResult', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, '_makeResult', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of '_makeResult(...)' code ##################

                
                # Call to OldTextResult(...): (line 214)
                # Processing the call keyword arguments (line 214)
                kwargs_206138 = {}
                # Getting the type of 'OldTextResult' (line 214)
                OldTextResult_206137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'OldTextResult', False)
                # Calling OldTextResult(args, kwargs) (line 214)
                OldTextResult_call_result_206139 = invoke(stypy.reporting.localization.Localization(__file__, 214, 23), OldTextResult_206137, *[], **kwargs_206138)
                
                # Assigning a type to the variable 'stypy_return_type' (line 214)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'stypy_return_type', OldTextResult_call_result_206139)
                
                # ################# End of '_makeResult(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function '_makeResult' in the type store
                # Getting the type of 'stypy_return_type' (line 213)
                stypy_return_type_206140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206140)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_makeResult'
                return stypy_return_type_206140

        
        # Assigning a type to the variable 'Runner' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'Runner', Runner)
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to Runner(...): (line 216)
        # Processing the call keyword arguments (line 216)
        kwargs_206142 = {}
        # Getting the type of 'Runner' (line 216)
        Runner_206141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'Runner', False)
        # Calling Runner(args, kwargs) (line 216)
        Runner_call_result_206143 = invoke(stypy.reporting.localization.Localization(__file__, 216, 17), Runner_206141, *[], **kwargs_206142)
        
        # Assigning a type to the variable 'runner' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'runner', Runner_call_result_206143)
        
        # Call to run(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Call to TestSuite(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_206148 = {}
        # Getting the type of 'unittest' (line 217)
        unittest_206146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 217)
        TestSuite_206147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 19), unittest_206146, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 217)
        TestSuite_call_result_206149 = invoke(stypy.reporting.localization.Localization(__file__, 217, 19), TestSuite_206147, *[], **kwargs_206148)
        
        # Processing the call keyword arguments (line 217)
        kwargs_206150 = {}
        # Getting the type of 'runner' (line 217)
        runner_206144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'runner', False)
        # Obtaining the member 'run' of a type (line 217)
        run_206145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), runner_206144, 'run')
        # Calling run(args, kwargs) (line 217)
        run_call_result_206151 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), run_206145, *[TestSuite_call_result_206149], **kwargs_206150)
        
        
        # ################# End of 'test_works_with_result_without_startTestRun_stopTestRun(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_works_with_result_without_startTestRun_stopTestRun' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_206152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206152)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_works_with_result_without_startTestRun_stopTestRun'
        return stypy_return_type_206152


    @norecursion
    def test_startTestRun_stopTestRun_called(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_startTestRun_stopTestRun_called'
        module_type_store = module_type_store.open_function_context('test_startTestRun_stopTestRun_called', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_localization', localization)
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_function_name', 'Test_TextTestRunner.test_startTestRun_stopTestRun_called')
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TextTestRunner.test_startTestRun_stopTestRun_called.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TextTestRunner.test_startTestRun_stopTestRun_called', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_startTestRun_stopTestRun_called', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_startTestRun_stopTestRun_called(...)' code ##################

        # Declaration of the 'LoggingTextResult' class
        # Getting the type of 'LoggingResult' (line 220)
        LoggingResult_206153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 32), 'LoggingResult')

        class LoggingTextResult(LoggingResult_206153, ):
            
            # Assigning a Str to a Name (line 221):
            
            # Assigning a Str to a Name (line 221):
            
            # Assigning a Str to a Name (line 221):
            
            # Assigning a Str to a Name (line 221):
            str_206154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 25), 'str', '')
            # Assigning a type to the variable 'separator2' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'separator2', str_206154)

            @norecursion
            def printErrors(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'printErrors'
                module_type_store = module_type_store.open_function_context('printErrors', 222, 12, False)
                # Assigning a type to the variable 'self' (line 223)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_localization', localization)
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_type_store', module_type_store)
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_function_name', 'LoggingTextResult.printErrors')
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_param_names_list', [])
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_varargs_param_name', None)
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_kwargs_param_name', None)
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_call_defaults', defaults)
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_call_varargs', varargs)
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                LoggingTextResult.printErrors.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingTextResult.printErrors', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'printErrors', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'printErrors(...)' code ##################

                pass
                
                # ################# End of 'printErrors(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'printErrors' in the type store
                # Getting the type of 'stypy_return_type' (line 222)
                stypy_return_type_206155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206155)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'printErrors'
                return stypy_return_type_206155

        
        # Assigning a type to the variable 'LoggingTextResult' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'LoggingTextResult', LoggingTextResult)
        # Declaration of the 'LoggingRunner' class
        # Getting the type of 'unittest' (line 225)
        unittest_206156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'unittest')
        # Obtaining the member 'TextTestRunner' of a type (line 225)
        TextTestRunner_206157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 28), unittest_206156, 'TextTestRunner')

        class LoggingRunner(TextTestRunner_206157, ):

            @norecursion
            def __init__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__init__'
                module_type_store = module_type_store.open_function_context('__init__', 226, 12, False)
                # Assigning a type to the variable 'self' (line 227)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingRunner.__init__', ['events'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return

                # Initialize method data
                init_call_information(module_type_store, '__init__', localization, ['events'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of '__init__(...)' code ##################

                
                # Call to __init__(...): (line 227)
                # Processing the call arguments (line 227)
                
                # Call to StringIO(...): (line 227)
                # Processing the call keyword arguments (line 227)
                kwargs_206165 = {}
                # Getting the type of 'StringIO' (line 227)
                StringIO_206164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'StringIO', False)
                # Calling StringIO(args, kwargs) (line 227)
                StringIO_call_result_206166 = invoke(stypy.reporting.localization.Localization(__file__, 227, 52), StringIO_206164, *[], **kwargs_206165)
                
                # Processing the call keyword arguments (line 227)
                kwargs_206167 = {}
                
                # Call to super(...): (line 227)
                # Processing the call arguments (line 227)
                # Getting the type of 'LoggingRunner' (line 227)
                LoggingRunner_206159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'LoggingRunner', False)
                # Getting the type of 'self' (line 227)
                self_206160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 37), 'self', False)
                # Processing the call keyword arguments (line 227)
                kwargs_206161 = {}
                # Getting the type of 'super' (line 227)
                super_206158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'super', False)
                # Calling super(args, kwargs) (line 227)
                super_call_result_206162 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), super_206158, *[LoggingRunner_206159, self_206160], **kwargs_206161)
                
                # Obtaining the member '__init__' of a type (line 227)
                init___206163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), super_call_result_206162, '__init__')
                # Calling __init__(args, kwargs) (line 227)
                init___call_result_206168 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), init___206163, *[StringIO_call_result_206166], **kwargs_206167)
                
                
                # Assigning a Name to a Attribute (line 228):
                
                # Assigning a Name to a Attribute (line 228):
                
                # Assigning a Name to a Attribute (line 228):
                
                # Assigning a Name to a Attribute (line 228):
                # Getting the type of 'events' (line 228)
                events_206169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 31), 'events')
                # Getting the type of 'self' (line 228)
                self_206170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'self')
                # Setting the type of the member '_events' of a type (line 228)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), self_206170, '_events', events_206169)
                
                # ################# End of '__init__(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()


            @norecursion
            def _makeResult(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_makeResult'
                module_type_store = module_type_store.open_function_context('_makeResult', 230, 12, False)
                # Assigning a type to the variable 'self' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_localization', localization)
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_type_store', module_type_store)
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_function_name', 'LoggingRunner._makeResult')
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_param_names_list', [])
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_varargs_param_name', None)
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_call_defaults', defaults)
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_call_varargs', varargs)
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                LoggingRunner._makeResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingRunner._makeResult', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, '_makeResult', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of '_makeResult(...)' code ##################

                
                # Call to LoggingTextResult(...): (line 231)
                # Processing the call arguments (line 231)
                # Getting the type of 'self' (line 231)
                self_206172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 41), 'self', False)
                # Obtaining the member '_events' of a type (line 231)
                _events_206173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 41), self_206172, '_events')
                # Processing the call keyword arguments (line 231)
                kwargs_206174 = {}
                # Getting the type of 'LoggingTextResult' (line 231)
                LoggingTextResult_206171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'LoggingTextResult', False)
                # Calling LoggingTextResult(args, kwargs) (line 231)
                LoggingTextResult_call_result_206175 = invoke(stypy.reporting.localization.Localization(__file__, 231, 23), LoggingTextResult_206171, *[_events_206173], **kwargs_206174)
                
                # Assigning a type to the variable 'stypy_return_type' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'stypy_return_type', LoggingTextResult_call_result_206175)
                
                # ################# End of '_makeResult(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function '_makeResult' in the type store
                # Getting the type of 'stypy_return_type' (line 230)
                stypy_return_type_206176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_206176)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_makeResult'
                return stypy_return_type_206176

        
        # Assigning a type to the variable 'LoggingRunner' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'LoggingRunner', LoggingRunner)
        
        # Assigning a List to a Name (line 233):
        
        # Assigning a List to a Name (line 233):
        
        # Assigning a List to a Name (line 233):
        
        # Assigning a List to a Name (line 233):
        
        # Obtaining an instance of the builtin type 'list' (line 233)
        list_206177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 233)
        
        # Assigning a type to the variable 'events' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'events', list_206177)
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Assigning a Call to a Name (line 234):
        
        # Call to LoggingRunner(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'events' (line 234)
        events_206179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 31), 'events', False)
        # Processing the call keyword arguments (line 234)
        kwargs_206180 = {}
        # Getting the type of 'LoggingRunner' (line 234)
        LoggingRunner_206178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'LoggingRunner', False)
        # Calling LoggingRunner(args, kwargs) (line 234)
        LoggingRunner_call_result_206181 = invoke(stypy.reporting.localization.Localization(__file__, 234, 17), LoggingRunner_206178, *[events_206179], **kwargs_206180)
        
        # Assigning a type to the variable 'runner' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'runner', LoggingRunner_call_result_206181)
        
        # Call to run(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Call to TestSuite(...): (line 235)
        # Processing the call keyword arguments (line 235)
        kwargs_206186 = {}
        # Getting the type of 'unittest' (line 235)
        unittest_206184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 235)
        TestSuite_206185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 19), unittest_206184, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 235)
        TestSuite_call_result_206187 = invoke(stypy.reporting.localization.Localization(__file__, 235, 19), TestSuite_206185, *[], **kwargs_206186)
        
        # Processing the call keyword arguments (line 235)
        kwargs_206188 = {}
        # Getting the type of 'runner' (line 235)
        runner_206182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'runner', False)
        # Obtaining the member 'run' of a type (line 235)
        run_206183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), runner_206182, 'run')
        # Calling run(args, kwargs) (line 235)
        run_call_result_206189 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), run_206183, *[TestSuite_call_result_206187], **kwargs_206188)
        
        
        # Assigning a List to a Name (line 236):
        
        # Assigning a List to a Name (line 236):
        
        # Assigning a List to a Name (line 236):
        
        # Assigning a List to a Name (line 236):
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_206190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        str_206191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 20), 'str', 'startTestRun')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 19), list_206190, str_206191)
        # Adding element type (line 236)
        str_206192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 36), 'str', 'stopTestRun')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 19), list_206190, str_206192)
        
        # Assigning a type to the variable 'expected' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'expected', list_206190)
        
        # Call to assertEqual(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'events' (line 237)
        events_206195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 25), 'events', False)
        # Getting the type of 'expected' (line 237)
        expected_206196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'expected', False)
        # Processing the call keyword arguments (line 237)
        kwargs_206197 = {}
        # Getting the type of 'self' (line 237)
        self_206193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 237)
        assertEqual_206194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_206193, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 237)
        assertEqual_call_result_206198 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), assertEqual_206194, *[events_206195, expected_206196], **kwargs_206197)
        
        
        # ################# End of 'test_startTestRun_stopTestRun_called(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_startTestRun_stopTestRun_called' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_206199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206199)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_startTestRun_stopTestRun_called'
        return stypy_return_type_206199


    @norecursion
    def test_pickle_unpickle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pickle_unpickle'
        module_type_store = module_type_store.open_function_context('test_pickle_unpickle', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_localization', localization)
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_function_name', 'Test_TextTestRunner.test_pickle_unpickle')
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TextTestRunner.test_pickle_unpickle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TextTestRunner.test_pickle_unpickle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pickle_unpickle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pickle_unpickle(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 242, 8))
        
        # 'from StringIO import PickleableIO' statement (line 242)
        from StringIO import StringIO as PickleableIO

        import_from_module(stypy.reporting.localization.Localization(__file__, 242, 8), 'StringIO', None, module_type_store, ['StringIO'], [PickleableIO])
        # Adding an alias
        module_type_store.add_alias('PickleableIO', 'StringIO')
        
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Call to PickleableIO(...): (line 244)
        # Processing the call arguments (line 244)
        str_206201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'foo')
        # Processing the call keyword arguments (line 244)
        kwargs_206202 = {}
        # Getting the type of 'PickleableIO' (line 244)
        PickleableIO_206200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 17), 'PickleableIO', False)
        # Calling PickleableIO(args, kwargs) (line 244)
        PickleableIO_call_result_206203 = invoke(stypy.reporting.localization.Localization(__file__, 244, 17), PickleableIO_206200, *[str_206201], **kwargs_206202)
        
        # Assigning a type to the variable 'stream' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'stream', PickleableIO_call_result_206203)
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to TextTestRunner(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'stream' (line 245)
        stream_206206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 41), 'stream', False)
        # Processing the call keyword arguments (line 245)
        kwargs_206207 = {}
        # Getting the type of 'unittest' (line 245)
        unittest_206204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 245)
        TextTestRunner_206205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 17), unittest_206204, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 245)
        TextTestRunner_call_result_206208 = invoke(stypy.reporting.localization.Localization(__file__, 245, 17), TextTestRunner_206205, *[stream_206206], **kwargs_206207)
        
        # Assigning a type to the variable 'runner' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'runner', TextTestRunner_call_result_206208)
        
        
        # Call to range(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'pickle' (line 246)
        pickle_206210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 30), 'pickle', False)
        # Obtaining the member 'HIGHEST_PROTOCOL' of a type (line 246)
        HIGHEST_PROTOCOL_206211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 30), pickle_206210, 'HIGHEST_PROTOCOL')
        int_206212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 56), 'int')
        # Applying the binary operator '+' (line 246)
        result_add_206213 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 30), '+', HIGHEST_PROTOCOL_206211, int_206212)
        
        # Processing the call keyword arguments (line 246)
        kwargs_206214 = {}
        # Getting the type of 'range' (line 246)
        range_206209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 24), 'range', False)
        # Calling range(args, kwargs) (line 246)
        range_call_result_206215 = invoke(stypy.reporting.localization.Localization(__file__, 246, 24), range_206209, *[result_add_206213], **kwargs_206214)
        
        # Testing the type of a for loop iterable (line 246)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 246, 8), range_call_result_206215)
        # Getting the type of the for loop variable (line 246)
        for_loop_var_206216 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 246, 8), range_call_result_206215)
        # Assigning a type to the variable 'protocol' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'protocol', for_loop_var_206216)
        # SSA begins for a for statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to dumps(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'runner' (line 247)
        runner_206219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'runner', False)
        # Processing the call keyword arguments (line 247)
        # Getting the type of 'protocol' (line 247)
        protocol_206220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 46), 'protocol', False)
        keyword_206221 = protocol_206220
        kwargs_206222 = {'protocol': keyword_206221}
        # Getting the type of 'pickle' (line 247)
        pickle_206217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'pickle', False)
        # Obtaining the member 'dumps' of a type (line 247)
        dumps_206218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 16), pickle_206217, 'dumps')
        # Calling dumps(args, kwargs) (line 247)
        dumps_call_result_206223 = invoke(stypy.reporting.localization.Localization(__file__, 247, 16), dumps_206218, *[runner_206219], **kwargs_206222)
        
        # Assigning a type to the variable 's' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 's', dumps_call_result_206223)
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to loads(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 's' (line 248)
        s_206226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 31), 's', False)
        # Processing the call keyword arguments (line 248)
        kwargs_206227 = {}
        # Getting the type of 'pickle' (line 248)
        pickle_206224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 18), 'pickle', False)
        # Obtaining the member 'loads' of a type (line 248)
        loads_206225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 18), pickle_206224, 'loads')
        # Calling loads(args, kwargs) (line 248)
        loads_call_result_206228 = invoke(stypy.reporting.localization.Localization(__file__, 248, 18), loads_206225, *[s_206226], **kwargs_206227)
        
        # Assigning a type to the variable 'obj' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'obj', loads_call_result_206228)
        
        # Call to assertEqual(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to getvalue(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_206234 = {}
        # Getting the type of 'obj' (line 250)
        obj_206231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 29), 'obj', False)
        # Obtaining the member 'stream' of a type (line 250)
        stream_206232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 29), obj_206231, 'stream')
        # Obtaining the member 'getvalue' of a type (line 250)
        getvalue_206233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 29), stream_206232, 'getvalue')
        # Calling getvalue(args, kwargs) (line 250)
        getvalue_call_result_206235 = invoke(stypy.reporting.localization.Localization(__file__, 250, 29), getvalue_206233, *[], **kwargs_206234)
        
        
        # Call to getvalue(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_206238 = {}
        # Getting the type of 'stream' (line 250)
        stream_206236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 52), 'stream', False)
        # Obtaining the member 'getvalue' of a type (line 250)
        getvalue_206237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 52), stream_206236, 'getvalue')
        # Calling getvalue(args, kwargs) (line 250)
        getvalue_call_result_206239 = invoke(stypy.reporting.localization.Localization(__file__, 250, 52), getvalue_206237, *[], **kwargs_206238)
        
        # Processing the call keyword arguments (line 250)
        kwargs_206240 = {}
        # Getting the type of 'self' (line 250)
        self_206229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 250)
        assertEqual_206230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), self_206229, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 250)
        assertEqual_call_result_206241 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), assertEqual_206230, *[getvalue_call_result_206235, getvalue_call_result_206239], **kwargs_206240)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_pickle_unpickle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pickle_unpickle' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_206242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206242)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pickle_unpickle'
        return stypy_return_type_206242


    @norecursion
    def test_resultclass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_resultclass'
        module_type_store = module_type_store.open_function_context('test_resultclass', 252, 4, False)
        # Assigning a type to the variable 'self' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_localization', localization)
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_function_name', 'Test_TextTestRunner.test_resultclass')
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TextTestRunner.test_resultclass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TextTestRunner.test_resultclass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_resultclass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_resultclass(...)' code ##################


        @norecursion
        def MockResultClass(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'MockResultClass'
            module_type_store = module_type_store.open_function_context('MockResultClass', 253, 8, False)
            
            # Passed parameters checking function
            MockResultClass.stypy_localization = localization
            MockResultClass.stypy_type_of_self = None
            MockResultClass.stypy_type_store = module_type_store
            MockResultClass.stypy_function_name = 'MockResultClass'
            MockResultClass.stypy_param_names_list = []
            MockResultClass.stypy_varargs_param_name = 'args'
            MockResultClass.stypy_kwargs_param_name = None
            MockResultClass.stypy_call_defaults = defaults
            MockResultClass.stypy_call_varargs = varargs
            MockResultClass.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'MockResultClass', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'MockResultClass', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'MockResultClass(...)' code ##################

            # Getting the type of 'args' (line 254)
            args_206243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'args')
            # Assigning a type to the variable 'stypy_return_type' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'stypy_return_type', args_206243)
            
            # ################# End of 'MockResultClass(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'MockResultClass' in the type store
            # Getting the type of 'stypy_return_type' (line 253)
            stypy_return_type_206244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_206244)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'MockResultClass'
            return stypy_return_type_206244

        # Assigning a type to the variable 'MockResultClass' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'MockResultClass', MockResultClass)
        
        # Assigning a Call to a Name (line 255):
        
        # Assigning a Call to a Name (line 255):
        
        # Assigning a Call to a Name (line 255):
        
        # Assigning a Call to a Name (line 255):
        
        # Call to object(...): (line 255)
        # Processing the call keyword arguments (line 255)
        kwargs_206246 = {}
        # Getting the type of 'object' (line 255)
        object_206245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 17), 'object', False)
        # Calling object(args, kwargs) (line 255)
        object_call_result_206247 = invoke(stypy.reporting.localization.Localization(__file__, 255, 17), object_206245, *[], **kwargs_206246)
        
        # Assigning a type to the variable 'STREAM' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'STREAM', object_call_result_206247)
        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Call to object(...): (line 256)
        # Processing the call keyword arguments (line 256)
        kwargs_206249 = {}
        # Getting the type of 'object' (line 256)
        object_206248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'object', False)
        # Calling object(args, kwargs) (line 256)
        object_call_result_206250 = invoke(stypy.reporting.localization.Localization(__file__, 256, 23), object_206248, *[], **kwargs_206249)
        
        # Assigning a type to the variable 'DESCRIPTIONS' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'DESCRIPTIONS', object_call_result_206250)
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to object(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_206252 = {}
        # Getting the type of 'object' (line 257)
        object_206251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'object', False)
        # Calling object(args, kwargs) (line 257)
        object_call_result_206253 = invoke(stypy.reporting.localization.Localization(__file__, 257, 20), object_206251, *[], **kwargs_206252)
        
        # Assigning a type to the variable 'VERBOSITY' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'VERBOSITY', object_call_result_206253)
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Call to TextTestRunner(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'STREAM' (line 258)
        STREAM_206256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 41), 'STREAM', False)
        # Getting the type of 'DESCRIPTIONS' (line 258)
        DESCRIPTIONS_206257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 49), 'DESCRIPTIONS', False)
        # Getting the type of 'VERBOSITY' (line 258)
        VERBOSITY_206258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 63), 'VERBOSITY', False)
        # Processing the call keyword arguments (line 258)
        # Getting the type of 'MockResultClass' (line 259)
        MockResultClass_206259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 53), 'MockResultClass', False)
        keyword_206260 = MockResultClass_206259
        kwargs_206261 = {'resultclass': keyword_206260}
        # Getting the type of 'unittest' (line 258)
        unittest_206254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 258)
        TextTestRunner_206255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 17), unittest_206254, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 258)
        TextTestRunner_call_result_206262 = invoke(stypy.reporting.localization.Localization(__file__, 258, 17), TextTestRunner_206255, *[STREAM_206256, DESCRIPTIONS_206257, VERBOSITY_206258], **kwargs_206261)
        
        # Assigning a type to the variable 'runner' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'runner', TextTestRunner_call_result_206262)
        
        # Call to assertEqual(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'runner' (line 260)
        runner_206265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 25), 'runner', False)
        # Obtaining the member 'resultclass' of a type (line 260)
        resultclass_206266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 25), runner_206265, 'resultclass')
        # Getting the type of 'MockResultClass' (line 260)
        MockResultClass_206267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 45), 'MockResultClass', False)
        # Processing the call keyword arguments (line 260)
        kwargs_206268 = {}
        # Getting the type of 'self' (line 260)
        self_206263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 260)
        assertEqual_206264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), self_206263, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 260)
        assertEqual_call_result_206269 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), assertEqual_206264, *[resultclass_206266, MockResultClass_206267], **kwargs_206268)
        
        
        # Assigning a Tuple to a Name (line 262):
        
        # Assigning a Tuple to a Name (line 262):
        
        # Assigning a Tuple to a Name (line 262):
        
        # Assigning a Tuple to a Name (line 262):
        
        # Obtaining an instance of the builtin type 'tuple' (line 262)
        tuple_206270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 262)
        # Adding element type (line 262)
        # Getting the type of 'runner' (line 262)
        runner_206271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 26), 'runner')
        # Obtaining the member 'stream' of a type (line 262)
        stream_206272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 26), runner_206271, 'stream')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 26), tuple_206270, stream_206272)
        # Adding element type (line 262)
        # Getting the type of 'DESCRIPTIONS' (line 262)
        DESCRIPTIONS_206273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 41), 'DESCRIPTIONS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 26), tuple_206270, DESCRIPTIONS_206273)
        # Adding element type (line 262)
        # Getting the type of 'VERBOSITY' (line 262)
        VERBOSITY_206274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 55), 'VERBOSITY')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 26), tuple_206270, VERBOSITY_206274)
        
        # Assigning a type to the variable 'expectedresult' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'expectedresult', tuple_206270)
        
        # Call to assertEqual(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Call to _makeResult(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_206279 = {}
        # Getting the type of 'runner' (line 263)
        runner_206277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), 'runner', False)
        # Obtaining the member '_makeResult' of a type (line 263)
        _makeResult_206278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 25), runner_206277, '_makeResult')
        # Calling _makeResult(args, kwargs) (line 263)
        _makeResult_call_result_206280 = invoke(stypy.reporting.localization.Localization(__file__, 263, 25), _makeResult_206278, *[], **kwargs_206279)
        
        # Getting the type of 'expectedresult' (line 263)
        expectedresult_206281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 47), 'expectedresult', False)
        # Processing the call keyword arguments (line 263)
        kwargs_206282 = {}
        # Getting the type of 'self' (line 263)
        self_206275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 263)
        assertEqual_206276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), self_206275, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 263)
        assertEqual_call_result_206283 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), assertEqual_206276, *[_makeResult_call_result_206280, expectedresult_206281], **kwargs_206282)
        
        
        # ################# End of 'test_resultclass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_resultclass' in the type store
        # Getting the type of 'stypy_return_type' (line 252)
        stypy_return_type_206284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_206284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_resultclass'
        return stypy_return_type_206284


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 141, 0, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TextTestRunner.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_TextTestRunner' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'Test_TextTestRunner', Test_TextTestRunner)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 267)
    # Processing the call keyword arguments (line 267)
    kwargs_206287 = {}
    # Getting the type of 'unittest' (line 267)
    unittest_206285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 267)
    main_206286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 4), unittest_206285, 'main')
    # Calling main(args, kwargs) (line 267)
    main_call_result_206288 = invoke(stypy.reporting.localization.Localization(__file__, 267, 4), main_206286, *[], **kwargs_206287)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
