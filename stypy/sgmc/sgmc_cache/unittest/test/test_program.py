
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from cStringIO import StringIO
2: 
3: import os
4: import sys
5: import unittest
6: import unittest.test
7: 
8: 
9: class Test_TestProgram(unittest.TestCase):
10: 
11:     def test_discovery_from_dotted_path(self):
12:         loader = unittest.TestLoader()
13: 
14:         tests = [self]
15:         expectedPath = os.path.abspath(os.path.dirname(unittest.test.__file__))
16: 
17:         self.wasRun = False
18:         def _find_tests(start_dir, pattern):
19:             self.wasRun = True
20:             self.assertEqual(start_dir, expectedPath)
21:             return tests
22:         loader._find_tests = _find_tests
23:         suite = loader.discover('unittest.test')
24:         self.assertTrue(self.wasRun)
25:         self.assertEqual(suite._tests, tests)
26: 
27:     # Horrible white box test
28:     def testNoExit(self):
29:         result = object()
30:         test = object()
31: 
32:         class FakeRunner(object):
33:             def run(self, test):
34:                 self.test = test
35:                 return result
36: 
37:         runner = FakeRunner()
38: 
39:         oldParseArgs = unittest.TestProgram.parseArgs
40:         def restoreParseArgs():
41:             unittest.TestProgram.parseArgs = oldParseArgs
42:         unittest.TestProgram.parseArgs = lambda *args: None
43:         self.addCleanup(restoreParseArgs)
44: 
45:         def removeTest():
46:             del unittest.TestProgram.test
47:         unittest.TestProgram.test = test
48:         self.addCleanup(removeTest)
49: 
50:         program = unittest.TestProgram(testRunner=runner, exit=False, verbosity=2)
51: 
52:         self.assertEqual(program.result, result)
53:         self.assertEqual(runner.test, test)
54:         self.assertEqual(program.verbosity, 2)
55: 
56:     class FooBar(unittest.TestCase):
57:         def testPass(self):
58:             assert True
59:         def testFail(self):
60:             assert False
61: 
62:     class FooBarLoader(unittest.TestLoader):
63:         '''Test loader that returns a suite containing FooBar.'''
64:         def loadTestsFromModule(self, module):
65:             return self.suiteClass(
66:                 [self.loadTestsFromTestCase(Test_TestProgram.FooBar)])
67: 
68: 
69:     def test_NonExit(self):
70:         program = unittest.main(exit=False,
71:                                 argv=["foobar"],
72:                                 testRunner=unittest.TextTestRunner(stream=StringIO()),
73:                                 testLoader=self.FooBarLoader())
74:         self.assertTrue(hasattr(program, 'result'))
75: 
76: 
77:     def test_Exit(self):
78:         self.assertRaises(
79:             SystemExit,
80:             unittest.main,
81:             argv=["foobar"],
82:             testRunner=unittest.TextTestRunner(stream=StringIO()),
83:             exit=True,
84:             testLoader=self.FooBarLoader())
85: 
86: 
87:     def test_ExitAsDefault(self):
88:         self.assertRaises(
89:             SystemExit,
90:             unittest.main,
91:             argv=["foobar"],
92:             testRunner=unittest.TextTestRunner(stream=StringIO()),
93:             testLoader=self.FooBarLoader())
94: 
95: 
96: class InitialisableProgram(unittest.TestProgram):
97:     exit = False
98:     result = None
99:     verbosity = 1
100:     defaultTest = None
101:     testRunner = None
102:     testLoader = unittest.defaultTestLoader
103:     progName = 'test'
104:     test = 'test'
105:     def __init__(self, *args):
106:         pass
107: 
108: RESULT = object()
109: 
110: class FakeRunner(object):
111:     initArgs = None
112:     test = None
113:     raiseError = False
114: 
115:     def __init__(self, **kwargs):
116:         FakeRunner.initArgs = kwargs
117:         if FakeRunner.raiseError:
118:             FakeRunner.raiseError = False
119:             raise TypeError
120: 
121:     def run(self, test):
122:         FakeRunner.test = test
123:         return RESULT
124: 
125: class TestCommandLineArgs(unittest.TestCase):
126: 
127:     def setUp(self):
128:         self.program = InitialisableProgram()
129:         self.program.createTests = lambda: None
130:         FakeRunner.initArgs = None
131:         FakeRunner.test = None
132:         FakeRunner.raiseError = False
133: 
134:     def testHelpAndUnknown(self):
135:         program = self.program
136:         def usageExit(msg=None):
137:             program.msg = msg
138:             program.exit = True
139:         program.usageExit = usageExit
140: 
141:         for opt in '-h', '-H', '--help':
142:             program.exit = False
143:             program.parseArgs([None, opt])
144:             self.assertTrue(program.exit)
145:             self.assertIsNone(program.msg)
146: 
147:         program.parseArgs([None, '-$'])
148:         self.assertTrue(program.exit)
149:         self.assertIsNotNone(program.msg)
150: 
151:     def testVerbosity(self):
152:         program = self.program
153: 
154:         for opt in '-q', '--quiet':
155:             program.verbosity = 1
156:             program.parseArgs([None, opt])
157:             self.assertEqual(program.verbosity, 0)
158: 
159:         for opt in '-v', '--verbose':
160:             program.verbosity = 1
161:             program.parseArgs([None, opt])
162:             self.assertEqual(program.verbosity, 2)
163: 
164:     def testBufferCatchFailfast(self):
165:         program = self.program
166:         for arg, attr in (('buffer', 'buffer'), ('failfast', 'failfast'),
167:                       ('catch', 'catchbreak')):
168:             if attr == 'catch' and not hasInstallHandler:
169:                 continue
170: 
171:             short_opt = '-%s' % arg[0]
172:             long_opt = '--%s' % arg
173:             for opt in short_opt, long_opt:
174:                 setattr(program, attr, None)
175: 
176:                 program.parseArgs([None, opt])
177:                 self.assertTrue(getattr(program, attr))
178: 
179:             for opt in short_opt, long_opt:
180:                 not_none = object()
181:                 setattr(program, attr, not_none)
182: 
183:                 program.parseArgs([None, opt])
184:                 self.assertEqual(getattr(program, attr), not_none)
185: 
186:     def testRunTestsRunnerClass(self):
187:         program = self.program
188: 
189:         program.testRunner = FakeRunner
190:         program.verbosity = 'verbosity'
191:         program.failfast = 'failfast'
192:         program.buffer = 'buffer'
193: 
194:         program.runTests()
195: 
196:         self.assertEqual(FakeRunner.initArgs, {'verbosity': 'verbosity',
197:                                                 'failfast': 'failfast',
198:                                                 'buffer': 'buffer'})
199:         self.assertEqual(FakeRunner.test, 'test')
200:         self.assertIs(program.result, RESULT)
201: 
202:     def testRunTestsRunnerInstance(self):
203:         program = self.program
204: 
205:         program.testRunner = FakeRunner()
206:         FakeRunner.initArgs = None
207: 
208:         program.runTests()
209: 
210:         # A new FakeRunner should not have been instantiated
211:         self.assertIsNone(FakeRunner.initArgs)
212: 
213:         self.assertEqual(FakeRunner.test, 'test')
214:         self.assertIs(program.result, RESULT)
215: 
216:     def testRunTestsOldRunnerClass(self):
217:         program = self.program
218: 
219:         FakeRunner.raiseError = True
220:         program.testRunner = FakeRunner
221:         program.verbosity = 'verbosity'
222:         program.failfast = 'failfast'
223:         program.buffer = 'buffer'
224:         program.test = 'test'
225: 
226:         program.runTests()
227: 
228:         # If initializing raises a type error it should be retried
229:         # without the new keyword arguments
230:         self.assertEqual(FakeRunner.initArgs, {})
231:         self.assertEqual(FakeRunner.test, 'test')
232:         self.assertIs(program.result, RESULT)
233: 
234:     def testCatchBreakInstallsHandler(self):
235:         module = sys.modules['unittest.main']
236:         original = module.installHandler
237:         def restore():
238:             module.installHandler = original
239:         self.addCleanup(restore)
240: 
241:         self.installed = False
242:         def fakeInstallHandler():
243:             self.installed = True
244:         module.installHandler = fakeInstallHandler
245: 
246:         program = self.program
247:         program.catchbreak = True
248: 
249:         program.testRunner = FakeRunner
250: 
251:         program.runTests()
252:         self.assertTrue(self.installed)
253: 
254: 
255: if __name__ == '__main__':
256:     unittest.main()
257: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from cStringIO import StringIO' statement (line 1)
from cStringIO import StringIO

import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import unittest' statement (line 5)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import unittest.test' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/unittest/test/')
import_203374 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test')

if (type(import_203374) is not StypyTypeError):

    if (import_203374 != 'pyd_module'):
        __import__(import_203374)
        sys_modules_203375 = sys.modules[import_203374]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test', sys_modules_203375.module_type_store, module_type_store)
    else:
        import unittest.test

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test', unittest.test, module_type_store)

else:
    # Assigning a type to the variable 'unittest.test' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test', import_203374)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/test/')

# Declaration of the 'Test_TestProgram' class
# Getting the type of 'unittest' (line 9)
unittest_203376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'unittest')
# Obtaining the member 'TestCase' of a type (line 9)
TestCase_203377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 23), unittest_203376, 'TestCase')

class Test_TestProgram(TestCase_203377, ):

    @norecursion
    def test_discovery_from_dotted_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_discovery_from_dotted_path'
        module_type_store = module_type_store.open_function_context('test_discovery_from_dotted_path', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_localization', localization)
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_function_name', 'Test_TestProgram.test_discovery_from_dotted_path')
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestProgram.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestProgram.test_discovery_from_dotted_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_discovery_from_dotted_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_discovery_from_dotted_path(...)' code ##################

        
        # Assigning a Call to a Name (line 12):
        
        # Call to TestLoader(...): (line 12)
        # Processing the call keyword arguments (line 12)
        kwargs_203380 = {}
        # Getting the type of 'unittest' (line 12)
        unittest_203378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'unittest', False)
        # Obtaining the member 'TestLoader' of a type (line 12)
        TestLoader_203379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 17), unittest_203378, 'TestLoader')
        # Calling TestLoader(args, kwargs) (line 12)
        TestLoader_call_result_203381 = invoke(stypy.reporting.localization.Localization(__file__, 12, 17), TestLoader_203379, *[], **kwargs_203380)
        
        # Assigning a type to the variable 'loader' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'loader', TestLoader_call_result_203381)
        
        # Assigning a List to a Name (line 14):
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_203382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        # Getting the type of 'self' (line 14)
        self_203383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'self')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_203382, self_203383)
        
        # Assigning a type to the variable 'tests' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'tests', list_203382)
        
        # Assigning a Call to a Name (line 15):
        
        # Call to abspath(...): (line 15)
        # Processing the call arguments (line 15)
        
        # Call to dirname(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'unittest' (line 15)
        unittest_203390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 55), 'unittest', False)
        # Obtaining the member 'test' of a type (line 15)
        test_203391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 55), unittest_203390, 'test')
        # Obtaining the member '__file__' of a type (line 15)
        file___203392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 55), test_203391, '__file__')
        # Processing the call keyword arguments (line 15)
        kwargs_203393 = {}
        # Getting the type of 'os' (line 15)
        os_203387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 15)
        path_203388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 39), os_203387, 'path')
        # Obtaining the member 'dirname' of a type (line 15)
        dirname_203389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 39), path_203388, 'dirname')
        # Calling dirname(args, kwargs) (line 15)
        dirname_call_result_203394 = invoke(stypy.reporting.localization.Localization(__file__, 15, 39), dirname_203389, *[file___203392], **kwargs_203393)
        
        # Processing the call keyword arguments (line 15)
        kwargs_203395 = {}
        # Getting the type of 'os' (line 15)
        os_203384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 15)
        path_203385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 23), os_203384, 'path')
        # Obtaining the member 'abspath' of a type (line 15)
        abspath_203386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 23), path_203385, 'abspath')
        # Calling abspath(args, kwargs) (line 15)
        abspath_call_result_203396 = invoke(stypy.reporting.localization.Localization(__file__, 15, 23), abspath_203386, *[dirname_call_result_203394], **kwargs_203395)
        
        # Assigning a type to the variable 'expectedPath' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'expectedPath', abspath_call_result_203396)
        
        # Assigning a Name to a Attribute (line 17):
        # Getting the type of 'False' (line 17)
        False_203397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'False')
        # Getting the type of 'self' (line 17)
        self_203398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self')
        # Setting the type of the member 'wasRun' of a type (line 17)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_203398, 'wasRun', False_203397)

        @norecursion
        def _find_tests(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_find_tests'
            module_type_store = module_type_store.open_function_context('_find_tests', 18, 8, False)
            
            # Passed parameters checking function
            _find_tests.stypy_localization = localization
            _find_tests.stypy_type_of_self = None
            _find_tests.stypy_type_store = module_type_store
            _find_tests.stypy_function_name = '_find_tests'
            _find_tests.stypy_param_names_list = ['start_dir', 'pattern']
            _find_tests.stypy_varargs_param_name = None
            _find_tests.stypy_kwargs_param_name = None
            _find_tests.stypy_call_defaults = defaults
            _find_tests.stypy_call_varargs = varargs
            _find_tests.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_find_tests', ['start_dir', 'pattern'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_find_tests', localization, ['start_dir', 'pattern'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_find_tests(...)' code ##################

            
            # Assigning a Name to a Attribute (line 19):
            # Getting the type of 'True' (line 19)
            True_203399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 26), 'True')
            # Getting the type of 'self' (line 19)
            self_203400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'self')
            # Setting the type of the member 'wasRun' of a type (line 19)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), self_203400, 'wasRun', True_203399)
            
            # Call to assertEqual(...): (line 20)
            # Processing the call arguments (line 20)
            # Getting the type of 'start_dir' (line 20)
            start_dir_203403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 29), 'start_dir', False)
            # Getting the type of 'expectedPath' (line 20)
            expectedPath_203404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 40), 'expectedPath', False)
            # Processing the call keyword arguments (line 20)
            kwargs_203405 = {}
            # Getting the type of 'self' (line 20)
            self_203401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 20)
            assertEqual_203402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), self_203401, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 20)
            assertEqual_call_result_203406 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), assertEqual_203402, *[start_dir_203403, expectedPath_203404], **kwargs_203405)
            
            # Getting the type of 'tests' (line 21)
            tests_203407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'tests')
            # Assigning a type to the variable 'stypy_return_type' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'stypy_return_type', tests_203407)
            
            # ################# End of '_find_tests(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_find_tests' in the type store
            # Getting the type of 'stypy_return_type' (line 18)
            stypy_return_type_203408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203408)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_find_tests'
            return stypy_return_type_203408

        # Assigning a type to the variable '_find_tests' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), '_find_tests', _find_tests)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of '_find_tests' (line 22)
        _find_tests_203409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 29), '_find_tests')
        # Getting the type of 'loader' (line 22)
        loader_203410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'loader')
        # Setting the type of the member '_find_tests' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), loader_203410, '_find_tests', _find_tests_203409)
        
        # Assigning a Call to a Name (line 23):
        
        # Call to discover(...): (line 23)
        # Processing the call arguments (line 23)
        str_203413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'str', 'unittest.test')
        # Processing the call keyword arguments (line 23)
        kwargs_203414 = {}
        # Getting the type of 'loader' (line 23)
        loader_203411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'loader', False)
        # Obtaining the member 'discover' of a type (line 23)
        discover_203412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 16), loader_203411, 'discover')
        # Calling discover(args, kwargs) (line 23)
        discover_call_result_203415 = invoke(stypy.reporting.localization.Localization(__file__, 23, 16), discover_203412, *[str_203413], **kwargs_203414)
        
        # Assigning a type to the variable 'suite' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'suite', discover_call_result_203415)
        
        # Call to assertTrue(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'self' (line 24)
        self_203418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'self', False)
        # Obtaining the member 'wasRun' of a type (line 24)
        wasRun_203419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 24), self_203418, 'wasRun')
        # Processing the call keyword arguments (line 24)
        kwargs_203420 = {}
        # Getting the type of 'self' (line 24)
        self_203416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 24)
        assertTrue_203417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_203416, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 24)
        assertTrue_call_result_203421 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), assertTrue_203417, *[wasRun_203419], **kwargs_203420)
        
        
        # Call to assertEqual(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'suite' (line 25)
        suite_203424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'suite', False)
        # Obtaining the member '_tests' of a type (line 25)
        _tests_203425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 25), suite_203424, '_tests')
        # Getting the type of 'tests' (line 25)
        tests_203426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'tests', False)
        # Processing the call keyword arguments (line 25)
        kwargs_203427 = {}
        # Getting the type of 'self' (line 25)
        self_203422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 25)
        assertEqual_203423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_203422, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 25)
        assertEqual_call_result_203428 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), assertEqual_203423, *[_tests_203425, tests_203426], **kwargs_203427)
        
        
        # ################# End of 'test_discovery_from_dotted_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_discovery_from_dotted_path' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_203429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_discovery_from_dotted_path'
        return stypy_return_type_203429


    @norecursion
    def testNoExit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testNoExit'
        module_type_store = module_type_store.open_function_context('testNoExit', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_localization', localization)
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_function_name', 'Test_TestProgram.testNoExit')
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestProgram.testNoExit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestProgram.testNoExit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testNoExit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testNoExit(...)' code ##################

        
        # Assigning a Call to a Name (line 29):
        
        # Call to object(...): (line 29)
        # Processing the call keyword arguments (line 29)
        kwargs_203431 = {}
        # Getting the type of 'object' (line 29)
        object_203430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'object', False)
        # Calling object(args, kwargs) (line 29)
        object_call_result_203432 = invoke(stypy.reporting.localization.Localization(__file__, 29, 17), object_203430, *[], **kwargs_203431)
        
        # Assigning a type to the variable 'result' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'result', object_call_result_203432)
        
        # Assigning a Call to a Name (line 30):
        
        # Call to object(...): (line 30)
        # Processing the call keyword arguments (line 30)
        kwargs_203434 = {}
        # Getting the type of 'object' (line 30)
        object_203433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'object', False)
        # Calling object(args, kwargs) (line 30)
        object_call_result_203435 = invoke(stypy.reporting.localization.Localization(__file__, 30, 15), object_203433, *[], **kwargs_203434)
        
        # Assigning a type to the variable 'test' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'test', object_call_result_203435)
        # Declaration of the 'FakeRunner' class

        class FakeRunner(object, ):

            @norecursion
            def run(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'run'
                module_type_store = module_type_store.open_function_context('run', 33, 12, False)
                # Assigning a type to the variable 'self' (line 34)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                FakeRunner.run.__dict__.__setitem__('stypy_localization', localization)
                FakeRunner.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                FakeRunner.run.__dict__.__setitem__('stypy_type_store', module_type_store)
                FakeRunner.run.__dict__.__setitem__('stypy_function_name', 'FakeRunner.run')
                FakeRunner.run.__dict__.__setitem__('stypy_param_names_list', ['test'])
                FakeRunner.run.__dict__.__setitem__('stypy_varargs_param_name', None)
                FakeRunner.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
                FakeRunner.run.__dict__.__setitem__('stypy_call_defaults', defaults)
                FakeRunner.run.__dict__.__setitem__('stypy_call_varargs', varargs)
                FakeRunner.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                FakeRunner.run.__dict__.__setitem__('stypy_declared_arg_number', 2)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeRunner.run', ['test'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'run', localization, ['test'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'run(...)' code ##################

                
                # Assigning a Name to a Attribute (line 34):
                # Getting the type of 'test' (line 34)
                test_203436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'test')
                # Getting the type of 'self' (line 34)
                self_203437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'self')
                # Setting the type of the member 'test' of a type (line 34)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), self_203437, 'test', test_203436)
                # Getting the type of 'result' (line 35)
                result_203438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'result')
                # Assigning a type to the variable 'stypy_return_type' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'stypy_return_type', result_203438)
                
                # ################# End of 'run(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'run' in the type store
                # Getting the type of 'stypy_return_type' (line 33)
                stypy_return_type_203439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_203439)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'run'
                return stypy_return_type_203439

        
        # Assigning a type to the variable 'FakeRunner' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'FakeRunner', FakeRunner)
        
        # Assigning a Call to a Name (line 37):
        
        # Call to FakeRunner(...): (line 37)
        # Processing the call keyword arguments (line 37)
        kwargs_203441 = {}
        # Getting the type of 'FakeRunner' (line 37)
        FakeRunner_203440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'FakeRunner', False)
        # Calling FakeRunner(args, kwargs) (line 37)
        FakeRunner_call_result_203442 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), FakeRunner_203440, *[], **kwargs_203441)
        
        # Assigning a type to the variable 'runner' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'runner', FakeRunner_call_result_203442)
        
        # Assigning a Attribute to a Name (line 39):
        # Getting the type of 'unittest' (line 39)
        unittest_203443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'unittest')
        # Obtaining the member 'TestProgram' of a type (line 39)
        TestProgram_203444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 23), unittest_203443, 'TestProgram')
        # Obtaining the member 'parseArgs' of a type (line 39)
        parseArgs_203445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 23), TestProgram_203444, 'parseArgs')
        # Assigning a type to the variable 'oldParseArgs' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'oldParseArgs', parseArgs_203445)

        @norecursion
        def restoreParseArgs(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restoreParseArgs'
            module_type_store = module_type_store.open_function_context('restoreParseArgs', 40, 8, False)
            
            # Passed parameters checking function
            restoreParseArgs.stypy_localization = localization
            restoreParseArgs.stypy_type_of_self = None
            restoreParseArgs.stypy_type_store = module_type_store
            restoreParseArgs.stypy_function_name = 'restoreParseArgs'
            restoreParseArgs.stypy_param_names_list = []
            restoreParseArgs.stypy_varargs_param_name = None
            restoreParseArgs.stypy_kwargs_param_name = None
            restoreParseArgs.stypy_call_defaults = defaults
            restoreParseArgs.stypy_call_varargs = varargs
            restoreParseArgs.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restoreParseArgs', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restoreParseArgs', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restoreParseArgs(...)' code ##################

            
            # Assigning a Name to a Attribute (line 41):
            # Getting the type of 'oldParseArgs' (line 41)
            oldParseArgs_203446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 45), 'oldParseArgs')
            # Getting the type of 'unittest' (line 41)
            unittest_203447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'unittest')
            # Obtaining the member 'TestProgram' of a type (line 41)
            TestProgram_203448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), unittest_203447, 'TestProgram')
            # Setting the type of the member 'parseArgs' of a type (line 41)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), TestProgram_203448, 'parseArgs', oldParseArgs_203446)
            
            # ################# End of 'restoreParseArgs(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restoreParseArgs' in the type store
            # Getting the type of 'stypy_return_type' (line 40)
            stypy_return_type_203449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203449)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restoreParseArgs'
            return stypy_return_type_203449

        # Assigning a type to the variable 'restoreParseArgs' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'restoreParseArgs', restoreParseArgs)
        
        # Assigning a Lambda to a Attribute (line 42):

        @norecursion
        def _stypy_temp_lambda_89(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_89'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_89', 42, 41, True)
            # Passed parameters checking function
            _stypy_temp_lambda_89.stypy_localization = localization
            _stypy_temp_lambda_89.stypy_type_of_self = None
            _stypy_temp_lambda_89.stypy_type_store = module_type_store
            _stypy_temp_lambda_89.stypy_function_name = '_stypy_temp_lambda_89'
            _stypy_temp_lambda_89.stypy_param_names_list = []
            _stypy_temp_lambda_89.stypy_varargs_param_name = 'args'
            _stypy_temp_lambda_89.stypy_kwargs_param_name = None
            _stypy_temp_lambda_89.stypy_call_defaults = defaults
            _stypy_temp_lambda_89.stypy_call_varargs = varargs
            _stypy_temp_lambda_89.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_89', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_89', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 42)
            None_203450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 55), 'None')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), 'stypy_return_type', None_203450)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_89' in the type store
            # Getting the type of 'stypy_return_type' (line 42)
            stypy_return_type_203451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203451)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_89'
            return stypy_return_type_203451

        # Assigning a type to the variable '_stypy_temp_lambda_89' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), '_stypy_temp_lambda_89', _stypy_temp_lambda_89)
        # Getting the type of '_stypy_temp_lambda_89' (line 42)
        _stypy_temp_lambda_89_203452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), '_stypy_temp_lambda_89')
        # Getting the type of 'unittest' (line 42)
        unittest_203453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'unittest')
        # Obtaining the member 'TestProgram' of a type (line 42)
        TestProgram_203454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), unittest_203453, 'TestProgram')
        # Setting the type of the member 'parseArgs' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), TestProgram_203454, 'parseArgs', _stypy_temp_lambda_89_203452)
        
        # Call to addCleanup(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'restoreParseArgs' (line 43)
        restoreParseArgs_203457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'restoreParseArgs', False)
        # Processing the call keyword arguments (line 43)
        kwargs_203458 = {}
        # Getting the type of 'self' (line 43)
        self_203455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 43)
        addCleanup_203456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_203455, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 43)
        addCleanup_call_result_203459 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), addCleanup_203456, *[restoreParseArgs_203457], **kwargs_203458)
        

        @norecursion
        def removeTest(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'removeTest'
            module_type_store = module_type_store.open_function_context('removeTest', 45, 8, False)
            
            # Passed parameters checking function
            removeTest.stypy_localization = localization
            removeTest.stypy_type_of_self = None
            removeTest.stypy_type_store = module_type_store
            removeTest.stypy_function_name = 'removeTest'
            removeTest.stypy_param_names_list = []
            removeTest.stypy_varargs_param_name = None
            removeTest.stypy_kwargs_param_name = None
            removeTest.stypy_call_defaults = defaults
            removeTest.stypy_call_varargs = varargs
            removeTest.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'removeTest', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'removeTest', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'removeTest(...)' code ##################

            # Deleting a member
            # Getting the type of 'unittest' (line 46)
            unittest_203460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'unittest')
            # Obtaining the member 'TestProgram' of a type (line 46)
            TestProgram_203461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), unittest_203460, 'TestProgram')
            module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 46, 12), TestProgram_203461, 'test')
            
            # ################# End of 'removeTest(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'removeTest' in the type store
            # Getting the type of 'stypy_return_type' (line 45)
            stypy_return_type_203462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203462)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'removeTest'
            return stypy_return_type_203462

        # Assigning a type to the variable 'removeTest' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'removeTest', removeTest)
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'test' (line 47)
        test_203463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 36), 'test')
        # Getting the type of 'unittest' (line 47)
        unittest_203464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'unittest')
        # Obtaining the member 'TestProgram' of a type (line 47)
        TestProgram_203465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), unittest_203464, 'TestProgram')
        # Setting the type of the member 'test' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), TestProgram_203465, 'test', test_203463)
        
        # Call to addCleanup(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'removeTest' (line 48)
        removeTest_203468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'removeTest', False)
        # Processing the call keyword arguments (line 48)
        kwargs_203469 = {}
        # Getting the type of 'self' (line 48)
        self_203466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 48)
        addCleanup_203467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_203466, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 48)
        addCleanup_call_result_203470 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), addCleanup_203467, *[removeTest_203468], **kwargs_203469)
        
        
        # Assigning a Call to a Name (line 50):
        
        # Call to TestProgram(...): (line 50)
        # Processing the call keyword arguments (line 50)
        # Getting the type of 'runner' (line 50)
        runner_203473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 50), 'runner', False)
        keyword_203474 = runner_203473
        # Getting the type of 'False' (line 50)
        False_203475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 63), 'False', False)
        keyword_203476 = False_203475
        int_203477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 80), 'int')
        keyword_203478 = int_203477
        kwargs_203479 = {'testRunner': keyword_203474, 'verbosity': keyword_203478, 'exit': keyword_203476}
        # Getting the type of 'unittest' (line 50)
        unittest_203471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 50)
        TestProgram_203472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 18), unittest_203471, 'TestProgram')
        # Calling TestProgram(args, kwargs) (line 50)
        TestProgram_call_result_203480 = invoke(stypy.reporting.localization.Localization(__file__, 50, 18), TestProgram_203472, *[], **kwargs_203479)
        
        # Assigning a type to the variable 'program' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'program', TestProgram_call_result_203480)
        
        # Call to assertEqual(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'program' (line 52)
        program_203483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'program', False)
        # Obtaining the member 'result' of a type (line 52)
        result_203484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 25), program_203483, 'result')
        # Getting the type of 'result' (line 52)
        result_203485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'result', False)
        # Processing the call keyword arguments (line 52)
        kwargs_203486 = {}
        # Getting the type of 'self' (line 52)
        self_203481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 52)
        assertEqual_203482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_203481, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 52)
        assertEqual_call_result_203487 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assertEqual_203482, *[result_203484, result_203485], **kwargs_203486)
        
        
        # Call to assertEqual(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'runner' (line 53)
        runner_203490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'runner', False)
        # Obtaining the member 'test' of a type (line 53)
        test_203491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 25), runner_203490, 'test')
        # Getting the type of 'test' (line 53)
        test_203492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 38), 'test', False)
        # Processing the call keyword arguments (line 53)
        kwargs_203493 = {}
        # Getting the type of 'self' (line 53)
        self_203488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 53)
        assertEqual_203489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_203488, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 53)
        assertEqual_call_result_203494 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assertEqual_203489, *[test_203491, test_203492], **kwargs_203493)
        
        
        # Call to assertEqual(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'program' (line 54)
        program_203497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'program', False)
        # Obtaining the member 'verbosity' of a type (line 54)
        verbosity_203498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), program_203497, 'verbosity')
        int_203499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'int')
        # Processing the call keyword arguments (line 54)
        kwargs_203500 = {}
        # Getting the type of 'self' (line 54)
        self_203495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 54)
        assertEqual_203496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_203495, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 54)
        assertEqual_call_result_203501 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assertEqual_203496, *[verbosity_203498, int_203499], **kwargs_203500)
        
        
        # ################# End of 'testNoExit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testNoExit' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_203502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203502)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testNoExit'
        return stypy_return_type_203502

    # Declaration of the 'FooBar' class
    # Getting the type of 'unittest' (line 56)
    unittest_203503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'unittest')
    # Obtaining the member 'TestCase' of a type (line 56)
    TestCase_203504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 17), unittest_203503, 'TestCase')

    class FooBar(TestCase_203504, ):

        @norecursion
        def testPass(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'testPass'
            module_type_store = module_type_store.open_function_context('testPass', 57, 8, False)
            # Assigning a type to the variable 'self' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            FooBar.testPass.__dict__.__setitem__('stypy_localization', localization)
            FooBar.testPass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            FooBar.testPass.__dict__.__setitem__('stypy_type_store', module_type_store)
            FooBar.testPass.__dict__.__setitem__('stypy_function_name', 'FooBar.testPass')
            FooBar.testPass.__dict__.__setitem__('stypy_param_names_list', [])
            FooBar.testPass.__dict__.__setitem__('stypy_varargs_param_name', None)
            FooBar.testPass.__dict__.__setitem__('stypy_kwargs_param_name', None)
            FooBar.testPass.__dict__.__setitem__('stypy_call_defaults', defaults)
            FooBar.testPass.__dict__.__setitem__('stypy_call_varargs', varargs)
            FooBar.testPass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            FooBar.testPass.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'FooBar.testPass', [], None, None, defaults, varargs, kwargs)

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

            # Evaluating assert statement condition
            # Getting the type of 'True' (line 58)
            True_203505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'True')
            
            # ################# End of 'testPass(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'testPass' in the type store
            # Getting the type of 'stypy_return_type' (line 57)
            stypy_return_type_203506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203506)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'testPass'
            return stypy_return_type_203506


        @norecursion
        def testFail(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'testFail'
            module_type_store = module_type_store.open_function_context('testFail', 59, 8, False)
            # Assigning a type to the variable 'self' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            FooBar.testFail.__dict__.__setitem__('stypy_localization', localization)
            FooBar.testFail.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            FooBar.testFail.__dict__.__setitem__('stypy_type_store', module_type_store)
            FooBar.testFail.__dict__.__setitem__('stypy_function_name', 'FooBar.testFail')
            FooBar.testFail.__dict__.__setitem__('stypy_param_names_list', [])
            FooBar.testFail.__dict__.__setitem__('stypy_varargs_param_name', None)
            FooBar.testFail.__dict__.__setitem__('stypy_kwargs_param_name', None)
            FooBar.testFail.__dict__.__setitem__('stypy_call_defaults', defaults)
            FooBar.testFail.__dict__.__setitem__('stypy_call_varargs', varargs)
            FooBar.testFail.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            FooBar.testFail.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'FooBar.testFail', [], None, None, defaults, varargs, kwargs)

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

            # Evaluating assert statement condition
            # Getting the type of 'False' (line 60)
            False_203507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'False')
            
            # ################# End of 'testFail(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'testFail' in the type store
            # Getting the type of 'stypy_return_type' (line 59)
            stypy_return_type_203508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203508)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'testFail'
            return stypy_return_type_203508

    
    # Assigning a type to the variable 'FooBar' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'FooBar', FooBar)
    # Declaration of the 'FooBarLoader' class
    # Getting the type of 'unittest' (line 62)
    unittest_203509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'unittest')
    # Obtaining the member 'TestLoader' of a type (line 62)
    TestLoader_203510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 23), unittest_203509, 'TestLoader')

    class FooBarLoader(TestLoader_203510, ):
        str_203511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'str', 'Test loader that returns a suite containing FooBar.')

        @norecursion
        def loadTestsFromModule(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'loadTestsFromModule'
            module_type_store = module_type_store.open_function_context('loadTestsFromModule', 64, 8, False)
            # Assigning a type to the variable 'self' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_localization', localization)
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_type_store', module_type_store)
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_function_name', 'FooBarLoader.loadTestsFromModule')
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_param_names_list', ['module'])
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_varargs_param_name', None)
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_call_defaults', defaults)
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_call_varargs', varargs)
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            FooBarLoader.loadTestsFromModule.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'FooBarLoader.loadTestsFromModule', ['module'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'loadTestsFromModule', localization, ['module'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'loadTestsFromModule(...)' code ##################

            
            # Call to suiteClass(...): (line 65)
            # Processing the call arguments (line 65)
            
            # Obtaining an instance of the builtin type 'list' (line 66)
            list_203514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'list')
            # Adding type elements to the builtin type 'list' instance (line 66)
            # Adding element type (line 66)
            
            # Call to loadTestsFromTestCase(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'Test_TestProgram' (line 66)
            Test_TestProgram_203517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 44), 'Test_TestProgram', False)
            # Obtaining the member 'FooBar' of a type (line 66)
            FooBar_203518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 44), Test_TestProgram_203517, 'FooBar')
            # Processing the call keyword arguments (line 66)
            kwargs_203519 = {}
            # Getting the type of 'self' (line 66)
            self_203515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'self', False)
            # Obtaining the member 'loadTestsFromTestCase' of a type (line 66)
            loadTestsFromTestCase_203516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 17), self_203515, 'loadTestsFromTestCase')
            # Calling loadTestsFromTestCase(args, kwargs) (line 66)
            loadTestsFromTestCase_call_result_203520 = invoke(stypy.reporting.localization.Localization(__file__, 66, 17), loadTestsFromTestCase_203516, *[FooBar_203518], **kwargs_203519)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 16), list_203514, loadTestsFromTestCase_call_result_203520)
            
            # Processing the call keyword arguments (line 65)
            kwargs_203521 = {}
            # Getting the type of 'self' (line 65)
            self_203512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'self', False)
            # Obtaining the member 'suiteClass' of a type (line 65)
            suiteClass_203513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), self_203512, 'suiteClass')
            # Calling suiteClass(args, kwargs) (line 65)
            suiteClass_call_result_203522 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), suiteClass_203513, *[list_203514], **kwargs_203521)
            
            # Assigning a type to the variable 'stypy_return_type' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', suiteClass_call_result_203522)
            
            # ################# End of 'loadTestsFromModule(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'loadTestsFromModule' in the type store
            # Getting the type of 'stypy_return_type' (line 64)
            stypy_return_type_203523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203523)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'loadTestsFromModule'
            return stypy_return_type_203523

    
    # Assigning a type to the variable 'FooBarLoader' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'FooBarLoader', FooBarLoader)

    @norecursion
    def test_NonExit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_NonExit'
        module_type_store = module_type_store.open_function_context('test_NonExit', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_localization', localization)
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_function_name', 'Test_TestProgram.test_NonExit')
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestProgram.test_NonExit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestProgram.test_NonExit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_NonExit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_NonExit(...)' code ##################

        
        # Assigning a Call to a Name (line 70):
        
        # Call to main(...): (line 70)
        # Processing the call keyword arguments (line 70)
        # Getting the type of 'False' (line 70)
        False_203526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 37), 'False', False)
        keyword_203527 = False_203526
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_203528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        str_203529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'str', 'foobar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 37), list_203528, str_203529)
        
        keyword_203530 = list_203528
        
        # Call to TextTestRunner(...): (line 72)
        # Processing the call keyword arguments (line 72)
        
        # Call to StringIO(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_203534 = {}
        # Getting the type of 'StringIO' (line 72)
        StringIO_203533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 74), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 72)
        StringIO_call_result_203535 = invoke(stypy.reporting.localization.Localization(__file__, 72, 74), StringIO_203533, *[], **kwargs_203534)
        
        keyword_203536 = StringIO_call_result_203535
        kwargs_203537 = {'stream': keyword_203536}
        # Getting the type of 'unittest' (line 72)
        unittest_203531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 43), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 72)
        TextTestRunner_203532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 43), unittest_203531, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 72)
        TextTestRunner_call_result_203538 = invoke(stypy.reporting.localization.Localization(__file__, 72, 43), TextTestRunner_203532, *[], **kwargs_203537)
        
        keyword_203539 = TextTestRunner_call_result_203538
        
        # Call to FooBarLoader(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_203542 = {}
        # Getting the type of 'self' (line 73)
        self_203540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 43), 'self', False)
        # Obtaining the member 'FooBarLoader' of a type (line 73)
        FooBarLoader_203541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 43), self_203540, 'FooBarLoader')
        # Calling FooBarLoader(args, kwargs) (line 73)
        FooBarLoader_call_result_203543 = invoke(stypy.reporting.localization.Localization(__file__, 73, 43), FooBarLoader_203541, *[], **kwargs_203542)
        
        keyword_203544 = FooBarLoader_call_result_203543
        kwargs_203545 = {'testRunner': keyword_203539, 'exit': keyword_203527, 'argv': keyword_203530, 'testLoader': keyword_203544}
        # Getting the type of 'unittest' (line 70)
        unittest_203524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'unittest', False)
        # Obtaining the member 'main' of a type (line 70)
        main_203525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 18), unittest_203524, 'main')
        # Calling main(args, kwargs) (line 70)
        main_call_result_203546 = invoke(stypy.reporting.localization.Localization(__file__, 70, 18), main_203525, *[], **kwargs_203545)
        
        # Assigning a type to the variable 'program' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'program', main_call_result_203546)
        
        # Call to assertTrue(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to hasattr(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'program' (line 74)
        program_203550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'program', False)
        str_203551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 41), 'str', 'result')
        # Processing the call keyword arguments (line 74)
        kwargs_203552 = {}
        # Getting the type of 'hasattr' (line 74)
        hasattr_203549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 74)
        hasattr_call_result_203553 = invoke(stypy.reporting.localization.Localization(__file__, 74, 24), hasattr_203549, *[program_203550, str_203551], **kwargs_203552)
        
        # Processing the call keyword arguments (line 74)
        kwargs_203554 = {}
        # Getting the type of 'self' (line 74)
        self_203547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 74)
        assertTrue_203548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_203547, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 74)
        assertTrue_call_result_203555 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assertTrue_203548, *[hasattr_call_result_203553], **kwargs_203554)
        
        
        # ################# End of 'test_NonExit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_NonExit' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_203556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203556)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_NonExit'
        return stypy_return_type_203556


    @norecursion
    def test_Exit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_Exit'
        module_type_store = module_type_store.open_function_context('test_Exit', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_localization', localization)
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_function_name', 'Test_TestProgram.test_Exit')
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestProgram.test_Exit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestProgram.test_Exit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_Exit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_Exit(...)' code ##################

        
        # Call to assertRaises(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'SystemExit' (line 79)
        SystemExit_203559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'SystemExit', False)
        # Getting the type of 'unittest' (line 80)
        unittest_203560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'unittest', False)
        # Obtaining the member 'main' of a type (line 80)
        main_203561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), unittest_203560, 'main')
        # Processing the call keyword arguments (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_203562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        str_203563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 18), 'str', 'foobar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 17), list_203562, str_203563)
        
        keyword_203564 = list_203562
        
        # Call to TextTestRunner(...): (line 82)
        # Processing the call keyword arguments (line 82)
        
        # Call to StringIO(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_203568 = {}
        # Getting the type of 'StringIO' (line 82)
        StringIO_203567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 54), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 82)
        StringIO_call_result_203569 = invoke(stypy.reporting.localization.Localization(__file__, 82, 54), StringIO_203567, *[], **kwargs_203568)
        
        keyword_203570 = StringIO_call_result_203569
        kwargs_203571 = {'stream': keyword_203570}
        # Getting the type of 'unittest' (line 82)
        unittest_203565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 82)
        TextTestRunner_203566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 23), unittest_203565, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 82)
        TextTestRunner_call_result_203572 = invoke(stypy.reporting.localization.Localization(__file__, 82, 23), TextTestRunner_203566, *[], **kwargs_203571)
        
        keyword_203573 = TextTestRunner_call_result_203572
        # Getting the type of 'True' (line 83)
        True_203574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'True', False)
        keyword_203575 = True_203574
        
        # Call to FooBarLoader(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_203578 = {}
        # Getting the type of 'self' (line 84)
        self_203576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'self', False)
        # Obtaining the member 'FooBarLoader' of a type (line 84)
        FooBarLoader_203577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), self_203576, 'FooBarLoader')
        # Calling FooBarLoader(args, kwargs) (line 84)
        FooBarLoader_call_result_203579 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), FooBarLoader_203577, *[], **kwargs_203578)
        
        keyword_203580 = FooBarLoader_call_result_203579
        kwargs_203581 = {'testRunner': keyword_203573, 'exit': keyword_203575, 'argv': keyword_203564, 'testLoader': keyword_203580}
        # Getting the type of 'self' (line 78)
        self_203557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 78)
        assertRaises_203558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_203557, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 78)
        assertRaises_call_result_203582 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assertRaises_203558, *[SystemExit_203559, main_203561], **kwargs_203581)
        
        
        # ################# End of 'test_Exit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_Exit' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_203583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203583)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_Exit'
        return stypy_return_type_203583


    @norecursion
    def test_ExitAsDefault(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ExitAsDefault'
        module_type_store = module_type_store.open_function_context('test_ExitAsDefault', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_localization', localization)
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_function_name', 'Test_TestProgram.test_ExitAsDefault')
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_param_names_list', [])
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_TestProgram.test_ExitAsDefault.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestProgram.test_ExitAsDefault', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ExitAsDefault', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ExitAsDefault(...)' code ##################

        
        # Call to assertRaises(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'SystemExit' (line 89)
        SystemExit_203586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'SystemExit', False)
        # Getting the type of 'unittest' (line 90)
        unittest_203587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'unittest', False)
        # Obtaining the member 'main' of a type (line 90)
        main_203588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), unittest_203587, 'main')
        # Processing the call keyword arguments (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_203589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        str_203590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 18), 'str', 'foobar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 17), list_203589, str_203590)
        
        keyword_203591 = list_203589
        
        # Call to TextTestRunner(...): (line 92)
        # Processing the call keyword arguments (line 92)
        
        # Call to StringIO(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_203595 = {}
        # Getting the type of 'StringIO' (line 92)
        StringIO_203594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 54), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 92)
        StringIO_call_result_203596 = invoke(stypy.reporting.localization.Localization(__file__, 92, 54), StringIO_203594, *[], **kwargs_203595)
        
        keyword_203597 = StringIO_call_result_203596
        kwargs_203598 = {'stream': keyword_203597}
        # Getting the type of 'unittest' (line 92)
        unittest_203592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 92)
        TextTestRunner_203593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 23), unittest_203592, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 92)
        TextTestRunner_call_result_203599 = invoke(stypy.reporting.localization.Localization(__file__, 92, 23), TextTestRunner_203593, *[], **kwargs_203598)
        
        keyword_203600 = TextTestRunner_call_result_203599
        
        # Call to FooBarLoader(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_203603 = {}
        # Getting the type of 'self' (line 93)
        self_203601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'self', False)
        # Obtaining the member 'FooBarLoader' of a type (line 93)
        FooBarLoader_203602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), self_203601, 'FooBarLoader')
        # Calling FooBarLoader(args, kwargs) (line 93)
        FooBarLoader_call_result_203604 = invoke(stypy.reporting.localization.Localization(__file__, 93, 23), FooBarLoader_203602, *[], **kwargs_203603)
        
        keyword_203605 = FooBarLoader_call_result_203604
        kwargs_203606 = {'testRunner': keyword_203600, 'argv': keyword_203591, 'testLoader': keyword_203605}
        # Getting the type of 'self' (line 88)
        self_203584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 88)
        assertRaises_203585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_203584, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 88)
        assertRaises_call_result_203607 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), assertRaises_203585, *[SystemExit_203586, main_203588], **kwargs_203606)
        
        
        # ################# End of 'test_ExitAsDefault(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ExitAsDefault' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_203608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ExitAsDefault'
        return stypy_return_type_203608


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 0, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_TestProgram.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_TestProgram' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'Test_TestProgram', Test_TestProgram)
# Declaration of the 'InitialisableProgram' class
# Getting the type of 'unittest' (line 96)
unittest_203609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'unittest')
# Obtaining the member 'TestProgram' of a type (line 96)
TestProgram_203610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 27), unittest_203609, 'TestProgram')

class InitialisableProgram(TestProgram_203610, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InitialisableProgram.__init__', [], 'args', None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InitialisableProgram' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'InitialisableProgram', InitialisableProgram)

# Assigning a Name to a Name (line 97):
# Getting the type of 'False' (line 97)
False_203611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'False')
# Getting the type of 'InitialisableProgram'
InitialisableProgram_203612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InitialisableProgram')
# Setting the type of the member 'exit' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InitialisableProgram_203612, 'exit', False_203611)

# Assigning a Name to a Name (line 98):
# Getting the type of 'None' (line 98)
None_203613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'None')
# Getting the type of 'InitialisableProgram'
InitialisableProgram_203614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InitialisableProgram')
# Setting the type of the member 'result' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InitialisableProgram_203614, 'result', None_203613)

# Assigning a Num to a Name (line 99):
int_203615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'int')
# Getting the type of 'InitialisableProgram'
InitialisableProgram_203616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InitialisableProgram')
# Setting the type of the member 'verbosity' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InitialisableProgram_203616, 'verbosity', int_203615)

# Assigning a Name to a Name (line 100):
# Getting the type of 'None' (line 100)
None_203617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 18), 'None')
# Getting the type of 'InitialisableProgram'
InitialisableProgram_203618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InitialisableProgram')
# Setting the type of the member 'defaultTest' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InitialisableProgram_203618, 'defaultTest', None_203617)

# Assigning a Name to a Name (line 101):
# Getting the type of 'None' (line 101)
None_203619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'None')
# Getting the type of 'InitialisableProgram'
InitialisableProgram_203620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InitialisableProgram')
# Setting the type of the member 'testRunner' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InitialisableProgram_203620, 'testRunner', None_203619)

# Assigning a Attribute to a Name (line 102):
# Getting the type of 'unittest' (line 102)
unittest_203621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'unittest')
# Obtaining the member 'defaultTestLoader' of a type (line 102)
defaultTestLoader_203622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 17), unittest_203621, 'defaultTestLoader')
# Getting the type of 'InitialisableProgram'
InitialisableProgram_203623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InitialisableProgram')
# Setting the type of the member 'testLoader' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InitialisableProgram_203623, 'testLoader', defaultTestLoader_203622)

# Assigning a Str to a Name (line 103):
str_203624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'str', 'test')
# Getting the type of 'InitialisableProgram'
InitialisableProgram_203625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InitialisableProgram')
# Setting the type of the member 'progName' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InitialisableProgram_203625, 'progName', str_203624)

# Assigning a Str to a Name (line 104):
str_203626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 11), 'str', 'test')
# Getting the type of 'InitialisableProgram'
InitialisableProgram_203627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'InitialisableProgram')
# Setting the type of the member 'test' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), InitialisableProgram_203627, 'test', str_203626)

# Assigning a Call to a Name (line 108):

# Call to object(...): (line 108)
# Processing the call keyword arguments (line 108)
kwargs_203629 = {}
# Getting the type of 'object' (line 108)
object_203628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 9), 'object', False)
# Calling object(args, kwargs) (line 108)
object_call_result_203630 = invoke(stypy.reporting.localization.Localization(__file__, 108, 9), object_203628, *[], **kwargs_203629)

# Assigning a type to the variable 'RESULT' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'RESULT', object_call_result_203630)
# Declaration of the 'FakeRunner' class

class FakeRunner(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeRunner.__init__', [], None, 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 116):
        # Getting the type of 'kwargs' (line 116)
        kwargs_203631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'kwargs')
        # Getting the type of 'FakeRunner' (line 116)
        FakeRunner_203632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'FakeRunner')
        # Setting the type of the member 'initArgs' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), FakeRunner_203632, 'initArgs', kwargs_203631)
        
        # Getting the type of 'FakeRunner' (line 117)
        FakeRunner_203633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'FakeRunner')
        # Obtaining the member 'raiseError' of a type (line 117)
        raiseError_203634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 11), FakeRunner_203633, 'raiseError')
        # Testing the type of an if condition (line 117)
        if_condition_203635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), raiseError_203634)
        # Assigning a type to the variable 'if_condition_203635' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'if_condition_203635', if_condition_203635)
        # SSA begins for if statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 118):
        # Getting the type of 'False' (line 118)
        False_203636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 36), 'False')
        # Getting the type of 'FakeRunner' (line 118)
        FakeRunner_203637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'FakeRunner')
        # Setting the type of the member 'raiseError' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), FakeRunner_203637, 'raiseError', False_203636)
        # Getting the type of 'TypeError' (line 119)
        TypeError_203638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'TypeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 119, 12), TypeError_203638, 'raise parameter', BaseException)
        # SSA join for if statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeRunner.run.__dict__.__setitem__('stypy_localization', localization)
        FakeRunner.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeRunner.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeRunner.run.__dict__.__setitem__('stypy_function_name', 'FakeRunner.run')
        FakeRunner.run.__dict__.__setitem__('stypy_param_names_list', ['test'])
        FakeRunner.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeRunner.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeRunner.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeRunner.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeRunner.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeRunner.run.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeRunner.run', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Assigning a Name to a Attribute (line 122):
        # Getting the type of 'test' (line 122)
        test_203639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 26), 'test')
        # Getting the type of 'FakeRunner' (line 122)
        FakeRunner_203640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'FakeRunner')
        # Setting the type of the member 'test' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), FakeRunner_203640, 'test', test_203639)
        # Getting the type of 'RESULT' (line 123)
        RESULT_203641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'RESULT')
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', RESULT_203641)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_203642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203642)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_203642


# Assigning a type to the variable 'FakeRunner' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'FakeRunner', FakeRunner)

# Assigning a Name to a Name (line 111):
# Getting the type of 'None' (line 111)
None_203643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'None')
# Getting the type of 'FakeRunner'
FakeRunner_203644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FakeRunner')
# Setting the type of the member 'initArgs' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FakeRunner_203644, 'initArgs', None_203643)

# Assigning a Name to a Name (line 112):
# Getting the type of 'None' (line 112)
None_203645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'None')
# Getting the type of 'FakeRunner'
FakeRunner_203646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FakeRunner')
# Setting the type of the member 'test' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FakeRunner_203646, 'test', None_203645)

# Assigning a Name to a Name (line 113):
# Getting the type of 'False' (line 113)
False_203647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'False')
# Getting the type of 'FakeRunner'
FakeRunner_203648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FakeRunner')
# Setting the type of the member 'raiseError' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FakeRunner_203648, 'raiseError', False_203647)
# Declaration of the 'TestCommandLineArgs' class
# Getting the type of 'unittest' (line 125)
unittest_203649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), 'unittest')
# Obtaining the member 'TestCase' of a type (line 125)
TestCase_203650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 26), unittest_203649, 'TestCase')

class TestCommandLineArgs(TestCase_203650, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_localization', localization)
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_function_name', 'TestCommandLineArgs.setUp')
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCommandLineArgs.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCommandLineArgs.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 128):
        
        # Call to InitialisableProgram(...): (line 128)
        # Processing the call keyword arguments (line 128)
        kwargs_203652 = {}
        # Getting the type of 'InitialisableProgram' (line 128)
        InitialisableProgram_203651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'InitialisableProgram', False)
        # Calling InitialisableProgram(args, kwargs) (line 128)
        InitialisableProgram_call_result_203653 = invoke(stypy.reporting.localization.Localization(__file__, 128, 23), InitialisableProgram_203651, *[], **kwargs_203652)
        
        # Getting the type of 'self' (line 128)
        self_203654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member 'program' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_203654, 'program', InitialisableProgram_call_result_203653)
        
        # Assigning a Lambda to a Attribute (line 129):

        @norecursion
        def _stypy_temp_lambda_90(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_90'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_90', 129, 35, True)
            # Passed parameters checking function
            _stypy_temp_lambda_90.stypy_localization = localization
            _stypy_temp_lambda_90.stypy_type_of_self = None
            _stypy_temp_lambda_90.stypy_type_store = module_type_store
            _stypy_temp_lambda_90.stypy_function_name = '_stypy_temp_lambda_90'
            _stypy_temp_lambda_90.stypy_param_names_list = []
            _stypy_temp_lambda_90.stypy_varargs_param_name = None
            _stypy_temp_lambda_90.stypy_kwargs_param_name = None
            _stypy_temp_lambda_90.stypy_call_defaults = defaults
            _stypy_temp_lambda_90.stypy_call_varargs = varargs
            _stypy_temp_lambda_90.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_90', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_90', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 129)
            None_203655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 43), 'None')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 35), 'stypy_return_type', None_203655)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_90' in the type store
            # Getting the type of 'stypy_return_type' (line 129)
            stypy_return_type_203656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 35), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203656)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_90'
            return stypy_return_type_203656

        # Assigning a type to the variable '_stypy_temp_lambda_90' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 35), '_stypy_temp_lambda_90', _stypy_temp_lambda_90)
        # Getting the type of '_stypy_temp_lambda_90' (line 129)
        _stypy_temp_lambda_90_203657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 35), '_stypy_temp_lambda_90')
        # Getting the type of 'self' (line 129)
        self_203658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Obtaining the member 'program' of a type (line 129)
        program_203659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_203658, 'program')
        # Setting the type of the member 'createTests' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), program_203659, 'createTests', _stypy_temp_lambda_90_203657)
        
        # Assigning a Name to a Attribute (line 130):
        # Getting the type of 'None' (line 130)
        None_203660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 30), 'None')
        # Getting the type of 'FakeRunner' (line 130)
        FakeRunner_203661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'FakeRunner')
        # Setting the type of the member 'initArgs' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), FakeRunner_203661, 'initArgs', None_203660)
        
        # Assigning a Name to a Attribute (line 131):
        # Getting the type of 'None' (line 131)
        None_203662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'None')
        # Getting the type of 'FakeRunner' (line 131)
        FakeRunner_203663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'FakeRunner')
        # Setting the type of the member 'test' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), FakeRunner_203663, 'test', None_203662)
        
        # Assigning a Name to a Attribute (line 132):
        # Getting the type of 'False' (line 132)
        False_203664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 32), 'False')
        # Getting the type of 'FakeRunner' (line 132)
        FakeRunner_203665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'FakeRunner')
        # Setting the type of the member 'raiseError' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), FakeRunner_203665, 'raiseError', False_203664)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_203666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203666)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_203666


    @norecursion
    def testHelpAndUnknown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testHelpAndUnknown'
        module_type_store = module_type_store.open_function_context('testHelpAndUnknown', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_localization', localization)
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_function_name', 'TestCommandLineArgs.testHelpAndUnknown')
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_param_names_list', [])
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCommandLineArgs.testHelpAndUnknown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCommandLineArgs.testHelpAndUnknown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testHelpAndUnknown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testHelpAndUnknown(...)' code ##################

        
        # Assigning a Attribute to a Name (line 135):
        # Getting the type of 'self' (line 135)
        self_203667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'self')
        # Obtaining the member 'program' of a type (line 135)
        program_203668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 18), self_203667, 'program')
        # Assigning a type to the variable 'program' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'program', program_203668)

        @norecursion
        def usageExit(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'None' (line 136)
            None_203669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 26), 'None')
            defaults = [None_203669]
            # Create a new context for function 'usageExit'
            module_type_store = module_type_store.open_function_context('usageExit', 136, 8, False)
            
            # Passed parameters checking function
            usageExit.stypy_localization = localization
            usageExit.stypy_type_of_self = None
            usageExit.stypy_type_store = module_type_store
            usageExit.stypy_function_name = 'usageExit'
            usageExit.stypy_param_names_list = ['msg']
            usageExit.stypy_varargs_param_name = None
            usageExit.stypy_kwargs_param_name = None
            usageExit.stypy_call_defaults = defaults
            usageExit.stypy_call_varargs = varargs
            usageExit.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'usageExit', ['msg'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'usageExit', localization, ['msg'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'usageExit(...)' code ##################

            
            # Assigning a Name to a Attribute (line 137):
            # Getting the type of 'msg' (line 137)
            msg_203670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'msg')
            # Getting the type of 'program' (line 137)
            program_203671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'program')
            # Setting the type of the member 'msg' of a type (line 137)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), program_203671, 'msg', msg_203670)
            
            # Assigning a Name to a Attribute (line 138):
            # Getting the type of 'True' (line 138)
            True_203672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'True')
            # Getting the type of 'program' (line 138)
            program_203673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'program')
            # Setting the type of the member 'exit' of a type (line 138)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), program_203673, 'exit', True_203672)
            
            # ################# End of 'usageExit(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'usageExit' in the type store
            # Getting the type of 'stypy_return_type' (line 136)
            stypy_return_type_203674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203674)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'usageExit'
            return stypy_return_type_203674

        # Assigning a type to the variable 'usageExit' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'usageExit', usageExit)
        
        # Assigning a Name to a Attribute (line 139):
        # Getting the type of 'usageExit' (line 139)
        usageExit_203675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'usageExit')
        # Getting the type of 'program' (line 139)
        program_203676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'program')
        # Setting the type of the member 'usageExit' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), program_203676, 'usageExit', usageExit_203675)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 141)
        tuple_203677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 141)
        # Adding element type (line 141)
        str_203678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'str', '-h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 19), tuple_203677, str_203678)
        # Adding element type (line 141)
        str_203679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 25), 'str', '-H')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 19), tuple_203677, str_203679)
        # Adding element type (line 141)
        str_203680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 31), 'str', '--help')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 19), tuple_203677, str_203680)
        
        # Testing the type of a for loop iterable (line 141)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 8), tuple_203677)
        # Getting the type of the for loop variable (line 141)
        for_loop_var_203681 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 8), tuple_203677)
        # Assigning a type to the variable 'opt' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'opt', for_loop_var_203681)
        # SSA begins for a for statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Attribute (line 142):
        # Getting the type of 'False' (line 142)
        False_203682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'False')
        # Getting the type of 'program' (line 142)
        program_203683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'program')
        # Setting the type of the member 'exit' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), program_203683, 'exit', False_203682)
        
        # Call to parseArgs(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_203686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        # Getting the type of 'None' (line 143)
        None_203687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 30), list_203686, None_203687)
        # Adding element type (line 143)
        # Getting the type of 'opt' (line 143)
        opt_203688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 37), 'opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 30), list_203686, opt_203688)
        
        # Processing the call keyword arguments (line 143)
        kwargs_203689 = {}
        # Getting the type of 'program' (line 143)
        program_203684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'program', False)
        # Obtaining the member 'parseArgs' of a type (line 143)
        parseArgs_203685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), program_203684, 'parseArgs')
        # Calling parseArgs(args, kwargs) (line 143)
        parseArgs_call_result_203690 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), parseArgs_203685, *[list_203686], **kwargs_203689)
        
        
        # Call to assertTrue(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'program' (line 144)
        program_203693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'program', False)
        # Obtaining the member 'exit' of a type (line 144)
        exit_203694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), program_203693, 'exit')
        # Processing the call keyword arguments (line 144)
        kwargs_203695 = {}
        # Getting the type of 'self' (line 144)
        self_203691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 144)
        assertTrue_203692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), self_203691, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 144)
        assertTrue_call_result_203696 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), assertTrue_203692, *[exit_203694], **kwargs_203695)
        
        
        # Call to assertIsNone(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'program' (line 145)
        program_203699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), 'program', False)
        # Obtaining the member 'msg' of a type (line 145)
        msg_203700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 30), program_203699, 'msg')
        # Processing the call keyword arguments (line 145)
        kwargs_203701 = {}
        # Getting the type of 'self' (line 145)
        self_203697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 145)
        assertIsNone_203698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), self_203697, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 145)
        assertIsNone_call_result_203702 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), assertIsNone_203698, *[msg_203700], **kwargs_203701)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to parseArgs(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_203705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        # Getting the type of 'None' (line 147)
        None_203706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 26), list_203705, None_203706)
        # Adding element type (line 147)
        str_203707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 33), 'str', '-$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 26), list_203705, str_203707)
        
        # Processing the call keyword arguments (line 147)
        kwargs_203708 = {}
        # Getting the type of 'program' (line 147)
        program_203703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'program', False)
        # Obtaining the member 'parseArgs' of a type (line 147)
        parseArgs_203704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), program_203703, 'parseArgs')
        # Calling parseArgs(args, kwargs) (line 147)
        parseArgs_call_result_203709 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), parseArgs_203704, *[list_203705], **kwargs_203708)
        
        
        # Call to assertTrue(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'program' (line 148)
        program_203712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'program', False)
        # Obtaining the member 'exit' of a type (line 148)
        exit_203713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 24), program_203712, 'exit')
        # Processing the call keyword arguments (line 148)
        kwargs_203714 = {}
        # Getting the type of 'self' (line 148)
        self_203710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 148)
        assertTrue_203711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_203710, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 148)
        assertTrue_call_result_203715 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), assertTrue_203711, *[exit_203713], **kwargs_203714)
        
        
        # Call to assertIsNotNone(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'program' (line 149)
        program_203718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 29), 'program', False)
        # Obtaining the member 'msg' of a type (line 149)
        msg_203719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 29), program_203718, 'msg')
        # Processing the call keyword arguments (line 149)
        kwargs_203720 = {}
        # Getting the type of 'self' (line 149)
        self_203716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self', False)
        # Obtaining the member 'assertIsNotNone' of a type (line 149)
        assertIsNotNone_203717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_203716, 'assertIsNotNone')
        # Calling assertIsNotNone(args, kwargs) (line 149)
        assertIsNotNone_call_result_203721 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assertIsNotNone_203717, *[msg_203719], **kwargs_203720)
        
        
        # ################# End of 'testHelpAndUnknown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testHelpAndUnknown' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_203722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203722)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testHelpAndUnknown'
        return stypy_return_type_203722


    @norecursion
    def testVerbosity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testVerbosity'
        module_type_store = module_type_store.open_function_context('testVerbosity', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_localization', localization)
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_function_name', 'TestCommandLineArgs.testVerbosity')
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_param_names_list', [])
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCommandLineArgs.testVerbosity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCommandLineArgs.testVerbosity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testVerbosity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testVerbosity(...)' code ##################

        
        # Assigning a Attribute to a Name (line 152):
        # Getting the type of 'self' (line 152)
        self_203723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'self')
        # Obtaining the member 'program' of a type (line 152)
        program_203724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 18), self_203723, 'program')
        # Assigning a type to the variable 'program' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'program', program_203724)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 154)
        tuple_203725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 154)
        # Adding element type (line 154)
        str_203726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 19), 'str', '-q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), tuple_203725, str_203726)
        # Adding element type (line 154)
        str_203727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 25), 'str', '--quiet')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 19), tuple_203725, str_203727)
        
        # Testing the type of a for loop iterable (line 154)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 154, 8), tuple_203725)
        # Getting the type of the for loop variable (line 154)
        for_loop_var_203728 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 154, 8), tuple_203725)
        # Assigning a type to the variable 'opt' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'opt', for_loop_var_203728)
        # SSA begins for a for statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Attribute (line 155):
        int_203729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 32), 'int')
        # Getting the type of 'program' (line 155)
        program_203730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'program')
        # Setting the type of the member 'verbosity' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), program_203730, 'verbosity', int_203729)
        
        # Call to parseArgs(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_203733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        # Getting the type of 'None' (line 156)
        None_203734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 31), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 30), list_203733, None_203734)
        # Adding element type (line 156)
        # Getting the type of 'opt' (line 156)
        opt_203735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 37), 'opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 30), list_203733, opt_203735)
        
        # Processing the call keyword arguments (line 156)
        kwargs_203736 = {}
        # Getting the type of 'program' (line 156)
        program_203731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'program', False)
        # Obtaining the member 'parseArgs' of a type (line 156)
        parseArgs_203732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 12), program_203731, 'parseArgs')
        # Calling parseArgs(args, kwargs) (line 156)
        parseArgs_call_result_203737 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), parseArgs_203732, *[list_203733], **kwargs_203736)
        
        
        # Call to assertEqual(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'program' (line 157)
        program_203740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'program', False)
        # Obtaining the member 'verbosity' of a type (line 157)
        verbosity_203741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 29), program_203740, 'verbosity')
        int_203742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 48), 'int')
        # Processing the call keyword arguments (line 157)
        kwargs_203743 = {}
        # Getting the type of 'self' (line 157)
        self_203738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 157)
        assertEqual_203739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), self_203738, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 157)
        assertEqual_call_result_203744 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), assertEqual_203739, *[verbosity_203741, int_203742], **kwargs_203743)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 159)
        tuple_203745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 159)
        # Adding element type (line 159)
        str_203746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 19), 'str', '-v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), tuple_203745, str_203746)
        # Adding element type (line 159)
        str_203747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'str', '--verbose')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 19), tuple_203745, str_203747)
        
        # Testing the type of a for loop iterable (line 159)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 8), tuple_203745)
        # Getting the type of the for loop variable (line 159)
        for_loop_var_203748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 8), tuple_203745)
        # Assigning a type to the variable 'opt' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'opt', for_loop_var_203748)
        # SSA begins for a for statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Attribute (line 160):
        int_203749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 32), 'int')
        # Getting the type of 'program' (line 160)
        program_203750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'program')
        # Setting the type of the member 'verbosity' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), program_203750, 'verbosity', int_203749)
        
        # Call to parseArgs(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Obtaining an instance of the builtin type 'list' (line 161)
        list_203753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 161)
        # Adding element type (line 161)
        # Getting the type of 'None' (line 161)
        None_203754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 31), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 30), list_203753, None_203754)
        # Adding element type (line 161)
        # Getting the type of 'opt' (line 161)
        opt_203755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 30), list_203753, opt_203755)
        
        # Processing the call keyword arguments (line 161)
        kwargs_203756 = {}
        # Getting the type of 'program' (line 161)
        program_203751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'program', False)
        # Obtaining the member 'parseArgs' of a type (line 161)
        parseArgs_203752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), program_203751, 'parseArgs')
        # Calling parseArgs(args, kwargs) (line 161)
        parseArgs_call_result_203757 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), parseArgs_203752, *[list_203753], **kwargs_203756)
        
        
        # Call to assertEqual(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'program' (line 162)
        program_203760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'program', False)
        # Obtaining the member 'verbosity' of a type (line 162)
        verbosity_203761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 29), program_203760, 'verbosity')
        int_203762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 48), 'int')
        # Processing the call keyword arguments (line 162)
        kwargs_203763 = {}
        # Getting the type of 'self' (line 162)
        self_203758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 162)
        assertEqual_203759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), self_203758, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 162)
        assertEqual_call_result_203764 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), assertEqual_203759, *[verbosity_203761, int_203762], **kwargs_203763)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'testVerbosity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testVerbosity' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_203765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testVerbosity'
        return stypy_return_type_203765


    @norecursion
    def testBufferCatchFailfast(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testBufferCatchFailfast'
        module_type_store = module_type_store.open_function_context('testBufferCatchFailfast', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_localization', localization)
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_function_name', 'TestCommandLineArgs.testBufferCatchFailfast')
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_param_names_list', [])
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCommandLineArgs.testBufferCatchFailfast.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCommandLineArgs.testBufferCatchFailfast', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testBufferCatchFailfast', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testBufferCatchFailfast(...)' code ##################

        
        # Assigning a Attribute to a Name (line 165):
        # Getting the type of 'self' (line 165)
        self_203766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'self')
        # Obtaining the member 'program' of a type (line 165)
        program_203767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 18), self_203766, 'program')
        # Assigning a type to the variable 'program' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'program', program_203767)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 166)
        tuple_203768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 166)
        # Adding element type (line 166)
        
        # Obtaining an instance of the builtin type 'tuple' (line 166)
        tuple_203769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 166)
        # Adding element type (line 166)
        str_203770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 27), 'str', 'buffer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 27), tuple_203769, str_203770)
        # Adding element type (line 166)
        str_203771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 37), 'str', 'buffer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 27), tuple_203769, str_203771)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 26), tuple_203768, tuple_203769)
        # Adding element type (line 166)
        
        # Obtaining an instance of the builtin type 'tuple' (line 166)
        tuple_203772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 166)
        # Adding element type (line 166)
        str_203773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 49), 'str', 'failfast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 49), tuple_203772, str_203773)
        # Adding element type (line 166)
        str_203774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 61), 'str', 'failfast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 49), tuple_203772, str_203774)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 26), tuple_203768, tuple_203772)
        # Adding element type (line 166)
        
        # Obtaining an instance of the builtin type 'tuple' (line 167)
        tuple_203775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 167)
        # Adding element type (line 167)
        str_203776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'str', 'catch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), tuple_203775, str_203776)
        # Adding element type (line 167)
        str_203777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 32), 'str', 'catchbreak')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), tuple_203775, str_203777)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 26), tuple_203768, tuple_203775)
        
        # Testing the type of a for loop iterable (line 166)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 166, 8), tuple_203768)
        # Getting the type of the for loop variable (line 166)
        for_loop_var_203778 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 166, 8), tuple_203768)
        # Assigning a type to the variable 'arg' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 8), for_loop_var_203778))
        # Assigning a type to the variable 'attr' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'attr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 8), for_loop_var_203778))
        # SSA begins for a for statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'attr' (line 168)
        attr_203779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'attr')
        str_203780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'str', 'catch')
        # Applying the binary operator '==' (line 168)
        result_eq_203781 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), '==', attr_203779, str_203780)
        
        
        # Getting the type of 'hasInstallHandler' (line 168)
        hasInstallHandler_203782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'hasInstallHandler')
        # Applying the 'not' unary operator (line 168)
        result_not__203783 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 35), 'not', hasInstallHandler_203782)
        
        # Applying the binary operator 'and' (line 168)
        result_and_keyword_203784 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), 'and', result_eq_203781, result_not__203783)
        
        # Testing the type of an if condition (line 168)
        if_condition_203785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 12), result_and_keyword_203784)
        # Assigning a type to the variable 'if_condition_203785' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'if_condition_203785', if_condition_203785)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 171):
        str_203786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 24), 'str', '-%s')
        
        # Obtaining the type of the subscript
        int_203787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 36), 'int')
        # Getting the type of 'arg' (line 171)
        arg_203788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 32), 'arg')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___203789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 32), arg_203788, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_203790 = invoke(stypy.reporting.localization.Localization(__file__, 171, 32), getitem___203789, int_203787)
        
        # Applying the binary operator '%' (line 171)
        result_mod_203791 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 24), '%', str_203786, subscript_call_result_203790)
        
        # Assigning a type to the variable 'short_opt' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'short_opt', result_mod_203791)
        
        # Assigning a BinOp to a Name (line 172):
        str_203792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 23), 'str', '--%s')
        # Getting the type of 'arg' (line 172)
        arg_203793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'arg')
        # Applying the binary operator '%' (line 172)
        result_mod_203794 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 23), '%', str_203792, arg_203793)
        
        # Assigning a type to the variable 'long_opt' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'long_opt', result_mod_203794)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 173)
        tuple_203795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 173)
        # Adding element type (line 173)
        # Getting the type of 'short_opt' (line 173)
        short_opt_203796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'short_opt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 23), tuple_203795, short_opt_203796)
        # Adding element type (line 173)
        # Getting the type of 'long_opt' (line 173)
        long_opt_203797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 34), 'long_opt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 23), tuple_203795, long_opt_203797)
        
        # Testing the type of a for loop iterable (line 173)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 173, 12), tuple_203795)
        # Getting the type of the for loop variable (line 173)
        for_loop_var_203798 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 173, 12), tuple_203795)
        # Assigning a type to the variable 'opt' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'opt', for_loop_var_203798)
        # SSA begins for a for statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'program' (line 174)
        program_203800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'program', False)
        # Getting the type of 'attr' (line 174)
        attr_203801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 33), 'attr', False)
        # Getting the type of 'None' (line 174)
        None_203802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 39), 'None', False)
        # Processing the call keyword arguments (line 174)
        kwargs_203803 = {}
        # Getting the type of 'setattr' (line 174)
        setattr_203799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'setattr', False)
        # Calling setattr(args, kwargs) (line 174)
        setattr_call_result_203804 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), setattr_203799, *[program_203800, attr_203801, None_203802], **kwargs_203803)
        
        
        # Call to parseArgs(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_203807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        # Getting the type of 'None' (line 176)
        None_203808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 34), list_203807, None_203808)
        # Adding element type (line 176)
        # Getting the type of 'opt' (line 176)
        opt_203809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 41), 'opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 34), list_203807, opt_203809)
        
        # Processing the call keyword arguments (line 176)
        kwargs_203810 = {}
        # Getting the type of 'program' (line 176)
        program_203805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'program', False)
        # Obtaining the member 'parseArgs' of a type (line 176)
        parseArgs_203806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), program_203805, 'parseArgs')
        # Calling parseArgs(args, kwargs) (line 176)
        parseArgs_call_result_203811 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), parseArgs_203806, *[list_203807], **kwargs_203810)
        
        
        # Call to assertTrue(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to getattr(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'program' (line 177)
        program_203815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 40), 'program', False)
        # Getting the type of 'attr' (line 177)
        attr_203816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 49), 'attr', False)
        # Processing the call keyword arguments (line 177)
        kwargs_203817 = {}
        # Getting the type of 'getattr' (line 177)
        getattr_203814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'getattr', False)
        # Calling getattr(args, kwargs) (line 177)
        getattr_call_result_203818 = invoke(stypy.reporting.localization.Localization(__file__, 177, 32), getattr_203814, *[program_203815, attr_203816], **kwargs_203817)
        
        # Processing the call keyword arguments (line 177)
        kwargs_203819 = {}
        # Getting the type of 'self' (line 177)
        self_203812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 177)
        assertTrue_203813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 16), self_203812, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 177)
        assertTrue_call_result_203820 = invoke(stypy.reporting.localization.Localization(__file__, 177, 16), assertTrue_203813, *[getattr_call_result_203818], **kwargs_203819)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 179)
        tuple_203821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 179)
        # Adding element type (line 179)
        # Getting the type of 'short_opt' (line 179)
        short_opt_203822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 23), 'short_opt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 23), tuple_203821, short_opt_203822)
        # Adding element type (line 179)
        # Getting the type of 'long_opt' (line 179)
        long_opt_203823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'long_opt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 23), tuple_203821, long_opt_203823)
        
        # Testing the type of a for loop iterable (line 179)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 179, 12), tuple_203821)
        # Getting the type of the for loop variable (line 179)
        for_loop_var_203824 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 179, 12), tuple_203821)
        # Assigning a type to the variable 'opt' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'opt', for_loop_var_203824)
        # SSA begins for a for statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 180):
        
        # Call to object(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_203826 = {}
        # Getting the type of 'object' (line 180)
        object_203825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'object', False)
        # Calling object(args, kwargs) (line 180)
        object_call_result_203827 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), object_203825, *[], **kwargs_203826)
        
        # Assigning a type to the variable 'not_none' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'not_none', object_call_result_203827)
        
        # Call to setattr(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'program' (line 181)
        program_203829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'program', False)
        # Getting the type of 'attr' (line 181)
        attr_203830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'attr', False)
        # Getting the type of 'not_none' (line 181)
        not_none_203831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 39), 'not_none', False)
        # Processing the call keyword arguments (line 181)
        kwargs_203832 = {}
        # Getting the type of 'setattr' (line 181)
        setattr_203828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'setattr', False)
        # Calling setattr(args, kwargs) (line 181)
        setattr_call_result_203833 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), setattr_203828, *[program_203829, attr_203830, not_none_203831], **kwargs_203832)
        
        
        # Call to parseArgs(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_203836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        # Getting the type of 'None' (line 183)
        None_203837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 35), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 34), list_203836, None_203837)
        # Adding element type (line 183)
        # Getting the type of 'opt' (line 183)
        opt_203838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 41), 'opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 34), list_203836, opt_203838)
        
        # Processing the call keyword arguments (line 183)
        kwargs_203839 = {}
        # Getting the type of 'program' (line 183)
        program_203834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'program', False)
        # Obtaining the member 'parseArgs' of a type (line 183)
        parseArgs_203835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 16), program_203834, 'parseArgs')
        # Calling parseArgs(args, kwargs) (line 183)
        parseArgs_call_result_203840 = invoke(stypy.reporting.localization.Localization(__file__, 183, 16), parseArgs_203835, *[list_203836], **kwargs_203839)
        
        
        # Call to assertEqual(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Call to getattr(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'program' (line 184)
        program_203844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 41), 'program', False)
        # Getting the type of 'attr' (line 184)
        attr_203845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 50), 'attr', False)
        # Processing the call keyword arguments (line 184)
        kwargs_203846 = {}
        # Getting the type of 'getattr' (line 184)
        getattr_203843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 33), 'getattr', False)
        # Calling getattr(args, kwargs) (line 184)
        getattr_call_result_203847 = invoke(stypy.reporting.localization.Localization(__file__, 184, 33), getattr_203843, *[program_203844, attr_203845], **kwargs_203846)
        
        # Getting the type of 'not_none' (line 184)
        not_none_203848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 57), 'not_none', False)
        # Processing the call keyword arguments (line 184)
        kwargs_203849 = {}
        # Getting the type of 'self' (line 184)
        self_203841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 184)
        assertEqual_203842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), self_203841, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 184)
        assertEqual_call_result_203850 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), assertEqual_203842, *[getattr_call_result_203847, not_none_203848], **kwargs_203849)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'testBufferCatchFailfast(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testBufferCatchFailfast' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_203851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testBufferCatchFailfast'
        return stypy_return_type_203851


    @norecursion
    def testRunTestsRunnerClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testRunTestsRunnerClass'
        module_type_store = module_type_store.open_function_context('testRunTestsRunnerClass', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_localization', localization)
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_function_name', 'TestCommandLineArgs.testRunTestsRunnerClass')
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_param_names_list', [])
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCommandLineArgs.testRunTestsRunnerClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCommandLineArgs.testRunTestsRunnerClass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testRunTestsRunnerClass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testRunTestsRunnerClass(...)' code ##################

        
        # Assigning a Attribute to a Name (line 187):
        # Getting the type of 'self' (line 187)
        self_203852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 18), 'self')
        # Obtaining the member 'program' of a type (line 187)
        program_203853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 18), self_203852, 'program')
        # Assigning a type to the variable 'program' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'program', program_203853)
        
        # Assigning a Name to a Attribute (line 189):
        # Getting the type of 'FakeRunner' (line 189)
        FakeRunner_203854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'FakeRunner')
        # Getting the type of 'program' (line 189)
        program_203855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'program')
        # Setting the type of the member 'testRunner' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), program_203855, 'testRunner', FakeRunner_203854)
        
        # Assigning a Str to a Attribute (line 190):
        str_203856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 28), 'str', 'verbosity')
        # Getting the type of 'program' (line 190)
        program_203857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'program')
        # Setting the type of the member 'verbosity' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), program_203857, 'verbosity', str_203856)
        
        # Assigning a Str to a Attribute (line 191):
        str_203858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 27), 'str', 'failfast')
        # Getting the type of 'program' (line 191)
        program_203859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'program')
        # Setting the type of the member 'failfast' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), program_203859, 'failfast', str_203858)
        
        # Assigning a Str to a Attribute (line 192):
        str_203860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 25), 'str', 'buffer')
        # Getting the type of 'program' (line 192)
        program_203861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'program')
        # Setting the type of the member 'buffer' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), program_203861, 'buffer', str_203860)
        
        # Call to runTests(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_203864 = {}
        # Getting the type of 'program' (line 194)
        program_203862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'program', False)
        # Obtaining the member 'runTests' of a type (line 194)
        runTests_203863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), program_203862, 'runTests')
        # Calling runTests(args, kwargs) (line 194)
        runTests_call_result_203865 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), runTests_203863, *[], **kwargs_203864)
        
        
        # Call to assertEqual(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'FakeRunner' (line 196)
        FakeRunner_203868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 25), 'FakeRunner', False)
        # Obtaining the member 'initArgs' of a type (line 196)
        initArgs_203869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 25), FakeRunner_203868, 'initArgs')
        
        # Obtaining an instance of the builtin type 'dict' (line 196)
        dict_203870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 46), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 196)
        # Adding element type (key, value) (line 196)
        str_203871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 47), 'str', 'verbosity')
        str_203872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 60), 'str', 'verbosity')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 46), dict_203870, (str_203871, str_203872))
        # Adding element type (key, value) (line 196)
        str_203873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 48), 'str', 'failfast')
        str_203874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 60), 'str', 'failfast')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 46), dict_203870, (str_203873, str_203874))
        # Adding element type (key, value) (line 196)
        str_203875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 48), 'str', 'buffer')
        str_203876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 58), 'str', 'buffer')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 46), dict_203870, (str_203875, str_203876))
        
        # Processing the call keyword arguments (line 196)
        kwargs_203877 = {}
        # Getting the type of 'self' (line 196)
        self_203866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 196)
        assertEqual_203867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_203866, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 196)
        assertEqual_call_result_203878 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), assertEqual_203867, *[initArgs_203869, dict_203870], **kwargs_203877)
        
        
        # Call to assertEqual(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'FakeRunner' (line 199)
        FakeRunner_203881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 25), 'FakeRunner', False)
        # Obtaining the member 'test' of a type (line 199)
        test_203882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 25), FakeRunner_203881, 'test')
        str_203883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 42), 'str', 'test')
        # Processing the call keyword arguments (line 199)
        kwargs_203884 = {}
        # Getting the type of 'self' (line 199)
        self_203879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 199)
        assertEqual_203880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_203879, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 199)
        assertEqual_call_result_203885 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), assertEqual_203880, *[test_203882, str_203883], **kwargs_203884)
        
        
        # Call to assertIs(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'program' (line 200)
        program_203888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'program', False)
        # Obtaining the member 'result' of a type (line 200)
        result_203889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 22), program_203888, 'result')
        # Getting the type of 'RESULT' (line 200)
        RESULT_203890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 38), 'RESULT', False)
        # Processing the call keyword arguments (line 200)
        kwargs_203891 = {}
        # Getting the type of 'self' (line 200)
        self_203886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 200)
        assertIs_203887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_203886, 'assertIs')
        # Calling assertIs(args, kwargs) (line 200)
        assertIs_call_result_203892 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), assertIs_203887, *[result_203889, RESULT_203890], **kwargs_203891)
        
        
        # ################# End of 'testRunTestsRunnerClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testRunTestsRunnerClass' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_203893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203893)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testRunTestsRunnerClass'
        return stypy_return_type_203893


    @norecursion
    def testRunTestsRunnerInstance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testRunTestsRunnerInstance'
        module_type_store = module_type_store.open_function_context('testRunTestsRunnerInstance', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_localization', localization)
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_function_name', 'TestCommandLineArgs.testRunTestsRunnerInstance')
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_param_names_list', [])
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCommandLineArgs.testRunTestsRunnerInstance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCommandLineArgs.testRunTestsRunnerInstance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testRunTestsRunnerInstance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testRunTestsRunnerInstance(...)' code ##################

        
        # Assigning a Attribute to a Name (line 203):
        # Getting the type of 'self' (line 203)
        self_203894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'self')
        # Obtaining the member 'program' of a type (line 203)
        program_203895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 18), self_203894, 'program')
        # Assigning a type to the variable 'program' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'program', program_203895)
        
        # Assigning a Call to a Attribute (line 205):
        
        # Call to FakeRunner(...): (line 205)
        # Processing the call keyword arguments (line 205)
        kwargs_203897 = {}
        # Getting the type of 'FakeRunner' (line 205)
        FakeRunner_203896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'FakeRunner', False)
        # Calling FakeRunner(args, kwargs) (line 205)
        FakeRunner_call_result_203898 = invoke(stypy.reporting.localization.Localization(__file__, 205, 29), FakeRunner_203896, *[], **kwargs_203897)
        
        # Getting the type of 'program' (line 205)
        program_203899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'program')
        # Setting the type of the member 'testRunner' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), program_203899, 'testRunner', FakeRunner_call_result_203898)
        
        # Assigning a Name to a Attribute (line 206):
        # Getting the type of 'None' (line 206)
        None_203900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 30), 'None')
        # Getting the type of 'FakeRunner' (line 206)
        FakeRunner_203901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'FakeRunner')
        # Setting the type of the member 'initArgs' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), FakeRunner_203901, 'initArgs', None_203900)
        
        # Call to runTests(...): (line 208)
        # Processing the call keyword arguments (line 208)
        kwargs_203904 = {}
        # Getting the type of 'program' (line 208)
        program_203902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'program', False)
        # Obtaining the member 'runTests' of a type (line 208)
        runTests_203903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), program_203902, 'runTests')
        # Calling runTests(args, kwargs) (line 208)
        runTests_call_result_203905 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), runTests_203903, *[], **kwargs_203904)
        
        
        # Call to assertIsNone(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'FakeRunner' (line 211)
        FakeRunner_203908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 26), 'FakeRunner', False)
        # Obtaining the member 'initArgs' of a type (line 211)
        initArgs_203909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 26), FakeRunner_203908, 'initArgs')
        # Processing the call keyword arguments (line 211)
        kwargs_203910 = {}
        # Getting the type of 'self' (line 211)
        self_203906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 211)
        assertIsNone_203907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_203906, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 211)
        assertIsNone_call_result_203911 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assertIsNone_203907, *[initArgs_203909], **kwargs_203910)
        
        
        # Call to assertEqual(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'FakeRunner' (line 213)
        FakeRunner_203914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'FakeRunner', False)
        # Obtaining the member 'test' of a type (line 213)
        test_203915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 25), FakeRunner_203914, 'test')
        str_203916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 42), 'str', 'test')
        # Processing the call keyword arguments (line 213)
        kwargs_203917 = {}
        # Getting the type of 'self' (line 213)
        self_203912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 213)
        assertEqual_203913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), self_203912, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 213)
        assertEqual_call_result_203918 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assertEqual_203913, *[test_203915, str_203916], **kwargs_203917)
        
        
        # Call to assertIs(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'program' (line 214)
        program_203921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 22), 'program', False)
        # Obtaining the member 'result' of a type (line 214)
        result_203922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 22), program_203921, 'result')
        # Getting the type of 'RESULT' (line 214)
        RESULT_203923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 38), 'RESULT', False)
        # Processing the call keyword arguments (line 214)
        kwargs_203924 = {}
        # Getting the type of 'self' (line 214)
        self_203919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 214)
        assertIs_203920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_203919, 'assertIs')
        # Calling assertIs(args, kwargs) (line 214)
        assertIs_call_result_203925 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assertIs_203920, *[result_203922, RESULT_203923], **kwargs_203924)
        
        
        # ################# End of 'testRunTestsRunnerInstance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testRunTestsRunnerInstance' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_203926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203926)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testRunTestsRunnerInstance'
        return stypy_return_type_203926


    @norecursion
    def testRunTestsOldRunnerClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testRunTestsOldRunnerClass'
        module_type_store = module_type_store.open_function_context('testRunTestsOldRunnerClass', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_localization', localization)
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_function_name', 'TestCommandLineArgs.testRunTestsOldRunnerClass')
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_param_names_list', [])
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCommandLineArgs.testRunTestsOldRunnerClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCommandLineArgs.testRunTestsOldRunnerClass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testRunTestsOldRunnerClass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testRunTestsOldRunnerClass(...)' code ##################

        
        # Assigning a Attribute to a Name (line 217):
        # Getting the type of 'self' (line 217)
        self_203927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 18), 'self')
        # Obtaining the member 'program' of a type (line 217)
        program_203928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 18), self_203927, 'program')
        # Assigning a type to the variable 'program' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'program', program_203928)
        
        # Assigning a Name to a Attribute (line 219):
        # Getting the type of 'True' (line 219)
        True_203929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 32), 'True')
        # Getting the type of 'FakeRunner' (line 219)
        FakeRunner_203930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'FakeRunner')
        # Setting the type of the member 'raiseError' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), FakeRunner_203930, 'raiseError', True_203929)
        
        # Assigning a Name to a Attribute (line 220):
        # Getting the type of 'FakeRunner' (line 220)
        FakeRunner_203931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 29), 'FakeRunner')
        # Getting the type of 'program' (line 220)
        program_203932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'program')
        # Setting the type of the member 'testRunner' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), program_203932, 'testRunner', FakeRunner_203931)
        
        # Assigning a Str to a Attribute (line 221):
        str_203933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 28), 'str', 'verbosity')
        # Getting the type of 'program' (line 221)
        program_203934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'program')
        # Setting the type of the member 'verbosity' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), program_203934, 'verbosity', str_203933)
        
        # Assigning a Str to a Attribute (line 222):
        str_203935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 27), 'str', 'failfast')
        # Getting the type of 'program' (line 222)
        program_203936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'program')
        # Setting the type of the member 'failfast' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), program_203936, 'failfast', str_203935)
        
        # Assigning a Str to a Attribute (line 223):
        str_203937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 25), 'str', 'buffer')
        # Getting the type of 'program' (line 223)
        program_203938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'program')
        # Setting the type of the member 'buffer' of a type (line 223)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), program_203938, 'buffer', str_203937)
        
        # Assigning a Str to a Attribute (line 224):
        str_203939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 23), 'str', 'test')
        # Getting the type of 'program' (line 224)
        program_203940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'program')
        # Setting the type of the member 'test' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), program_203940, 'test', str_203939)
        
        # Call to runTests(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_203943 = {}
        # Getting the type of 'program' (line 226)
        program_203941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'program', False)
        # Obtaining the member 'runTests' of a type (line 226)
        runTests_203942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), program_203941, 'runTests')
        # Calling runTests(args, kwargs) (line 226)
        runTests_call_result_203944 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), runTests_203942, *[], **kwargs_203943)
        
        
        # Call to assertEqual(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'FakeRunner' (line 230)
        FakeRunner_203947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'FakeRunner', False)
        # Obtaining the member 'initArgs' of a type (line 230)
        initArgs_203948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 25), FakeRunner_203947, 'initArgs')
        
        # Obtaining an instance of the builtin type 'dict' (line 230)
        dict_203949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 46), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 230)
        
        # Processing the call keyword arguments (line 230)
        kwargs_203950 = {}
        # Getting the type of 'self' (line 230)
        self_203945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 230)
        assertEqual_203946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_203945, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 230)
        assertEqual_call_result_203951 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), assertEqual_203946, *[initArgs_203948, dict_203949], **kwargs_203950)
        
        
        # Call to assertEqual(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'FakeRunner' (line 231)
        FakeRunner_203954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 25), 'FakeRunner', False)
        # Obtaining the member 'test' of a type (line 231)
        test_203955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 25), FakeRunner_203954, 'test')
        str_203956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 42), 'str', 'test')
        # Processing the call keyword arguments (line 231)
        kwargs_203957 = {}
        # Getting the type of 'self' (line 231)
        self_203952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 231)
        assertEqual_203953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), self_203952, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 231)
        assertEqual_call_result_203958 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), assertEqual_203953, *[test_203955, str_203956], **kwargs_203957)
        
        
        # Call to assertIs(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'program' (line 232)
        program_203961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'program', False)
        # Obtaining the member 'result' of a type (line 232)
        result_203962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 22), program_203961, 'result')
        # Getting the type of 'RESULT' (line 232)
        RESULT_203963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 38), 'RESULT', False)
        # Processing the call keyword arguments (line 232)
        kwargs_203964 = {}
        # Getting the type of 'self' (line 232)
        self_203959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 232)
        assertIs_203960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_203959, 'assertIs')
        # Calling assertIs(args, kwargs) (line 232)
        assertIs_call_result_203965 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), assertIs_203960, *[result_203962, RESULT_203963], **kwargs_203964)
        
        
        # ################# End of 'testRunTestsOldRunnerClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testRunTestsOldRunnerClass' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_203966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_203966)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testRunTestsOldRunnerClass'
        return stypy_return_type_203966


    @norecursion
    def testCatchBreakInstallsHandler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testCatchBreakInstallsHandler'
        module_type_store = module_type_store.open_function_context('testCatchBreakInstallsHandler', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_localization', localization)
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_function_name', 'TestCommandLineArgs.testCatchBreakInstallsHandler')
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_param_names_list', [])
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCommandLineArgs.testCatchBreakInstallsHandler.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCommandLineArgs.testCatchBreakInstallsHandler', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testCatchBreakInstallsHandler', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testCatchBreakInstallsHandler(...)' code ##################

        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        str_203967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 29), 'str', 'unittest.main')
        # Getting the type of 'sys' (line 235)
        sys_203968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 17), 'sys')
        # Obtaining the member 'modules' of a type (line 235)
        modules_203969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), sys_203968, 'modules')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___203970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), modules_203969, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_203971 = invoke(stypy.reporting.localization.Localization(__file__, 235, 17), getitem___203970, str_203967)
        
        # Assigning a type to the variable 'module' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'module', subscript_call_result_203971)
        
        # Assigning a Attribute to a Name (line 236):
        # Getting the type of 'module' (line 236)
        module_203972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 19), 'module')
        # Obtaining the member 'installHandler' of a type (line 236)
        installHandler_203973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 19), module_203972, 'installHandler')
        # Assigning a type to the variable 'original' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'original', installHandler_203973)

        @norecursion
        def restore(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore'
            module_type_store = module_type_store.open_function_context('restore', 237, 8, False)
            
            # Passed parameters checking function
            restore.stypy_localization = localization
            restore.stypy_type_of_self = None
            restore.stypy_type_store = module_type_store
            restore.stypy_function_name = 'restore'
            restore.stypy_param_names_list = []
            restore.stypy_varargs_param_name = None
            restore.stypy_kwargs_param_name = None
            restore.stypy_call_defaults = defaults
            restore.stypy_call_varargs = varargs
            restore.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore(...)' code ##################

            
            # Assigning a Name to a Attribute (line 238):
            # Getting the type of 'original' (line 238)
            original_203974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'original')
            # Getting the type of 'module' (line 238)
            module_203975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'module')
            # Setting the type of the member 'installHandler' of a type (line 238)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), module_203975, 'installHandler', original_203974)
            
            # ################# End of 'restore(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore' in the type store
            # Getting the type of 'stypy_return_type' (line 237)
            stypy_return_type_203976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203976)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore'
            return stypy_return_type_203976

        # Assigning a type to the variable 'restore' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'restore', restore)
        
        # Call to addCleanup(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'restore' (line 239)
        restore_203979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 24), 'restore', False)
        # Processing the call keyword arguments (line 239)
        kwargs_203980 = {}
        # Getting the type of 'self' (line 239)
        self_203977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 239)
        addCleanup_203978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_203977, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 239)
        addCleanup_call_result_203981 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), addCleanup_203978, *[restore_203979], **kwargs_203980)
        
        
        # Assigning a Name to a Attribute (line 241):
        # Getting the type of 'False' (line 241)
        False_203982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 25), 'False')
        # Getting the type of 'self' (line 241)
        self_203983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'self')
        # Setting the type of the member 'installed' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), self_203983, 'installed', False_203982)

        @norecursion
        def fakeInstallHandler(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fakeInstallHandler'
            module_type_store = module_type_store.open_function_context('fakeInstallHandler', 242, 8, False)
            
            # Passed parameters checking function
            fakeInstallHandler.stypy_localization = localization
            fakeInstallHandler.stypy_type_of_self = None
            fakeInstallHandler.stypy_type_store = module_type_store
            fakeInstallHandler.stypy_function_name = 'fakeInstallHandler'
            fakeInstallHandler.stypy_param_names_list = []
            fakeInstallHandler.stypy_varargs_param_name = None
            fakeInstallHandler.stypy_kwargs_param_name = None
            fakeInstallHandler.stypy_call_defaults = defaults
            fakeInstallHandler.stypy_call_varargs = varargs
            fakeInstallHandler.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fakeInstallHandler', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fakeInstallHandler', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fakeInstallHandler(...)' code ##################

            
            # Assigning a Name to a Attribute (line 243):
            # Getting the type of 'True' (line 243)
            True_203984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'True')
            # Getting the type of 'self' (line 243)
            self_203985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self')
            # Setting the type of the member 'installed' of a type (line 243)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), self_203985, 'installed', True_203984)
            
            # ################# End of 'fakeInstallHandler(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fakeInstallHandler' in the type store
            # Getting the type of 'stypy_return_type' (line 242)
            stypy_return_type_203986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_203986)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fakeInstallHandler'
            return stypy_return_type_203986

        # Assigning a type to the variable 'fakeInstallHandler' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'fakeInstallHandler', fakeInstallHandler)
        
        # Assigning a Name to a Attribute (line 244):
        # Getting the type of 'fakeInstallHandler' (line 244)
        fakeInstallHandler_203987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 32), 'fakeInstallHandler')
        # Getting the type of 'module' (line 244)
        module_203988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'module')
        # Setting the type of the member 'installHandler' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), module_203988, 'installHandler', fakeInstallHandler_203987)
        
        # Assigning a Attribute to a Name (line 246):
        # Getting the type of 'self' (line 246)
        self_203989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 18), 'self')
        # Obtaining the member 'program' of a type (line 246)
        program_203990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 18), self_203989, 'program')
        # Assigning a type to the variable 'program' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'program', program_203990)
        
        # Assigning a Name to a Attribute (line 247):
        # Getting the type of 'True' (line 247)
        True_203991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'True')
        # Getting the type of 'program' (line 247)
        program_203992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'program')
        # Setting the type of the member 'catchbreak' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), program_203992, 'catchbreak', True_203991)
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'FakeRunner' (line 249)
        FakeRunner_203993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 29), 'FakeRunner')
        # Getting the type of 'program' (line 249)
        program_203994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'program')
        # Setting the type of the member 'testRunner' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), program_203994, 'testRunner', FakeRunner_203993)
        
        # Call to runTests(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_203997 = {}
        # Getting the type of 'program' (line 251)
        program_203995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'program', False)
        # Obtaining the member 'runTests' of a type (line 251)
        runTests_203996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), program_203995, 'runTests')
        # Calling runTests(args, kwargs) (line 251)
        runTests_call_result_203998 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), runTests_203996, *[], **kwargs_203997)
        
        
        # Call to assertTrue(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'self' (line 252)
        self_204001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 24), 'self', False)
        # Obtaining the member 'installed' of a type (line 252)
        installed_204002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 24), self_204001, 'installed')
        # Processing the call keyword arguments (line 252)
        kwargs_204003 = {}
        # Getting the type of 'self' (line 252)
        self_203999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 252)
        assertTrue_204000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), self_203999, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 252)
        assertTrue_call_result_204004 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), assertTrue_204000, *[installed_204002], **kwargs_204003)
        
        
        # ################# End of 'testCatchBreakInstallsHandler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testCatchBreakInstallsHandler' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_204005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_204005)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testCatchBreakInstallsHandler'
        return stypy_return_type_204005


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 125, 0, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCommandLineArgs.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCommandLineArgs' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'TestCommandLineArgs', TestCommandLineArgs)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 256)
    # Processing the call keyword arguments (line 256)
    kwargs_204008 = {}
    # Getting the type of 'unittest' (line 256)
    unittest_204006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 256)
    main_204007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), unittest_204006, 'main')
    # Calling main(args, kwargs) (line 256)
    main_call_result_204009 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), main_204007, *[], **kwargs_204008)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
