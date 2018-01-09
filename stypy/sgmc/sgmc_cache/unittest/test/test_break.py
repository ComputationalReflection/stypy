
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import gc
2: import os
3: import sys
4: import signal
5: import weakref
6: 
7: from cStringIO import StringIO
8: 
9: 
10: import unittest
11: 
12: 
13: @unittest.skipUnless(hasattr(os, 'kill'), "Test requires os.kill")
14: @unittest.skipIf(sys.platform =="win32", "Test cannot run on Windows")
15: @unittest.skipIf(sys.platform == 'freebsd6', "Test kills regrtest on freebsd6 "
16:     "if threads have been used")
17: class TestBreak(unittest.TestCase):
18:     int_handler = None
19: 
20:     def setUp(self):
21:         self._default_handler = signal.getsignal(signal.SIGINT)
22:         if self.int_handler is not None:
23:             signal.signal(signal.SIGINT, self.int_handler)
24: 
25:     def tearDown(self):
26:         signal.signal(signal.SIGINT, self._default_handler)
27:         unittest.signals._results = weakref.WeakKeyDictionary()
28:         unittest.signals._interrupt_handler = None
29: 
30: 
31:     def testInstallHandler(self):
32:         default_handler = signal.getsignal(signal.SIGINT)
33:         unittest.installHandler()
34:         self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)
35: 
36:         try:
37:             pid = os.getpid()
38:             os.kill(pid, signal.SIGINT)
39:         except KeyboardInterrupt:
40:             self.fail("KeyboardInterrupt not handled")
41: 
42:         self.assertTrue(unittest.signals._interrupt_handler.called)
43: 
44:     def testRegisterResult(self):
45:         result = unittest.TestResult()
46:         unittest.registerResult(result)
47: 
48:         for ref in unittest.signals._results:
49:             if ref is result:
50:                 break
51:             elif ref is not result:
52:                 self.fail("odd object in result set")
53:         else:
54:             self.fail("result not found")
55: 
56: 
57:     def testInterruptCaught(self):
58:         default_handler = signal.getsignal(signal.SIGINT)
59: 
60:         result = unittest.TestResult()
61:         unittest.installHandler()
62:         unittest.registerResult(result)
63: 
64:         self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)
65: 
66:         def test(result):
67:             pid = os.getpid()
68:             os.kill(pid, signal.SIGINT)
69:             result.breakCaught = True
70:             self.assertTrue(result.shouldStop)
71: 
72:         try:
73:             test(result)
74:         except KeyboardInterrupt:
75:             self.fail("KeyboardInterrupt not handled")
76:         self.assertTrue(result.breakCaught)
77: 
78: 
79:     def testSecondInterrupt(self):
80:         # Can't use skipIf decorator because the signal handler may have
81:         # been changed after defining this method.
82:         if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
83:             self.skipTest("test requires SIGINT to not be ignored")
84:         result = unittest.TestResult()
85:         unittest.installHandler()
86:         unittest.registerResult(result)
87: 
88:         def test(result):
89:             pid = os.getpid()
90:             os.kill(pid, signal.SIGINT)
91:             result.breakCaught = True
92:             self.assertTrue(result.shouldStop)
93:             os.kill(pid, signal.SIGINT)
94:             self.fail("Second KeyboardInterrupt not raised")
95: 
96:         try:
97:             test(result)
98:         except KeyboardInterrupt:
99:             pass
100:         else:
101:             self.fail("Second KeyboardInterrupt not raised")
102:         self.assertTrue(result.breakCaught)
103: 
104: 
105:     def testTwoResults(self):
106:         unittest.installHandler()
107: 
108:         result = unittest.TestResult()
109:         unittest.registerResult(result)
110:         new_handler = signal.getsignal(signal.SIGINT)
111: 
112:         result2 = unittest.TestResult()
113:         unittest.registerResult(result2)
114:         self.assertEqual(signal.getsignal(signal.SIGINT), new_handler)
115: 
116:         result3 = unittest.TestResult()
117: 
118:         def test(result):
119:             pid = os.getpid()
120:             os.kill(pid, signal.SIGINT)
121: 
122:         try:
123:             test(result)
124:         except KeyboardInterrupt:
125:             self.fail("KeyboardInterrupt not handled")
126: 
127:         self.assertTrue(result.shouldStop)
128:         self.assertTrue(result2.shouldStop)
129:         self.assertFalse(result3.shouldStop)
130: 
131: 
132:     def testHandlerReplacedButCalled(self):
133:         # Can't use skipIf decorator because the signal handler may have
134:         # been changed after defining this method.
135:         if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
136:             self.skipTest("test requires SIGINT to not be ignored")
137:         # If our handler has been replaced (is no longer installed) but is
138:         # called by the *new* handler, then it isn't safe to delay the
139:         # SIGINT and we should immediately delegate to the default handler
140:         unittest.installHandler()
141: 
142:         handler = signal.getsignal(signal.SIGINT)
143:         def new_handler(frame, signum):
144:             handler(frame, signum)
145:         signal.signal(signal.SIGINT, new_handler)
146: 
147:         try:
148:             pid = os.getpid()
149:             os.kill(pid, signal.SIGINT)
150:         except KeyboardInterrupt:
151:             pass
152:         else:
153:             self.fail("replaced but delegated handler doesn't raise interrupt")
154: 
155:     def testRunner(self):
156:         # Creating a TextTestRunner with the appropriate argument should
157:         # register the TextTestResult it creates
158:         runner = unittest.TextTestRunner(stream=StringIO())
159: 
160:         result = runner.run(unittest.TestSuite())
161:         self.assertIn(result, unittest.signals._results)
162: 
163:     def testWeakReferences(self):
164:         # Calling registerResult on a result should not keep it alive
165:         result = unittest.TestResult()
166:         unittest.registerResult(result)
167: 
168:         ref = weakref.ref(result)
169:         del result
170: 
171:         # For non-reference counting implementations
172:         gc.collect();gc.collect()
173:         self.assertIsNone(ref())
174: 
175: 
176:     def testRemoveResult(self):
177:         result = unittest.TestResult()
178:         unittest.registerResult(result)
179: 
180:         unittest.installHandler()
181:         self.assertTrue(unittest.removeResult(result))
182: 
183:         # Should this raise an error instead?
184:         self.assertFalse(unittest.removeResult(unittest.TestResult()))
185: 
186:         try:
187:             pid = os.getpid()
188:             os.kill(pid, signal.SIGINT)
189:         except KeyboardInterrupt:
190:             pass
191: 
192:         self.assertFalse(result.shouldStop)
193: 
194:     def testMainInstallsHandler(self):
195:         failfast = object()
196:         test = object()
197:         verbosity = object()
198:         result = object()
199:         default_handler = signal.getsignal(signal.SIGINT)
200: 
201:         class FakeRunner(object):
202:             initArgs = []
203:             runArgs = []
204:             def __init__(self, *args, **kwargs):
205:                 self.initArgs.append((args, kwargs))
206:             def run(self, test):
207:                 self.runArgs.append(test)
208:                 return result
209: 
210:         class Program(unittest.TestProgram):
211:             def __init__(self, catchbreak):
212:                 self.exit = False
213:                 self.verbosity = verbosity
214:                 self.failfast = failfast
215:                 self.catchbreak = catchbreak
216:                 self.testRunner = FakeRunner
217:                 self.test = test
218:                 self.result = None
219: 
220:         p = Program(False)
221:         p.runTests()
222: 
223:         self.assertEqual(FakeRunner.initArgs, [((), {'buffer': None,
224:                                                      'verbosity': verbosity,
225:                                                      'failfast': failfast})])
226:         self.assertEqual(FakeRunner.runArgs, [test])
227:         self.assertEqual(p.result, result)
228: 
229:         self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
230: 
231:         FakeRunner.initArgs = []
232:         FakeRunner.runArgs = []
233:         p = Program(True)
234:         p.runTests()
235: 
236:         self.assertEqual(FakeRunner.initArgs, [((), {'buffer': None,
237:                                                      'verbosity': verbosity,
238:                                                      'failfast': failfast})])
239:         self.assertEqual(FakeRunner.runArgs, [test])
240:         self.assertEqual(p.result, result)
241: 
242:         self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)
243: 
244:     def testRemoveHandler(self):
245:         default_handler = signal.getsignal(signal.SIGINT)
246:         unittest.installHandler()
247:         unittest.removeHandler()
248:         self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
249: 
250:         # check that calling removeHandler multiple times has no ill-effect
251:         unittest.removeHandler()
252:         self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
253: 
254:     def testRemoveHandlerAsDecorator(self):
255:         default_handler = signal.getsignal(signal.SIGINT)
256:         unittest.installHandler()
257: 
258:         @unittest.removeHandler
259:         def test():
260:             self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
261: 
262:         test()
263:         self.assertNotEqual(signal.getsignal(signal.SIGINT), default_handler)
264: 
265: @unittest.skipUnless(hasattr(os, 'kill'), "Test requires os.kill")
266: @unittest.skipIf(sys.platform =="win32", "Test cannot run on Windows")
267: @unittest.skipIf(sys.platform == 'freebsd6', "Test kills regrtest on freebsd6 "
268:     "if threads have been used")
269: class TestBreakDefaultIntHandler(TestBreak):
270:     int_handler = signal.default_int_handler
271: 
272: @unittest.skipUnless(hasattr(os, 'kill'), "Test requires os.kill")
273: @unittest.skipIf(sys.platform =="win32", "Test cannot run on Windows")
274: @unittest.skipIf(sys.platform == 'freebsd6', "Test kills regrtest on freebsd6 "
275:     "if threads have been used")
276: class TestBreakSignalIgnored(TestBreak):
277:     int_handler = signal.SIG_IGN
278: 
279: @unittest.skipUnless(hasattr(os, 'kill'), "Test requires os.kill")
280: @unittest.skipIf(sys.platform =="win32", "Test cannot run on Windows")
281: @unittest.skipIf(sys.platform == 'freebsd6', "Test kills regrtest on freebsd6 "
282:     "if threads have been used")
283: class TestBreakSignalDefault(TestBreak):
284:     int_handler = signal.SIG_DFL
285: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import gc' statement (line 1)
import gc

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'gc', gc, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import signal' statement (line 4)
import signal

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'signal', signal, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import weakref' statement (line 5)
import weakref

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'weakref', weakref, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from cStringIO import StringIO' statement (line 7)
from cStringIO import StringIO

import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import unittest' statement (line 10)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'unittest', unittest, module_type_store)

# Declaration of the 'TestBreak' class
# Getting the type of 'unittest' (line 17)
unittest_194322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'unittest')
# Obtaining the member 'TestCase' of a type (line 17)
TestCase_194323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), unittest_194322, 'TestCase')

class TestBreak(TestCase_194323, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.setUp.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.setUp.__dict__.__setitem__('stypy_function_name', 'TestBreak.setUp')
        TestBreak.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 21):
        
        # Call to getsignal(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'signal' (line 21)
        signal_194326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 49), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 21)
        SIGINT_194327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 49), signal_194326, 'SIGINT')
        # Processing the call keyword arguments (line 21)
        kwargs_194328 = {}
        # Getting the type of 'signal' (line 21)
        signal_194324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 32), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 21)
        getsignal_194325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 32), signal_194324, 'getsignal')
        # Calling getsignal(args, kwargs) (line 21)
        getsignal_call_result_194329 = invoke(stypy.reporting.localization.Localization(__file__, 21, 32), getsignal_194325, *[SIGINT_194327], **kwargs_194328)
        
        # Getting the type of 'self' (line 21)
        self_194330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member '_default_handler' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_194330, '_default_handler', getsignal_call_result_194329)
        
        
        # Getting the type of 'self' (line 22)
        self_194331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'self')
        # Obtaining the member 'int_handler' of a type (line 22)
        int_handler_194332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), self_194331, 'int_handler')
        # Getting the type of 'None' (line 22)
        None_194333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 35), 'None')
        # Applying the binary operator 'isnot' (line 22)
        result_is_not_194334 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), 'isnot', int_handler_194332, None_194333)
        
        # Testing the type of an if condition (line 22)
        if_condition_194335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 8), result_is_not_194334)
        # Assigning a type to the variable 'if_condition_194335' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'if_condition_194335', if_condition_194335)
        # SSA begins for if statement (line 22)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to signal(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'signal' (line 23)
        signal_194338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 23)
        SIGINT_194339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 26), signal_194338, 'SIGINT')
        # Getting the type of 'self' (line 23)
        self_194340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 41), 'self', False)
        # Obtaining the member 'int_handler' of a type (line 23)
        int_handler_194341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 41), self_194340, 'int_handler')
        # Processing the call keyword arguments (line 23)
        kwargs_194342 = {}
        # Getting the type of 'signal' (line 23)
        signal_194336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'signal', False)
        # Obtaining the member 'signal' of a type (line 23)
        signal_194337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), signal_194336, 'signal')
        # Calling signal(args, kwargs) (line 23)
        signal_call_result_194343 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), signal_194337, *[SIGINT_194339, int_handler_194341], **kwargs_194342)
        
        # SSA join for if statement (line 22)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_194344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194344)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_194344


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.tearDown.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.tearDown.__dict__.__setitem__('stypy_function_name', 'TestBreak.tearDown')
        TestBreak.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to signal(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'signal' (line 26)
        signal_194347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 22), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 26)
        SIGINT_194348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 22), signal_194347, 'SIGINT')
        # Getting the type of 'self' (line 26)
        self_194349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 37), 'self', False)
        # Obtaining the member '_default_handler' of a type (line 26)
        _default_handler_194350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 37), self_194349, '_default_handler')
        # Processing the call keyword arguments (line 26)
        kwargs_194351 = {}
        # Getting the type of 'signal' (line 26)
        signal_194345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'signal', False)
        # Obtaining the member 'signal' of a type (line 26)
        signal_194346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), signal_194345, 'signal')
        # Calling signal(args, kwargs) (line 26)
        signal_call_result_194352 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), signal_194346, *[SIGINT_194348, _default_handler_194350], **kwargs_194351)
        
        
        # Assigning a Call to a Attribute (line 27):
        
        # Call to WeakKeyDictionary(...): (line 27)
        # Processing the call keyword arguments (line 27)
        kwargs_194355 = {}
        # Getting the type of 'weakref' (line 27)
        weakref_194353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 36), 'weakref', False)
        # Obtaining the member 'WeakKeyDictionary' of a type (line 27)
        WeakKeyDictionary_194354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 36), weakref_194353, 'WeakKeyDictionary')
        # Calling WeakKeyDictionary(args, kwargs) (line 27)
        WeakKeyDictionary_call_result_194356 = invoke(stypy.reporting.localization.Localization(__file__, 27, 36), WeakKeyDictionary_194354, *[], **kwargs_194355)
        
        # Getting the type of 'unittest' (line 27)
        unittest_194357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'unittest')
        # Obtaining the member 'signals' of a type (line 27)
        signals_194358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), unittest_194357, 'signals')
        # Setting the type of the member '_results' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), signals_194358, '_results', WeakKeyDictionary_call_result_194356)
        
        # Assigning a Name to a Attribute (line 28):
        # Getting the type of 'None' (line 28)
        None_194359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'None')
        # Getting the type of 'unittest' (line 28)
        unittest_194360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'unittest')
        # Obtaining the member 'signals' of a type (line 28)
        signals_194361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), unittest_194360, 'signals')
        # Setting the type of the member '_interrupt_handler' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), signals_194361, '_interrupt_handler', None_194359)
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_194362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194362)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_194362


    @norecursion
    def testInstallHandler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testInstallHandler'
        module_type_store = module_type_store.open_function_context('testInstallHandler', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_function_name', 'TestBreak.testInstallHandler')
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testInstallHandler.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testInstallHandler', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testInstallHandler', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testInstallHandler(...)' code ##################

        
        # Assigning a Call to a Name (line 32):
        
        # Call to getsignal(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'signal' (line 32)
        signal_194365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 43), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 32)
        SIGINT_194366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 43), signal_194365, 'SIGINT')
        # Processing the call keyword arguments (line 32)
        kwargs_194367 = {}
        # Getting the type of 'signal' (line 32)
        signal_194363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 32)
        getsignal_194364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 26), signal_194363, 'getsignal')
        # Calling getsignal(args, kwargs) (line 32)
        getsignal_call_result_194368 = invoke(stypy.reporting.localization.Localization(__file__, 32, 26), getsignal_194364, *[SIGINT_194366], **kwargs_194367)
        
        # Assigning a type to the variable 'default_handler' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'default_handler', getsignal_call_result_194368)
        
        # Call to installHandler(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_194371 = {}
        # Getting the type of 'unittest' (line 33)
        unittest_194369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'unittest', False)
        # Obtaining the member 'installHandler' of a type (line 33)
        installHandler_194370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), unittest_194369, 'installHandler')
        # Calling installHandler(args, kwargs) (line 33)
        installHandler_call_result_194372 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), installHandler_194370, *[], **kwargs_194371)
        
        
        # Call to assertNotEqual(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Call to getsignal(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'signal' (line 34)
        signal_194377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 45), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 34)
        SIGINT_194378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 45), signal_194377, 'SIGINT')
        # Processing the call keyword arguments (line 34)
        kwargs_194379 = {}
        # Getting the type of 'signal' (line 34)
        signal_194375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 34)
        getsignal_194376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 28), signal_194375, 'getsignal')
        # Calling getsignal(args, kwargs) (line 34)
        getsignal_call_result_194380 = invoke(stypy.reporting.localization.Localization(__file__, 34, 28), getsignal_194376, *[SIGINT_194378], **kwargs_194379)
        
        # Getting the type of 'default_handler' (line 34)
        default_handler_194381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 61), 'default_handler', False)
        # Processing the call keyword arguments (line 34)
        kwargs_194382 = {}
        # Getting the type of 'self' (line 34)
        self_194373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'assertNotEqual' of a type (line 34)
        assertNotEqual_194374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_194373, 'assertNotEqual')
        # Calling assertNotEqual(args, kwargs) (line 34)
        assertNotEqual_call_result_194383 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assertNotEqual_194374, *[getsignal_call_result_194380, default_handler_194381], **kwargs_194382)
        
        
        
        # SSA begins for try-except statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 37):
        
        # Call to getpid(...): (line 37)
        # Processing the call keyword arguments (line 37)
        kwargs_194386 = {}
        # Getting the type of 'os' (line 37)
        os_194384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'os', False)
        # Obtaining the member 'getpid' of a type (line 37)
        getpid_194385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 18), os_194384, 'getpid')
        # Calling getpid(args, kwargs) (line 37)
        getpid_call_result_194387 = invoke(stypy.reporting.localization.Localization(__file__, 37, 18), getpid_194385, *[], **kwargs_194386)
        
        # Assigning a type to the variable 'pid' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'pid', getpid_call_result_194387)
        
        # Call to kill(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'pid' (line 38)
        pid_194390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'pid', False)
        # Getting the type of 'signal' (line 38)
        signal_194391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 38)
        SIGINT_194392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 25), signal_194391, 'SIGINT')
        # Processing the call keyword arguments (line 38)
        kwargs_194393 = {}
        # Getting the type of 'os' (line 38)
        os_194388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'os', False)
        # Obtaining the member 'kill' of a type (line 38)
        kill_194389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), os_194388, 'kill')
        # Calling kill(args, kwargs) (line 38)
        kill_call_result_194394 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), kill_194389, *[pid_194390, SIGINT_194392], **kwargs_194393)
        
        # SSA branch for the except part of a try statement (line 36)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 36)
        module_type_store.open_ssa_branch('except')
        
        # Call to fail(...): (line 40)
        # Processing the call arguments (line 40)
        str_194397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 22), 'str', 'KeyboardInterrupt not handled')
        # Processing the call keyword arguments (line 40)
        kwargs_194398 = {}
        # Getting the type of 'self' (line 40)
        self_194395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 40)
        fail_194396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 12), self_194395, 'fail')
        # Calling fail(args, kwargs) (line 40)
        fail_call_result_194399 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), fail_194396, *[str_194397], **kwargs_194398)
        
        # SSA join for try-except statement (line 36)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertTrue(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'unittest' (line 42)
        unittest_194402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'unittest', False)
        # Obtaining the member 'signals' of a type (line 42)
        signals_194403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), unittest_194402, 'signals')
        # Obtaining the member '_interrupt_handler' of a type (line 42)
        _interrupt_handler_194404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), signals_194403, '_interrupt_handler')
        # Obtaining the member 'called' of a type (line 42)
        called_194405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), _interrupt_handler_194404, 'called')
        # Processing the call keyword arguments (line 42)
        kwargs_194406 = {}
        # Getting the type of 'self' (line 42)
        self_194400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 42)
        assertTrue_194401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_194400, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 42)
        assertTrue_call_result_194407 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assertTrue_194401, *[called_194405], **kwargs_194406)
        
        
        # ################# End of 'testInstallHandler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testInstallHandler' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_194408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194408)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testInstallHandler'
        return stypy_return_type_194408


    @norecursion
    def testRegisterResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testRegisterResult'
        module_type_store = module_type_store.open_function_context('testRegisterResult', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_function_name', 'TestBreak.testRegisterResult')
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testRegisterResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testRegisterResult', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testRegisterResult', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testRegisterResult(...)' code ##################

        
        # Assigning a Call to a Name (line 45):
        
        # Call to TestResult(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_194411 = {}
        # Getting the type of 'unittest' (line 45)
        unittest_194409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 45)
        TestResult_194410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), unittest_194409, 'TestResult')
        # Calling TestResult(args, kwargs) (line 45)
        TestResult_call_result_194412 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), TestResult_194410, *[], **kwargs_194411)
        
        # Assigning a type to the variable 'result' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'result', TestResult_call_result_194412)
        
        # Call to registerResult(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'result' (line 46)
        result_194415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'result', False)
        # Processing the call keyword arguments (line 46)
        kwargs_194416 = {}
        # Getting the type of 'unittest' (line 46)
        unittest_194413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'unittest', False)
        # Obtaining the member 'registerResult' of a type (line 46)
        registerResult_194414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), unittest_194413, 'registerResult')
        # Calling registerResult(args, kwargs) (line 46)
        registerResult_call_result_194417 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), registerResult_194414, *[result_194415], **kwargs_194416)
        
        
        # Getting the type of 'unittest' (line 48)
        unittest_194418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'unittest')
        # Obtaining the member 'signals' of a type (line 48)
        signals_194419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), unittest_194418, 'signals')
        # Obtaining the member '_results' of a type (line 48)
        _results_194420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), signals_194419, '_results')
        # Testing the type of a for loop iterable (line 48)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 8), _results_194420)
        # Getting the type of the for loop variable (line 48)
        for_loop_var_194421 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 8), _results_194420)
        # Assigning a type to the variable 'ref' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'ref', for_loop_var_194421)
        # SSA begins for a for statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'ref' (line 49)
        ref_194422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'ref')
        # Getting the type of 'result' (line 49)
        result_194423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'result')
        # Applying the binary operator 'is' (line 49)
        result_is__194424 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 15), 'is', ref_194422, result_194423)
        
        # Testing the type of an if condition (line 49)
        if_condition_194425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 12), result_is__194424)
        # Assigning a type to the variable 'if_condition_194425' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'if_condition_194425', if_condition_194425)
        # SSA begins for if statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA branch for the else part of an if statement (line 49)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ref' (line 51)
        ref_194426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'ref')
        # Getting the type of 'result' (line 51)
        result_194427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'result')
        # Applying the binary operator 'isnot' (line 51)
        result_is_not_194428 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 17), 'isnot', ref_194426, result_194427)
        
        # Testing the type of an if condition (line 51)
        if_condition_194429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 17), result_is_not_194428)
        # Assigning a type to the variable 'if_condition_194429' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'if_condition_194429', if_condition_194429)
        # SSA begins for if statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to fail(...): (line 52)
        # Processing the call arguments (line 52)
        str_194432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 26), 'str', 'odd object in result set')
        # Processing the call keyword arguments (line 52)
        kwargs_194433 = {}
        # Getting the type of 'self' (line 52)
        self_194430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'self', False)
        # Obtaining the member 'fail' of a type (line 52)
        fail_194431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), self_194430, 'fail')
        # Calling fail(args, kwargs) (line 52)
        fail_call_result_194434 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), fail_194431, *[str_194432], **kwargs_194433)
        
        # SSA join for if statement (line 51)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 49)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 48)
        module_type_store.open_ssa_branch('for loop else')
        
        # Call to fail(...): (line 54)
        # Processing the call arguments (line 54)
        str_194437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 22), 'str', 'result not found')
        # Processing the call keyword arguments (line 54)
        kwargs_194438 = {}
        # Getting the type of 'self' (line 54)
        self_194435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 54)
        fail_194436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_194435, 'fail')
        # Calling fail(args, kwargs) (line 54)
        fail_call_result_194439 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), fail_194436, *[str_194437], **kwargs_194438)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'testRegisterResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testRegisterResult' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_194440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testRegisterResult'
        return stypy_return_type_194440


    @norecursion
    def testInterruptCaught(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testInterruptCaught'
        module_type_store = module_type_store.open_function_context('testInterruptCaught', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_function_name', 'TestBreak.testInterruptCaught')
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testInterruptCaught.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testInterruptCaught', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testInterruptCaught', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testInterruptCaught(...)' code ##################

        
        # Assigning a Call to a Name (line 58):
        
        # Call to getsignal(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'signal' (line 58)
        signal_194443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 58)
        SIGINT_194444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 43), signal_194443, 'SIGINT')
        # Processing the call keyword arguments (line 58)
        kwargs_194445 = {}
        # Getting the type of 'signal' (line 58)
        signal_194441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 58)
        getsignal_194442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 26), signal_194441, 'getsignal')
        # Calling getsignal(args, kwargs) (line 58)
        getsignal_call_result_194446 = invoke(stypy.reporting.localization.Localization(__file__, 58, 26), getsignal_194442, *[SIGINT_194444], **kwargs_194445)
        
        # Assigning a type to the variable 'default_handler' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'default_handler', getsignal_call_result_194446)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to TestResult(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_194449 = {}
        # Getting the type of 'unittest' (line 60)
        unittest_194447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 60)
        TestResult_194448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 17), unittest_194447, 'TestResult')
        # Calling TestResult(args, kwargs) (line 60)
        TestResult_call_result_194450 = invoke(stypy.reporting.localization.Localization(__file__, 60, 17), TestResult_194448, *[], **kwargs_194449)
        
        # Assigning a type to the variable 'result' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'result', TestResult_call_result_194450)
        
        # Call to installHandler(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_194453 = {}
        # Getting the type of 'unittest' (line 61)
        unittest_194451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'unittest', False)
        # Obtaining the member 'installHandler' of a type (line 61)
        installHandler_194452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), unittest_194451, 'installHandler')
        # Calling installHandler(args, kwargs) (line 61)
        installHandler_call_result_194454 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), installHandler_194452, *[], **kwargs_194453)
        
        
        # Call to registerResult(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'result' (line 62)
        result_194457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 32), 'result', False)
        # Processing the call keyword arguments (line 62)
        kwargs_194458 = {}
        # Getting the type of 'unittest' (line 62)
        unittest_194455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'unittest', False)
        # Obtaining the member 'registerResult' of a type (line 62)
        registerResult_194456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), unittest_194455, 'registerResult')
        # Calling registerResult(args, kwargs) (line 62)
        registerResult_call_result_194459 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), registerResult_194456, *[result_194457], **kwargs_194458)
        
        
        # Call to assertNotEqual(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to getsignal(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'signal' (line 64)
        signal_194464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 45), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 64)
        SIGINT_194465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 45), signal_194464, 'SIGINT')
        # Processing the call keyword arguments (line 64)
        kwargs_194466 = {}
        # Getting the type of 'signal' (line 64)
        signal_194462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 64)
        getsignal_194463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 28), signal_194462, 'getsignal')
        # Calling getsignal(args, kwargs) (line 64)
        getsignal_call_result_194467 = invoke(stypy.reporting.localization.Localization(__file__, 64, 28), getsignal_194463, *[SIGINT_194465], **kwargs_194466)
        
        # Getting the type of 'default_handler' (line 64)
        default_handler_194468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 61), 'default_handler', False)
        # Processing the call keyword arguments (line 64)
        kwargs_194469 = {}
        # Getting the type of 'self' (line 64)
        self_194460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'assertNotEqual' of a type (line 64)
        assertNotEqual_194461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_194460, 'assertNotEqual')
        # Calling assertNotEqual(args, kwargs) (line 64)
        assertNotEqual_call_result_194470 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assertNotEqual_194461, *[getsignal_call_result_194467, default_handler_194468], **kwargs_194469)
        

        @norecursion
        def test(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test'
            module_type_store = module_type_store.open_function_context('test', 66, 8, False)
            
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

            
            # Assigning a Call to a Name (line 67):
            
            # Call to getpid(...): (line 67)
            # Processing the call keyword arguments (line 67)
            kwargs_194473 = {}
            # Getting the type of 'os' (line 67)
            os_194471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'os', False)
            # Obtaining the member 'getpid' of a type (line 67)
            getpid_194472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), os_194471, 'getpid')
            # Calling getpid(args, kwargs) (line 67)
            getpid_call_result_194474 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), getpid_194472, *[], **kwargs_194473)
            
            # Assigning a type to the variable 'pid' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'pid', getpid_call_result_194474)
            
            # Call to kill(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'pid' (line 68)
            pid_194477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'pid', False)
            # Getting the type of 'signal' (line 68)
            signal_194478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'signal', False)
            # Obtaining the member 'SIGINT' of a type (line 68)
            SIGINT_194479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 25), signal_194478, 'SIGINT')
            # Processing the call keyword arguments (line 68)
            kwargs_194480 = {}
            # Getting the type of 'os' (line 68)
            os_194475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'os', False)
            # Obtaining the member 'kill' of a type (line 68)
            kill_194476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), os_194475, 'kill')
            # Calling kill(args, kwargs) (line 68)
            kill_call_result_194481 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), kill_194476, *[pid_194477, SIGINT_194479], **kwargs_194480)
            
            
            # Assigning a Name to a Attribute (line 69):
            # Getting the type of 'True' (line 69)
            True_194482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 33), 'True')
            # Getting the type of 'result' (line 69)
            result_194483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'result')
            # Setting the type of the member 'breakCaught' of a type (line 69)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), result_194483, 'breakCaught', True_194482)
            
            # Call to assertTrue(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 'result' (line 70)
            result_194486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'result', False)
            # Obtaining the member 'shouldStop' of a type (line 70)
            shouldStop_194487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 28), result_194486, 'shouldStop')
            # Processing the call keyword arguments (line 70)
            kwargs_194488 = {}
            # Getting the type of 'self' (line 70)
            self_194484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'self', False)
            # Obtaining the member 'assertTrue' of a type (line 70)
            assertTrue_194485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), self_194484, 'assertTrue')
            # Calling assertTrue(args, kwargs) (line 70)
            assertTrue_call_result_194489 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), assertTrue_194485, *[shouldStop_194487], **kwargs_194488)
            
            
            # ################# End of 'test(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test' in the type store
            # Getting the type of 'stypy_return_type' (line 66)
            stypy_return_type_194490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_194490)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test'
            return stypy_return_type_194490

        # Assigning a type to the variable 'test' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'test', test)
        
        
        # SSA begins for try-except statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to test(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'result' (line 73)
        result_194492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'result', False)
        # Processing the call keyword arguments (line 73)
        kwargs_194493 = {}
        # Getting the type of 'test' (line 73)
        test_194491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'test', False)
        # Calling test(args, kwargs) (line 73)
        test_call_result_194494 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), test_194491, *[result_194492], **kwargs_194493)
        
        # SSA branch for the except part of a try statement (line 72)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 72)
        module_type_store.open_ssa_branch('except')
        
        # Call to fail(...): (line 75)
        # Processing the call arguments (line 75)
        str_194497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'str', 'KeyboardInterrupt not handled')
        # Processing the call keyword arguments (line 75)
        kwargs_194498 = {}
        # Getting the type of 'self' (line 75)
        self_194495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 75)
        fail_194496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), self_194495, 'fail')
        # Calling fail(args, kwargs) (line 75)
        fail_call_result_194499 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), fail_194496, *[str_194497], **kwargs_194498)
        
        # SSA join for try-except statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertTrue(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'result' (line 76)
        result_194502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'result', False)
        # Obtaining the member 'breakCaught' of a type (line 76)
        breakCaught_194503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 24), result_194502, 'breakCaught')
        # Processing the call keyword arguments (line 76)
        kwargs_194504 = {}
        # Getting the type of 'self' (line 76)
        self_194500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 76)
        assertTrue_194501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_194500, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 76)
        assertTrue_call_result_194505 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assertTrue_194501, *[breakCaught_194503], **kwargs_194504)
        
        
        # ################# End of 'testInterruptCaught(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testInterruptCaught' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_194506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194506)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testInterruptCaught'
        return stypy_return_type_194506


    @norecursion
    def testSecondInterrupt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testSecondInterrupt'
        module_type_store = module_type_store.open_function_context('testSecondInterrupt', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_function_name', 'TestBreak.testSecondInterrupt')
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testSecondInterrupt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testSecondInterrupt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testSecondInterrupt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testSecondInterrupt(...)' code ##################

        
        
        
        # Call to getsignal(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'signal' (line 82)
        signal_194509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 82)
        SIGINT_194510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 28), signal_194509, 'SIGINT')
        # Processing the call keyword arguments (line 82)
        kwargs_194511 = {}
        # Getting the type of 'signal' (line 82)
        signal_194507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 82)
        getsignal_194508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 11), signal_194507, 'getsignal')
        # Calling getsignal(args, kwargs) (line 82)
        getsignal_call_result_194512 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), getsignal_194508, *[SIGINT_194510], **kwargs_194511)
        
        # Getting the type of 'signal' (line 82)
        signal_194513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 46), 'signal')
        # Obtaining the member 'SIG_IGN' of a type (line 82)
        SIG_IGN_194514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 46), signal_194513, 'SIG_IGN')
        # Applying the binary operator '==' (line 82)
        result_eq_194515 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 11), '==', getsignal_call_result_194512, SIG_IGN_194514)
        
        # Testing the type of an if condition (line 82)
        if_condition_194516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_eq_194515)
        # Assigning a type to the variable 'if_condition_194516' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'if_condition_194516', if_condition_194516)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to skipTest(...): (line 83)
        # Processing the call arguments (line 83)
        str_194519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'str', 'test requires SIGINT to not be ignored')
        # Processing the call keyword arguments (line 83)
        kwargs_194520 = {}
        # Getting the type of 'self' (line 83)
        self_194517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'self', False)
        # Obtaining the member 'skipTest' of a type (line 83)
        skipTest_194518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), self_194517, 'skipTest')
        # Calling skipTest(args, kwargs) (line 83)
        skipTest_call_result_194521 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), skipTest_194518, *[str_194519], **kwargs_194520)
        
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 84):
        
        # Call to TestResult(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_194524 = {}
        # Getting the type of 'unittest' (line 84)
        unittest_194522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 84)
        TestResult_194523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 17), unittest_194522, 'TestResult')
        # Calling TestResult(args, kwargs) (line 84)
        TestResult_call_result_194525 = invoke(stypy.reporting.localization.Localization(__file__, 84, 17), TestResult_194523, *[], **kwargs_194524)
        
        # Assigning a type to the variable 'result' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'result', TestResult_call_result_194525)
        
        # Call to installHandler(...): (line 85)
        # Processing the call keyword arguments (line 85)
        kwargs_194528 = {}
        # Getting the type of 'unittest' (line 85)
        unittest_194526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'unittest', False)
        # Obtaining the member 'installHandler' of a type (line 85)
        installHandler_194527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), unittest_194526, 'installHandler')
        # Calling installHandler(args, kwargs) (line 85)
        installHandler_call_result_194529 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), installHandler_194527, *[], **kwargs_194528)
        
        
        # Call to registerResult(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'result' (line 86)
        result_194532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'result', False)
        # Processing the call keyword arguments (line 86)
        kwargs_194533 = {}
        # Getting the type of 'unittest' (line 86)
        unittest_194530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'unittest', False)
        # Obtaining the member 'registerResult' of a type (line 86)
        registerResult_194531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), unittest_194530, 'registerResult')
        # Calling registerResult(args, kwargs) (line 86)
        registerResult_call_result_194534 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), registerResult_194531, *[result_194532], **kwargs_194533)
        

        @norecursion
        def test(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test'
            module_type_store = module_type_store.open_function_context('test', 88, 8, False)
            
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

            
            # Assigning a Call to a Name (line 89):
            
            # Call to getpid(...): (line 89)
            # Processing the call keyword arguments (line 89)
            kwargs_194537 = {}
            # Getting the type of 'os' (line 89)
            os_194535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'os', False)
            # Obtaining the member 'getpid' of a type (line 89)
            getpid_194536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 18), os_194535, 'getpid')
            # Calling getpid(args, kwargs) (line 89)
            getpid_call_result_194538 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), getpid_194536, *[], **kwargs_194537)
            
            # Assigning a type to the variable 'pid' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'pid', getpid_call_result_194538)
            
            # Call to kill(...): (line 90)
            # Processing the call arguments (line 90)
            # Getting the type of 'pid' (line 90)
            pid_194541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'pid', False)
            # Getting the type of 'signal' (line 90)
            signal_194542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'signal', False)
            # Obtaining the member 'SIGINT' of a type (line 90)
            SIGINT_194543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 25), signal_194542, 'SIGINT')
            # Processing the call keyword arguments (line 90)
            kwargs_194544 = {}
            # Getting the type of 'os' (line 90)
            os_194539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'os', False)
            # Obtaining the member 'kill' of a type (line 90)
            kill_194540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), os_194539, 'kill')
            # Calling kill(args, kwargs) (line 90)
            kill_call_result_194545 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), kill_194540, *[pid_194541, SIGINT_194543], **kwargs_194544)
            
            
            # Assigning a Name to a Attribute (line 91):
            # Getting the type of 'True' (line 91)
            True_194546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'True')
            # Getting the type of 'result' (line 91)
            result_194547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'result')
            # Setting the type of the member 'breakCaught' of a type (line 91)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), result_194547, 'breakCaught', True_194546)
            
            # Call to assertTrue(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'result' (line 92)
            result_194550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'result', False)
            # Obtaining the member 'shouldStop' of a type (line 92)
            shouldStop_194551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 28), result_194550, 'shouldStop')
            # Processing the call keyword arguments (line 92)
            kwargs_194552 = {}
            # Getting the type of 'self' (line 92)
            self_194548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self', False)
            # Obtaining the member 'assertTrue' of a type (line 92)
            assertTrue_194549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_194548, 'assertTrue')
            # Calling assertTrue(args, kwargs) (line 92)
            assertTrue_call_result_194553 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), assertTrue_194549, *[shouldStop_194551], **kwargs_194552)
            
            
            # Call to kill(...): (line 93)
            # Processing the call arguments (line 93)
            # Getting the type of 'pid' (line 93)
            pid_194556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'pid', False)
            # Getting the type of 'signal' (line 93)
            signal_194557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'signal', False)
            # Obtaining the member 'SIGINT' of a type (line 93)
            SIGINT_194558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 25), signal_194557, 'SIGINT')
            # Processing the call keyword arguments (line 93)
            kwargs_194559 = {}
            # Getting the type of 'os' (line 93)
            os_194554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'os', False)
            # Obtaining the member 'kill' of a type (line 93)
            kill_194555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), os_194554, 'kill')
            # Calling kill(args, kwargs) (line 93)
            kill_call_result_194560 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), kill_194555, *[pid_194556, SIGINT_194558], **kwargs_194559)
            
            
            # Call to fail(...): (line 94)
            # Processing the call arguments (line 94)
            str_194563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'str', 'Second KeyboardInterrupt not raised')
            # Processing the call keyword arguments (line 94)
            kwargs_194564 = {}
            # Getting the type of 'self' (line 94)
            self_194561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'self', False)
            # Obtaining the member 'fail' of a type (line 94)
            fail_194562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), self_194561, 'fail')
            # Calling fail(args, kwargs) (line 94)
            fail_call_result_194565 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), fail_194562, *[str_194563], **kwargs_194564)
            
            
            # ################# End of 'test(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test' in the type store
            # Getting the type of 'stypy_return_type' (line 88)
            stypy_return_type_194566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_194566)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test'
            return stypy_return_type_194566

        # Assigning a type to the variable 'test' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'test', test)
        
        
        # SSA begins for try-except statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to test(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'result' (line 97)
        result_194568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'result', False)
        # Processing the call keyword arguments (line 97)
        kwargs_194569 = {}
        # Getting the type of 'test' (line 97)
        test_194567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'test', False)
        # Calling test(args, kwargs) (line 97)
        test_call_result_194570 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), test_194567, *[result_194568], **kwargs_194569)
        
        # SSA branch for the except part of a try statement (line 96)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 96)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 96)
        module_type_store.open_ssa_branch('except else')
        
        # Call to fail(...): (line 101)
        # Processing the call arguments (line 101)
        str_194573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'str', 'Second KeyboardInterrupt not raised')
        # Processing the call keyword arguments (line 101)
        kwargs_194574 = {}
        # Getting the type of 'self' (line 101)
        self_194571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 101)
        fail_194572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_194571, 'fail')
        # Calling fail(args, kwargs) (line 101)
        fail_call_result_194575 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), fail_194572, *[str_194573], **kwargs_194574)
        
        # SSA join for try-except statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertTrue(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'result' (line 102)
        result_194578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'result', False)
        # Obtaining the member 'breakCaught' of a type (line 102)
        breakCaught_194579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), result_194578, 'breakCaught')
        # Processing the call keyword arguments (line 102)
        kwargs_194580 = {}
        # Getting the type of 'self' (line 102)
        self_194576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 102)
        assertTrue_194577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_194576, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 102)
        assertTrue_call_result_194581 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assertTrue_194577, *[breakCaught_194579], **kwargs_194580)
        
        
        # ################# End of 'testSecondInterrupt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testSecondInterrupt' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_194582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194582)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testSecondInterrupt'
        return stypy_return_type_194582


    @norecursion
    def testTwoResults(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testTwoResults'
        module_type_store = module_type_store.open_function_context('testTwoResults', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_function_name', 'TestBreak.testTwoResults')
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testTwoResults.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testTwoResults', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testTwoResults', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testTwoResults(...)' code ##################

        
        # Call to installHandler(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_194585 = {}
        # Getting the type of 'unittest' (line 106)
        unittest_194583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'unittest', False)
        # Obtaining the member 'installHandler' of a type (line 106)
        installHandler_194584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), unittest_194583, 'installHandler')
        # Calling installHandler(args, kwargs) (line 106)
        installHandler_call_result_194586 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), installHandler_194584, *[], **kwargs_194585)
        
        
        # Assigning a Call to a Name (line 108):
        
        # Call to TestResult(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_194589 = {}
        # Getting the type of 'unittest' (line 108)
        unittest_194587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 108)
        TestResult_194588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 17), unittest_194587, 'TestResult')
        # Calling TestResult(args, kwargs) (line 108)
        TestResult_call_result_194590 = invoke(stypy.reporting.localization.Localization(__file__, 108, 17), TestResult_194588, *[], **kwargs_194589)
        
        # Assigning a type to the variable 'result' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'result', TestResult_call_result_194590)
        
        # Call to registerResult(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'result' (line 109)
        result_194593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 32), 'result', False)
        # Processing the call keyword arguments (line 109)
        kwargs_194594 = {}
        # Getting the type of 'unittest' (line 109)
        unittest_194591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'unittest', False)
        # Obtaining the member 'registerResult' of a type (line 109)
        registerResult_194592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), unittest_194591, 'registerResult')
        # Calling registerResult(args, kwargs) (line 109)
        registerResult_call_result_194595 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), registerResult_194592, *[result_194593], **kwargs_194594)
        
        
        # Assigning a Call to a Name (line 110):
        
        # Call to getsignal(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'signal' (line 110)
        signal_194598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 39), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 110)
        SIGINT_194599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 39), signal_194598, 'SIGINT')
        # Processing the call keyword arguments (line 110)
        kwargs_194600 = {}
        # Getting the type of 'signal' (line 110)
        signal_194596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 110)
        getsignal_194597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 22), signal_194596, 'getsignal')
        # Calling getsignal(args, kwargs) (line 110)
        getsignal_call_result_194601 = invoke(stypy.reporting.localization.Localization(__file__, 110, 22), getsignal_194597, *[SIGINT_194599], **kwargs_194600)
        
        # Assigning a type to the variable 'new_handler' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'new_handler', getsignal_call_result_194601)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to TestResult(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_194604 = {}
        # Getting the type of 'unittest' (line 112)
        unittest_194602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 112)
        TestResult_194603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 18), unittest_194602, 'TestResult')
        # Calling TestResult(args, kwargs) (line 112)
        TestResult_call_result_194605 = invoke(stypy.reporting.localization.Localization(__file__, 112, 18), TestResult_194603, *[], **kwargs_194604)
        
        # Assigning a type to the variable 'result2' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'result2', TestResult_call_result_194605)
        
        # Call to registerResult(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'result2' (line 113)
        result2_194608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 32), 'result2', False)
        # Processing the call keyword arguments (line 113)
        kwargs_194609 = {}
        # Getting the type of 'unittest' (line 113)
        unittest_194606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'unittest', False)
        # Obtaining the member 'registerResult' of a type (line 113)
        registerResult_194607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), unittest_194606, 'registerResult')
        # Calling registerResult(args, kwargs) (line 113)
        registerResult_call_result_194610 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), registerResult_194607, *[result2_194608], **kwargs_194609)
        
        
        # Call to assertEqual(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to getsignal(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'signal' (line 114)
        signal_194615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 42), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 114)
        SIGINT_194616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 42), signal_194615, 'SIGINT')
        # Processing the call keyword arguments (line 114)
        kwargs_194617 = {}
        # Getting the type of 'signal' (line 114)
        signal_194613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 25), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 114)
        getsignal_194614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 25), signal_194613, 'getsignal')
        # Calling getsignal(args, kwargs) (line 114)
        getsignal_call_result_194618 = invoke(stypy.reporting.localization.Localization(__file__, 114, 25), getsignal_194614, *[SIGINT_194616], **kwargs_194617)
        
        # Getting the type of 'new_handler' (line 114)
        new_handler_194619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 58), 'new_handler', False)
        # Processing the call keyword arguments (line 114)
        kwargs_194620 = {}
        # Getting the type of 'self' (line 114)
        self_194611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 114)
        assertEqual_194612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_194611, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 114)
        assertEqual_call_result_194621 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assertEqual_194612, *[getsignal_call_result_194618, new_handler_194619], **kwargs_194620)
        
        
        # Assigning a Call to a Name (line 116):
        
        # Call to TestResult(...): (line 116)
        # Processing the call keyword arguments (line 116)
        kwargs_194624 = {}
        # Getting the type of 'unittest' (line 116)
        unittest_194622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 116)
        TestResult_194623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 18), unittest_194622, 'TestResult')
        # Calling TestResult(args, kwargs) (line 116)
        TestResult_call_result_194625 = invoke(stypy.reporting.localization.Localization(__file__, 116, 18), TestResult_194623, *[], **kwargs_194624)
        
        # Assigning a type to the variable 'result3' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'result3', TestResult_call_result_194625)

        @norecursion
        def test(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test'
            module_type_store = module_type_store.open_function_context('test', 118, 8, False)
            
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

            
            # Assigning a Call to a Name (line 119):
            
            # Call to getpid(...): (line 119)
            # Processing the call keyword arguments (line 119)
            kwargs_194628 = {}
            # Getting the type of 'os' (line 119)
            os_194626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'os', False)
            # Obtaining the member 'getpid' of a type (line 119)
            getpid_194627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 18), os_194626, 'getpid')
            # Calling getpid(args, kwargs) (line 119)
            getpid_call_result_194629 = invoke(stypy.reporting.localization.Localization(__file__, 119, 18), getpid_194627, *[], **kwargs_194628)
            
            # Assigning a type to the variable 'pid' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'pid', getpid_call_result_194629)
            
            # Call to kill(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'pid' (line 120)
            pid_194632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'pid', False)
            # Getting the type of 'signal' (line 120)
            signal_194633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 25), 'signal', False)
            # Obtaining the member 'SIGINT' of a type (line 120)
            SIGINT_194634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 25), signal_194633, 'SIGINT')
            # Processing the call keyword arguments (line 120)
            kwargs_194635 = {}
            # Getting the type of 'os' (line 120)
            os_194630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'os', False)
            # Obtaining the member 'kill' of a type (line 120)
            kill_194631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), os_194630, 'kill')
            # Calling kill(args, kwargs) (line 120)
            kill_call_result_194636 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), kill_194631, *[pid_194632, SIGINT_194634], **kwargs_194635)
            
            
            # ################# End of 'test(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test' in the type store
            # Getting the type of 'stypy_return_type' (line 118)
            stypy_return_type_194637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_194637)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test'
            return stypy_return_type_194637

        # Assigning a type to the variable 'test' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'test', test)
        
        
        # SSA begins for try-except statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to test(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'result' (line 123)
        result_194639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'result', False)
        # Processing the call keyword arguments (line 123)
        kwargs_194640 = {}
        # Getting the type of 'test' (line 123)
        test_194638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'test', False)
        # Calling test(args, kwargs) (line 123)
        test_call_result_194641 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), test_194638, *[result_194639], **kwargs_194640)
        
        # SSA branch for the except part of a try statement (line 122)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 122)
        module_type_store.open_ssa_branch('except')
        
        # Call to fail(...): (line 125)
        # Processing the call arguments (line 125)
        str_194644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 22), 'str', 'KeyboardInterrupt not handled')
        # Processing the call keyword arguments (line 125)
        kwargs_194645 = {}
        # Getting the type of 'self' (line 125)
        self_194642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 125)
        fail_194643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), self_194642, 'fail')
        # Calling fail(args, kwargs) (line 125)
        fail_call_result_194646 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), fail_194643, *[str_194644], **kwargs_194645)
        
        # SSA join for try-except statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertTrue(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'result' (line 127)
        result_194649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 127)
        shouldStop_194650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), result_194649, 'shouldStop')
        # Processing the call keyword arguments (line 127)
        kwargs_194651 = {}
        # Getting the type of 'self' (line 127)
        self_194647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 127)
        assertTrue_194648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_194647, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 127)
        assertTrue_call_result_194652 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), assertTrue_194648, *[shouldStop_194650], **kwargs_194651)
        
        
        # Call to assertTrue(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'result2' (line 128)
        result2_194655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'result2', False)
        # Obtaining the member 'shouldStop' of a type (line 128)
        shouldStop_194656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 24), result2_194655, 'shouldStop')
        # Processing the call keyword arguments (line 128)
        kwargs_194657 = {}
        # Getting the type of 'self' (line 128)
        self_194653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 128)
        assertTrue_194654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_194653, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 128)
        assertTrue_call_result_194658 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), assertTrue_194654, *[shouldStop_194656], **kwargs_194657)
        
        
        # Call to assertFalse(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'result3' (line 129)
        result3_194661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'result3', False)
        # Obtaining the member 'shouldStop' of a type (line 129)
        shouldStop_194662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), result3_194661, 'shouldStop')
        # Processing the call keyword arguments (line 129)
        kwargs_194663 = {}
        # Getting the type of 'self' (line 129)
        self_194659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 129)
        assertFalse_194660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_194659, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 129)
        assertFalse_call_result_194664 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assertFalse_194660, *[shouldStop_194662], **kwargs_194663)
        
        
        # ################# End of 'testTwoResults(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testTwoResults' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_194665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194665)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testTwoResults'
        return stypy_return_type_194665


    @norecursion
    def testHandlerReplacedButCalled(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testHandlerReplacedButCalled'
        module_type_store = module_type_store.open_function_context('testHandlerReplacedButCalled', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_function_name', 'TestBreak.testHandlerReplacedButCalled')
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testHandlerReplacedButCalled.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testHandlerReplacedButCalled', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testHandlerReplacedButCalled', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testHandlerReplacedButCalled(...)' code ##################

        
        
        
        # Call to getsignal(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'signal' (line 135)
        signal_194668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 135)
        SIGINT_194669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 28), signal_194668, 'SIGINT')
        # Processing the call keyword arguments (line 135)
        kwargs_194670 = {}
        # Getting the type of 'signal' (line 135)
        signal_194666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 135)
        getsignal_194667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), signal_194666, 'getsignal')
        # Calling getsignal(args, kwargs) (line 135)
        getsignal_call_result_194671 = invoke(stypy.reporting.localization.Localization(__file__, 135, 11), getsignal_194667, *[SIGINT_194669], **kwargs_194670)
        
        # Getting the type of 'signal' (line 135)
        signal_194672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 46), 'signal')
        # Obtaining the member 'SIG_IGN' of a type (line 135)
        SIG_IGN_194673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 46), signal_194672, 'SIG_IGN')
        # Applying the binary operator '==' (line 135)
        result_eq_194674 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 11), '==', getsignal_call_result_194671, SIG_IGN_194673)
        
        # Testing the type of an if condition (line 135)
        if_condition_194675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 8), result_eq_194674)
        # Assigning a type to the variable 'if_condition_194675' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'if_condition_194675', if_condition_194675)
        # SSA begins for if statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to skipTest(...): (line 136)
        # Processing the call arguments (line 136)
        str_194678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 26), 'str', 'test requires SIGINT to not be ignored')
        # Processing the call keyword arguments (line 136)
        kwargs_194679 = {}
        # Getting the type of 'self' (line 136)
        self_194676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'self', False)
        # Obtaining the member 'skipTest' of a type (line 136)
        skipTest_194677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 12), self_194676, 'skipTest')
        # Calling skipTest(args, kwargs) (line 136)
        skipTest_call_result_194680 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), skipTest_194677, *[str_194678], **kwargs_194679)
        
        # SSA join for if statement (line 135)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to installHandler(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_194683 = {}
        # Getting the type of 'unittest' (line 140)
        unittest_194681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'unittest', False)
        # Obtaining the member 'installHandler' of a type (line 140)
        installHandler_194682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), unittest_194681, 'installHandler')
        # Calling installHandler(args, kwargs) (line 140)
        installHandler_call_result_194684 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), installHandler_194682, *[], **kwargs_194683)
        
        
        # Assigning a Call to a Name (line 142):
        
        # Call to getsignal(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'signal' (line 142)
        signal_194687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 35), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 142)
        SIGINT_194688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 35), signal_194687, 'SIGINT')
        # Processing the call keyword arguments (line 142)
        kwargs_194689 = {}
        # Getting the type of 'signal' (line 142)
        signal_194685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 142)
        getsignal_194686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 18), signal_194685, 'getsignal')
        # Calling getsignal(args, kwargs) (line 142)
        getsignal_call_result_194690 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), getsignal_194686, *[SIGINT_194688], **kwargs_194689)
        
        # Assigning a type to the variable 'handler' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'handler', getsignal_call_result_194690)

        @norecursion
        def new_handler(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'new_handler'
            module_type_store = module_type_store.open_function_context('new_handler', 143, 8, False)
            
            # Passed parameters checking function
            new_handler.stypy_localization = localization
            new_handler.stypy_type_of_self = None
            new_handler.stypy_type_store = module_type_store
            new_handler.stypy_function_name = 'new_handler'
            new_handler.stypy_param_names_list = ['frame', 'signum']
            new_handler.stypy_varargs_param_name = None
            new_handler.stypy_kwargs_param_name = None
            new_handler.stypy_call_defaults = defaults
            new_handler.stypy_call_varargs = varargs
            new_handler.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'new_handler', ['frame', 'signum'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'new_handler', localization, ['frame', 'signum'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'new_handler(...)' code ##################

            
            # Call to handler(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'frame' (line 144)
            frame_194692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'frame', False)
            # Getting the type of 'signum' (line 144)
            signum_194693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'signum', False)
            # Processing the call keyword arguments (line 144)
            kwargs_194694 = {}
            # Getting the type of 'handler' (line 144)
            handler_194691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'handler', False)
            # Calling handler(args, kwargs) (line 144)
            handler_call_result_194695 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), handler_194691, *[frame_194692, signum_194693], **kwargs_194694)
            
            
            # ################# End of 'new_handler(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'new_handler' in the type store
            # Getting the type of 'stypy_return_type' (line 143)
            stypy_return_type_194696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_194696)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'new_handler'
            return stypy_return_type_194696

        # Assigning a type to the variable 'new_handler' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'new_handler', new_handler)
        
        # Call to signal(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'signal' (line 145)
        signal_194699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 22), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 145)
        SIGINT_194700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 22), signal_194699, 'SIGINT')
        # Getting the type of 'new_handler' (line 145)
        new_handler_194701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 37), 'new_handler', False)
        # Processing the call keyword arguments (line 145)
        kwargs_194702 = {}
        # Getting the type of 'signal' (line 145)
        signal_194697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'signal', False)
        # Obtaining the member 'signal' of a type (line 145)
        signal_194698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), signal_194697, 'signal')
        # Calling signal(args, kwargs) (line 145)
        signal_call_result_194703 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), signal_194698, *[SIGINT_194700, new_handler_194701], **kwargs_194702)
        
        
        
        # SSA begins for try-except statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 148):
        
        # Call to getpid(...): (line 148)
        # Processing the call keyword arguments (line 148)
        kwargs_194706 = {}
        # Getting the type of 'os' (line 148)
        os_194704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 18), 'os', False)
        # Obtaining the member 'getpid' of a type (line 148)
        getpid_194705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 18), os_194704, 'getpid')
        # Calling getpid(args, kwargs) (line 148)
        getpid_call_result_194707 = invoke(stypy.reporting.localization.Localization(__file__, 148, 18), getpid_194705, *[], **kwargs_194706)
        
        # Assigning a type to the variable 'pid' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'pid', getpid_call_result_194707)
        
        # Call to kill(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'pid' (line 149)
        pid_194710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'pid', False)
        # Getting the type of 'signal' (line 149)
        signal_194711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 149)
        SIGINT_194712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 25), signal_194711, 'SIGINT')
        # Processing the call keyword arguments (line 149)
        kwargs_194713 = {}
        # Getting the type of 'os' (line 149)
        os_194708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'os', False)
        # Obtaining the member 'kill' of a type (line 149)
        kill_194709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), os_194708, 'kill')
        # Calling kill(args, kwargs) (line 149)
        kill_call_result_194714 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), kill_194709, *[pid_194710, SIGINT_194712], **kwargs_194713)
        
        # SSA branch for the except part of a try statement (line 147)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 147)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 147)
        module_type_store.open_ssa_branch('except else')
        
        # Call to fail(...): (line 153)
        # Processing the call arguments (line 153)
        str_194717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 22), 'str', "replaced but delegated handler doesn't raise interrupt")
        # Processing the call keyword arguments (line 153)
        kwargs_194718 = {}
        # Getting the type of 'self' (line 153)
        self_194715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 153)
        fail_194716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), self_194715, 'fail')
        # Calling fail(args, kwargs) (line 153)
        fail_call_result_194719 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), fail_194716, *[str_194717], **kwargs_194718)
        
        # SSA join for try-except statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'testHandlerReplacedButCalled(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testHandlerReplacedButCalled' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_194720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194720)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testHandlerReplacedButCalled'
        return stypy_return_type_194720


    @norecursion
    def testRunner(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testRunner'
        module_type_store = module_type_store.open_function_context('testRunner', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testRunner.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testRunner.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testRunner.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testRunner.__dict__.__setitem__('stypy_function_name', 'TestBreak.testRunner')
        TestBreak.testRunner.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testRunner.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testRunner.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testRunner.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testRunner.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testRunner.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testRunner.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testRunner', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testRunner', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testRunner(...)' code ##################

        
        # Assigning a Call to a Name (line 158):
        
        # Call to TextTestRunner(...): (line 158)
        # Processing the call keyword arguments (line 158)
        
        # Call to StringIO(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_194724 = {}
        # Getting the type of 'StringIO' (line 158)
        StringIO_194723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 48), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 158)
        StringIO_call_result_194725 = invoke(stypy.reporting.localization.Localization(__file__, 158, 48), StringIO_194723, *[], **kwargs_194724)
        
        keyword_194726 = StringIO_call_result_194725
        kwargs_194727 = {'stream': keyword_194726}
        # Getting the type of 'unittest' (line 158)
        unittest_194721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), 'unittest', False)
        # Obtaining the member 'TextTestRunner' of a type (line 158)
        TextTestRunner_194722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 17), unittest_194721, 'TextTestRunner')
        # Calling TextTestRunner(args, kwargs) (line 158)
        TextTestRunner_call_result_194728 = invoke(stypy.reporting.localization.Localization(__file__, 158, 17), TextTestRunner_194722, *[], **kwargs_194727)
        
        # Assigning a type to the variable 'runner' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'runner', TextTestRunner_call_result_194728)
        
        # Assigning a Call to a Name (line 160):
        
        # Call to run(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Call to TestSuite(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_194733 = {}
        # Getting the type of 'unittest' (line 160)
        unittest_194731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'unittest', False)
        # Obtaining the member 'TestSuite' of a type (line 160)
        TestSuite_194732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 28), unittest_194731, 'TestSuite')
        # Calling TestSuite(args, kwargs) (line 160)
        TestSuite_call_result_194734 = invoke(stypy.reporting.localization.Localization(__file__, 160, 28), TestSuite_194732, *[], **kwargs_194733)
        
        # Processing the call keyword arguments (line 160)
        kwargs_194735 = {}
        # Getting the type of 'runner' (line 160)
        runner_194729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'runner', False)
        # Obtaining the member 'run' of a type (line 160)
        run_194730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 17), runner_194729, 'run')
        # Calling run(args, kwargs) (line 160)
        run_call_result_194736 = invoke(stypy.reporting.localization.Localization(__file__, 160, 17), run_194730, *[TestSuite_call_result_194734], **kwargs_194735)
        
        # Assigning a type to the variable 'result' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'result', run_call_result_194736)
        
        # Call to assertIn(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'result' (line 161)
        result_194739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 22), 'result', False)
        # Getting the type of 'unittest' (line 161)
        unittest_194740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 30), 'unittest', False)
        # Obtaining the member 'signals' of a type (line 161)
        signals_194741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 30), unittest_194740, 'signals')
        # Obtaining the member '_results' of a type (line 161)
        _results_194742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 30), signals_194741, '_results')
        # Processing the call keyword arguments (line 161)
        kwargs_194743 = {}
        # Getting the type of 'self' (line 161)
        self_194737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 161)
        assertIn_194738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), self_194737, 'assertIn')
        # Calling assertIn(args, kwargs) (line 161)
        assertIn_call_result_194744 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), assertIn_194738, *[result_194739, _results_194742], **kwargs_194743)
        
        
        # ################# End of 'testRunner(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testRunner' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_194745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testRunner'
        return stypy_return_type_194745


    @norecursion
    def testWeakReferences(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testWeakReferences'
        module_type_store = module_type_store.open_function_context('testWeakReferences', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_function_name', 'TestBreak.testWeakReferences')
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testWeakReferences.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testWeakReferences', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testWeakReferences', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testWeakReferences(...)' code ##################

        
        # Assigning a Call to a Name (line 165):
        
        # Call to TestResult(...): (line 165)
        # Processing the call keyword arguments (line 165)
        kwargs_194748 = {}
        # Getting the type of 'unittest' (line 165)
        unittest_194746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 165)
        TestResult_194747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 17), unittest_194746, 'TestResult')
        # Calling TestResult(args, kwargs) (line 165)
        TestResult_call_result_194749 = invoke(stypy.reporting.localization.Localization(__file__, 165, 17), TestResult_194747, *[], **kwargs_194748)
        
        # Assigning a type to the variable 'result' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'result', TestResult_call_result_194749)
        
        # Call to registerResult(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'result' (line 166)
        result_194752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'result', False)
        # Processing the call keyword arguments (line 166)
        kwargs_194753 = {}
        # Getting the type of 'unittest' (line 166)
        unittest_194750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'unittest', False)
        # Obtaining the member 'registerResult' of a type (line 166)
        registerResult_194751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), unittest_194750, 'registerResult')
        # Calling registerResult(args, kwargs) (line 166)
        registerResult_call_result_194754 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), registerResult_194751, *[result_194752], **kwargs_194753)
        
        
        # Assigning a Call to a Name (line 168):
        
        # Call to ref(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'result' (line 168)
        result_194757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 26), 'result', False)
        # Processing the call keyword arguments (line 168)
        kwargs_194758 = {}
        # Getting the type of 'weakref' (line 168)
        weakref_194755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 14), 'weakref', False)
        # Obtaining the member 'ref' of a type (line 168)
        ref_194756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 14), weakref_194755, 'ref')
        # Calling ref(args, kwargs) (line 168)
        ref_call_result_194759 = invoke(stypy.reporting.localization.Localization(__file__, 168, 14), ref_194756, *[result_194757], **kwargs_194758)
        
        # Assigning a type to the variable 'ref' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'ref', ref_call_result_194759)
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 169, 8), module_type_store, 'result')
        
        # Call to collect(...): (line 172)
        # Processing the call keyword arguments (line 172)
        kwargs_194762 = {}
        # Getting the type of 'gc' (line 172)
        gc_194760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'gc', False)
        # Obtaining the member 'collect' of a type (line 172)
        collect_194761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), gc_194760, 'collect')
        # Calling collect(args, kwargs) (line 172)
        collect_call_result_194763 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), collect_194761, *[], **kwargs_194762)
        
        
        # Call to collect(...): (line 172)
        # Processing the call keyword arguments (line 172)
        kwargs_194766 = {}
        # Getting the type of 'gc' (line 172)
        gc_194764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'gc', False)
        # Obtaining the member 'collect' of a type (line 172)
        collect_194765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 21), gc_194764, 'collect')
        # Calling collect(args, kwargs) (line 172)
        collect_call_result_194767 = invoke(stypy.reporting.localization.Localization(__file__, 172, 21), collect_194765, *[], **kwargs_194766)
        
        
        # Call to assertIsNone(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Call to ref(...): (line 173)
        # Processing the call keyword arguments (line 173)
        kwargs_194771 = {}
        # Getting the type of 'ref' (line 173)
        ref_194770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 26), 'ref', False)
        # Calling ref(args, kwargs) (line 173)
        ref_call_result_194772 = invoke(stypy.reporting.localization.Localization(__file__, 173, 26), ref_194770, *[], **kwargs_194771)
        
        # Processing the call keyword arguments (line 173)
        kwargs_194773 = {}
        # Getting the type of 'self' (line 173)
        self_194768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self', False)
        # Obtaining the member 'assertIsNone' of a type (line 173)
        assertIsNone_194769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_194768, 'assertIsNone')
        # Calling assertIsNone(args, kwargs) (line 173)
        assertIsNone_call_result_194774 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), assertIsNone_194769, *[ref_call_result_194772], **kwargs_194773)
        
        
        # ################# End of 'testWeakReferences(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testWeakReferences' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_194775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194775)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testWeakReferences'
        return stypy_return_type_194775


    @norecursion
    def testRemoveResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testRemoveResult'
        module_type_store = module_type_store.open_function_context('testRemoveResult', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_function_name', 'TestBreak.testRemoveResult')
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testRemoveResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testRemoveResult', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testRemoveResult', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testRemoveResult(...)' code ##################

        
        # Assigning a Call to a Name (line 177):
        
        # Call to TestResult(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_194778 = {}
        # Getting the type of 'unittest' (line 177)
        unittest_194776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 177)
        TestResult_194777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 17), unittest_194776, 'TestResult')
        # Calling TestResult(args, kwargs) (line 177)
        TestResult_call_result_194779 = invoke(stypy.reporting.localization.Localization(__file__, 177, 17), TestResult_194777, *[], **kwargs_194778)
        
        # Assigning a type to the variable 'result' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'result', TestResult_call_result_194779)
        
        # Call to registerResult(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'result' (line 178)
        result_194782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 32), 'result', False)
        # Processing the call keyword arguments (line 178)
        kwargs_194783 = {}
        # Getting the type of 'unittest' (line 178)
        unittest_194780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'unittest', False)
        # Obtaining the member 'registerResult' of a type (line 178)
        registerResult_194781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), unittest_194780, 'registerResult')
        # Calling registerResult(args, kwargs) (line 178)
        registerResult_call_result_194784 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), registerResult_194781, *[result_194782], **kwargs_194783)
        
        
        # Call to installHandler(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_194787 = {}
        # Getting the type of 'unittest' (line 180)
        unittest_194785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'unittest', False)
        # Obtaining the member 'installHandler' of a type (line 180)
        installHandler_194786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), unittest_194785, 'installHandler')
        # Calling installHandler(args, kwargs) (line 180)
        installHandler_call_result_194788 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), installHandler_194786, *[], **kwargs_194787)
        
        
        # Call to assertTrue(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to removeResult(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'result' (line 181)
        result_194793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'result', False)
        # Processing the call keyword arguments (line 181)
        kwargs_194794 = {}
        # Getting the type of 'unittest' (line 181)
        unittest_194791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'unittest', False)
        # Obtaining the member 'removeResult' of a type (line 181)
        removeResult_194792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 24), unittest_194791, 'removeResult')
        # Calling removeResult(args, kwargs) (line 181)
        removeResult_call_result_194795 = invoke(stypy.reporting.localization.Localization(__file__, 181, 24), removeResult_194792, *[result_194793], **kwargs_194794)
        
        # Processing the call keyword arguments (line 181)
        kwargs_194796 = {}
        # Getting the type of 'self' (line 181)
        self_194789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 181)
        assertTrue_194790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_194789, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 181)
        assertTrue_call_result_194797 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assertTrue_194790, *[removeResult_call_result_194795], **kwargs_194796)
        
        
        # Call to assertFalse(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Call to removeResult(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Call to TestResult(...): (line 184)
        # Processing the call keyword arguments (line 184)
        kwargs_194804 = {}
        # Getting the type of 'unittest' (line 184)
        unittest_194802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 47), 'unittest', False)
        # Obtaining the member 'TestResult' of a type (line 184)
        TestResult_194803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 47), unittest_194802, 'TestResult')
        # Calling TestResult(args, kwargs) (line 184)
        TestResult_call_result_194805 = invoke(stypy.reporting.localization.Localization(__file__, 184, 47), TestResult_194803, *[], **kwargs_194804)
        
        # Processing the call keyword arguments (line 184)
        kwargs_194806 = {}
        # Getting the type of 'unittest' (line 184)
        unittest_194800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'unittest', False)
        # Obtaining the member 'removeResult' of a type (line 184)
        removeResult_194801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 25), unittest_194800, 'removeResult')
        # Calling removeResult(args, kwargs) (line 184)
        removeResult_call_result_194807 = invoke(stypy.reporting.localization.Localization(__file__, 184, 25), removeResult_194801, *[TestResult_call_result_194805], **kwargs_194806)
        
        # Processing the call keyword arguments (line 184)
        kwargs_194808 = {}
        # Getting the type of 'self' (line 184)
        self_194798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 184)
        assertFalse_194799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_194798, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 184)
        assertFalse_call_result_194809 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), assertFalse_194799, *[removeResult_call_result_194807], **kwargs_194808)
        
        
        
        # SSA begins for try-except statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 187):
        
        # Call to getpid(...): (line 187)
        # Processing the call keyword arguments (line 187)
        kwargs_194812 = {}
        # Getting the type of 'os' (line 187)
        os_194810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 18), 'os', False)
        # Obtaining the member 'getpid' of a type (line 187)
        getpid_194811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 18), os_194810, 'getpid')
        # Calling getpid(args, kwargs) (line 187)
        getpid_call_result_194813 = invoke(stypy.reporting.localization.Localization(__file__, 187, 18), getpid_194811, *[], **kwargs_194812)
        
        # Assigning a type to the variable 'pid' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'pid', getpid_call_result_194813)
        
        # Call to kill(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'pid' (line 188)
        pid_194816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'pid', False)
        # Getting the type of 'signal' (line 188)
        signal_194817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 25), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 188)
        SIGINT_194818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 25), signal_194817, 'SIGINT')
        # Processing the call keyword arguments (line 188)
        kwargs_194819 = {}
        # Getting the type of 'os' (line 188)
        os_194814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'os', False)
        # Obtaining the member 'kill' of a type (line 188)
        kill_194815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), os_194814, 'kill')
        # Calling kill(args, kwargs) (line 188)
        kill_call_result_194820 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), kill_194815, *[pid_194816, SIGINT_194818], **kwargs_194819)
        
        # SSA branch for the except part of a try statement (line 186)
        # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 186)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertFalse(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'result' (line 192)
        result_194823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 25), 'result', False)
        # Obtaining the member 'shouldStop' of a type (line 192)
        shouldStop_194824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 25), result_194823, 'shouldStop')
        # Processing the call keyword arguments (line 192)
        kwargs_194825 = {}
        # Getting the type of 'self' (line 192)
        self_194821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 192)
        assertFalse_194822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_194821, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 192)
        assertFalse_call_result_194826 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), assertFalse_194822, *[shouldStop_194824], **kwargs_194825)
        
        
        # ################# End of 'testRemoveResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testRemoveResult' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_194827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194827)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testRemoveResult'
        return stypy_return_type_194827


    @norecursion
    def testMainInstallsHandler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testMainInstallsHandler'
        module_type_store = module_type_store.open_function_context('testMainInstallsHandler', 194, 4, False)
        # Assigning a type to the variable 'self' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_function_name', 'TestBreak.testMainInstallsHandler')
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testMainInstallsHandler.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testMainInstallsHandler', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testMainInstallsHandler', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testMainInstallsHandler(...)' code ##################

        
        # Assigning a Call to a Name (line 195):
        
        # Call to object(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_194829 = {}
        # Getting the type of 'object' (line 195)
        object_194828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'object', False)
        # Calling object(args, kwargs) (line 195)
        object_call_result_194830 = invoke(stypy.reporting.localization.Localization(__file__, 195, 19), object_194828, *[], **kwargs_194829)
        
        # Assigning a type to the variable 'failfast' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'failfast', object_call_result_194830)
        
        # Assigning a Call to a Name (line 196):
        
        # Call to object(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_194832 = {}
        # Getting the type of 'object' (line 196)
        object_194831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'object', False)
        # Calling object(args, kwargs) (line 196)
        object_call_result_194833 = invoke(stypy.reporting.localization.Localization(__file__, 196, 15), object_194831, *[], **kwargs_194832)
        
        # Assigning a type to the variable 'test' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'test', object_call_result_194833)
        
        # Assigning a Call to a Name (line 197):
        
        # Call to object(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_194835 = {}
        # Getting the type of 'object' (line 197)
        object_194834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'object', False)
        # Calling object(args, kwargs) (line 197)
        object_call_result_194836 = invoke(stypy.reporting.localization.Localization(__file__, 197, 20), object_194834, *[], **kwargs_194835)
        
        # Assigning a type to the variable 'verbosity' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'verbosity', object_call_result_194836)
        
        # Assigning a Call to a Name (line 198):
        
        # Call to object(...): (line 198)
        # Processing the call keyword arguments (line 198)
        kwargs_194838 = {}
        # Getting the type of 'object' (line 198)
        object_194837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'object', False)
        # Calling object(args, kwargs) (line 198)
        object_call_result_194839 = invoke(stypy.reporting.localization.Localization(__file__, 198, 17), object_194837, *[], **kwargs_194838)
        
        # Assigning a type to the variable 'result' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'result', object_call_result_194839)
        
        # Assigning a Call to a Name (line 199):
        
        # Call to getsignal(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'signal' (line 199)
        signal_194842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 43), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 199)
        SIGINT_194843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 43), signal_194842, 'SIGINT')
        # Processing the call keyword arguments (line 199)
        kwargs_194844 = {}
        # Getting the type of 'signal' (line 199)
        signal_194840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 199)
        getsignal_194841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 26), signal_194840, 'getsignal')
        # Calling getsignal(args, kwargs) (line 199)
        getsignal_call_result_194845 = invoke(stypy.reporting.localization.Localization(__file__, 199, 26), getsignal_194841, *[SIGINT_194843], **kwargs_194844)
        
        # Assigning a type to the variable 'default_handler' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'default_handler', getsignal_call_result_194845)
        # Declaration of the 'FakeRunner' class

        class FakeRunner(object, ):
            
            # Assigning a List to a Name (line 202):
            
            # Obtaining an instance of the builtin type 'list' (line 202)
            list_194846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 202)
            
            # Assigning a type to the variable 'initArgs' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'initArgs', list_194846)
            
            # Assigning a List to a Name (line 203):
            
            # Obtaining an instance of the builtin type 'list' (line 203)
            list_194847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 22), 'list')
            # Adding type elements to the builtin type 'list' instance (line 203)
            
            # Assigning a type to the variable 'runArgs' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'runArgs', list_194847)

            @norecursion
            def __init__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__init__'
                module_type_store = module_type_store.open_function_context('__init__', 204, 12, False)
                # Assigning a type to the variable 'self' (line 205)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeRunner.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 205)
                # Processing the call arguments (line 205)
                
                # Obtaining an instance of the builtin type 'tuple' (line 205)
                tuple_194851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 38), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 205)
                # Adding element type (line 205)
                # Getting the type of 'args' (line 205)
                args_194852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 38), 'args', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 38), tuple_194851, args_194852)
                # Adding element type (line 205)
                # Getting the type of 'kwargs' (line 205)
                kwargs_194853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 44), 'kwargs', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 38), tuple_194851, kwargs_194853)
                
                # Processing the call keyword arguments (line 205)
                kwargs_194854 = {}
                # Getting the type of 'self' (line 205)
                self_194848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'self', False)
                # Obtaining the member 'initArgs' of a type (line 205)
                initArgs_194849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 16), self_194848, 'initArgs')
                # Obtaining the member 'append' of a type (line 205)
                append_194850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 16), initArgs_194849, 'append')
                # Calling append(args, kwargs) (line 205)
                append_call_result_194855 = invoke(stypy.reporting.localization.Localization(__file__, 205, 16), append_194850, *[tuple_194851], **kwargs_194854)
                
                
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
                module_type_store = module_type_store.open_function_context('run', 206, 12, False)
                # Assigning a type to the variable 'self' (line 207)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'self', type_of_self)
                
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

                
                # Call to append(...): (line 207)
                # Processing the call arguments (line 207)
                # Getting the type of 'test' (line 207)
                test_194859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 36), 'test', False)
                # Processing the call keyword arguments (line 207)
                kwargs_194860 = {}
                # Getting the type of 'self' (line 207)
                self_194856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'self', False)
                # Obtaining the member 'runArgs' of a type (line 207)
                runArgs_194857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), self_194856, 'runArgs')
                # Obtaining the member 'append' of a type (line 207)
                append_194858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 16), runArgs_194857, 'append')
                # Calling append(args, kwargs) (line 207)
                append_call_result_194861 = invoke(stypy.reporting.localization.Localization(__file__, 207, 16), append_194858, *[test_194859], **kwargs_194860)
                
                # Getting the type of 'result' (line 208)
                result_194862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'result')
                # Assigning a type to the variable 'stypy_return_type' (line 208)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'stypy_return_type', result_194862)
                
                # ################# End of 'run(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'run' in the type store
                # Getting the type of 'stypy_return_type' (line 206)
                stypy_return_type_194863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_194863)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'run'
                return stypy_return_type_194863

        
        # Assigning a type to the variable 'FakeRunner' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'FakeRunner', FakeRunner)
        # Declaration of the 'Program' class
        # Getting the type of 'unittest' (line 210)
        unittest_194864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'unittest')
        # Obtaining the member 'TestProgram' of a type (line 210)
        TestProgram_194865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 22), unittest_194864, 'TestProgram')

        class Program(TestProgram_194865, ):

            @norecursion
            def __init__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__init__'
                module_type_store = module_type_store.open_function_context('__init__', 211, 12, False)
                # Assigning a type to the variable 'self' (line 212)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Program.__init__', ['catchbreak'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return

                # Initialize method data
                init_call_information(module_type_store, '__init__', localization, ['catchbreak'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of '__init__(...)' code ##################

                
                # Assigning a Name to a Attribute (line 212):
                # Getting the type of 'False' (line 212)
                False_194866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'False')
                # Getting the type of 'self' (line 212)
                self_194867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'self')
                # Setting the type of the member 'exit' of a type (line 212)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 16), self_194867, 'exit', False_194866)
                
                # Assigning a Name to a Attribute (line 213):
                # Getting the type of 'verbosity' (line 213)
                verbosity_194868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'verbosity')
                # Getting the type of 'self' (line 213)
                self_194869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'self')
                # Setting the type of the member 'verbosity' of a type (line 213)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), self_194869, 'verbosity', verbosity_194868)
                
                # Assigning a Name to a Attribute (line 214):
                # Getting the type of 'failfast' (line 214)
                failfast_194870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 32), 'failfast')
                # Getting the type of 'self' (line 214)
                self_194871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'self')
                # Setting the type of the member 'failfast' of a type (line 214)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), self_194871, 'failfast', failfast_194870)
                
                # Assigning a Name to a Attribute (line 215):
                # Getting the type of 'catchbreak' (line 215)
                catchbreak_194872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 34), 'catchbreak')
                # Getting the type of 'self' (line 215)
                self_194873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'self')
                # Setting the type of the member 'catchbreak' of a type (line 215)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 16), self_194873, 'catchbreak', catchbreak_194872)
                
                # Assigning a Name to a Attribute (line 216):
                # Getting the type of 'FakeRunner' (line 216)
                FakeRunner_194874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 'FakeRunner')
                # Getting the type of 'self' (line 216)
                self_194875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'self')
                # Setting the type of the member 'testRunner' of a type (line 216)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), self_194875, 'testRunner', FakeRunner_194874)
                
                # Assigning a Name to a Attribute (line 217):
                # Getting the type of 'test' (line 217)
                test_194876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'test')
                # Getting the type of 'self' (line 217)
                self_194877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'self')
                # Setting the type of the member 'test' of a type (line 217)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 16), self_194877, 'test', test_194876)
                
                # Assigning a Name to a Attribute (line 218):
                # Getting the type of 'None' (line 218)
                None_194878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 30), 'None')
                # Getting the type of 'self' (line 218)
                self_194879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'self')
                # Setting the type of the member 'result' of a type (line 218)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 16), self_194879, 'result', None_194878)
                
                # ################# End of '__init__(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()

        
        # Assigning a type to the variable 'Program' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'Program', Program)
        
        # Assigning a Call to a Name (line 220):
        
        # Call to Program(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'False' (line 220)
        False_194881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), 'False', False)
        # Processing the call keyword arguments (line 220)
        kwargs_194882 = {}
        # Getting the type of 'Program' (line 220)
        Program_194880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'Program', False)
        # Calling Program(args, kwargs) (line 220)
        Program_call_result_194883 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), Program_194880, *[False_194881], **kwargs_194882)
        
        # Assigning a type to the variable 'p' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'p', Program_call_result_194883)
        
        # Call to runTests(...): (line 221)
        # Processing the call keyword arguments (line 221)
        kwargs_194886 = {}
        # Getting the type of 'p' (line 221)
        p_194884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'p', False)
        # Obtaining the member 'runTests' of a type (line 221)
        runTests_194885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), p_194884, 'runTests')
        # Calling runTests(args, kwargs) (line 221)
        runTests_call_result_194887 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), runTests_194885, *[], **kwargs_194886)
        
        
        # Call to assertEqual(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'FakeRunner' (line 223)
        FakeRunner_194890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'FakeRunner', False)
        # Obtaining the member 'initArgs' of a type (line 223)
        initArgs_194891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 25), FakeRunner_194890, 'initArgs')
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_194892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        
        # Obtaining an instance of the builtin type 'tuple' (line 223)
        tuple_194893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 223)
        # Adding element type (line 223)
        
        # Obtaining an instance of the builtin type 'tuple' (line 223)
        tuple_194894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 223)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 48), tuple_194893, tuple_194894)
        # Adding element type (line 223)
        
        # Obtaining an instance of the builtin type 'dict' (line 223)
        dict_194895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 52), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 223)
        # Adding element type (key, value) (line 223)
        str_194896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 53), 'str', 'buffer')
        # Getting the type of 'None' (line 223)
        None_194897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 63), 'None', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 52), dict_194895, (str_194896, None_194897))
        # Adding element type (key, value) (line 223)
        str_194898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 53), 'str', 'verbosity')
        # Getting the type of 'verbosity' (line 224)
        verbosity_194899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 66), 'verbosity', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 52), dict_194895, (str_194898, verbosity_194899))
        # Adding element type (key, value) (line 223)
        str_194900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 53), 'str', 'failfast')
        # Getting the type of 'failfast' (line 225)
        failfast_194901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 65), 'failfast', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 52), dict_194895, (str_194900, failfast_194901))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 48), tuple_194893, dict_194895)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 46), list_194892, tuple_194893)
        
        # Processing the call keyword arguments (line 223)
        kwargs_194902 = {}
        # Getting the type of 'self' (line 223)
        self_194888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 223)
        assertEqual_194889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_194888, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 223)
        assertEqual_call_result_194903 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), assertEqual_194889, *[initArgs_194891, list_194892], **kwargs_194902)
        
        
        # Call to assertEqual(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'FakeRunner' (line 226)
        FakeRunner_194906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 25), 'FakeRunner', False)
        # Obtaining the member 'runArgs' of a type (line 226)
        runArgs_194907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 25), FakeRunner_194906, 'runArgs')
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_194908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'test' (line 226)
        test_194909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 46), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 45), list_194908, test_194909)
        
        # Processing the call keyword arguments (line 226)
        kwargs_194910 = {}
        # Getting the type of 'self' (line 226)
        self_194904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 226)
        assertEqual_194905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), self_194904, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 226)
        assertEqual_call_result_194911 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), assertEqual_194905, *[runArgs_194907, list_194908], **kwargs_194910)
        
        
        # Call to assertEqual(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'p' (line 227)
        p_194914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'p', False)
        # Obtaining the member 'result' of a type (line 227)
        result_194915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 25), p_194914, 'result')
        # Getting the type of 'result' (line 227)
        result_194916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 35), 'result', False)
        # Processing the call keyword arguments (line 227)
        kwargs_194917 = {}
        # Getting the type of 'self' (line 227)
        self_194912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 227)
        assertEqual_194913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_194912, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 227)
        assertEqual_call_result_194918 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), assertEqual_194913, *[result_194915, result_194916], **kwargs_194917)
        
        
        # Call to assertEqual(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Call to getsignal(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'signal' (line 229)
        signal_194923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 42), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 229)
        SIGINT_194924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 42), signal_194923, 'SIGINT')
        # Processing the call keyword arguments (line 229)
        kwargs_194925 = {}
        # Getting the type of 'signal' (line 229)
        signal_194921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 25), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 229)
        getsignal_194922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 25), signal_194921, 'getsignal')
        # Calling getsignal(args, kwargs) (line 229)
        getsignal_call_result_194926 = invoke(stypy.reporting.localization.Localization(__file__, 229, 25), getsignal_194922, *[SIGINT_194924], **kwargs_194925)
        
        # Getting the type of 'default_handler' (line 229)
        default_handler_194927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 58), 'default_handler', False)
        # Processing the call keyword arguments (line 229)
        kwargs_194928 = {}
        # Getting the type of 'self' (line 229)
        self_194919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 229)
        assertEqual_194920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_194919, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 229)
        assertEqual_call_result_194929 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), assertEqual_194920, *[getsignal_call_result_194926, default_handler_194927], **kwargs_194928)
        
        
        # Assigning a List to a Attribute (line 231):
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_194930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        
        # Getting the type of 'FakeRunner' (line 231)
        FakeRunner_194931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'FakeRunner')
        # Setting the type of the member 'initArgs' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), FakeRunner_194931, 'initArgs', list_194930)
        
        # Assigning a List to a Attribute (line 232):
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_194932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        
        # Getting the type of 'FakeRunner' (line 232)
        FakeRunner_194933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'FakeRunner')
        # Setting the type of the member 'runArgs' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), FakeRunner_194933, 'runArgs', list_194932)
        
        # Assigning a Call to a Name (line 233):
        
        # Call to Program(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'True' (line 233)
        True_194935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'True', False)
        # Processing the call keyword arguments (line 233)
        kwargs_194936 = {}
        # Getting the type of 'Program' (line 233)
        Program_194934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'Program', False)
        # Calling Program(args, kwargs) (line 233)
        Program_call_result_194937 = invoke(stypy.reporting.localization.Localization(__file__, 233, 12), Program_194934, *[True_194935], **kwargs_194936)
        
        # Assigning a type to the variable 'p' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'p', Program_call_result_194937)
        
        # Call to runTests(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_194940 = {}
        # Getting the type of 'p' (line 234)
        p_194938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'p', False)
        # Obtaining the member 'runTests' of a type (line 234)
        runTests_194939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), p_194938, 'runTests')
        # Calling runTests(args, kwargs) (line 234)
        runTests_call_result_194941 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), runTests_194939, *[], **kwargs_194940)
        
        
        # Call to assertEqual(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'FakeRunner' (line 236)
        FakeRunner_194944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 25), 'FakeRunner', False)
        # Obtaining the member 'initArgs' of a type (line 236)
        initArgs_194945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 25), FakeRunner_194944, 'initArgs')
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_194946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'tuple' (line 236)
        tuple_194947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 236)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'tuple' (line 236)
        tuple_194948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 236)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 48), tuple_194947, tuple_194948)
        # Adding element type (line 236)
        
        # Obtaining an instance of the builtin type 'dict' (line 236)
        dict_194949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 52), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 236)
        # Adding element type (key, value) (line 236)
        str_194950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 53), 'str', 'buffer')
        # Getting the type of 'None' (line 236)
        None_194951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 63), 'None', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 52), dict_194949, (str_194950, None_194951))
        # Adding element type (key, value) (line 236)
        str_194952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 53), 'str', 'verbosity')
        # Getting the type of 'verbosity' (line 237)
        verbosity_194953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 66), 'verbosity', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 52), dict_194949, (str_194952, verbosity_194953))
        # Adding element type (key, value) (line 236)
        str_194954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 53), 'str', 'failfast')
        # Getting the type of 'failfast' (line 238)
        failfast_194955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 65), 'failfast', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 52), dict_194949, (str_194954, failfast_194955))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 48), tuple_194947, dict_194949)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 46), list_194946, tuple_194947)
        
        # Processing the call keyword arguments (line 236)
        kwargs_194956 = {}
        # Getting the type of 'self' (line 236)
        self_194942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 236)
        assertEqual_194943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_194942, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 236)
        assertEqual_call_result_194957 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), assertEqual_194943, *[initArgs_194945, list_194946], **kwargs_194956)
        
        
        # Call to assertEqual(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'FakeRunner' (line 239)
        FakeRunner_194960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 25), 'FakeRunner', False)
        # Obtaining the member 'runArgs' of a type (line 239)
        runArgs_194961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 25), FakeRunner_194960, 'runArgs')
        
        # Obtaining an instance of the builtin type 'list' (line 239)
        list_194962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 239)
        # Adding element type (line 239)
        # Getting the type of 'test' (line 239)
        test_194963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 46), 'test', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 45), list_194962, test_194963)
        
        # Processing the call keyword arguments (line 239)
        kwargs_194964 = {}
        # Getting the type of 'self' (line 239)
        self_194958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 239)
        assertEqual_194959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_194958, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 239)
        assertEqual_call_result_194965 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), assertEqual_194959, *[runArgs_194961, list_194962], **kwargs_194964)
        
        
        # Call to assertEqual(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'p' (line 240)
        p_194968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 25), 'p', False)
        # Obtaining the member 'result' of a type (line 240)
        result_194969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 25), p_194968, 'result')
        # Getting the type of 'result' (line 240)
        result_194970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 35), 'result', False)
        # Processing the call keyword arguments (line 240)
        kwargs_194971 = {}
        # Getting the type of 'self' (line 240)
        self_194966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 240)
        assertEqual_194967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), self_194966, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 240)
        assertEqual_call_result_194972 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), assertEqual_194967, *[result_194969, result_194970], **kwargs_194971)
        
        
        # Call to assertNotEqual(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Call to getsignal(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'signal' (line 242)
        signal_194977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 45), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 242)
        SIGINT_194978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 45), signal_194977, 'SIGINT')
        # Processing the call keyword arguments (line 242)
        kwargs_194979 = {}
        # Getting the type of 'signal' (line 242)
        signal_194975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 242)
        getsignal_194976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 28), signal_194975, 'getsignal')
        # Calling getsignal(args, kwargs) (line 242)
        getsignal_call_result_194980 = invoke(stypy.reporting.localization.Localization(__file__, 242, 28), getsignal_194976, *[SIGINT_194978], **kwargs_194979)
        
        # Getting the type of 'default_handler' (line 242)
        default_handler_194981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 61), 'default_handler', False)
        # Processing the call keyword arguments (line 242)
        kwargs_194982 = {}
        # Getting the type of 'self' (line 242)
        self_194973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self', False)
        # Obtaining the member 'assertNotEqual' of a type (line 242)
        assertNotEqual_194974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_194973, 'assertNotEqual')
        # Calling assertNotEqual(args, kwargs) (line 242)
        assertNotEqual_call_result_194983 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assertNotEqual_194974, *[getsignal_call_result_194980, default_handler_194981], **kwargs_194982)
        
        
        # ################# End of 'testMainInstallsHandler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testMainInstallsHandler' in the type store
        # Getting the type of 'stypy_return_type' (line 194)
        stypy_return_type_194984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194984)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testMainInstallsHandler'
        return stypy_return_type_194984


    @norecursion
    def testRemoveHandler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testRemoveHandler'
        module_type_store = module_type_store.open_function_context('testRemoveHandler', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_function_name', 'TestBreak.testRemoveHandler')
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testRemoveHandler.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testRemoveHandler', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testRemoveHandler', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testRemoveHandler(...)' code ##################

        
        # Assigning a Call to a Name (line 245):
        
        # Call to getsignal(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'signal' (line 245)
        signal_194987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 43), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 245)
        SIGINT_194988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 43), signal_194987, 'SIGINT')
        # Processing the call keyword arguments (line 245)
        kwargs_194989 = {}
        # Getting the type of 'signal' (line 245)
        signal_194985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 26), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 245)
        getsignal_194986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 26), signal_194985, 'getsignal')
        # Calling getsignal(args, kwargs) (line 245)
        getsignal_call_result_194990 = invoke(stypy.reporting.localization.Localization(__file__, 245, 26), getsignal_194986, *[SIGINT_194988], **kwargs_194989)
        
        # Assigning a type to the variable 'default_handler' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'default_handler', getsignal_call_result_194990)
        
        # Call to installHandler(...): (line 246)
        # Processing the call keyword arguments (line 246)
        kwargs_194993 = {}
        # Getting the type of 'unittest' (line 246)
        unittest_194991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'unittest', False)
        # Obtaining the member 'installHandler' of a type (line 246)
        installHandler_194992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), unittest_194991, 'installHandler')
        # Calling installHandler(args, kwargs) (line 246)
        installHandler_call_result_194994 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), installHandler_194992, *[], **kwargs_194993)
        
        
        # Call to removeHandler(...): (line 247)
        # Processing the call keyword arguments (line 247)
        kwargs_194997 = {}
        # Getting the type of 'unittest' (line 247)
        unittest_194995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'unittest', False)
        # Obtaining the member 'removeHandler' of a type (line 247)
        removeHandler_194996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), unittest_194995, 'removeHandler')
        # Calling removeHandler(args, kwargs) (line 247)
        removeHandler_call_result_194998 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), removeHandler_194996, *[], **kwargs_194997)
        
        
        # Call to assertEqual(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Call to getsignal(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'signal' (line 248)
        signal_195003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 248)
        SIGINT_195004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 42), signal_195003, 'SIGINT')
        # Processing the call keyword arguments (line 248)
        kwargs_195005 = {}
        # Getting the type of 'signal' (line 248)
        signal_195001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 25), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 248)
        getsignal_195002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 25), signal_195001, 'getsignal')
        # Calling getsignal(args, kwargs) (line 248)
        getsignal_call_result_195006 = invoke(stypy.reporting.localization.Localization(__file__, 248, 25), getsignal_195002, *[SIGINT_195004], **kwargs_195005)
        
        # Getting the type of 'default_handler' (line 248)
        default_handler_195007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 58), 'default_handler', False)
        # Processing the call keyword arguments (line 248)
        kwargs_195008 = {}
        # Getting the type of 'self' (line 248)
        self_194999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 248)
        assertEqual_195000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_194999, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 248)
        assertEqual_call_result_195009 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), assertEqual_195000, *[getsignal_call_result_195006, default_handler_195007], **kwargs_195008)
        
        
        # Call to removeHandler(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_195012 = {}
        # Getting the type of 'unittest' (line 251)
        unittest_195010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'unittest', False)
        # Obtaining the member 'removeHandler' of a type (line 251)
        removeHandler_195011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), unittest_195010, 'removeHandler')
        # Calling removeHandler(args, kwargs) (line 251)
        removeHandler_call_result_195013 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), removeHandler_195011, *[], **kwargs_195012)
        
        
        # Call to assertEqual(...): (line 252)
        # Processing the call arguments (line 252)
        
        # Call to getsignal(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'signal' (line 252)
        signal_195018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 42), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 252)
        SIGINT_195019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 42), signal_195018, 'SIGINT')
        # Processing the call keyword arguments (line 252)
        kwargs_195020 = {}
        # Getting the type of 'signal' (line 252)
        signal_195016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 25), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 252)
        getsignal_195017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 25), signal_195016, 'getsignal')
        # Calling getsignal(args, kwargs) (line 252)
        getsignal_call_result_195021 = invoke(stypy.reporting.localization.Localization(__file__, 252, 25), getsignal_195017, *[SIGINT_195019], **kwargs_195020)
        
        # Getting the type of 'default_handler' (line 252)
        default_handler_195022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 58), 'default_handler', False)
        # Processing the call keyword arguments (line 252)
        kwargs_195023 = {}
        # Getting the type of 'self' (line 252)
        self_195014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 252)
        assertEqual_195015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), self_195014, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 252)
        assertEqual_call_result_195024 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), assertEqual_195015, *[getsignal_call_result_195021, default_handler_195022], **kwargs_195023)
        
        
        # ################# End of 'testRemoveHandler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testRemoveHandler' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_195025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_195025)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testRemoveHandler'
        return stypy_return_type_195025


    @norecursion
    def testRemoveHandlerAsDecorator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testRemoveHandlerAsDecorator'
        module_type_store = module_type_store.open_function_context('testRemoveHandlerAsDecorator', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_localization', localization)
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_function_name', 'TestBreak.testRemoveHandlerAsDecorator')
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_param_names_list', [])
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBreak.testRemoveHandlerAsDecorator.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.testRemoveHandlerAsDecorator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testRemoveHandlerAsDecorator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testRemoveHandlerAsDecorator(...)' code ##################

        
        # Assigning a Call to a Name (line 255):
        
        # Call to getsignal(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'signal' (line 255)
        signal_195028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 43), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 255)
        SIGINT_195029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 43), signal_195028, 'SIGINT')
        # Processing the call keyword arguments (line 255)
        kwargs_195030 = {}
        # Getting the type of 'signal' (line 255)
        signal_195026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 26), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 255)
        getsignal_195027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 26), signal_195026, 'getsignal')
        # Calling getsignal(args, kwargs) (line 255)
        getsignal_call_result_195031 = invoke(stypy.reporting.localization.Localization(__file__, 255, 26), getsignal_195027, *[SIGINT_195029], **kwargs_195030)
        
        # Assigning a type to the variable 'default_handler' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'default_handler', getsignal_call_result_195031)
        
        # Call to installHandler(...): (line 256)
        # Processing the call keyword arguments (line 256)
        kwargs_195034 = {}
        # Getting the type of 'unittest' (line 256)
        unittest_195032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'unittest', False)
        # Obtaining the member 'installHandler' of a type (line 256)
        installHandler_195033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), unittest_195032, 'installHandler')
        # Calling installHandler(args, kwargs) (line 256)
        installHandler_call_result_195035 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), installHandler_195033, *[], **kwargs_195034)
        

        @norecursion
        def test(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'test'
            module_type_store = module_type_store.open_function_context('test', 258, 8, False)
            
            # Passed parameters checking function
            test.stypy_localization = localization
            test.stypy_type_of_self = None
            test.stypy_type_store = module_type_store
            test.stypy_function_name = 'test'
            test.stypy_param_names_list = []
            test.stypy_varargs_param_name = None
            test.stypy_kwargs_param_name = None
            test.stypy_call_defaults = defaults
            test.stypy_call_varargs = varargs
            test.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'test', [], None, None, defaults, varargs, kwargs)

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

            
            # Call to assertEqual(...): (line 260)
            # Processing the call arguments (line 260)
            
            # Call to getsignal(...): (line 260)
            # Processing the call arguments (line 260)
            # Getting the type of 'signal' (line 260)
            signal_195040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 46), 'signal', False)
            # Obtaining the member 'SIGINT' of a type (line 260)
            SIGINT_195041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 46), signal_195040, 'SIGINT')
            # Processing the call keyword arguments (line 260)
            kwargs_195042 = {}
            # Getting the type of 'signal' (line 260)
            signal_195038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 29), 'signal', False)
            # Obtaining the member 'getsignal' of a type (line 260)
            getsignal_195039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 29), signal_195038, 'getsignal')
            # Calling getsignal(args, kwargs) (line 260)
            getsignal_call_result_195043 = invoke(stypy.reporting.localization.Localization(__file__, 260, 29), getsignal_195039, *[SIGINT_195041], **kwargs_195042)
            
            # Getting the type of 'default_handler' (line 260)
            default_handler_195044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 62), 'default_handler', False)
            # Processing the call keyword arguments (line 260)
            kwargs_195045 = {}
            # Getting the type of 'self' (line 260)
            self_195036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 260)
            assertEqual_195037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), self_195036, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 260)
            assertEqual_call_result_195046 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), assertEqual_195037, *[getsignal_call_result_195043, default_handler_195044], **kwargs_195045)
            
            
            # ################# End of 'test(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'test' in the type store
            # Getting the type of 'stypy_return_type' (line 258)
            stypy_return_type_195047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_195047)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'test'
            return stypy_return_type_195047

        # Assigning a type to the variable 'test' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'test', test)
        
        # Call to test(...): (line 262)
        # Processing the call keyword arguments (line 262)
        kwargs_195049 = {}
        # Getting the type of 'test' (line 262)
        test_195048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'test', False)
        # Calling test(args, kwargs) (line 262)
        test_call_result_195050 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), test_195048, *[], **kwargs_195049)
        
        
        # Call to assertNotEqual(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Call to getsignal(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'signal' (line 263)
        signal_195055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 263)
        SIGINT_195056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 45), signal_195055, 'SIGINT')
        # Processing the call keyword arguments (line 263)
        kwargs_195057 = {}
        # Getting the type of 'signal' (line 263)
        signal_195053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 28), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 263)
        getsignal_195054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 28), signal_195053, 'getsignal')
        # Calling getsignal(args, kwargs) (line 263)
        getsignal_call_result_195058 = invoke(stypy.reporting.localization.Localization(__file__, 263, 28), getsignal_195054, *[SIGINT_195056], **kwargs_195057)
        
        # Getting the type of 'default_handler' (line 263)
        default_handler_195059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 61), 'default_handler', False)
        # Processing the call keyword arguments (line 263)
        kwargs_195060 = {}
        # Getting the type of 'self' (line 263)
        self_195051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self', False)
        # Obtaining the member 'assertNotEqual' of a type (line 263)
        assertNotEqual_195052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), self_195051, 'assertNotEqual')
        # Calling assertNotEqual(args, kwargs) (line 263)
        assertNotEqual_call_result_195061 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), assertNotEqual_195052, *[getsignal_call_result_195058, default_handler_195059], **kwargs_195060)
        
        
        # ################# End of 'testRemoveHandlerAsDecorator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testRemoveHandlerAsDecorator' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_195062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_195062)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testRemoveHandlerAsDecorator'
        return stypy_return_type_195062


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 0, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreak.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBreak' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'TestBreak', TestBreak)

# Assigning a Name to a Name (line 18):
# Getting the type of 'None' (line 18)
None_195063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'None')
# Getting the type of 'TestBreak'
TestBreak_195064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestBreak')
# Setting the type of the member 'int_handler' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestBreak_195064, 'int_handler', None_195063)
# Declaration of the 'TestBreakDefaultIntHandler' class
# Getting the type of 'TestBreak' (line 269)
TestBreak_195065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 33), 'TestBreak')

class TestBreakDefaultIntHandler(TestBreak_195065, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 265, 0, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreakDefaultIntHandler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBreakDefaultIntHandler' (line 265)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'TestBreakDefaultIntHandler', TestBreakDefaultIntHandler)

# Assigning a Attribute to a Name (line 270):
# Getting the type of 'signal' (line 270)
signal_195066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 18), 'signal')
# Obtaining the member 'default_int_handler' of a type (line 270)
default_int_handler_195067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 18), signal_195066, 'default_int_handler')
# Getting the type of 'TestBreakDefaultIntHandler'
TestBreakDefaultIntHandler_195068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestBreakDefaultIntHandler')
# Setting the type of the member 'int_handler' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestBreakDefaultIntHandler_195068, 'int_handler', default_int_handler_195067)
# Declaration of the 'TestBreakSignalIgnored' class
# Getting the type of 'TestBreak' (line 276)
TestBreak_195069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 29), 'TestBreak')

class TestBreakSignalIgnored(TestBreak_195069, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 272, 0, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreakSignalIgnored.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBreakSignalIgnored' (line 272)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 0), 'TestBreakSignalIgnored', TestBreakSignalIgnored)

# Assigning a Attribute to a Name (line 277):
# Getting the type of 'signal' (line 277)
signal_195070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 18), 'signal')
# Obtaining the member 'SIG_IGN' of a type (line 277)
SIG_IGN_195071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 18), signal_195070, 'SIG_IGN')
# Getting the type of 'TestBreakSignalIgnored'
TestBreakSignalIgnored_195072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestBreakSignalIgnored')
# Setting the type of the member 'int_handler' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestBreakSignalIgnored_195072, 'int_handler', SIG_IGN_195071)
# Declaration of the 'TestBreakSignalDefault' class
# Getting the type of 'TestBreak' (line 283)
TestBreak_195073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 29), 'TestBreak')

class TestBreakSignalDefault(TestBreak_195073, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 279, 0, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBreakSignalDefault.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBreakSignalDefault' (line 279)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 0), 'TestBreakSignalDefault', TestBreakSignalDefault)

# Assigning a Attribute to a Name (line 284):
# Getting the type of 'signal' (line 284)
signal_195074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 18), 'signal')
# Obtaining the member 'SIG_DFL' of a type (line 284)
SIG_DFL_195075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 18), signal_195074, 'SIG_DFL')
# Getting the type of 'TestBreakSignalDefault'
TestBreakSignalDefault_195076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestBreakSignalDefault')
# Setting the type of the member 'int_handler' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestBreakSignalDefault_195076, 'int_handler', SIG_DFL_195075)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
