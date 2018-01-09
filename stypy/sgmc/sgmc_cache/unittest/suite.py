
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''TestSuite'''
2: 
3: import sys
4: 
5: from . import case
6: from . import util
7: 
8: __unittest = True
9: 
10: 
11: def _call_if_exists(parent, attr):
12:     func = getattr(parent, attr, lambda: None)
13:     func()
14: 
15: 
16: class BaseTestSuite(object):
17:     '''A simple test suite that doesn't provide class or module shared fixtures.
18:     '''
19:     def __init__(self, tests=()):
20:         self._tests = []
21:         self.addTests(tests)
22: 
23:     def __repr__(self):
24:         return "<%s tests=%s>" % (util.strclass(self.__class__), list(self))
25: 
26:     def __eq__(self, other):
27:         if not isinstance(other, self.__class__):
28:             return NotImplemented
29:         return list(self) == list(other)
30: 
31:     def __ne__(self, other):
32:         return not self == other
33: 
34:     # Can't guarantee hash invariant, so flag as unhashable
35:     __hash__ = None
36: 
37:     def __iter__(self):
38:         return iter(self._tests)
39: 
40:     def countTestCases(self):
41:         cases = 0
42:         for test in self:
43:             cases += test.countTestCases()
44:         return cases
45: 
46:     def addTest(self, test):
47:         # sanity checks
48:         if not hasattr(test, '__call__'):
49:             raise TypeError("{} is not callable".format(repr(test)))
50:         if isinstance(test, type) and issubclass(test,
51:                                                  (case.TestCase, TestSuite)):
52:             raise TypeError("TestCases and TestSuites must be instantiated "
53:                             "before passing them to addTest()")
54:         self._tests.append(test)
55: 
56:     def addTests(self, tests):
57:         if isinstance(tests, basestring):
58:             raise TypeError("tests must be an iterable of tests, not a string")
59:         for test in tests:
60:             self.addTest(test)
61: 
62:     def run(self, result):
63:         for test in self:
64:             if result.shouldStop:
65:                 break
66:             test(result)
67:         return result
68: 
69:     def __call__(self, *args, **kwds):
70:         return self.run(*args, **kwds)
71: 
72:     def debug(self):
73:         '''Run the tests without collecting errors in a TestResult'''
74:         for test in self:
75:             test.debug()
76: 
77: 
78: class TestSuite(BaseTestSuite):
79:     '''A test suite is a composite test consisting of a number of TestCases.
80: 
81:     For use, create an instance of TestSuite, then add test case instances.
82:     When all tests have been added, the suite can be passed to a test
83:     runner, such as TextTestRunner. It will run the individual test cases
84:     in the order in which they were added, aggregating the results. When
85:     subclassing, do not forget to call the base class constructor.
86:     '''
87: 
88:     def run(self, result, debug=False):
89:         topLevel = False
90:         if getattr(result, '_testRunEntered', False) is False:
91:             result._testRunEntered = topLevel = True
92: 
93:         for test in self:
94:             if result.shouldStop:
95:                 break
96: 
97:             if _isnotsuite(test):
98:                 self._tearDownPreviousClass(test, result)
99:                 self._handleModuleFixture(test, result)
100:                 self._handleClassSetUp(test, result)
101:                 result._previousTestClass = test.__class__
102: 
103:                 if (getattr(test.__class__, '_classSetupFailed', False) or
104:                     getattr(result, '_moduleSetUpFailed', False)):
105:                     continue
106: 
107:             if not debug:
108:                 test(result)
109:             else:
110:                 test.debug()
111: 
112:         if topLevel:
113:             self._tearDownPreviousClass(None, result)
114:             self._handleModuleTearDown(result)
115:             result._testRunEntered = False
116:         return result
117: 
118:     def debug(self):
119:         '''Run the tests without collecting errors in a TestResult'''
120:         debug = _DebugResult()
121:         self.run(debug, True)
122: 
123:     ################################
124: 
125:     def _handleClassSetUp(self, test, result):
126:         previousClass = getattr(result, '_previousTestClass', None)
127:         currentClass = test.__class__
128:         if currentClass == previousClass:
129:             return
130:         if result._moduleSetUpFailed:
131:             return
132:         if getattr(currentClass, "__unittest_skip__", False):
133:             return
134: 
135:         try:
136:             currentClass._classSetupFailed = False
137:         except TypeError:
138:             # test may actually be a function
139:             # so its class will be a builtin-type
140:             pass
141: 
142:         setUpClass = getattr(currentClass, 'setUpClass', None)
143:         if setUpClass is not None:
144:             _call_if_exists(result, '_setupStdout')
145:             try:
146:                 setUpClass()
147:             except Exception as e:
148:                 if isinstance(result, _DebugResult):
149:                     raise
150:                 currentClass._classSetupFailed = True
151:                 className = util.strclass(currentClass)
152:                 errorName = 'setUpClass (%s)' % className
153:                 self._addClassOrModuleLevelException(result, e, errorName)
154:             finally:
155:                 _call_if_exists(result, '_restoreStdout')
156: 
157:     def _get_previous_module(self, result):
158:         previousModule = None
159:         previousClass = getattr(result, '_previousTestClass', None)
160:         if previousClass is not None:
161:             previousModule = previousClass.__module__
162:         return previousModule
163: 
164: 
165:     def _handleModuleFixture(self, test, result):
166:         previousModule = self._get_previous_module(result)
167:         currentModule = test.__class__.__module__
168:         if currentModule == previousModule:
169:             return
170: 
171:         self._handleModuleTearDown(result)
172: 
173:         result._moduleSetUpFailed = False
174:         try:
175:             module = sys.modules[currentModule]
176:         except KeyError:
177:             return
178:         setUpModule = getattr(module, 'setUpModule', None)
179:         if setUpModule is not None:
180:             _call_if_exists(result, '_setupStdout')
181:             try:
182:                 setUpModule()
183:             except Exception, e:
184:                 if isinstance(result, _DebugResult):
185:                     raise
186:                 result._moduleSetUpFailed = True
187:                 errorName = 'setUpModule (%s)' % currentModule
188:                 self._addClassOrModuleLevelException(result, e, errorName)
189:             finally:
190:                 _call_if_exists(result, '_restoreStdout')
191: 
192:     def _addClassOrModuleLevelException(self, result, exception, errorName):
193:         error = _ErrorHolder(errorName)
194:         addSkip = getattr(result, 'addSkip', None)
195:         if addSkip is not None and isinstance(exception, case.SkipTest):
196:             addSkip(error, str(exception))
197:         else:
198:             result.addError(error, sys.exc_info())
199: 
200:     def _handleModuleTearDown(self, result):
201:         previousModule = self._get_previous_module(result)
202:         if previousModule is None:
203:             return
204:         if result._moduleSetUpFailed:
205:             return
206: 
207:         try:
208:             module = sys.modules[previousModule]
209:         except KeyError:
210:             return
211: 
212:         tearDownModule = getattr(module, 'tearDownModule', None)
213:         if tearDownModule is not None:
214:             _call_if_exists(result, '_setupStdout')
215:             try:
216:                 tearDownModule()
217:             except Exception as e:
218:                 if isinstance(result, _DebugResult):
219:                     raise
220:                 errorName = 'tearDownModule (%s)' % previousModule
221:                 self._addClassOrModuleLevelException(result, e, errorName)
222:             finally:
223:                 _call_if_exists(result, '_restoreStdout')
224: 
225:     def _tearDownPreviousClass(self, test, result):
226:         previousClass = getattr(result, '_previousTestClass', None)
227:         currentClass = test.__class__
228:         if currentClass == previousClass:
229:             return
230:         if getattr(previousClass, '_classSetupFailed', False):
231:             return
232:         if getattr(result, '_moduleSetUpFailed', False):
233:             return
234:         if getattr(previousClass, "__unittest_skip__", False):
235:             return
236: 
237:         tearDownClass = getattr(previousClass, 'tearDownClass', None)
238:         if tearDownClass is not None:
239:             _call_if_exists(result, '_setupStdout')
240:             try:
241:                 tearDownClass()
242:             except Exception, e:
243:                 if isinstance(result, _DebugResult):
244:                     raise
245:                 className = util.strclass(previousClass)
246:                 errorName = 'tearDownClass (%s)' % className
247:                 self._addClassOrModuleLevelException(result, e, errorName)
248:             finally:
249:                 _call_if_exists(result, '_restoreStdout')
250: 
251: 
252: class _ErrorHolder(object):
253:     '''
254:     Placeholder for a TestCase inside a result. As far as a TestResult
255:     is concerned, this looks exactly like a unit test. Used to insert
256:     arbitrary errors into a test suite run.
257:     '''
258:     # Inspired by the ErrorHolder from Twisted:
259:     # http://twistedmatrix.com/trac/browser/trunk/twisted/trial/runner.py
260: 
261:     # attribute used by TestResult._exc_info_to_string
262:     failureException = None
263: 
264:     def __init__(self, description):
265:         self.description = description
266: 
267:     def id(self):
268:         return self.description
269: 
270:     def shortDescription(self):
271:         return None
272: 
273:     def __repr__(self):
274:         return "<ErrorHolder description=%r>" % (self.description,)
275: 
276:     def __str__(self):
277:         return self.id()
278: 
279:     def run(self, result):
280:         # could call result.addError(...) - but this test-like object
281:         # shouldn't be run anyway
282:         pass
283: 
284:     def __call__(self, result):
285:         return self.run(result)
286: 
287:     def countTestCases(self):
288:         return 0
289: 
290: def _isnotsuite(test):
291:     "A crude way to tell apart testcases and suites with duck-typing"
292:     try:
293:         iter(test)
294:     except TypeError:
295:         return True
296:     return False
297: 
298: 
299: class _DebugResult(object):
300:     "Used by the TestSuite to hold previous class when running in debug."
301:     _previousTestClass = None
302:     _moduleSetUpFailed = False
303:     shouldStop = False
304: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_191993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'TestSuite')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from unittest import case' statement (line 5)
from unittest import case

import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'unittest', None, module_type_store, ['case'], [case])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from unittest import util' statement (line 6)
from unittest import util

import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest', None, module_type_store, ['util'], [util])


# Assigning a Name to a Name (line 8):
# Getting the type of 'True' (line 8)
True_191994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'True')
# Assigning a type to the variable '__unittest' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__unittest', True_191994)

@norecursion
def _call_if_exists(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_call_if_exists'
    module_type_store = module_type_store.open_function_context('_call_if_exists', 11, 0, False)
    
    # Passed parameters checking function
    _call_if_exists.stypy_localization = localization
    _call_if_exists.stypy_type_of_self = None
    _call_if_exists.stypy_type_store = module_type_store
    _call_if_exists.stypy_function_name = '_call_if_exists'
    _call_if_exists.stypy_param_names_list = ['parent', 'attr']
    _call_if_exists.stypy_varargs_param_name = None
    _call_if_exists.stypy_kwargs_param_name = None
    _call_if_exists.stypy_call_defaults = defaults
    _call_if_exists.stypy_call_varargs = varargs
    _call_if_exists.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_call_if_exists', ['parent', 'attr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_call_if_exists', localization, ['parent', 'attr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_call_if_exists(...)' code ##################

    
    # Assigning a Call to a Name (line 12):
    
    # Call to getattr(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'parent' (line 12)
    parent_191996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'parent', False)
    # Getting the type of 'attr' (line 12)
    attr_191997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 27), 'attr', False)

    @norecursion
    def _stypy_temp_lambda_53(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_53'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_53', 12, 33, True)
        # Passed parameters checking function
        _stypy_temp_lambda_53.stypy_localization = localization
        _stypy_temp_lambda_53.stypy_type_of_self = None
        _stypy_temp_lambda_53.stypy_type_store = module_type_store
        _stypy_temp_lambda_53.stypy_function_name = '_stypy_temp_lambda_53'
        _stypy_temp_lambda_53.stypy_param_names_list = []
        _stypy_temp_lambda_53.stypy_varargs_param_name = None
        _stypy_temp_lambda_53.stypy_kwargs_param_name = None
        _stypy_temp_lambda_53.stypy_call_defaults = defaults
        _stypy_temp_lambda_53.stypy_call_varargs = varargs
        _stypy_temp_lambda_53.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_53', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_53', [], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'None' (line 12)
        None_191998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 41), 'None', False)
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 33), 'stypy_return_type', None_191998)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_53' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_191999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 33), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_53'
        return stypy_return_type_191999

    # Assigning a type to the variable '_stypy_temp_lambda_53' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 33), '_stypy_temp_lambda_53', _stypy_temp_lambda_53)
    # Getting the type of '_stypy_temp_lambda_53' (line 12)
    _stypy_temp_lambda_53_192000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 33), '_stypy_temp_lambda_53')
    # Processing the call keyword arguments (line 12)
    kwargs_192001 = {}
    # Getting the type of 'getattr' (line 12)
    getattr_191995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'getattr', False)
    # Calling getattr(args, kwargs) (line 12)
    getattr_call_result_192002 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), getattr_191995, *[parent_191996, attr_191997, _stypy_temp_lambda_53_192000], **kwargs_192001)
    
    # Assigning a type to the variable 'func' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'func', getattr_call_result_192002)
    
    # Call to func(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_192004 = {}
    # Getting the type of 'func' (line 13)
    func_192003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'func', False)
    # Calling func(args, kwargs) (line 13)
    func_call_result_192005 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), func_192003, *[], **kwargs_192004)
    
    
    # ################# End of '_call_if_exists(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_call_if_exists' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_192006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192006)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_call_if_exists'
    return stypy_return_type_192006

# Assigning a type to the variable '_call_if_exists' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '_call_if_exists', _call_if_exists)
# Declaration of the 'BaseTestSuite' class

class BaseTestSuite(object, ):
    str_192007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', "A simple test suite that doesn't provide class or module shared fixtures.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 19)
        tuple_192008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 19)
        
        defaults = [tuple_192008]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.__init__', ['tests'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['tests'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a List to a Attribute (line 20):
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_192009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        
        # Getting the type of 'self' (line 20)
        self_192010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member '_tests' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_192010, '_tests', list_192009)
        
        # Call to addTests(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'tests' (line 21)
        tests_192013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'tests', False)
        # Processing the call keyword arguments (line 21)
        kwargs_192014 = {}
        # Getting the type of 'self' (line 21)
        self_192011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self', False)
        # Obtaining the member 'addTests' of a type (line 21)
        addTests_192012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_192011, 'addTests')
        # Calling addTests(args, kwargs) (line 21)
        addTests_call_result_192015 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), addTests_192012, *[tests_192013], **kwargs_192014)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.__repr__')
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_192016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'str', '<%s tests=%s>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_192017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        
        # Call to strclass(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'self' (line 24)
        self_192020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 48), 'self', False)
        # Obtaining the member '__class__' of a type (line 24)
        class___192021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 48), self_192020, '__class__')
        # Processing the call keyword arguments (line 24)
        kwargs_192022 = {}
        # Getting the type of 'util' (line 24)
        util_192018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 34), 'util', False)
        # Obtaining the member 'strclass' of a type (line 24)
        strclass_192019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 34), util_192018, 'strclass')
        # Calling strclass(args, kwargs) (line 24)
        strclass_call_result_192023 = invoke(stypy.reporting.localization.Localization(__file__, 24, 34), strclass_192019, *[class___192021], **kwargs_192022)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 34), tuple_192017, strclass_call_result_192023)
        # Adding element type (line 24)
        
        # Call to list(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'self' (line 24)
        self_192025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 70), 'self', False)
        # Processing the call keyword arguments (line 24)
        kwargs_192026 = {}
        # Getting the type of 'list' (line 24)
        list_192024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 65), 'list', False)
        # Calling list(args, kwargs) (line 24)
        list_call_result_192027 = invoke(stypy.reporting.localization.Localization(__file__, 24, 65), list_192024, *[self_192025], **kwargs_192026)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 34), tuple_192017, list_call_result_192027)
        
        # Applying the binary operator '%' (line 24)
        result_mod_192028 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 15), '%', str_192016, tuple_192017)
        
        # Assigning a type to the variable 'stypy_return_type' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'stypy_return_type', result_mod_192028)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_192029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192029)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_192029


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.__eq__')
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Call to isinstance(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'other' (line 27)
        other_192031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'other', False)
        # Getting the type of 'self' (line 27)
        self_192032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'self', False)
        # Obtaining the member '__class__' of a type (line 27)
        class___192033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 33), self_192032, '__class__')
        # Processing the call keyword arguments (line 27)
        kwargs_192034 = {}
        # Getting the type of 'isinstance' (line 27)
        isinstance_192030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 27)
        isinstance_call_result_192035 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), isinstance_192030, *[other_192031, class___192033], **kwargs_192034)
        
        # Applying the 'not' unary operator (line 27)
        result_not__192036 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 11), 'not', isinstance_call_result_192035)
        
        # Testing the type of an if condition (line 27)
        if_condition_192037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 8), result_not__192036)
        # Assigning a type to the variable 'if_condition_192037' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'if_condition_192037', if_condition_192037)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 28)
        NotImplemented_192038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'stypy_return_type', NotImplemented_192038)
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to list(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'self' (line 29)
        self_192040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'self', False)
        # Processing the call keyword arguments (line 29)
        kwargs_192041 = {}
        # Getting the type of 'list' (line 29)
        list_192039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'list', False)
        # Calling list(args, kwargs) (line 29)
        list_call_result_192042 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), list_192039, *[self_192040], **kwargs_192041)
        
        
        # Call to list(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'other' (line 29)
        other_192044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 34), 'other', False)
        # Processing the call keyword arguments (line 29)
        kwargs_192045 = {}
        # Getting the type of 'list' (line 29)
        list_192043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'list', False)
        # Calling list(args, kwargs) (line 29)
        list_call_result_192046 = invoke(stypy.reporting.localization.Localization(__file__, 29, 29), list_192043, *[other_192044], **kwargs_192045)
        
        # Applying the binary operator '==' (line 29)
        result_eq_192047 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 15), '==', list_call_result_192042, list_call_result_192046)
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', result_eq_192047)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_192048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192048)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_192048


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.__ne__')
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.__ne__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 32)
        self_192049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'self')
        # Getting the type of 'other' (line 32)
        other_192050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'other')
        # Applying the binary operator '==' (line 32)
        result_eq_192051 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 19), '==', self_192049, other_192050)
        
        # Applying the 'not' unary operator (line 32)
        result_not__192052 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 15), 'not', result_eq_192051)
        
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'stypy_return_type', result_not__192052)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_192053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192053)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_192053


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.__iter__')
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        
        # Call to iter(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_192055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'self', False)
        # Obtaining the member '_tests' of a type (line 38)
        _tests_192056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 20), self_192055, '_tests')
        # Processing the call keyword arguments (line 38)
        kwargs_192057 = {}
        # Getting the type of 'iter' (line 38)
        iter_192054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'iter', False)
        # Calling iter(args, kwargs) (line 38)
        iter_call_result_192058 = invoke(stypy.reporting.localization.Localization(__file__, 38, 15), iter_192054, *[_tests_192056], **kwargs_192057)
        
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', iter_call_result_192058)
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_192059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_192059


    @norecursion
    def countTestCases(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'countTestCases'
        module_type_store = module_type_store.open_function_context('countTestCases', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.countTestCases')
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_param_names_list', [])
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.countTestCases.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.countTestCases', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Num to a Name (line 41):
        int_192060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'int')
        # Assigning a type to the variable 'cases' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'cases', int_192060)
        
        # Getting the type of 'self' (line 42)
        self_192061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'self')
        # Testing the type of a for loop iterable (line 42)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 8), self_192061)
        # Getting the type of the for loop variable (line 42)
        for_loop_var_192062 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 8), self_192061)
        # Assigning a type to the variable 'test' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'test', for_loop_var_192062)
        # SSA begins for a for statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'cases' (line 43)
        cases_192063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'cases')
        
        # Call to countTestCases(...): (line 43)
        # Processing the call keyword arguments (line 43)
        kwargs_192066 = {}
        # Getting the type of 'test' (line 43)
        test_192064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'test', False)
        # Obtaining the member 'countTestCases' of a type (line 43)
        countTestCases_192065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 21), test_192064, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 43)
        countTestCases_call_result_192067 = invoke(stypy.reporting.localization.Localization(__file__, 43, 21), countTestCases_192065, *[], **kwargs_192066)
        
        # Applying the binary operator '+=' (line 43)
        result_iadd_192068 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 12), '+=', cases_192063, countTestCases_call_result_192067)
        # Assigning a type to the variable 'cases' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'cases', result_iadd_192068)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'cases' (line 44)
        cases_192069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'cases')
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', cases_192069)
        
        # ################# End of 'countTestCases(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'countTestCases' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_192070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192070)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'countTestCases'
        return stypy_return_type_192070


    @norecursion
    def addTest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addTest'
        module_type_store = module_type_store.open_function_context('addTest', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.addTest')
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_param_names_list', ['test'])
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.addTest.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.addTest', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addTest', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addTest(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 48)
        str_192071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'str', '__call__')
        # Getting the type of 'test' (line 48)
        test_192072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'test')
        
        (may_be_192073, more_types_in_union_192074) = may_not_provide_member(str_192071, test_192072)

        if may_be_192073:

            if more_types_in_union_192074:
                # Runtime conditional SSA (line 48)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'test' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'test', remove_member_provider_from_union(test_192072, '__call__'))
            
            # Call to TypeError(...): (line 49)
            # Processing the call arguments (line 49)
            
            # Call to format(...): (line 49)
            # Processing the call arguments (line 49)
            
            # Call to repr(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'test' (line 49)
            test_192079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 61), 'test', False)
            # Processing the call keyword arguments (line 49)
            kwargs_192080 = {}
            # Getting the type of 'repr' (line 49)
            repr_192078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 56), 'repr', False)
            # Calling repr(args, kwargs) (line 49)
            repr_call_result_192081 = invoke(stypy.reporting.localization.Localization(__file__, 49, 56), repr_192078, *[test_192079], **kwargs_192080)
            
            # Processing the call keyword arguments (line 49)
            kwargs_192082 = {}
            str_192076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'str', '{} is not callable')
            # Obtaining the member 'format' of a type (line 49)
            format_192077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 28), str_192076, 'format')
            # Calling format(args, kwargs) (line 49)
            format_call_result_192083 = invoke(stypy.reporting.localization.Localization(__file__, 49, 28), format_192077, *[repr_call_result_192081], **kwargs_192082)
            
            # Processing the call keyword arguments (line 49)
            kwargs_192084 = {}
            # Getting the type of 'TypeError' (line 49)
            TypeError_192075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 49)
            TypeError_call_result_192085 = invoke(stypy.reporting.localization.Localization(__file__, 49, 18), TypeError_192075, *[format_call_result_192083], **kwargs_192084)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 49, 12), TypeError_call_result_192085, 'raise parameter', BaseException)

            if more_types_in_union_192074:
                # SSA join for if statement (line 48)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'test' (line 50)
        test_192087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'test', False)
        # Getting the type of 'type' (line 50)
        type_192088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'type', False)
        # Processing the call keyword arguments (line 50)
        kwargs_192089 = {}
        # Getting the type of 'isinstance' (line 50)
        isinstance_192086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 50)
        isinstance_call_result_192090 = invoke(stypy.reporting.localization.Localization(__file__, 50, 11), isinstance_192086, *[test_192087, type_192088], **kwargs_192089)
        
        
        # Call to issubclass(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'test' (line 50)
        test_192092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 49), 'test', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 51)
        tuple_192093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 51)
        # Adding element type (line 51)
        # Getting the type of 'case' (line 51)
        case_192094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 50), 'case', False)
        # Obtaining the member 'TestCase' of a type (line 51)
        TestCase_192095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 50), case_192094, 'TestCase')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 50), tuple_192093, TestCase_192095)
        # Adding element type (line 51)
        # Getting the type of 'TestSuite' (line 51)
        TestSuite_192096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 65), 'TestSuite', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 50), tuple_192093, TestSuite_192096)
        
        # Processing the call keyword arguments (line 50)
        kwargs_192097 = {}
        # Getting the type of 'issubclass' (line 50)
        issubclass_192091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 38), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 50)
        issubclass_call_result_192098 = invoke(stypy.reporting.localization.Localization(__file__, 50, 38), issubclass_192091, *[test_192092, tuple_192093], **kwargs_192097)
        
        # Applying the binary operator 'and' (line 50)
        result_and_keyword_192099 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), 'and', isinstance_call_result_192090, issubclass_call_result_192098)
        
        # Testing the type of an if condition (line 50)
        if_condition_192100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), result_and_keyword_192099)
        # Assigning a type to the variable 'if_condition_192100' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_192100', if_condition_192100)
        # SSA begins for if statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 52)
        # Processing the call arguments (line 52)
        str_192102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'str', 'TestCases and TestSuites must be instantiated before passing them to addTest()')
        # Processing the call keyword arguments (line 52)
        kwargs_192103 = {}
        # Getting the type of 'TypeError' (line 52)
        TypeError_192101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 52)
        TypeError_call_result_192104 = invoke(stypy.reporting.localization.Localization(__file__, 52, 18), TypeError_192101, *[str_192102], **kwargs_192103)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 52, 12), TypeError_call_result_192104, 'raise parameter', BaseException)
        # SSA join for if statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'test' (line 54)
        test_192108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'test', False)
        # Processing the call keyword arguments (line 54)
        kwargs_192109 = {}
        # Getting the type of 'self' (line 54)
        self_192105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', False)
        # Obtaining the member '_tests' of a type (line 54)
        _tests_192106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_192105, '_tests')
        # Obtaining the member 'append' of a type (line 54)
        append_192107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), _tests_192106, 'append')
        # Calling append(args, kwargs) (line 54)
        append_call_result_192110 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), append_192107, *[test_192108], **kwargs_192109)
        
        
        # ################# End of 'addTest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addTest' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_192111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addTest'
        return stypy_return_type_192111


    @norecursion
    def addTests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addTests'
        module_type_store = module_type_store.open_function_context('addTests', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.addTests')
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_param_names_list', ['tests'])
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.addTests.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.addTests', ['tests'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addTests', localization, ['tests'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addTests(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 57)
        # Getting the type of 'basestring' (line 57)
        basestring_192112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'basestring')
        # Getting the type of 'tests' (line 57)
        tests_192113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 22), 'tests')
        
        (may_be_192114, more_types_in_union_192115) = may_be_subtype(basestring_192112, tests_192113)

        if may_be_192114:

            if more_types_in_union_192115:
                # Runtime conditional SSA (line 57)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'tests' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'tests', remove_not_subtype_from_union(tests_192113, basestring))
            
            # Call to TypeError(...): (line 58)
            # Processing the call arguments (line 58)
            str_192117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'str', 'tests must be an iterable of tests, not a string')
            # Processing the call keyword arguments (line 58)
            kwargs_192118 = {}
            # Getting the type of 'TypeError' (line 58)
            TypeError_192116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 58)
            TypeError_call_result_192119 = invoke(stypy.reporting.localization.Localization(__file__, 58, 18), TypeError_192116, *[str_192117], **kwargs_192118)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 58, 12), TypeError_call_result_192119, 'raise parameter', BaseException)

            if more_types_in_union_192115:
                # SSA join for if statement (line 57)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'tests' (line 59)
        tests_192120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'tests')
        # Testing the type of a for loop iterable (line 59)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 8), tests_192120)
        # Getting the type of the for loop variable (line 59)
        for_loop_var_192121 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 8), tests_192120)
        # Assigning a type to the variable 'test' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'test', for_loop_var_192121)
        # SSA begins for a for statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to addTest(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'test' (line 60)
        test_192124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'test', False)
        # Processing the call keyword arguments (line 60)
        kwargs_192125 = {}
        # Getting the type of 'self' (line 60)
        self_192122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'self', False)
        # Obtaining the member 'addTest' of a type (line 60)
        addTest_192123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), self_192122, 'addTest')
        # Calling addTest(args, kwargs) (line 60)
        addTest_call_result_192126 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), addTest_192123, *[test_192124], **kwargs_192125)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'addTests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addTests' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_192127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192127)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addTests'
        return stypy_return_type_192127


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.run.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.run.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.run')
        BaseTestSuite.run.__dict__.__setitem__('stypy_param_names_list', ['result'])
        BaseTestSuite.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTestSuite.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTestSuite.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.run.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.run', ['result'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 63)
        self_192128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'self')
        # Testing the type of a for loop iterable (line 63)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 8), self_192128)
        # Getting the type of the for loop variable (line 63)
        for_loop_var_192129 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 8), self_192128)
        # Assigning a type to the variable 'test' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'test', for_loop_var_192129)
        # SSA begins for a for statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'result' (line 64)
        result_192130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'result')
        # Obtaining the member 'shouldStop' of a type (line 64)
        shouldStop_192131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 15), result_192130, 'shouldStop')
        # Testing the type of an if condition (line 64)
        if_condition_192132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 12), shouldStop_192131)
        # Assigning a type to the variable 'if_condition_192132' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'if_condition_192132', if_condition_192132)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to test(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'result' (line 66)
        result_192134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'result', False)
        # Processing the call keyword arguments (line 66)
        kwargs_192135 = {}
        # Getting the type of 'test' (line 66)
        test_192133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'test', False)
        # Calling test(args, kwargs) (line 66)
        test_call_result_192136 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), test_192133, *[result_192134], **kwargs_192135)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 67)
        result_192137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', result_192137)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_192138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192138)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_192138


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.__call__')
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwds')
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.__call__', [], 'args', 'kwds', defaults, varargs, kwargs)

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

        
        # Call to run(...): (line 70)
        # Getting the type of 'args' (line 70)
        args_192141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'args', False)
        # Processing the call keyword arguments (line 70)
        # Getting the type of 'kwds' (line 70)
        kwds_192142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 33), 'kwds', False)
        kwargs_192143 = {'kwds_192142': kwds_192142}
        # Getting the type of 'self' (line 70)
        self_192139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'self', False)
        # Obtaining the member 'run' of a type (line 70)
        run_192140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), self_192139, 'run')
        # Calling run(args, kwargs) (line 70)
        run_call_result_192144 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), run_192140, *[args_192141], **kwargs_192143)
        
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', run_call_result_192144)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_192145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192145)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_192145


    @norecursion
    def debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'debug'
        module_type_store = module_type_store.open_function_context('debug', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTestSuite.debug.__dict__.__setitem__('stypy_localization', localization)
        BaseTestSuite.debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTestSuite.debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTestSuite.debug.__dict__.__setitem__('stypy_function_name', 'BaseTestSuite.debug')
        BaseTestSuite.debug.__dict__.__setitem__('stypy_param_names_list', [])
        BaseTestSuite.debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTestSuite.debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTestSuite.debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTestSuite.debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTestSuite.debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTestSuite.debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTestSuite.debug', [], None, None, defaults, varargs, kwargs)

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

        str_192146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'str', 'Run the tests without collecting errors in a TestResult')
        
        # Getting the type of 'self' (line 74)
        self_192147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'self')
        # Testing the type of a for loop iterable (line 74)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 8), self_192147)
        # Getting the type of the for loop variable (line 74)
        for_loop_var_192148 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 8), self_192147)
        # Assigning a type to the variable 'test' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'test', for_loop_var_192148)
        # SSA begins for a for statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to debug(...): (line 75)
        # Processing the call keyword arguments (line 75)
        kwargs_192151 = {}
        # Getting the type of 'test' (line 75)
        test_192149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'test', False)
        # Obtaining the member 'debug' of a type (line 75)
        debug_192150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), test_192149, 'debug')
        # Calling debug(args, kwargs) (line 75)
        debug_call_result_192152 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), debug_192150, *[], **kwargs_192151)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'debug' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_192153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192153)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'debug'
        return stypy_return_type_192153


# Assigning a type to the variable 'BaseTestSuite' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'BaseTestSuite', BaseTestSuite)

# Assigning a Name to a Name (line 35):
# Getting the type of 'None' (line 35)
None_192154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'None')
# Getting the type of 'BaseTestSuite'
BaseTestSuite_192155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BaseTestSuite')
# Setting the type of the member '__hash__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BaseTestSuite_192155, '__hash__', None_192154)
# Declaration of the 'TestSuite' class
# Getting the type of 'BaseTestSuite' (line 78)
BaseTestSuite_192156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'BaseTestSuite')

class TestSuite(BaseTestSuite_192156, ):
    str_192157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', 'A test suite is a composite test consisting of a number of TestCases.\n\n    For use, create an instance of TestSuite, then add test case instances.\n    When all tests have been added, the suite can be passed to a test\n    runner, such as TextTestRunner. It will run the individual test cases\n    in the order in which they were added, aggregating the results. When\n    subclassing, do not forget to call the base class constructor.\n    ')

    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 88)
        False_192158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'False')
        defaults = [False_192158]
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSuite.run.__dict__.__setitem__('stypy_localization', localization)
        TestSuite.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSuite.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSuite.run.__dict__.__setitem__('stypy_function_name', 'TestSuite.run')
        TestSuite.run.__dict__.__setitem__('stypy_param_names_list', ['result', 'debug'])
        TestSuite.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSuite.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSuite.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSuite.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSuite.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSuite.run.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSuite.run', ['result', 'debug'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, ['result', 'debug'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'False' (line 89)
        False_192159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'False')
        # Assigning a type to the variable 'topLevel' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'topLevel', False_192159)
        
        
        
        # Call to getattr(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'result' (line 90)
        result_192161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'result', False)
        str_192162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 27), 'str', '_testRunEntered')
        # Getting the type of 'False' (line 90)
        False_192163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 46), 'False', False)
        # Processing the call keyword arguments (line 90)
        kwargs_192164 = {}
        # Getting the type of 'getattr' (line 90)
        getattr_192160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'getattr', False)
        # Calling getattr(args, kwargs) (line 90)
        getattr_call_result_192165 = invoke(stypy.reporting.localization.Localization(__file__, 90, 11), getattr_192160, *[result_192161, str_192162, False_192163], **kwargs_192164)
        
        # Getting the type of 'False' (line 90)
        False_192166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 56), 'False')
        # Applying the binary operator 'is' (line 90)
        result_is__192167 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 11), 'is', getattr_call_result_192165, False_192166)
        
        # Testing the type of an if condition (line 90)
        if_condition_192168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 8), result_is__192167)
        # Assigning a type to the variable 'if_condition_192168' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'if_condition_192168', if_condition_192168)
        # SSA begins for if statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Multiple assignment of 2 elements.
        # Getting the type of 'True' (line 91)
        True_192169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'True')
        # Assigning a type to the variable 'topLevel' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'topLevel', True_192169)
        # Getting the type of 'topLevel' (line 91)
        topLevel_192170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'topLevel')
        # Getting the type of 'result' (line 91)
        result_192171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'result')
        # Setting the type of the member '_testRunEntered' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), result_192171, '_testRunEntered', topLevel_192170)
        # SSA join for if statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 93)
        self_192172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'self')
        # Testing the type of a for loop iterable (line 93)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 8), self_192172)
        # Getting the type of the for loop variable (line 93)
        for_loop_var_192173 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 8), self_192172)
        # Assigning a type to the variable 'test' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'test', for_loop_var_192173)
        # SSA begins for a for statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'result' (line 94)
        result_192174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'result')
        # Obtaining the member 'shouldStop' of a type (line 94)
        shouldStop_192175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), result_192174, 'shouldStop')
        # Testing the type of an if condition (line 94)
        if_condition_192176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 12), shouldStop_192175)
        # Assigning a type to the variable 'if_condition_192176' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'if_condition_192176', if_condition_192176)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to _isnotsuite(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'test' (line 97)
        test_192178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'test', False)
        # Processing the call keyword arguments (line 97)
        kwargs_192179 = {}
        # Getting the type of '_isnotsuite' (line 97)
        _isnotsuite_192177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), '_isnotsuite', False)
        # Calling _isnotsuite(args, kwargs) (line 97)
        _isnotsuite_call_result_192180 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), _isnotsuite_192177, *[test_192178], **kwargs_192179)
        
        # Testing the type of an if condition (line 97)
        if_condition_192181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 12), _isnotsuite_call_result_192180)
        # Assigning a type to the variable 'if_condition_192181' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'if_condition_192181', if_condition_192181)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _tearDownPreviousClass(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'test' (line 98)
        test_192184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 44), 'test', False)
        # Getting the type of 'result' (line 98)
        result_192185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 50), 'result', False)
        # Processing the call keyword arguments (line 98)
        kwargs_192186 = {}
        # Getting the type of 'self' (line 98)
        self_192182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'self', False)
        # Obtaining the member '_tearDownPreviousClass' of a type (line 98)
        _tearDownPreviousClass_192183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), self_192182, '_tearDownPreviousClass')
        # Calling _tearDownPreviousClass(args, kwargs) (line 98)
        _tearDownPreviousClass_call_result_192187 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), _tearDownPreviousClass_192183, *[test_192184, result_192185], **kwargs_192186)
        
        
        # Call to _handleModuleFixture(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'test' (line 99)
        test_192190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 42), 'test', False)
        # Getting the type of 'result' (line 99)
        result_192191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 48), 'result', False)
        # Processing the call keyword arguments (line 99)
        kwargs_192192 = {}
        # Getting the type of 'self' (line 99)
        self_192188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'self', False)
        # Obtaining the member '_handleModuleFixture' of a type (line 99)
        _handleModuleFixture_192189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), self_192188, '_handleModuleFixture')
        # Calling _handleModuleFixture(args, kwargs) (line 99)
        _handleModuleFixture_call_result_192193 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), _handleModuleFixture_192189, *[test_192190, result_192191], **kwargs_192192)
        
        
        # Call to _handleClassSetUp(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'test' (line 100)
        test_192196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'test', False)
        # Getting the type of 'result' (line 100)
        result_192197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 45), 'result', False)
        # Processing the call keyword arguments (line 100)
        kwargs_192198 = {}
        # Getting the type of 'self' (line 100)
        self_192194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'self', False)
        # Obtaining the member '_handleClassSetUp' of a type (line 100)
        _handleClassSetUp_192195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 16), self_192194, '_handleClassSetUp')
        # Calling _handleClassSetUp(args, kwargs) (line 100)
        _handleClassSetUp_call_result_192199 = invoke(stypy.reporting.localization.Localization(__file__, 100, 16), _handleClassSetUp_192195, *[test_192196, result_192197], **kwargs_192198)
        
        
        # Assigning a Attribute to a Attribute (line 101):
        # Getting the type of 'test' (line 101)
        test_192200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'test')
        # Obtaining the member '__class__' of a type (line 101)
        class___192201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 44), test_192200, '__class__')
        # Getting the type of 'result' (line 101)
        result_192202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'result')
        # Setting the type of the member '_previousTestClass' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 16), result_192202, '_previousTestClass', class___192201)
        
        
        # Evaluating a boolean operation
        
        # Call to getattr(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'test' (line 103)
        test_192204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'test', False)
        # Obtaining the member '__class__' of a type (line 103)
        class___192205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 28), test_192204, '__class__')
        str_192206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 44), 'str', '_classSetupFailed')
        # Getting the type of 'False' (line 103)
        False_192207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 65), 'False', False)
        # Processing the call keyword arguments (line 103)
        kwargs_192208 = {}
        # Getting the type of 'getattr' (line 103)
        getattr_192203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 103)
        getattr_call_result_192209 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), getattr_192203, *[class___192205, str_192206, False_192207], **kwargs_192208)
        
        
        # Call to getattr(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'result' (line 104)
        result_192211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'result', False)
        str_192212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'str', '_moduleSetUpFailed')
        # Getting the type of 'False' (line 104)
        False_192213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 58), 'False', False)
        # Processing the call keyword arguments (line 104)
        kwargs_192214 = {}
        # Getting the type of 'getattr' (line 104)
        getattr_192210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 104)
        getattr_call_result_192215 = invoke(stypy.reporting.localization.Localization(__file__, 104, 20), getattr_192210, *[result_192211, str_192212, False_192213], **kwargs_192214)
        
        # Applying the binary operator 'or' (line 103)
        result_or_keyword_192216 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 20), 'or', getattr_call_result_192209, getattr_call_result_192215)
        
        # Testing the type of an if condition (line 103)
        if_condition_192217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 16), result_or_keyword_192216)
        # Assigning a type to the variable 'if_condition_192217' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'if_condition_192217', if_condition_192217)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'debug' (line 107)
        debug_192218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'debug')
        # Applying the 'not' unary operator (line 107)
        result_not__192219 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), 'not', debug_192218)
        
        # Testing the type of an if condition (line 107)
        if_condition_192220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 12), result_not__192219)
        # Assigning a type to the variable 'if_condition_192220' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'if_condition_192220', if_condition_192220)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to test(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'result' (line 108)
        result_192222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'result', False)
        # Processing the call keyword arguments (line 108)
        kwargs_192223 = {}
        # Getting the type of 'test' (line 108)
        test_192221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'test', False)
        # Calling test(args, kwargs) (line 108)
        test_call_result_192224 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), test_192221, *[result_192222], **kwargs_192223)
        
        # SSA branch for the else part of an if statement (line 107)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_192227 = {}
        # Getting the type of 'test' (line 110)
        test_192225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'test', False)
        # Obtaining the member 'debug' of a type (line 110)
        debug_192226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), test_192225, 'debug')
        # Calling debug(args, kwargs) (line 110)
        debug_call_result_192228 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), debug_192226, *[], **kwargs_192227)
        
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'topLevel' (line 112)
        topLevel_192229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'topLevel')
        # Testing the type of an if condition (line 112)
        if_condition_192230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), topLevel_192229)
        # Assigning a type to the variable 'if_condition_192230' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_192230', if_condition_192230)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _tearDownPreviousClass(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'None' (line 113)
        None_192233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 40), 'None', False)
        # Getting the type of 'result' (line 113)
        result_192234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 46), 'result', False)
        # Processing the call keyword arguments (line 113)
        kwargs_192235 = {}
        # Getting the type of 'self' (line 113)
        self_192231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self', False)
        # Obtaining the member '_tearDownPreviousClass' of a type (line 113)
        _tearDownPreviousClass_192232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_192231, '_tearDownPreviousClass')
        # Calling _tearDownPreviousClass(args, kwargs) (line 113)
        _tearDownPreviousClass_call_result_192236 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), _tearDownPreviousClass_192232, *[None_192233, result_192234], **kwargs_192235)
        
        
        # Call to _handleModuleTearDown(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'result' (line 114)
        result_192239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 39), 'result', False)
        # Processing the call keyword arguments (line 114)
        kwargs_192240 = {}
        # Getting the type of 'self' (line 114)
        self_192237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'self', False)
        # Obtaining the member '_handleModuleTearDown' of a type (line 114)
        _handleModuleTearDown_192238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), self_192237, '_handleModuleTearDown')
        # Calling _handleModuleTearDown(args, kwargs) (line 114)
        _handleModuleTearDown_call_result_192241 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), _handleModuleTearDown_192238, *[result_192239], **kwargs_192240)
        
        
        # Assigning a Name to a Attribute (line 115):
        # Getting the type of 'False' (line 115)
        False_192242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'False')
        # Getting the type of 'result' (line 115)
        result_192243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'result')
        # Setting the type of the member '_testRunEntered' of a type (line 115)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), result_192243, '_testRunEntered', False_192242)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 116)
        result_192244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type', result_192244)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_192245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192245)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_192245


    @norecursion
    def debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'debug'
        module_type_store = module_type_store.open_function_context('debug', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSuite.debug.__dict__.__setitem__('stypy_localization', localization)
        TestSuite.debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSuite.debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSuite.debug.__dict__.__setitem__('stypy_function_name', 'TestSuite.debug')
        TestSuite.debug.__dict__.__setitem__('stypy_param_names_list', [])
        TestSuite.debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSuite.debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSuite.debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSuite.debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSuite.debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSuite.debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSuite.debug', [], None, None, defaults, varargs, kwargs)

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

        str_192246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'str', 'Run the tests without collecting errors in a TestResult')
        
        # Assigning a Call to a Name (line 120):
        
        # Call to _DebugResult(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_192248 = {}
        # Getting the type of '_DebugResult' (line 120)
        _DebugResult_192247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), '_DebugResult', False)
        # Calling _DebugResult(args, kwargs) (line 120)
        _DebugResult_call_result_192249 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), _DebugResult_192247, *[], **kwargs_192248)
        
        # Assigning a type to the variable 'debug' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'debug', _DebugResult_call_result_192249)
        
        # Call to run(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'debug' (line 121)
        debug_192252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'debug', False)
        # Getting the type of 'True' (line 121)
        True_192253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'True', False)
        # Processing the call keyword arguments (line 121)
        kwargs_192254 = {}
        # Getting the type of 'self' (line 121)
        self_192250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self', False)
        # Obtaining the member 'run' of a type (line 121)
        run_192251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_192250, 'run')
        # Calling run(args, kwargs) (line 121)
        run_call_result_192255 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), run_192251, *[debug_192252, True_192253], **kwargs_192254)
        
        
        # ################# End of 'debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'debug' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_192256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'debug'
        return stypy_return_type_192256


    @norecursion
    def _handleClassSetUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handleClassSetUp'
        module_type_store = module_type_store.open_function_context('_handleClassSetUp', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_localization', localization)
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_function_name', 'TestSuite._handleClassSetUp')
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_param_names_list', ['test', 'result'])
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSuite._handleClassSetUp.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSuite._handleClassSetUp', ['test', 'result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handleClassSetUp', localization, ['test', 'result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handleClassSetUp(...)' code ##################

        
        # Assigning a Call to a Name (line 126):
        
        # Call to getattr(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'result' (line 126)
        result_192258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'result', False)
        str_192259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 40), 'str', '_previousTestClass')
        # Getting the type of 'None' (line 126)
        None_192260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 62), 'None', False)
        # Processing the call keyword arguments (line 126)
        kwargs_192261 = {}
        # Getting the type of 'getattr' (line 126)
        getattr_192257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 126)
        getattr_call_result_192262 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), getattr_192257, *[result_192258, str_192259, None_192260], **kwargs_192261)
        
        # Assigning a type to the variable 'previousClass' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'previousClass', getattr_call_result_192262)
        
        # Assigning a Attribute to a Name (line 127):
        # Getting the type of 'test' (line 127)
        test_192263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'test')
        # Obtaining the member '__class__' of a type (line 127)
        class___192264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 23), test_192263, '__class__')
        # Assigning a type to the variable 'currentClass' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'currentClass', class___192264)
        
        
        # Getting the type of 'currentClass' (line 128)
        currentClass_192265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'currentClass')
        # Getting the type of 'previousClass' (line 128)
        previousClass_192266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'previousClass')
        # Applying the binary operator '==' (line 128)
        result_eq_192267 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 11), '==', currentClass_192265, previousClass_192266)
        
        # Testing the type of an if condition (line 128)
        if_condition_192268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 8), result_eq_192267)
        # Assigning a type to the variable 'if_condition_192268' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'if_condition_192268', if_condition_192268)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'result' (line 130)
        result_192269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'result')
        # Obtaining the member '_moduleSetUpFailed' of a type (line 130)
        _moduleSetUpFailed_192270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 11), result_192269, '_moduleSetUpFailed')
        # Testing the type of an if condition (line 130)
        if_condition_192271 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), _moduleSetUpFailed_192270)
        # Assigning a type to the variable 'if_condition_192271' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_192271', if_condition_192271)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to getattr(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'currentClass' (line 132)
        currentClass_192273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'currentClass', False)
        str_192274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 33), 'str', '__unittest_skip__')
        # Getting the type of 'False' (line 132)
        False_192275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 54), 'False', False)
        # Processing the call keyword arguments (line 132)
        kwargs_192276 = {}
        # Getting the type of 'getattr' (line 132)
        getattr_192272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'getattr', False)
        # Calling getattr(args, kwargs) (line 132)
        getattr_call_result_192277 = invoke(stypy.reporting.localization.Localization(__file__, 132, 11), getattr_192272, *[currentClass_192273, str_192274, False_192275], **kwargs_192276)
        
        # Testing the type of an if condition (line 132)
        if_condition_192278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 8), getattr_call_result_192277)
        # Assigning a type to the variable 'if_condition_192278' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'if_condition_192278', if_condition_192278)
        # SSA begins for if statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 132)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Attribute (line 136):
        # Getting the type of 'False' (line 136)
        False_192279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 45), 'False')
        # Getting the type of 'currentClass' (line 136)
        currentClass_192280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'currentClass')
        # Setting the type of the member '_classSetupFailed' of a type (line 136)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 12), currentClass_192280, '_classSetupFailed', False_192279)
        # SSA branch for the except part of a try statement (line 135)
        # SSA branch for the except 'TypeError' branch of a try statement (line 135)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 135)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 142):
        
        # Call to getattr(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'currentClass' (line 142)
        currentClass_192282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'currentClass', False)
        str_192283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 43), 'str', 'setUpClass')
        # Getting the type of 'None' (line 142)
        None_192284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 57), 'None', False)
        # Processing the call keyword arguments (line 142)
        kwargs_192285 = {}
        # Getting the type of 'getattr' (line 142)
        getattr_192281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'getattr', False)
        # Calling getattr(args, kwargs) (line 142)
        getattr_call_result_192286 = invoke(stypy.reporting.localization.Localization(__file__, 142, 21), getattr_192281, *[currentClass_192282, str_192283, None_192284], **kwargs_192285)
        
        # Assigning a type to the variable 'setUpClass' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'setUpClass', getattr_call_result_192286)
        
        # Type idiom detected: calculating its left and rigth part (line 143)
        # Getting the type of 'setUpClass' (line 143)
        setUpClass_192287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'setUpClass')
        # Getting the type of 'None' (line 143)
        None_192288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'None')
        
        (may_be_192289, more_types_in_union_192290) = may_not_be_none(setUpClass_192287, None_192288)

        if may_be_192289:

            if more_types_in_union_192290:
                # Runtime conditional SSA (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _call_if_exists(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'result' (line 144)
            result_192292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'result', False)
            str_192293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 36), 'str', '_setupStdout')
            # Processing the call keyword arguments (line 144)
            kwargs_192294 = {}
            # Getting the type of '_call_if_exists' (line 144)
            _call_if_exists_192291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), '_call_if_exists', False)
            # Calling _call_if_exists(args, kwargs) (line 144)
            _call_if_exists_call_result_192295 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), _call_if_exists_192291, *[result_192292, str_192293], **kwargs_192294)
            
            
            # Try-finally block (line 145)
            
            
            # SSA begins for try-except statement (line 145)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to setUpClass(...): (line 146)
            # Processing the call keyword arguments (line 146)
            kwargs_192297 = {}
            # Getting the type of 'setUpClass' (line 146)
            setUpClass_192296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'setUpClass', False)
            # Calling setUpClass(args, kwargs) (line 146)
            setUpClass_call_result_192298 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), setUpClass_192296, *[], **kwargs_192297)
            
            # SSA branch for the except part of a try statement (line 145)
            # SSA branch for the except 'Exception' branch of a try statement (line 145)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'Exception' (line 147)
            Exception_192299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'Exception')
            # Assigning a type to the variable 'e' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'e', Exception_192299)
            
            
            # Call to isinstance(...): (line 148)
            # Processing the call arguments (line 148)
            # Getting the type of 'result' (line 148)
            result_192301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'result', False)
            # Getting the type of '_DebugResult' (line 148)
            _DebugResult_192302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 38), '_DebugResult', False)
            # Processing the call keyword arguments (line 148)
            kwargs_192303 = {}
            # Getting the type of 'isinstance' (line 148)
            isinstance_192300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 148)
            isinstance_call_result_192304 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), isinstance_192300, *[result_192301, _DebugResult_192302], **kwargs_192303)
            
            # Testing the type of an if condition (line 148)
            if_condition_192305 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 16), isinstance_call_result_192304)
            # Assigning a type to the variable 'if_condition_192305' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'if_condition_192305', if_condition_192305)
            # SSA begins for if statement (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 148)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 150):
            # Getting the type of 'True' (line 150)
            True_192306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 49), 'True')
            # Getting the type of 'currentClass' (line 150)
            currentClass_192307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'currentClass')
            # Setting the type of the member '_classSetupFailed' of a type (line 150)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), currentClass_192307, '_classSetupFailed', True_192306)
            
            # Assigning a Call to a Name (line 151):
            
            # Call to strclass(...): (line 151)
            # Processing the call arguments (line 151)
            # Getting the type of 'currentClass' (line 151)
            currentClass_192310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 42), 'currentClass', False)
            # Processing the call keyword arguments (line 151)
            kwargs_192311 = {}
            # Getting the type of 'util' (line 151)
            util_192308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), 'util', False)
            # Obtaining the member 'strclass' of a type (line 151)
            strclass_192309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 28), util_192308, 'strclass')
            # Calling strclass(args, kwargs) (line 151)
            strclass_call_result_192312 = invoke(stypy.reporting.localization.Localization(__file__, 151, 28), strclass_192309, *[currentClass_192310], **kwargs_192311)
            
            # Assigning a type to the variable 'className' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'className', strclass_call_result_192312)
            
            # Assigning a BinOp to a Name (line 152):
            str_192313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'str', 'setUpClass (%s)')
            # Getting the type of 'className' (line 152)
            className_192314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 48), 'className')
            # Applying the binary operator '%' (line 152)
            result_mod_192315 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 28), '%', str_192313, className_192314)
            
            # Assigning a type to the variable 'errorName' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'errorName', result_mod_192315)
            
            # Call to _addClassOrModuleLevelException(...): (line 153)
            # Processing the call arguments (line 153)
            # Getting the type of 'result' (line 153)
            result_192318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 53), 'result', False)
            # Getting the type of 'e' (line 153)
            e_192319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 61), 'e', False)
            # Getting the type of 'errorName' (line 153)
            errorName_192320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 64), 'errorName', False)
            # Processing the call keyword arguments (line 153)
            kwargs_192321 = {}
            # Getting the type of 'self' (line 153)
            self_192316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'self', False)
            # Obtaining the member '_addClassOrModuleLevelException' of a type (line 153)
            _addClassOrModuleLevelException_192317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), self_192316, '_addClassOrModuleLevelException')
            # Calling _addClassOrModuleLevelException(args, kwargs) (line 153)
            _addClassOrModuleLevelException_call_result_192322 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), _addClassOrModuleLevelException_192317, *[result_192318, e_192319, errorName_192320], **kwargs_192321)
            
            # SSA join for try-except statement (line 145)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # finally branch of the try-finally block (line 145)
            
            # Call to _call_if_exists(...): (line 155)
            # Processing the call arguments (line 155)
            # Getting the type of 'result' (line 155)
            result_192324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'result', False)
            str_192325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 40), 'str', '_restoreStdout')
            # Processing the call keyword arguments (line 155)
            kwargs_192326 = {}
            # Getting the type of '_call_if_exists' (line 155)
            _call_if_exists_192323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), '_call_if_exists', False)
            # Calling _call_if_exists(args, kwargs) (line 155)
            _call_if_exists_call_result_192327 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), _call_if_exists_192323, *[result_192324, str_192325], **kwargs_192326)
            
            

            if more_types_in_union_192290:
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_handleClassSetUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handleClassSetUp' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_192328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handleClassSetUp'
        return stypy_return_type_192328


    @norecursion
    def _get_previous_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_previous_module'
        module_type_store = module_type_store.open_function_context('_get_previous_module', 157, 4, False)
        # Assigning a type to the variable 'self' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_localization', localization)
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_function_name', 'TestSuite._get_previous_module')
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_param_names_list', ['result'])
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSuite._get_previous_module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSuite._get_previous_module', ['result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_previous_module', localization, ['result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_previous_module(...)' code ##################

        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'None' (line 158)
        None_192329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 25), 'None')
        # Assigning a type to the variable 'previousModule' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'previousModule', None_192329)
        
        # Assigning a Call to a Name (line 159):
        
        # Call to getattr(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'result' (line 159)
        result_192331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'result', False)
        str_192332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 40), 'str', '_previousTestClass')
        # Getting the type of 'None' (line 159)
        None_192333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 62), 'None', False)
        # Processing the call keyword arguments (line 159)
        kwargs_192334 = {}
        # Getting the type of 'getattr' (line 159)
        getattr_192330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 159)
        getattr_call_result_192335 = invoke(stypy.reporting.localization.Localization(__file__, 159, 24), getattr_192330, *[result_192331, str_192332, None_192333], **kwargs_192334)
        
        # Assigning a type to the variable 'previousClass' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'previousClass', getattr_call_result_192335)
        
        # Type idiom detected: calculating its left and rigth part (line 160)
        # Getting the type of 'previousClass' (line 160)
        previousClass_192336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'previousClass')
        # Getting the type of 'None' (line 160)
        None_192337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'None')
        
        (may_be_192338, more_types_in_union_192339) = may_not_be_none(previousClass_192336, None_192337)

        if may_be_192338:

            if more_types_in_union_192339:
                # Runtime conditional SSA (line 160)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 161):
            # Getting the type of 'previousClass' (line 161)
            previousClass_192340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'previousClass')
            # Obtaining the member '__module__' of a type (line 161)
            module___192341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 29), previousClass_192340, '__module__')
            # Assigning a type to the variable 'previousModule' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'previousModule', module___192341)

            if more_types_in_union_192339:
                # SSA join for if statement (line 160)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'previousModule' (line 162)
        previousModule_192342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'previousModule')
        # Assigning a type to the variable 'stypy_return_type' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'stypy_return_type', previousModule_192342)
        
        # ################# End of '_get_previous_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_previous_module' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_192343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192343)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_previous_module'
        return stypy_return_type_192343


    @norecursion
    def _handleModuleFixture(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handleModuleFixture'
        module_type_store = module_type_store.open_function_context('_handleModuleFixture', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_localization', localization)
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_function_name', 'TestSuite._handleModuleFixture')
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_param_names_list', ['test', 'result'])
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSuite._handleModuleFixture.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSuite._handleModuleFixture', ['test', 'result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handleModuleFixture', localization, ['test', 'result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handleModuleFixture(...)' code ##################

        
        # Assigning a Call to a Name (line 166):
        
        # Call to _get_previous_module(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'result' (line 166)
        result_192346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 51), 'result', False)
        # Processing the call keyword arguments (line 166)
        kwargs_192347 = {}
        # Getting the type of 'self' (line 166)
        self_192344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 25), 'self', False)
        # Obtaining the member '_get_previous_module' of a type (line 166)
        _get_previous_module_192345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), self_192344, '_get_previous_module')
        # Calling _get_previous_module(args, kwargs) (line 166)
        _get_previous_module_call_result_192348 = invoke(stypy.reporting.localization.Localization(__file__, 166, 25), _get_previous_module_192345, *[result_192346], **kwargs_192347)
        
        # Assigning a type to the variable 'previousModule' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'previousModule', _get_previous_module_call_result_192348)
        
        # Assigning a Attribute to a Name (line 167):
        # Getting the type of 'test' (line 167)
        test_192349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'test')
        # Obtaining the member '__class__' of a type (line 167)
        class___192350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 24), test_192349, '__class__')
        # Obtaining the member '__module__' of a type (line 167)
        module___192351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 24), class___192350, '__module__')
        # Assigning a type to the variable 'currentModule' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'currentModule', module___192351)
        
        
        # Getting the type of 'currentModule' (line 168)
        currentModule_192352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'currentModule')
        # Getting the type of 'previousModule' (line 168)
        previousModule_192353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'previousModule')
        # Applying the binary operator '==' (line 168)
        result_eq_192354 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), '==', currentModule_192352, previousModule_192353)
        
        # Testing the type of an if condition (line 168)
        if_condition_192355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 8), result_eq_192354)
        # Assigning a type to the variable 'if_condition_192355' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'if_condition_192355', if_condition_192355)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _handleModuleTearDown(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'result' (line 171)
        result_192358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'result', False)
        # Processing the call keyword arguments (line 171)
        kwargs_192359 = {}
        # Getting the type of 'self' (line 171)
        self_192356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self', False)
        # Obtaining the member '_handleModuleTearDown' of a type (line 171)
        _handleModuleTearDown_192357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_192356, '_handleModuleTearDown')
        # Calling _handleModuleTearDown(args, kwargs) (line 171)
        _handleModuleTearDown_call_result_192360 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), _handleModuleTearDown_192357, *[result_192358], **kwargs_192359)
        
        
        # Assigning a Name to a Attribute (line 173):
        # Getting the type of 'False' (line 173)
        False_192361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 36), 'False')
        # Getting the type of 'result' (line 173)
        result_192362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'result')
        # Setting the type of the member '_moduleSetUpFailed' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), result_192362, '_moduleSetUpFailed', False_192361)
        
        
        # SSA begins for try-except statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 175):
        
        # Obtaining the type of the subscript
        # Getting the type of 'currentModule' (line 175)
        currentModule_192363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 33), 'currentModule')
        # Getting the type of 'sys' (line 175)
        sys_192364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'sys')
        # Obtaining the member 'modules' of a type (line 175)
        modules_192365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 21), sys_192364, 'modules')
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___192366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 21), modules_192365, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_192367 = invoke(stypy.reporting.localization.Localization(__file__, 175, 21), getitem___192366, currentModule_192363)
        
        # Assigning a type to the variable 'module' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'module', subscript_call_result_192367)
        # SSA branch for the except part of a try statement (line 174)
        # SSA branch for the except 'KeyError' branch of a try statement (line 174)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 178):
        
        # Call to getattr(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'module' (line 178)
        module_192369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 30), 'module', False)
        str_192370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 38), 'str', 'setUpModule')
        # Getting the type of 'None' (line 178)
        None_192371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 53), 'None', False)
        # Processing the call keyword arguments (line 178)
        kwargs_192372 = {}
        # Getting the type of 'getattr' (line 178)
        getattr_192368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'getattr', False)
        # Calling getattr(args, kwargs) (line 178)
        getattr_call_result_192373 = invoke(stypy.reporting.localization.Localization(__file__, 178, 22), getattr_192368, *[module_192369, str_192370, None_192371], **kwargs_192372)
        
        # Assigning a type to the variable 'setUpModule' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'setUpModule', getattr_call_result_192373)
        
        # Type idiom detected: calculating its left and rigth part (line 179)
        # Getting the type of 'setUpModule' (line 179)
        setUpModule_192374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'setUpModule')
        # Getting the type of 'None' (line 179)
        None_192375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 30), 'None')
        
        (may_be_192376, more_types_in_union_192377) = may_not_be_none(setUpModule_192374, None_192375)

        if may_be_192376:

            if more_types_in_union_192377:
                # Runtime conditional SSA (line 179)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _call_if_exists(...): (line 180)
            # Processing the call arguments (line 180)
            # Getting the type of 'result' (line 180)
            result_192379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'result', False)
            str_192380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 36), 'str', '_setupStdout')
            # Processing the call keyword arguments (line 180)
            kwargs_192381 = {}
            # Getting the type of '_call_if_exists' (line 180)
            _call_if_exists_192378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), '_call_if_exists', False)
            # Calling _call_if_exists(args, kwargs) (line 180)
            _call_if_exists_call_result_192382 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), _call_if_exists_192378, *[result_192379, str_192380], **kwargs_192381)
            
            
            # Try-finally block (line 181)
            
            
            # SSA begins for try-except statement (line 181)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to setUpModule(...): (line 182)
            # Processing the call keyword arguments (line 182)
            kwargs_192384 = {}
            # Getting the type of 'setUpModule' (line 182)
            setUpModule_192383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'setUpModule', False)
            # Calling setUpModule(args, kwargs) (line 182)
            setUpModule_call_result_192385 = invoke(stypy.reporting.localization.Localization(__file__, 182, 16), setUpModule_192383, *[], **kwargs_192384)
            
            # SSA branch for the except part of a try statement (line 181)
            # SSA branch for the except 'Exception' branch of a try statement (line 181)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'Exception' (line 183)
            Exception_192386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'Exception')
            # Assigning a type to the variable 'e' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'e', Exception_192386)
            
            
            # Call to isinstance(...): (line 184)
            # Processing the call arguments (line 184)
            # Getting the type of 'result' (line 184)
            result_192388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 30), 'result', False)
            # Getting the type of '_DebugResult' (line 184)
            _DebugResult_192389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 38), '_DebugResult', False)
            # Processing the call keyword arguments (line 184)
            kwargs_192390 = {}
            # Getting the type of 'isinstance' (line 184)
            isinstance_192387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 184)
            isinstance_call_result_192391 = invoke(stypy.reporting.localization.Localization(__file__, 184, 19), isinstance_192387, *[result_192388, _DebugResult_192389], **kwargs_192390)
            
            # Testing the type of an if condition (line 184)
            if_condition_192392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 16), isinstance_call_result_192391)
            # Assigning a type to the variable 'if_condition_192392' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'if_condition_192392', if_condition_192392)
            # SSA begins for if statement (line 184)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 184)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 186):
            # Getting the type of 'True' (line 186)
            True_192393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 44), 'True')
            # Getting the type of 'result' (line 186)
            result_192394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'result')
            # Setting the type of the member '_moduleSetUpFailed' of a type (line 186)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 16), result_192394, '_moduleSetUpFailed', True_192393)
            
            # Assigning a BinOp to a Name (line 187):
            str_192395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 28), 'str', 'setUpModule (%s)')
            # Getting the type of 'currentModule' (line 187)
            currentModule_192396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 49), 'currentModule')
            # Applying the binary operator '%' (line 187)
            result_mod_192397 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 28), '%', str_192395, currentModule_192396)
            
            # Assigning a type to the variable 'errorName' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'errorName', result_mod_192397)
            
            # Call to _addClassOrModuleLevelException(...): (line 188)
            # Processing the call arguments (line 188)
            # Getting the type of 'result' (line 188)
            result_192400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 53), 'result', False)
            # Getting the type of 'e' (line 188)
            e_192401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 61), 'e', False)
            # Getting the type of 'errorName' (line 188)
            errorName_192402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 64), 'errorName', False)
            # Processing the call keyword arguments (line 188)
            kwargs_192403 = {}
            # Getting the type of 'self' (line 188)
            self_192398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'self', False)
            # Obtaining the member '_addClassOrModuleLevelException' of a type (line 188)
            _addClassOrModuleLevelException_192399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 16), self_192398, '_addClassOrModuleLevelException')
            # Calling _addClassOrModuleLevelException(args, kwargs) (line 188)
            _addClassOrModuleLevelException_call_result_192404 = invoke(stypy.reporting.localization.Localization(__file__, 188, 16), _addClassOrModuleLevelException_192399, *[result_192400, e_192401, errorName_192402], **kwargs_192403)
            
            # SSA join for try-except statement (line 181)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # finally branch of the try-finally block (line 181)
            
            # Call to _call_if_exists(...): (line 190)
            # Processing the call arguments (line 190)
            # Getting the type of 'result' (line 190)
            result_192406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 32), 'result', False)
            str_192407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 40), 'str', '_restoreStdout')
            # Processing the call keyword arguments (line 190)
            kwargs_192408 = {}
            # Getting the type of '_call_if_exists' (line 190)
            _call_if_exists_192405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), '_call_if_exists', False)
            # Calling _call_if_exists(args, kwargs) (line 190)
            _call_if_exists_call_result_192409 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), _call_if_exists_192405, *[result_192406, str_192407], **kwargs_192408)
            
            

            if more_types_in_union_192377:
                # SSA join for if statement (line 179)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_handleModuleFixture(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handleModuleFixture' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_192410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192410)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handleModuleFixture'
        return stypy_return_type_192410


    @norecursion
    def _addClassOrModuleLevelException(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_addClassOrModuleLevelException'
        module_type_store = module_type_store.open_function_context('_addClassOrModuleLevelException', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_localization', localization)
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_function_name', 'TestSuite._addClassOrModuleLevelException')
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_param_names_list', ['result', 'exception', 'errorName'])
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSuite._addClassOrModuleLevelException.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSuite._addClassOrModuleLevelException', ['result', 'exception', 'errorName'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_addClassOrModuleLevelException', localization, ['result', 'exception', 'errorName'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_addClassOrModuleLevelException(...)' code ##################

        
        # Assigning a Call to a Name (line 193):
        
        # Call to _ErrorHolder(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'errorName' (line 193)
        errorName_192412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'errorName', False)
        # Processing the call keyword arguments (line 193)
        kwargs_192413 = {}
        # Getting the type of '_ErrorHolder' (line 193)
        _ErrorHolder_192411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), '_ErrorHolder', False)
        # Calling _ErrorHolder(args, kwargs) (line 193)
        _ErrorHolder_call_result_192414 = invoke(stypy.reporting.localization.Localization(__file__, 193, 16), _ErrorHolder_192411, *[errorName_192412], **kwargs_192413)
        
        # Assigning a type to the variable 'error' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'error', _ErrorHolder_call_result_192414)
        
        # Assigning a Call to a Name (line 194):
        
        # Call to getattr(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'result' (line 194)
        result_192416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 26), 'result', False)
        str_192417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 34), 'str', 'addSkip')
        # Getting the type of 'None' (line 194)
        None_192418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 45), 'None', False)
        # Processing the call keyword arguments (line 194)
        kwargs_192419 = {}
        # Getting the type of 'getattr' (line 194)
        getattr_192415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 194)
        getattr_call_result_192420 = invoke(stypy.reporting.localization.Localization(__file__, 194, 18), getattr_192415, *[result_192416, str_192417, None_192418], **kwargs_192419)
        
        # Assigning a type to the variable 'addSkip' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'addSkip', getattr_call_result_192420)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'addSkip' (line 195)
        addSkip_192421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'addSkip')
        # Getting the type of 'None' (line 195)
        None_192422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 26), 'None')
        # Applying the binary operator 'isnot' (line 195)
        result_is_not_192423 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 11), 'isnot', addSkip_192421, None_192422)
        
        
        # Call to isinstance(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'exception' (line 195)
        exception_192425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 46), 'exception', False)
        # Getting the type of 'case' (line 195)
        case_192426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 57), 'case', False)
        # Obtaining the member 'SkipTest' of a type (line 195)
        SkipTest_192427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 57), case_192426, 'SkipTest')
        # Processing the call keyword arguments (line 195)
        kwargs_192428 = {}
        # Getting the type of 'isinstance' (line 195)
        isinstance_192424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 195)
        isinstance_call_result_192429 = invoke(stypy.reporting.localization.Localization(__file__, 195, 35), isinstance_192424, *[exception_192425, SkipTest_192427], **kwargs_192428)
        
        # Applying the binary operator 'and' (line 195)
        result_and_keyword_192430 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 11), 'and', result_is_not_192423, isinstance_call_result_192429)
        
        # Testing the type of an if condition (line 195)
        if_condition_192431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 8), result_and_keyword_192430)
        # Assigning a type to the variable 'if_condition_192431' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'if_condition_192431', if_condition_192431)
        # SSA begins for if statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to addSkip(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'error' (line 196)
        error_192433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'error', False)
        
        # Call to str(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'exception' (line 196)
        exception_192435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 'exception', False)
        # Processing the call keyword arguments (line 196)
        kwargs_192436 = {}
        # Getting the type of 'str' (line 196)
        str_192434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'str', False)
        # Calling str(args, kwargs) (line 196)
        str_call_result_192437 = invoke(stypy.reporting.localization.Localization(__file__, 196, 27), str_192434, *[exception_192435], **kwargs_192436)
        
        # Processing the call keyword arguments (line 196)
        kwargs_192438 = {}
        # Getting the type of 'addSkip' (line 196)
        addSkip_192432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'addSkip', False)
        # Calling addSkip(args, kwargs) (line 196)
        addSkip_call_result_192439 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), addSkip_192432, *[error_192433, str_call_result_192437], **kwargs_192438)
        
        # SSA branch for the else part of an if statement (line 195)
        module_type_store.open_ssa_branch('else')
        
        # Call to addError(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'error' (line 198)
        error_192442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'error', False)
        
        # Call to exc_info(...): (line 198)
        # Processing the call keyword arguments (line 198)
        kwargs_192445 = {}
        # Getting the type of 'sys' (line 198)
        sys_192443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 35), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 198)
        exc_info_192444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 35), sys_192443, 'exc_info')
        # Calling exc_info(args, kwargs) (line 198)
        exc_info_call_result_192446 = invoke(stypy.reporting.localization.Localization(__file__, 198, 35), exc_info_192444, *[], **kwargs_192445)
        
        # Processing the call keyword arguments (line 198)
        kwargs_192447 = {}
        # Getting the type of 'result' (line 198)
        result_192440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'result', False)
        # Obtaining the member 'addError' of a type (line 198)
        addError_192441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), result_192440, 'addError')
        # Calling addError(args, kwargs) (line 198)
        addError_call_result_192448 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), addError_192441, *[error_192442, exc_info_call_result_192446], **kwargs_192447)
        
        # SSA join for if statement (line 195)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_addClassOrModuleLevelException(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_addClassOrModuleLevelException' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_192449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192449)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_addClassOrModuleLevelException'
        return stypy_return_type_192449


    @norecursion
    def _handleModuleTearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handleModuleTearDown'
        module_type_store = module_type_store.open_function_context('_handleModuleTearDown', 200, 4, False)
        # Assigning a type to the variable 'self' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_localization', localization)
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_function_name', 'TestSuite._handleModuleTearDown')
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_param_names_list', ['result'])
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSuite._handleModuleTearDown.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSuite._handleModuleTearDown', ['result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handleModuleTearDown', localization, ['result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handleModuleTearDown(...)' code ##################

        
        # Assigning a Call to a Name (line 201):
        
        # Call to _get_previous_module(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'result' (line 201)
        result_192452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 51), 'result', False)
        # Processing the call keyword arguments (line 201)
        kwargs_192453 = {}
        # Getting the type of 'self' (line 201)
        self_192450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'self', False)
        # Obtaining the member '_get_previous_module' of a type (line 201)
        _get_previous_module_192451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 25), self_192450, '_get_previous_module')
        # Calling _get_previous_module(args, kwargs) (line 201)
        _get_previous_module_call_result_192454 = invoke(stypy.reporting.localization.Localization(__file__, 201, 25), _get_previous_module_192451, *[result_192452], **kwargs_192453)
        
        # Assigning a type to the variable 'previousModule' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'previousModule', _get_previous_module_call_result_192454)
        
        # Type idiom detected: calculating its left and rigth part (line 202)
        # Getting the type of 'previousModule' (line 202)
        previousModule_192455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'previousModule')
        # Getting the type of 'None' (line 202)
        None_192456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'None')
        
        (may_be_192457, more_types_in_union_192458) = may_be_none(previousModule_192455, None_192456)

        if may_be_192457:

            if more_types_in_union_192458:
                # Runtime conditional SSA (line 202)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_192458:
                # SSA join for if statement (line 202)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'result' (line 204)
        result_192459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'result')
        # Obtaining the member '_moduleSetUpFailed' of a type (line 204)
        _moduleSetUpFailed_192460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), result_192459, '_moduleSetUpFailed')
        # Testing the type of an if condition (line 204)
        if_condition_192461 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), _moduleSetUpFailed_192460)
        # Assigning a type to the variable 'if_condition_192461' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_192461', if_condition_192461)
        # SSA begins for if statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 204)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 208):
        
        # Obtaining the type of the subscript
        # Getting the type of 'previousModule' (line 208)
        previousModule_192462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 33), 'previousModule')
        # Getting the type of 'sys' (line 208)
        sys_192463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 21), 'sys')
        # Obtaining the member 'modules' of a type (line 208)
        modules_192464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 21), sys_192463, 'modules')
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___192465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 21), modules_192464, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_192466 = invoke(stypy.reporting.localization.Localization(__file__, 208, 21), getitem___192465, previousModule_192462)
        
        # Assigning a type to the variable 'module' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'module', subscript_call_result_192466)
        # SSA branch for the except part of a try statement (line 207)
        # SSA branch for the except 'KeyError' branch of a try statement (line 207)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 212):
        
        # Call to getattr(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'module' (line 212)
        module_192468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'module', False)
        str_192469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 41), 'str', 'tearDownModule')
        # Getting the type of 'None' (line 212)
        None_192470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 59), 'None', False)
        # Processing the call keyword arguments (line 212)
        kwargs_192471 = {}
        # Getting the type of 'getattr' (line 212)
        getattr_192467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 25), 'getattr', False)
        # Calling getattr(args, kwargs) (line 212)
        getattr_call_result_192472 = invoke(stypy.reporting.localization.Localization(__file__, 212, 25), getattr_192467, *[module_192468, str_192469, None_192470], **kwargs_192471)
        
        # Assigning a type to the variable 'tearDownModule' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tearDownModule', getattr_call_result_192472)
        
        # Type idiom detected: calculating its left and rigth part (line 213)
        # Getting the type of 'tearDownModule' (line 213)
        tearDownModule_192473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'tearDownModule')
        # Getting the type of 'None' (line 213)
        None_192474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'None')
        
        (may_be_192475, more_types_in_union_192476) = may_not_be_none(tearDownModule_192473, None_192474)

        if may_be_192475:

            if more_types_in_union_192476:
                # Runtime conditional SSA (line 213)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _call_if_exists(...): (line 214)
            # Processing the call arguments (line 214)
            # Getting the type of 'result' (line 214)
            result_192478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'result', False)
            str_192479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 36), 'str', '_setupStdout')
            # Processing the call keyword arguments (line 214)
            kwargs_192480 = {}
            # Getting the type of '_call_if_exists' (line 214)
            _call_if_exists_192477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), '_call_if_exists', False)
            # Calling _call_if_exists(args, kwargs) (line 214)
            _call_if_exists_call_result_192481 = invoke(stypy.reporting.localization.Localization(__file__, 214, 12), _call_if_exists_192477, *[result_192478, str_192479], **kwargs_192480)
            
            
            # Try-finally block (line 215)
            
            
            # SSA begins for try-except statement (line 215)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to tearDownModule(...): (line 216)
            # Processing the call keyword arguments (line 216)
            kwargs_192483 = {}
            # Getting the type of 'tearDownModule' (line 216)
            tearDownModule_192482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'tearDownModule', False)
            # Calling tearDownModule(args, kwargs) (line 216)
            tearDownModule_call_result_192484 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), tearDownModule_192482, *[], **kwargs_192483)
            
            # SSA branch for the except part of a try statement (line 215)
            # SSA branch for the except 'Exception' branch of a try statement (line 215)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'Exception' (line 217)
            Exception_192485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'Exception')
            # Assigning a type to the variable 'e' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'e', Exception_192485)
            
            
            # Call to isinstance(...): (line 218)
            # Processing the call arguments (line 218)
            # Getting the type of 'result' (line 218)
            result_192487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 30), 'result', False)
            # Getting the type of '_DebugResult' (line 218)
            _DebugResult_192488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 38), '_DebugResult', False)
            # Processing the call keyword arguments (line 218)
            kwargs_192489 = {}
            # Getting the type of 'isinstance' (line 218)
            isinstance_192486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 218)
            isinstance_call_result_192490 = invoke(stypy.reporting.localization.Localization(__file__, 218, 19), isinstance_192486, *[result_192487, _DebugResult_192488], **kwargs_192489)
            
            # Testing the type of an if condition (line 218)
            if_condition_192491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 16), isinstance_call_result_192490)
            # Assigning a type to the variable 'if_condition_192491' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'if_condition_192491', if_condition_192491)
            # SSA begins for if statement (line 218)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 218)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 220):
            str_192492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 28), 'str', 'tearDownModule (%s)')
            # Getting the type of 'previousModule' (line 220)
            previousModule_192493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 52), 'previousModule')
            # Applying the binary operator '%' (line 220)
            result_mod_192494 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 28), '%', str_192492, previousModule_192493)
            
            # Assigning a type to the variable 'errorName' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'errorName', result_mod_192494)
            
            # Call to _addClassOrModuleLevelException(...): (line 221)
            # Processing the call arguments (line 221)
            # Getting the type of 'result' (line 221)
            result_192497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 53), 'result', False)
            # Getting the type of 'e' (line 221)
            e_192498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 61), 'e', False)
            # Getting the type of 'errorName' (line 221)
            errorName_192499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'errorName', False)
            # Processing the call keyword arguments (line 221)
            kwargs_192500 = {}
            # Getting the type of 'self' (line 221)
            self_192495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'self', False)
            # Obtaining the member '_addClassOrModuleLevelException' of a type (line 221)
            _addClassOrModuleLevelException_192496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), self_192495, '_addClassOrModuleLevelException')
            # Calling _addClassOrModuleLevelException(args, kwargs) (line 221)
            _addClassOrModuleLevelException_call_result_192501 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), _addClassOrModuleLevelException_192496, *[result_192497, e_192498, errorName_192499], **kwargs_192500)
            
            # SSA join for try-except statement (line 215)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # finally branch of the try-finally block (line 215)
            
            # Call to _call_if_exists(...): (line 223)
            # Processing the call arguments (line 223)
            # Getting the type of 'result' (line 223)
            result_192503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'result', False)
            str_192504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 40), 'str', '_restoreStdout')
            # Processing the call keyword arguments (line 223)
            kwargs_192505 = {}
            # Getting the type of '_call_if_exists' (line 223)
            _call_if_exists_192502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), '_call_if_exists', False)
            # Calling _call_if_exists(args, kwargs) (line 223)
            _call_if_exists_call_result_192506 = invoke(stypy.reporting.localization.Localization(__file__, 223, 16), _call_if_exists_192502, *[result_192503, str_192504], **kwargs_192505)
            
            

            if more_types_in_union_192476:
                # SSA join for if statement (line 213)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_handleModuleTearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handleModuleTearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 200)
        stypy_return_type_192507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192507)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handleModuleTearDown'
        return stypy_return_type_192507


    @norecursion
    def _tearDownPreviousClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_tearDownPreviousClass'
        module_type_store = module_type_store.open_function_context('_tearDownPreviousClass', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_localization', localization)
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_function_name', 'TestSuite._tearDownPreviousClass')
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_param_names_list', ['test', 'result'])
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSuite._tearDownPreviousClass.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSuite._tearDownPreviousClass', ['test', 'result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_tearDownPreviousClass', localization, ['test', 'result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_tearDownPreviousClass(...)' code ##################

        
        # Assigning a Call to a Name (line 226):
        
        # Call to getattr(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'result' (line 226)
        result_192509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 32), 'result', False)
        str_192510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 40), 'str', '_previousTestClass')
        # Getting the type of 'None' (line 226)
        None_192511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 62), 'None', False)
        # Processing the call keyword arguments (line 226)
        kwargs_192512 = {}
        # Getting the type of 'getattr' (line 226)
        getattr_192508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 226)
        getattr_call_result_192513 = invoke(stypy.reporting.localization.Localization(__file__, 226, 24), getattr_192508, *[result_192509, str_192510, None_192511], **kwargs_192512)
        
        # Assigning a type to the variable 'previousClass' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'previousClass', getattr_call_result_192513)
        
        # Assigning a Attribute to a Name (line 227):
        # Getting the type of 'test' (line 227)
        test_192514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'test')
        # Obtaining the member '__class__' of a type (line 227)
        class___192515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 23), test_192514, '__class__')
        # Assigning a type to the variable 'currentClass' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'currentClass', class___192515)
        
        
        # Getting the type of 'currentClass' (line 228)
        currentClass_192516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'currentClass')
        # Getting the type of 'previousClass' (line 228)
        previousClass_192517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 27), 'previousClass')
        # Applying the binary operator '==' (line 228)
        result_eq_192518 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 11), '==', currentClass_192516, previousClass_192517)
        
        # Testing the type of an if condition (line 228)
        if_condition_192519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 8), result_eq_192518)
        # Assigning a type to the variable 'if_condition_192519' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'if_condition_192519', if_condition_192519)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to getattr(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'previousClass' (line 230)
        previousClass_192521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'previousClass', False)
        str_192522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 34), 'str', '_classSetupFailed')
        # Getting the type of 'False' (line 230)
        False_192523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 55), 'False', False)
        # Processing the call keyword arguments (line 230)
        kwargs_192524 = {}
        # Getting the type of 'getattr' (line 230)
        getattr_192520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'getattr', False)
        # Calling getattr(args, kwargs) (line 230)
        getattr_call_result_192525 = invoke(stypy.reporting.localization.Localization(__file__, 230, 11), getattr_192520, *[previousClass_192521, str_192522, False_192523], **kwargs_192524)
        
        # Testing the type of an if condition (line 230)
        if_condition_192526 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), getattr_call_result_192525)
        # Assigning a type to the variable 'if_condition_192526' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_192526', if_condition_192526)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to getattr(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'result' (line 232)
        result_192528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'result', False)
        str_192529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 27), 'str', '_moduleSetUpFailed')
        # Getting the type of 'False' (line 232)
        False_192530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 49), 'False', False)
        # Processing the call keyword arguments (line 232)
        kwargs_192531 = {}
        # Getting the type of 'getattr' (line 232)
        getattr_192527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'getattr', False)
        # Calling getattr(args, kwargs) (line 232)
        getattr_call_result_192532 = invoke(stypy.reporting.localization.Localization(__file__, 232, 11), getattr_192527, *[result_192528, str_192529, False_192530], **kwargs_192531)
        
        # Testing the type of an if condition (line 232)
        if_condition_192533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), getattr_call_result_192532)
        # Assigning a type to the variable 'if_condition_192533' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_192533', if_condition_192533)
        # SSA begins for if statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 232)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to getattr(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'previousClass' (line 234)
        previousClass_192535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'previousClass', False)
        str_192536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 34), 'str', '__unittest_skip__')
        # Getting the type of 'False' (line 234)
        False_192537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 55), 'False', False)
        # Processing the call keyword arguments (line 234)
        kwargs_192538 = {}
        # Getting the type of 'getattr' (line 234)
        getattr_192534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'getattr', False)
        # Calling getattr(args, kwargs) (line 234)
        getattr_call_result_192539 = invoke(stypy.reporting.localization.Localization(__file__, 234, 11), getattr_192534, *[previousClass_192535, str_192536, False_192537], **kwargs_192538)
        
        # Testing the type of an if condition (line 234)
        if_condition_192540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 8), getattr_call_result_192539)
        # Assigning a type to the variable 'if_condition_192540' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'if_condition_192540', if_condition_192540)
        # SSA begins for if statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 234)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 237):
        
        # Call to getattr(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'previousClass' (line 237)
        previousClass_192542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'previousClass', False)
        str_192543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 47), 'str', 'tearDownClass')
        # Getting the type of 'None' (line 237)
        None_192544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 64), 'None', False)
        # Processing the call keyword arguments (line 237)
        kwargs_192545 = {}
        # Getting the type of 'getattr' (line 237)
        getattr_192541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 237)
        getattr_call_result_192546 = invoke(stypy.reporting.localization.Localization(__file__, 237, 24), getattr_192541, *[previousClass_192542, str_192543, None_192544], **kwargs_192545)
        
        # Assigning a type to the variable 'tearDownClass' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tearDownClass', getattr_call_result_192546)
        
        # Type idiom detected: calculating its left and rigth part (line 238)
        # Getting the type of 'tearDownClass' (line 238)
        tearDownClass_192547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'tearDownClass')
        # Getting the type of 'None' (line 238)
        None_192548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 32), 'None')
        
        (may_be_192549, more_types_in_union_192550) = may_not_be_none(tearDownClass_192547, None_192548)

        if may_be_192549:

            if more_types_in_union_192550:
                # Runtime conditional SSA (line 238)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _call_if_exists(...): (line 239)
            # Processing the call arguments (line 239)
            # Getting the type of 'result' (line 239)
            result_192552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'result', False)
            str_192553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 36), 'str', '_setupStdout')
            # Processing the call keyword arguments (line 239)
            kwargs_192554 = {}
            # Getting the type of '_call_if_exists' (line 239)
            _call_if_exists_192551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), '_call_if_exists', False)
            # Calling _call_if_exists(args, kwargs) (line 239)
            _call_if_exists_call_result_192555 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), _call_if_exists_192551, *[result_192552, str_192553], **kwargs_192554)
            
            
            # Try-finally block (line 240)
            
            
            # SSA begins for try-except statement (line 240)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to tearDownClass(...): (line 241)
            # Processing the call keyword arguments (line 241)
            kwargs_192557 = {}
            # Getting the type of 'tearDownClass' (line 241)
            tearDownClass_192556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'tearDownClass', False)
            # Calling tearDownClass(args, kwargs) (line 241)
            tearDownClass_call_result_192558 = invoke(stypy.reporting.localization.Localization(__file__, 241, 16), tearDownClass_192556, *[], **kwargs_192557)
            
            # SSA branch for the except part of a try statement (line 240)
            # SSA branch for the except 'Exception' branch of a try statement (line 240)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'Exception' (line 242)
            Exception_192559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'Exception')
            # Assigning a type to the variable 'e' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'e', Exception_192559)
            
            
            # Call to isinstance(...): (line 243)
            # Processing the call arguments (line 243)
            # Getting the type of 'result' (line 243)
            result_192561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 30), 'result', False)
            # Getting the type of '_DebugResult' (line 243)
            _DebugResult_192562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 38), '_DebugResult', False)
            # Processing the call keyword arguments (line 243)
            kwargs_192563 = {}
            # Getting the type of 'isinstance' (line 243)
            isinstance_192560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 243)
            isinstance_call_result_192564 = invoke(stypy.reporting.localization.Localization(__file__, 243, 19), isinstance_192560, *[result_192561, _DebugResult_192562], **kwargs_192563)
            
            # Testing the type of an if condition (line 243)
            if_condition_192565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 16), isinstance_call_result_192564)
            # Assigning a type to the variable 'if_condition_192565' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'if_condition_192565', if_condition_192565)
            # SSA begins for if statement (line 243)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 243)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 245):
            
            # Call to strclass(...): (line 245)
            # Processing the call arguments (line 245)
            # Getting the type of 'previousClass' (line 245)
            previousClass_192568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 42), 'previousClass', False)
            # Processing the call keyword arguments (line 245)
            kwargs_192569 = {}
            # Getting the type of 'util' (line 245)
            util_192566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'util', False)
            # Obtaining the member 'strclass' of a type (line 245)
            strclass_192567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 28), util_192566, 'strclass')
            # Calling strclass(args, kwargs) (line 245)
            strclass_call_result_192570 = invoke(stypy.reporting.localization.Localization(__file__, 245, 28), strclass_192567, *[previousClass_192568], **kwargs_192569)
            
            # Assigning a type to the variable 'className' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'className', strclass_call_result_192570)
            
            # Assigning a BinOp to a Name (line 246):
            str_192571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 28), 'str', 'tearDownClass (%s)')
            # Getting the type of 'className' (line 246)
            className_192572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 51), 'className')
            # Applying the binary operator '%' (line 246)
            result_mod_192573 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 28), '%', str_192571, className_192572)
            
            # Assigning a type to the variable 'errorName' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'errorName', result_mod_192573)
            
            # Call to _addClassOrModuleLevelException(...): (line 247)
            # Processing the call arguments (line 247)
            # Getting the type of 'result' (line 247)
            result_192576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 53), 'result', False)
            # Getting the type of 'e' (line 247)
            e_192577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 61), 'e', False)
            # Getting the type of 'errorName' (line 247)
            errorName_192578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 64), 'errorName', False)
            # Processing the call keyword arguments (line 247)
            kwargs_192579 = {}
            # Getting the type of 'self' (line 247)
            self_192574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'self', False)
            # Obtaining the member '_addClassOrModuleLevelException' of a type (line 247)
            _addClassOrModuleLevelException_192575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 16), self_192574, '_addClassOrModuleLevelException')
            # Calling _addClassOrModuleLevelException(args, kwargs) (line 247)
            _addClassOrModuleLevelException_call_result_192580 = invoke(stypy.reporting.localization.Localization(__file__, 247, 16), _addClassOrModuleLevelException_192575, *[result_192576, e_192577, errorName_192578], **kwargs_192579)
            
            # SSA join for try-except statement (line 240)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # finally branch of the try-finally block (line 240)
            
            # Call to _call_if_exists(...): (line 249)
            # Processing the call arguments (line 249)
            # Getting the type of 'result' (line 249)
            result_192582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 32), 'result', False)
            str_192583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 40), 'str', '_restoreStdout')
            # Processing the call keyword arguments (line 249)
            kwargs_192584 = {}
            # Getting the type of '_call_if_exists' (line 249)
            _call_if_exists_192581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), '_call_if_exists', False)
            # Calling _call_if_exists(args, kwargs) (line 249)
            _call_if_exists_call_result_192585 = invoke(stypy.reporting.localization.Localization(__file__, 249, 16), _call_if_exists_192581, *[result_192582, str_192583], **kwargs_192584)
            
            

            if more_types_in_union_192550:
                # SSA join for if statement (line 238)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_tearDownPreviousClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_tearDownPreviousClass' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_192586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192586)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_tearDownPreviousClass'
        return stypy_return_type_192586


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 78, 0, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSuite.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSuite' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'TestSuite', TestSuite)
# Declaration of the '_ErrorHolder' class

class _ErrorHolder(object, ):
    str_192587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'str', '\n    Placeholder for a TestCase inside a result. As far as a TestResult\n    is concerned, this looks exactly like a unit test. Used to insert\n    arbitrary errors into a test suite run.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ErrorHolder.__init__', ['description'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['description'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 265):
        # Getting the type of 'description' (line 265)
        description_192588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 27), 'description')
        # Getting the type of 'self' (line 265)
        self_192589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'self')
        # Setting the type of the member 'description' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), self_192589, 'description', description_192588)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def id(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'id'
        module_type_store = module_type_store.open_function_context('id', 267, 4, False)
        # Assigning a type to the variable 'self' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ErrorHolder.id.__dict__.__setitem__('stypy_localization', localization)
        _ErrorHolder.id.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ErrorHolder.id.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ErrorHolder.id.__dict__.__setitem__('stypy_function_name', '_ErrorHolder.id')
        _ErrorHolder.id.__dict__.__setitem__('stypy_param_names_list', [])
        _ErrorHolder.id.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ErrorHolder.id.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ErrorHolder.id.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ErrorHolder.id.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ErrorHolder.id.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ErrorHolder.id.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ErrorHolder.id', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 268)
        self_192590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'self')
        # Obtaining the member 'description' of a type (line 268)
        description_192591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), self_192590, 'description')
        # Assigning a type to the variable 'stypy_return_type' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'stypy_return_type', description_192591)
        
        # ################# End of 'id(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'id' in the type store
        # Getting the type of 'stypy_return_type' (line 267)
        stypy_return_type_192592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'id'
        return stypy_return_type_192592


    @norecursion
    def shortDescription(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shortDescription'
        module_type_store = module_type_store.open_function_context('shortDescription', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_localization', localization)
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_function_name', '_ErrorHolder.shortDescription')
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_param_names_list', [])
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ErrorHolder.shortDescription.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ErrorHolder.shortDescription', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'None' (line 271)
        None_192593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'stypy_return_type', None_192593)
        
        # ################# End of 'shortDescription(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shortDescription' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_192594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shortDescription'
        return stypy_return_type_192594


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_function_name', '_ErrorHolder.__repr__')
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ErrorHolder.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ErrorHolder.__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_192595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 15), 'str', '<ErrorHolder description=%r>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 274)
        tuple_192596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 274)
        # Adding element type (line 274)
        # Getting the type of 'self' (line 274)
        self_192597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 49), 'self')
        # Obtaining the member 'description' of a type (line 274)
        description_192598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 49), self_192597, 'description')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 49), tuple_192596, description_192598)
        
        # Applying the binary operator '%' (line 274)
        result_mod_192599 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 15), '%', str_192595, tuple_192596)
        
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'stypy_return_type', result_mod_192599)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_192600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192600)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_192600


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 276, 4, False)
        # Assigning a type to the variable 'self' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_function_name', '_ErrorHolder.__str__')
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ErrorHolder.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ErrorHolder.__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to id(...): (line 277)
        # Processing the call keyword arguments (line 277)
        kwargs_192603 = {}
        # Getting the type of 'self' (line 277)
        self_192601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'self', False)
        # Obtaining the member 'id' of a type (line 277)
        id_192602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 15), self_192601, 'id')
        # Calling id(args, kwargs) (line 277)
        id_call_result_192604 = invoke(stypy.reporting.localization.Localization(__file__, 277, 15), id_192602, *[], **kwargs_192603)
        
        # Assigning a type to the variable 'stypy_return_type' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'stypy_return_type', id_call_result_192604)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 276)
        stypy_return_type_192605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192605)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_192605


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ErrorHolder.run.__dict__.__setitem__('stypy_localization', localization)
        _ErrorHolder.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ErrorHolder.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ErrorHolder.run.__dict__.__setitem__('stypy_function_name', '_ErrorHolder.run')
        _ErrorHolder.run.__dict__.__setitem__('stypy_param_names_list', ['result'])
        _ErrorHolder.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ErrorHolder.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ErrorHolder.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ErrorHolder.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ErrorHolder.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ErrorHolder.run.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ErrorHolder.run', ['result'], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_192606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_192606


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 284, 4, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_localization', localization)
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_function_name', '_ErrorHolder.__call__')
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_param_names_list', ['result'])
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ErrorHolder.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ErrorHolder.__call__', ['result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to run(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'result' (line 285)
        result_192609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'result', False)
        # Processing the call keyword arguments (line 285)
        kwargs_192610 = {}
        # Getting the type of 'self' (line 285)
        self_192607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'self', False)
        # Obtaining the member 'run' of a type (line 285)
        run_192608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 15), self_192607, 'run')
        # Calling run(args, kwargs) (line 285)
        run_call_result_192611 = invoke(stypy.reporting.localization.Localization(__file__, 285, 15), run_192608, *[result_192609], **kwargs_192610)
        
        # Assigning a type to the variable 'stypy_return_type' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'stypy_return_type', run_call_result_192611)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 284)
        stypy_return_type_192612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_192612


    @norecursion
    def countTestCases(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'countTestCases'
        module_type_store = module_type_store.open_function_context('countTestCases', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_localization', localization)
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_function_name', '_ErrorHolder.countTestCases')
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_param_names_list', [])
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ErrorHolder.countTestCases.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ErrorHolder.countTestCases', [], None, None, defaults, varargs, kwargs)

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

        int_192613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', int_192613)
        
        # ################# End of 'countTestCases(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'countTestCases' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_192614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_192614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'countTestCases'
        return stypy_return_type_192614


# Assigning a type to the variable '_ErrorHolder' (line 252)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), '_ErrorHolder', _ErrorHolder)

# Assigning a Name to a Name (line 262):
# Getting the type of 'None' (line 262)
None_192615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'None')
# Getting the type of '_ErrorHolder'
_ErrorHolder_192616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ErrorHolder')
# Setting the type of the member 'failureException' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ErrorHolder_192616, 'failureException', None_192615)

@norecursion
def _isnotsuite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_isnotsuite'
    module_type_store = module_type_store.open_function_context('_isnotsuite', 290, 0, False)
    
    # Passed parameters checking function
    _isnotsuite.stypy_localization = localization
    _isnotsuite.stypy_type_of_self = None
    _isnotsuite.stypy_type_store = module_type_store
    _isnotsuite.stypy_function_name = '_isnotsuite'
    _isnotsuite.stypy_param_names_list = ['test']
    _isnotsuite.stypy_varargs_param_name = None
    _isnotsuite.stypy_kwargs_param_name = None
    _isnotsuite.stypy_call_defaults = defaults
    _isnotsuite.stypy_call_varargs = varargs
    _isnotsuite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_isnotsuite', ['test'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_isnotsuite', localization, ['test'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_isnotsuite(...)' code ##################

    str_192617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 4), 'str', 'A crude way to tell apart testcases and suites with duck-typing')
    
    
    # SSA begins for try-except statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to iter(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'test' (line 293)
    test_192619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'test', False)
    # Processing the call keyword arguments (line 293)
    kwargs_192620 = {}
    # Getting the type of 'iter' (line 293)
    iter_192618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'iter', False)
    # Calling iter(args, kwargs) (line 293)
    iter_call_result_192621 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), iter_192618, *[test_192619], **kwargs_192620)
    
    # SSA branch for the except part of a try statement (line 292)
    # SSA branch for the except 'TypeError' branch of a try statement (line 292)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'True' (line 295)
    True_192622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'stypy_return_type', True_192622)
    # SSA join for try-except statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 296)
    False_192623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type', False_192623)
    
    # ################# End of '_isnotsuite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_isnotsuite' in the type store
    # Getting the type of 'stypy_return_type' (line 290)
    stypy_return_type_192624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192624)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_isnotsuite'
    return stypy_return_type_192624

# Assigning a type to the variable '_isnotsuite' (line 290)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), '_isnotsuite', _isnotsuite)
# Declaration of the '_DebugResult' class

class _DebugResult(object, ):
    str_192625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 4), 'str', 'Used by the TestSuite to hold previous class when running in debug.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 299, 0, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_DebugResult.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_DebugResult' (line 299)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), '_DebugResult', _DebugResult)

# Assigning a Name to a Name (line 301):
# Getting the type of 'None' (line 301)
None_192626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'None')
# Getting the type of '_DebugResult'
_DebugResult_192627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_DebugResult')
# Setting the type of the member '_previousTestClass' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _DebugResult_192627, '_previousTestClass', None_192626)

# Assigning a Name to a Name (line 302):
# Getting the type of 'False' (line 302)
False_192628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 25), 'False')
# Getting the type of '_DebugResult'
_DebugResult_192629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_DebugResult')
# Setting the type of the member '_moduleSetUpFailed' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _DebugResult_192629, '_moduleSetUpFailed', False_192628)

# Assigning a Name to a Name (line 303):
# Getting the type of 'False' (line 303)
False_192630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 17), 'False')
# Getting the type of '_DebugResult'
_DebugResult_192631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_DebugResult')
# Setting the type of the member 'shouldStop' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _DebugResult_192631, 'shouldStop', False_192630)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
