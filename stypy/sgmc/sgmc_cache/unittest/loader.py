
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Loading unittests.'''
2: 
3: import os
4: import re
5: import sys
6: import traceback
7: import types
8: 
9: from functools import cmp_to_key as _CmpToKey
10: from fnmatch import fnmatch
11: 
12: from . import case, suite
13: 
14: __unittest = True
15: 
16: # what about .pyc or .pyo (etc)
17: # we would need to avoid loading the same tests multiple times
18: # from '.py', '.pyc' *and* '.pyo'
19: VALID_MODULE_NAME = re.compile(r'[_a-z]\w*\.py$', re.IGNORECASE)
20: 
21: 
22: def _make_failed_import_test(name, suiteClass):
23:     message = 'Failed to import test module: %s\n%s' % (name, traceback.format_exc())
24:     return _make_failed_test('ModuleImportFailure', name, ImportError(message),
25:                              suiteClass)
26: 
27: def _make_failed_load_tests(name, exception, suiteClass):
28:     return _make_failed_test('LoadTestsFailure', name, exception, suiteClass)
29: 
30: def _make_failed_test(classname, methodname, exception, suiteClass):
31:     def testFailure(self):
32:         raise exception
33:     attrs = {methodname: testFailure}
34:     TestClass = type(classname, (case.TestCase,), attrs)
35:     return suiteClass((TestClass(methodname),))
36: 
37: 
38: class TestLoader(object):
39:     '''
40:     This class is responsible for loading tests according to various criteria
41:     and returning them wrapped in a TestSuite
42:     '''
43:     testMethodPrefix = 'test'
44:     sortTestMethodsUsing = cmp
45:     suiteClass = suite.TestSuite
46:     _top_level_dir = None
47: 
48:     def loadTestsFromTestCase(self, testCaseClass):
49:         '''Return a suite of all tests cases contained in testCaseClass'''
50:         if issubclass(testCaseClass, suite.TestSuite):
51:             raise TypeError("Test cases should not be derived from TestSuite." \
52:                                 " Maybe you meant to derive from TestCase?")
53:         testCaseNames = self.getTestCaseNames(testCaseClass)
54:         if not testCaseNames and hasattr(testCaseClass, 'runTest'):
55:             testCaseNames = ['runTest']
56:         loaded_suite = self.suiteClass(map(testCaseClass, testCaseNames))
57:         return loaded_suite
58: 
59:     def loadTestsFromModule(self, module, use_load_tests=True):
60:         '''Return a suite of all tests cases contained in the given module'''
61:         tests = []
62:         for name in dir(module):
63:             obj = getattr(module, name)
64:             if isinstance(obj, type) and issubclass(obj, case.TestCase):
65:                 tests.append(self.loadTestsFromTestCase(obj))
66: 
67:         load_tests = getattr(module, 'load_tests', None)
68:         tests = self.suiteClass(tests)
69:         if use_load_tests and load_tests is not None:
70:             try:
71:                 return load_tests(self, tests, None)
72:             except Exception, e:
73:                 return _make_failed_load_tests(module.__name__, e,
74:                                                self.suiteClass)
75:         return tests
76: 
77:     def loadTestsFromName(self, name, module=None):
78:         '''Return a suite of all tests cases given a string specifier.
79: 
80:         The name may resolve either to a module, a test case class, a
81:         test method within a test case class, or a callable object which
82:         returns a TestCase or TestSuite instance.
83: 
84:         The method optionally resolves the names relative to a given module.
85:         '''
86:         parts = name.split('.')
87:         if module is None:
88:             parts_copy = parts[:]
89:             while parts_copy:
90:                 try:
91:                     module = __import__('.'.join(parts_copy))
92:                     break
93:                 except ImportError:
94:                     del parts_copy[-1]
95:                     if not parts_copy:
96:                         raise
97:             parts = parts[1:]
98:         obj = module
99:         for part in parts:
100:             parent, obj = obj, getattr(obj, part)
101: 
102:         if isinstance(obj, types.ModuleType):
103:             return self.loadTestsFromModule(obj)
104:         elif isinstance(obj, type) and issubclass(obj, case.TestCase):
105:             return self.loadTestsFromTestCase(obj)
106:         elif (isinstance(obj, types.UnboundMethodType) and
107:               isinstance(parent, type) and
108:               issubclass(parent, case.TestCase)):
109:             name = parts[-1]
110:             inst = parent(name)
111:             return self.suiteClass([inst])
112:         elif isinstance(obj, suite.TestSuite):
113:             return obj
114:         elif hasattr(obj, '__call__'):
115:             test = obj()
116:             if isinstance(test, suite.TestSuite):
117:                 return test
118:             elif isinstance(test, case.TestCase):
119:                 return self.suiteClass([test])
120:             else:
121:                 raise TypeError("calling %s returned %s, not a test" %
122:                                 (obj, test))
123:         else:
124:             raise TypeError("don't know how to make test from: %s" % obj)
125: 
126:     def loadTestsFromNames(self, names, module=None):
127:         '''Return a suite of all tests cases found using the given sequence
128:         of string specifiers. See 'loadTestsFromName()'.
129:         '''
130:         suites = [self.loadTestsFromName(name, module) for name in names]
131:         return self.suiteClass(suites)
132: 
133:     def getTestCaseNames(self, testCaseClass):
134:         '''Return a sorted sequence of method names found within testCaseClass
135:         '''
136:         def isTestMethod(attrname, testCaseClass=testCaseClass,
137:                          prefix=self.testMethodPrefix):
138:             return attrname.startswith(prefix) and \
139:                 hasattr(getattr(testCaseClass, attrname), '__call__')
140:         testFnNames = filter(isTestMethod, dir(testCaseClass))
141:         if self.sortTestMethodsUsing:
142:             testFnNames.sort(key=_CmpToKey(self.sortTestMethodsUsing))
143:         return testFnNames
144: 
145:     def discover(self, start_dir, pattern='test*.py', top_level_dir=None):
146:         '''Find and return all test modules from the specified start
147:         directory, recursing into subdirectories to find them. Only test files
148:         that match the pattern will be loaded. (Using shell style pattern
149:         matching.)
150: 
151:         All test modules must be importable from the top level of the project.
152:         If the start directory is not the top level directory then the top
153:         level directory must be specified separately.
154: 
155:         If a test package name (directory with '__init__.py') matches the
156:         pattern then the package will be checked for a 'load_tests' function. If
157:         this exists then it will be called with loader, tests, pattern.
158: 
159:         If load_tests exists then discovery does  *not* recurse into the package,
160:         load_tests is responsible for loading all tests in the package.
161: 
162:         The pattern is deliberately not stored as a loader attribute so that
163:         packages can continue discovery themselves. top_level_dir is stored so
164:         load_tests does not need to pass this argument in to loader.discover().
165:         '''
166:         set_implicit_top = False
167:         if top_level_dir is None and self._top_level_dir is not None:
168:             # make top_level_dir optional if called from load_tests in a package
169:             top_level_dir = self._top_level_dir
170:         elif top_level_dir is None:
171:             set_implicit_top = True
172:             top_level_dir = start_dir
173: 
174:         top_level_dir = os.path.abspath(top_level_dir)
175: 
176:         if not top_level_dir in sys.path:
177:             # all test modules must be importable from the top level directory
178:             # should we *unconditionally* put the start directory in first
179:             # in sys.path to minimise likelihood of conflicts between installed
180:             # modules and development versions?
181:             sys.path.insert(0, top_level_dir)
182:         self._top_level_dir = top_level_dir
183: 
184:         is_not_importable = False
185:         if os.path.isdir(os.path.abspath(start_dir)):
186:             start_dir = os.path.abspath(start_dir)
187:             if start_dir != top_level_dir:
188:                 is_not_importable = not os.path.isfile(os.path.join(start_dir, '__init__.py'))
189:         else:
190:             # support for discovery from dotted module names
191:             try:
192:                 __import__(start_dir)
193:             except ImportError:
194:                 is_not_importable = True
195:             else:
196:                 the_module = sys.modules[start_dir]
197:                 top_part = start_dir.split('.')[0]
198:                 start_dir = os.path.abspath(os.path.dirname((the_module.__file__)))
199:                 if set_implicit_top:
200:                     self._top_level_dir = self._get_directory_containing_module(top_part)
201:                     sys.path.remove(top_level_dir)
202: 
203:         if is_not_importable:
204:             raise ImportError('Start directory is not importable: %r' % start_dir)
205: 
206:         tests = list(self._find_tests(start_dir, pattern))
207:         return self.suiteClass(tests)
208: 
209:     def _get_directory_containing_module(self, module_name):
210:         module = sys.modules[module_name]
211:         full_path = os.path.abspath(module.__file__)
212: 
213:         if os.path.basename(full_path).lower().startswith('__init__.py'):
214:             return os.path.dirname(os.path.dirname(full_path))
215:         else:
216:             # here we have been given a module rather than a package - so
217:             # all we can do is search the *same* directory the module is in
218:             # should an exception be raised instead
219:             return os.path.dirname(full_path)
220: 
221:     def _get_name_from_path(self, path):
222:         path = os.path.splitext(os.path.normpath(path))[0]
223: 
224:         _relpath = os.path.relpath(path, self._top_level_dir)
225:         assert not os.path.isabs(_relpath), "Path must be within the project"
226:         assert not _relpath.startswith('..'), "Path must be within the project"
227: 
228:         name = _relpath.replace(os.path.sep, '.')
229:         return name
230: 
231:     def _get_module_from_name(self, name):
232:         __import__(name)
233:         return sys.modules[name]
234: 
235:     def _match_path(self, path, full_path, pattern):
236:         # override this method to use alternative matching strategy
237:         return fnmatch(path, pattern)
238: 
239:     def _find_tests(self, start_dir, pattern):
240:         '''Used by discovery. Yields test suites it loads.'''
241:         paths = os.listdir(start_dir)
242: 
243:         for path in paths:
244:             full_path = os.path.join(start_dir, path)
245:             if os.path.isfile(full_path):
246:                 if not VALID_MODULE_NAME.match(path):
247:                     # valid Python identifiers only
248:                     continue
249:                 if not self._match_path(path, full_path, pattern):
250:                     continue
251:                 # if the test file matches, load it
252:                 name = self._get_name_from_path(full_path)
253:                 try:
254:                     module = self._get_module_from_name(name)
255:                 except:
256:                     yield _make_failed_import_test(name, self.suiteClass)
257:                 else:
258:                     mod_file = os.path.abspath(getattr(module, '__file__', full_path))
259:                     realpath = os.path.splitext(os.path.realpath(mod_file))[0]
260:                     fullpath_noext = os.path.splitext(os.path.realpath(full_path))[0]
261:                     if realpath.lower() != fullpath_noext.lower():
262:                         module_dir = os.path.dirname(realpath)
263:                         mod_name = os.path.splitext(os.path.basename(full_path))[0]
264:                         expected_dir = os.path.dirname(full_path)
265:                         msg = ("%r module incorrectly imported from %r. Expected %r. "
266:                                "Is this module globally installed?")
267:                         raise ImportError(msg % (mod_name, module_dir, expected_dir))
268:                     yield self.loadTestsFromModule(module)
269:             elif os.path.isdir(full_path):
270:                 if not os.path.isfile(os.path.join(full_path, '__init__.py')):
271:                     continue
272: 
273:                 load_tests = None
274:                 tests = None
275:                 if fnmatch(path, pattern):
276:                     # only check load_tests if the package directory itself matches the filter
277:                     name = self._get_name_from_path(full_path)
278:                     package = self._get_module_from_name(name)
279:                     load_tests = getattr(package, 'load_tests', None)
280:                     tests = self.loadTestsFromModule(package, use_load_tests=False)
281: 
282:                 if load_tests is None:
283:                     if tests is not None:
284:                         # tests loaded from package file
285:                         yield tests
286:                     # recurse into the package
287:                     for test in self._find_tests(full_path, pattern):
288:                         yield test
289:                 else:
290:                     try:
291:                         yield load_tests(self, tests, pattern)
292:                     except Exception, e:
293:                         yield _make_failed_load_tests(package.__name__, e,
294:                                                       self.suiteClass)
295: 
296: defaultTestLoader = TestLoader()
297: 
298: 
299: def _makeLoader(prefix, sortUsing, suiteClass=None):
300:     loader = TestLoader()
301:     loader.sortTestMethodsUsing = sortUsing
302:     loader.testMethodPrefix = prefix
303:     if suiteClass:
304:         loader.suiteClass = suiteClass
305:     return loader
306: 
307: def getTestCaseNames(testCaseClass, prefix, sortUsing=cmp):
308:     return _makeLoader(prefix, sortUsing).getTestCaseNames(testCaseClass)
309: 
310: def makeSuite(testCaseClass, prefix='test', sortUsing=cmp,
311:               suiteClass=suite.TestSuite):
312:     return _makeLoader(prefix, sortUsing, suiteClass).loadTestsFromTestCase(testCaseClass)
313: 
314: def findTestCases(module, prefix='test', sortUsing=cmp,
315:                   suiteClass=suite.TestSuite):
316:     return _makeLoader(prefix, sortUsing, suiteClass).loadTestsFromModule(module)
317: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_189154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Loading unittests.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import re' statement (line 4)
import re

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import traceback' statement (line 6)
import traceback

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'traceback', traceback, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import types' statement (line 7)
import types

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from functools import _CmpToKey' statement (line 9)
from functools import cmp_to_key as _CmpToKey

import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'functools', None, module_type_store, ['cmp_to_key'], [_CmpToKey])
# Adding an alias
module_type_store.add_alias('_CmpToKey', 'cmp_to_key')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from fnmatch import fnmatch' statement (line 10)
from fnmatch import fnmatch

import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'fnmatch', None, module_type_store, ['fnmatch'], [fnmatch])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from unittest import case, suite' statement (line 12)
from unittest import case, suite

import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'unittest', None, module_type_store, ['case', 'suite'], [case, suite])


# Assigning a Name to a Name (line 14):

# Assigning a Name to a Name (line 14):
# Getting the type of 'True' (line 14)
True_189155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'True')
# Assigning a type to the variable '__unittest' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__unittest', True_189155)

# Assigning a Call to a Name (line 19):

# Assigning a Call to a Name (line 19):

# Call to compile(...): (line 19)
# Processing the call arguments (line 19)
str_189158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'str', '[_a-z]\\w*\\.py$')
# Getting the type of 're' (line 19)
re_189159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 50), 're', False)
# Obtaining the member 'IGNORECASE' of a type (line 19)
IGNORECASE_189160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 50), re_189159, 'IGNORECASE')
# Processing the call keyword arguments (line 19)
kwargs_189161 = {}
# Getting the type of 're' (line 19)
re_189156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 're', False)
# Obtaining the member 'compile' of a type (line 19)
compile_189157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), re_189156, 'compile')
# Calling compile(args, kwargs) (line 19)
compile_call_result_189162 = invoke(stypy.reporting.localization.Localization(__file__, 19, 20), compile_189157, *[str_189158, IGNORECASE_189160], **kwargs_189161)

# Assigning a type to the variable 'VALID_MODULE_NAME' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'VALID_MODULE_NAME', compile_call_result_189162)

@norecursion
def _make_failed_import_test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_make_failed_import_test'
    module_type_store = module_type_store.open_function_context('_make_failed_import_test', 22, 0, False)
    
    # Passed parameters checking function
    _make_failed_import_test.stypy_localization = localization
    _make_failed_import_test.stypy_type_of_self = None
    _make_failed_import_test.stypy_type_store = module_type_store
    _make_failed_import_test.stypy_function_name = '_make_failed_import_test'
    _make_failed_import_test.stypy_param_names_list = ['name', 'suiteClass']
    _make_failed_import_test.stypy_varargs_param_name = None
    _make_failed_import_test.stypy_kwargs_param_name = None
    _make_failed_import_test.stypy_call_defaults = defaults
    _make_failed_import_test.stypy_call_varargs = varargs
    _make_failed_import_test.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_failed_import_test', ['name', 'suiteClass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_failed_import_test', localization, ['name', 'suiteClass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_failed_import_test(...)' code ##################

    
    # Assigning a BinOp to a Name (line 23):
    
    # Assigning a BinOp to a Name (line 23):
    str_189163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'str', 'Failed to import test module: %s\n%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_189164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    # Getting the type of 'name' (line 23)
    name_189165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 56), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 56), tuple_189164, name_189165)
    # Adding element type (line 23)
    
    # Call to format_exc(...): (line 23)
    # Processing the call keyword arguments (line 23)
    kwargs_189168 = {}
    # Getting the type of 'traceback' (line 23)
    traceback_189166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 62), 'traceback', False)
    # Obtaining the member 'format_exc' of a type (line 23)
    format_exc_189167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 62), traceback_189166, 'format_exc')
    # Calling format_exc(args, kwargs) (line 23)
    format_exc_call_result_189169 = invoke(stypy.reporting.localization.Localization(__file__, 23, 62), format_exc_189167, *[], **kwargs_189168)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 56), tuple_189164, format_exc_call_result_189169)
    
    # Applying the binary operator '%' (line 23)
    result_mod_189170 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 14), '%', str_189163, tuple_189164)
    
    # Assigning a type to the variable 'message' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'message', result_mod_189170)
    
    # Call to _make_failed_test(...): (line 24)
    # Processing the call arguments (line 24)
    str_189172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'str', 'ModuleImportFailure')
    # Getting the type of 'name' (line 24)
    name_189173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 52), 'name', False)
    
    # Call to ImportError(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'message' (line 24)
    message_189175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 70), 'message', False)
    # Processing the call keyword arguments (line 24)
    kwargs_189176 = {}
    # Getting the type of 'ImportError' (line 24)
    ImportError_189174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 58), 'ImportError', False)
    # Calling ImportError(args, kwargs) (line 24)
    ImportError_call_result_189177 = invoke(stypy.reporting.localization.Localization(__file__, 24, 58), ImportError_189174, *[message_189175], **kwargs_189176)
    
    # Getting the type of 'suiteClass' (line 25)
    suiteClass_189178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 29), 'suiteClass', False)
    # Processing the call keyword arguments (line 24)
    kwargs_189179 = {}
    # Getting the type of '_make_failed_test' (line 24)
    _make_failed_test_189171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), '_make_failed_test', False)
    # Calling _make_failed_test(args, kwargs) (line 24)
    _make_failed_test_call_result_189180 = invoke(stypy.reporting.localization.Localization(__file__, 24, 11), _make_failed_test_189171, *[str_189172, name_189173, ImportError_call_result_189177, suiteClass_189178], **kwargs_189179)
    
    # Assigning a type to the variable 'stypy_return_type' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type', _make_failed_test_call_result_189180)
    
    # ################# End of '_make_failed_import_test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_failed_import_test' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_189181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_189181)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_failed_import_test'
    return stypy_return_type_189181

# Assigning a type to the variable '_make_failed_import_test' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '_make_failed_import_test', _make_failed_import_test)

@norecursion
def _make_failed_load_tests(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_make_failed_load_tests'
    module_type_store = module_type_store.open_function_context('_make_failed_load_tests', 27, 0, False)
    
    # Passed parameters checking function
    _make_failed_load_tests.stypy_localization = localization
    _make_failed_load_tests.stypy_type_of_self = None
    _make_failed_load_tests.stypy_type_store = module_type_store
    _make_failed_load_tests.stypy_function_name = '_make_failed_load_tests'
    _make_failed_load_tests.stypy_param_names_list = ['name', 'exception', 'suiteClass']
    _make_failed_load_tests.stypy_varargs_param_name = None
    _make_failed_load_tests.stypy_kwargs_param_name = None
    _make_failed_load_tests.stypy_call_defaults = defaults
    _make_failed_load_tests.stypy_call_varargs = varargs
    _make_failed_load_tests.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_failed_load_tests', ['name', 'exception', 'suiteClass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_failed_load_tests', localization, ['name', 'exception', 'suiteClass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_failed_load_tests(...)' code ##################

    
    # Call to _make_failed_test(...): (line 28)
    # Processing the call arguments (line 28)
    str_189183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'str', 'LoadTestsFailure')
    # Getting the type of 'name' (line 28)
    name_189184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 49), 'name', False)
    # Getting the type of 'exception' (line 28)
    exception_189185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 55), 'exception', False)
    # Getting the type of 'suiteClass' (line 28)
    suiteClass_189186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 66), 'suiteClass', False)
    # Processing the call keyword arguments (line 28)
    kwargs_189187 = {}
    # Getting the type of '_make_failed_test' (line 28)
    _make_failed_test_189182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), '_make_failed_test', False)
    # Calling _make_failed_test(args, kwargs) (line 28)
    _make_failed_test_call_result_189188 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), _make_failed_test_189182, *[str_189183, name_189184, exception_189185, suiteClass_189186], **kwargs_189187)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type', _make_failed_test_call_result_189188)
    
    # ################# End of '_make_failed_load_tests(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_failed_load_tests' in the type store
    # Getting the type of 'stypy_return_type' (line 27)
    stypy_return_type_189189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_189189)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_failed_load_tests'
    return stypy_return_type_189189

# Assigning a type to the variable '_make_failed_load_tests' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), '_make_failed_load_tests', _make_failed_load_tests)

@norecursion
def _make_failed_test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_make_failed_test'
    module_type_store = module_type_store.open_function_context('_make_failed_test', 30, 0, False)
    
    # Passed parameters checking function
    _make_failed_test.stypy_localization = localization
    _make_failed_test.stypy_type_of_self = None
    _make_failed_test.stypy_type_store = module_type_store
    _make_failed_test.stypy_function_name = '_make_failed_test'
    _make_failed_test.stypy_param_names_list = ['classname', 'methodname', 'exception', 'suiteClass']
    _make_failed_test.stypy_varargs_param_name = None
    _make_failed_test.stypy_kwargs_param_name = None
    _make_failed_test.stypy_call_defaults = defaults
    _make_failed_test.stypy_call_varargs = varargs
    _make_failed_test.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_failed_test', ['classname', 'methodname', 'exception', 'suiteClass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_failed_test', localization, ['classname', 'methodname', 'exception', 'suiteClass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_failed_test(...)' code ##################


    @norecursion
    def testFailure(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testFailure'
        module_type_store = module_type_store.open_function_context('testFailure', 31, 4, False)
        
        # Passed parameters checking function
        testFailure.stypy_localization = localization
        testFailure.stypy_type_of_self = None
        testFailure.stypy_type_store = module_type_store
        testFailure.stypy_function_name = 'testFailure'
        testFailure.stypy_param_names_list = ['self']
        testFailure.stypy_varargs_param_name = None
        testFailure.stypy_kwargs_param_name = None
        testFailure.stypy_call_defaults = defaults
        testFailure.stypy_call_varargs = varargs
        testFailure.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'testFailure', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testFailure', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testFailure(...)' code ##################

        # Getting the type of 'exception' (line 32)
        exception_189190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'exception')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 32, 8), exception_189190, 'raise parameter', BaseException)
        
        # ################# End of 'testFailure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testFailure' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_189191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testFailure'
        return stypy_return_type_189191

    # Assigning a type to the variable 'testFailure' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'testFailure', testFailure)
    
    # Assigning a Dict to a Name (line 33):
    
    # Assigning a Dict to a Name (line 33):
    
    # Obtaining an instance of the builtin type 'dict' (line 33)
    dict_189192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 33)
    # Adding element type (key, value) (line 33)
    # Getting the type of 'methodname' (line 33)
    methodname_189193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'methodname')
    # Getting the type of 'testFailure' (line 33)
    testFailure_189194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'testFailure')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 12), dict_189192, (methodname_189193, testFailure_189194))
    
    # Assigning a type to the variable 'attrs' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'attrs', dict_189192)
    
    # Assigning a Call to a Name (line 34):
    
    # Assigning a Call to a Name (line 34):
    
    # Call to type(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'classname' (line 34)
    classname_189196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'classname', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_189197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    # Getting the type of 'case' (line 34)
    case_189198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 33), 'case', False)
    # Obtaining the member 'TestCase' of a type (line 34)
    TestCase_189199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 33), case_189198, 'TestCase')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 33), tuple_189197, TestCase_189199)
    
    # Getting the type of 'attrs' (line 34)
    attrs_189200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 50), 'attrs', False)
    # Processing the call keyword arguments (line 34)
    kwargs_189201 = {}
    # Getting the type of 'type' (line 34)
    type_189195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'type', False)
    # Calling type(args, kwargs) (line 34)
    type_call_result_189202 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), type_189195, *[classname_189196, tuple_189197, attrs_189200], **kwargs_189201)
    
    # Assigning a type to the variable 'TestClass' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'TestClass', type_call_result_189202)
    
    # Call to suiteClass(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_189204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    
    # Call to TestClass(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'methodname' (line 35)
    methodname_189206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'methodname', False)
    # Processing the call keyword arguments (line 35)
    kwargs_189207 = {}
    # Getting the type of 'TestClass' (line 35)
    TestClass_189205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'TestClass', False)
    # Calling TestClass(args, kwargs) (line 35)
    TestClass_call_result_189208 = invoke(stypy.reporting.localization.Localization(__file__, 35, 23), TestClass_189205, *[methodname_189206], **kwargs_189207)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 23), tuple_189204, TestClass_call_result_189208)
    
    # Processing the call keyword arguments (line 35)
    kwargs_189209 = {}
    # Getting the type of 'suiteClass' (line 35)
    suiteClass_189203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'suiteClass', False)
    # Calling suiteClass(args, kwargs) (line 35)
    suiteClass_call_result_189210 = invoke(stypy.reporting.localization.Localization(__file__, 35, 11), suiteClass_189203, *[tuple_189204], **kwargs_189209)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', suiteClass_call_result_189210)
    
    # ################# End of '_make_failed_test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_failed_test' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_189211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_189211)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_failed_test'
    return stypy_return_type_189211

# Assigning a type to the variable '_make_failed_test' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '_make_failed_test', _make_failed_test)
# Declaration of the 'TestLoader' class

class TestLoader(object, ):
    str_189212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'str', '\n    This class is responsible for loading tests according to various criteria\n    and returning them wrapped in a TestSuite\n    ')
    
    # Assigning a Str to a Name (line 43):
    
    # Assigning a Name to a Name (line 44):
    
    # Assigning a Attribute to a Name (line 45):
    
    # Assigning a Name to a Name (line 46):

    @norecursion
    def loadTestsFromTestCase(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'loadTestsFromTestCase'
        module_type_store = module_type_store.open_function_context('loadTestsFromTestCase', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_localization', localization)
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_function_name', 'TestLoader.loadTestsFromTestCase')
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_param_names_list', ['testCaseClass'])
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader.loadTestsFromTestCase.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader.loadTestsFromTestCase', ['testCaseClass'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'loadTestsFromTestCase', localization, ['testCaseClass'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'loadTestsFromTestCase(...)' code ##################

        str_189213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'str', 'Return a suite of all tests cases contained in testCaseClass')
        
        
        # Call to issubclass(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'testCaseClass' (line 50)
        testCaseClass_189215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'testCaseClass', False)
        # Getting the type of 'suite' (line 50)
        suite_189216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 37), 'suite', False)
        # Obtaining the member 'TestSuite' of a type (line 50)
        TestSuite_189217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 37), suite_189216, 'TestSuite')
        # Processing the call keyword arguments (line 50)
        kwargs_189218 = {}
        # Getting the type of 'issubclass' (line 50)
        issubclass_189214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 50)
        issubclass_call_result_189219 = invoke(stypy.reporting.localization.Localization(__file__, 50, 11), issubclass_189214, *[testCaseClass_189215, TestSuite_189217], **kwargs_189218)
        
        # Testing the type of an if condition (line 50)
        if_condition_189220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), issubclass_call_result_189219)
        # Assigning a type to the variable 'if_condition_189220' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_189220', if_condition_189220)
        # SSA begins for if statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 51)
        # Processing the call arguments (line 51)
        str_189222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'str', 'Test cases should not be derived from TestSuite. Maybe you meant to derive from TestCase?')
        # Processing the call keyword arguments (line 51)
        kwargs_189223 = {}
        # Getting the type of 'TypeError' (line 51)
        TypeError_189221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 51)
        TypeError_call_result_189224 = invoke(stypy.reporting.localization.Localization(__file__, 51, 18), TypeError_189221, *[str_189222], **kwargs_189223)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 51, 12), TypeError_call_result_189224, 'raise parameter', BaseException)
        # SSA join for if statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to getTestCaseNames(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'testCaseClass' (line 53)
        testCaseClass_189227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 46), 'testCaseClass', False)
        # Processing the call keyword arguments (line 53)
        kwargs_189228 = {}
        # Getting the type of 'self' (line 53)
        self_189225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 24), 'self', False)
        # Obtaining the member 'getTestCaseNames' of a type (line 53)
        getTestCaseNames_189226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 24), self_189225, 'getTestCaseNames')
        # Calling getTestCaseNames(args, kwargs) (line 53)
        getTestCaseNames_call_result_189229 = invoke(stypy.reporting.localization.Localization(__file__, 53, 24), getTestCaseNames_189226, *[testCaseClass_189227], **kwargs_189228)
        
        # Assigning a type to the variable 'testCaseNames' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'testCaseNames', getTestCaseNames_call_result_189229)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'testCaseNames' (line 54)
        testCaseNames_189230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'testCaseNames')
        # Applying the 'not' unary operator (line 54)
        result_not__189231 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'not', testCaseNames_189230)
        
        
        # Call to hasattr(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'testCaseClass' (line 54)
        testCaseClass_189233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 41), 'testCaseClass', False)
        str_189234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 56), 'str', 'runTest')
        # Processing the call keyword arguments (line 54)
        kwargs_189235 = {}
        # Getting the type of 'hasattr' (line 54)
        hasattr_189232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 54)
        hasattr_call_result_189236 = invoke(stypy.reporting.localization.Localization(__file__, 54, 33), hasattr_189232, *[testCaseClass_189233, str_189234], **kwargs_189235)
        
        # Applying the binary operator 'and' (line 54)
        result_and_keyword_189237 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'and', result_not__189231, hasattr_call_result_189236)
        
        # Testing the type of an if condition (line 54)
        if_condition_189238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), result_and_keyword_189237)
        # Assigning a type to the variable 'if_condition_189238' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_189238', if_condition_189238)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 55):
        
        # Assigning a List to a Name (line 55):
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_189239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        str_189240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'str', 'runTest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 28), list_189239, str_189240)
        
        # Assigning a type to the variable 'testCaseNames' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'testCaseNames', list_189239)
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to suiteClass(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Call to map(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'testCaseClass' (line 56)
        testCaseClass_189244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 43), 'testCaseClass', False)
        # Getting the type of 'testCaseNames' (line 56)
        testCaseNames_189245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 58), 'testCaseNames', False)
        # Processing the call keyword arguments (line 56)
        kwargs_189246 = {}
        # Getting the type of 'map' (line 56)
        map_189243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'map', False)
        # Calling map(args, kwargs) (line 56)
        map_call_result_189247 = invoke(stypy.reporting.localization.Localization(__file__, 56, 39), map_189243, *[testCaseClass_189244, testCaseNames_189245], **kwargs_189246)
        
        # Processing the call keyword arguments (line 56)
        kwargs_189248 = {}
        # Getting the type of 'self' (line 56)
        self_189241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'self', False)
        # Obtaining the member 'suiteClass' of a type (line 56)
        suiteClass_189242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 23), self_189241, 'suiteClass')
        # Calling suiteClass(args, kwargs) (line 56)
        suiteClass_call_result_189249 = invoke(stypy.reporting.localization.Localization(__file__, 56, 23), suiteClass_189242, *[map_call_result_189247], **kwargs_189248)
        
        # Assigning a type to the variable 'loaded_suite' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'loaded_suite', suiteClass_call_result_189249)
        # Getting the type of 'loaded_suite' (line 57)
        loaded_suite_189250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'loaded_suite')
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', loaded_suite_189250)
        
        # ################# End of 'loadTestsFromTestCase(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'loadTestsFromTestCase' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_189251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189251)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'loadTestsFromTestCase'
        return stypy_return_type_189251


    @norecursion
    def loadTestsFromModule(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 59)
        True_189252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 57), 'True')
        defaults = [True_189252]
        # Create a new context for function 'loadTestsFromModule'
        module_type_store = module_type_store.open_function_context('loadTestsFromModule', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_localization', localization)
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_function_name', 'TestLoader.loadTestsFromModule')
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_param_names_list', ['module', 'use_load_tests'])
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader.loadTestsFromModule.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader.loadTestsFromModule', ['module', 'use_load_tests'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'loadTestsFromModule', localization, ['module', 'use_load_tests'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'loadTestsFromModule(...)' code ##################

        str_189253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'str', 'Return a suite of all tests cases contained in the given module')
        
        # Assigning a List to a Name (line 61):
        
        # Assigning a List to a Name (line 61):
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_189254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        
        # Assigning a type to the variable 'tests' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tests', list_189254)
        
        
        # Call to dir(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'module' (line 62)
        module_189256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'module', False)
        # Processing the call keyword arguments (line 62)
        kwargs_189257 = {}
        # Getting the type of 'dir' (line 62)
        dir_189255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'dir', False)
        # Calling dir(args, kwargs) (line 62)
        dir_call_result_189258 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), dir_189255, *[module_189256], **kwargs_189257)
        
        # Testing the type of a for loop iterable (line 62)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 8), dir_call_result_189258)
        # Getting the type of the for loop variable (line 62)
        for_loop_var_189259 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 8), dir_call_result_189258)
        # Assigning a type to the variable 'name' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'name', for_loop_var_189259)
        # SSA begins for a for statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to getattr(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'module' (line 63)
        module_189261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 26), 'module', False)
        # Getting the type of 'name' (line 63)
        name_189262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 34), 'name', False)
        # Processing the call keyword arguments (line 63)
        kwargs_189263 = {}
        # Getting the type of 'getattr' (line 63)
        getattr_189260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 63)
        getattr_call_result_189264 = invoke(stypy.reporting.localization.Localization(__file__, 63, 18), getattr_189260, *[module_189261, name_189262], **kwargs_189263)
        
        # Assigning a type to the variable 'obj' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'obj', getattr_call_result_189264)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'obj' (line 64)
        obj_189266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'obj', False)
        # Getting the type of 'type' (line 64)
        type_189267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'type', False)
        # Processing the call keyword arguments (line 64)
        kwargs_189268 = {}
        # Getting the type of 'isinstance' (line 64)
        isinstance_189265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 64)
        isinstance_call_result_189269 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), isinstance_189265, *[obj_189266, type_189267], **kwargs_189268)
        
        
        # Call to issubclass(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'obj' (line 64)
        obj_189271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 52), 'obj', False)
        # Getting the type of 'case' (line 64)
        case_189272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 57), 'case', False)
        # Obtaining the member 'TestCase' of a type (line 64)
        TestCase_189273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 57), case_189272, 'TestCase')
        # Processing the call keyword arguments (line 64)
        kwargs_189274 = {}
        # Getting the type of 'issubclass' (line 64)
        issubclass_189270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 64)
        issubclass_call_result_189275 = invoke(stypy.reporting.localization.Localization(__file__, 64, 41), issubclass_189270, *[obj_189271, TestCase_189273], **kwargs_189274)
        
        # Applying the binary operator 'and' (line 64)
        result_and_keyword_189276 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 15), 'and', isinstance_call_result_189269, issubclass_call_result_189275)
        
        # Testing the type of an if condition (line 64)
        if_condition_189277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 12), result_and_keyword_189276)
        # Assigning a type to the variable 'if_condition_189277' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'if_condition_189277', if_condition_189277)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Call to loadTestsFromTestCase(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'obj' (line 65)
        obj_189282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 56), 'obj', False)
        # Processing the call keyword arguments (line 65)
        kwargs_189283 = {}
        # Getting the type of 'self' (line 65)
        self_189280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), 'self', False)
        # Obtaining the member 'loadTestsFromTestCase' of a type (line 65)
        loadTestsFromTestCase_189281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 29), self_189280, 'loadTestsFromTestCase')
        # Calling loadTestsFromTestCase(args, kwargs) (line 65)
        loadTestsFromTestCase_call_result_189284 = invoke(stypy.reporting.localization.Localization(__file__, 65, 29), loadTestsFromTestCase_189281, *[obj_189282], **kwargs_189283)
        
        # Processing the call keyword arguments (line 65)
        kwargs_189285 = {}
        # Getting the type of 'tests' (line 65)
        tests_189278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'tests', False)
        # Obtaining the member 'append' of a type (line 65)
        append_189279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), tests_189278, 'append')
        # Calling append(args, kwargs) (line 65)
        append_call_result_189286 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), append_189279, *[loadTestsFromTestCase_call_result_189284], **kwargs_189285)
        
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to getattr(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'module' (line 67)
        module_189288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'module', False)
        str_189289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 37), 'str', 'load_tests')
        # Getting the type of 'None' (line 67)
        None_189290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 51), 'None', False)
        # Processing the call keyword arguments (line 67)
        kwargs_189291 = {}
        # Getting the type of 'getattr' (line 67)
        getattr_189287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'getattr', False)
        # Calling getattr(args, kwargs) (line 67)
        getattr_call_result_189292 = invoke(stypy.reporting.localization.Localization(__file__, 67, 21), getattr_189287, *[module_189288, str_189289, None_189290], **kwargs_189291)
        
        # Assigning a type to the variable 'load_tests' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'load_tests', getattr_call_result_189292)
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to suiteClass(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'tests' (line 68)
        tests_189295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), 'tests', False)
        # Processing the call keyword arguments (line 68)
        kwargs_189296 = {}
        # Getting the type of 'self' (line 68)
        self_189293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'self', False)
        # Obtaining the member 'suiteClass' of a type (line 68)
        suiteClass_189294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), self_189293, 'suiteClass')
        # Calling suiteClass(args, kwargs) (line 68)
        suiteClass_call_result_189297 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), suiteClass_189294, *[tests_189295], **kwargs_189296)
        
        # Assigning a type to the variable 'tests' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tests', suiteClass_call_result_189297)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'use_load_tests' (line 69)
        use_load_tests_189298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'use_load_tests')
        
        # Getting the type of 'load_tests' (line 69)
        load_tests_189299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'load_tests')
        # Getting the type of 'None' (line 69)
        None_189300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 48), 'None')
        # Applying the binary operator 'isnot' (line 69)
        result_is_not_189301 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 30), 'isnot', load_tests_189299, None_189300)
        
        # Applying the binary operator 'and' (line 69)
        result_and_keyword_189302 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), 'and', use_load_tests_189298, result_is_not_189301)
        
        # Testing the type of an if condition (line 69)
        if_condition_189303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), result_and_keyword_189302)
        # Assigning a type to the variable 'if_condition_189303' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_189303', if_condition_189303)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to load_tests(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'self' (line 71)
        self_189305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'self', False)
        # Getting the type of 'tests' (line 71)
        tests_189306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'tests', False)
        # Getting the type of 'None' (line 71)
        None_189307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 47), 'None', False)
        # Processing the call keyword arguments (line 71)
        kwargs_189308 = {}
        # Getting the type of 'load_tests' (line 71)
        load_tests_189304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'load_tests', False)
        # Calling load_tests(args, kwargs) (line 71)
        load_tests_call_result_189309 = invoke(stypy.reporting.localization.Localization(__file__, 71, 23), load_tests_189304, *[self_189305, tests_189306, None_189307], **kwargs_189308)
        
        # Assigning a type to the variable 'stypy_return_type' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'stypy_return_type', load_tests_call_result_189309)
        # SSA branch for the except part of a try statement (line 70)
        # SSA branch for the except 'Exception' branch of a try statement (line 70)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 72)
        Exception_189310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'Exception')
        # Assigning a type to the variable 'e' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'e', Exception_189310)
        
        # Call to _make_failed_load_tests(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'module' (line 73)
        module_189312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 47), 'module', False)
        # Obtaining the member '__name__' of a type (line 73)
        name___189313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 47), module_189312, '__name__')
        # Getting the type of 'e' (line 73)
        e_189314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 64), 'e', False)
        # Getting the type of 'self' (line 74)
        self_189315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 47), 'self', False)
        # Obtaining the member 'suiteClass' of a type (line 74)
        suiteClass_189316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 47), self_189315, 'suiteClass')
        # Processing the call keyword arguments (line 73)
        kwargs_189317 = {}
        # Getting the type of '_make_failed_load_tests' (line 73)
        _make_failed_load_tests_189311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), '_make_failed_load_tests', False)
        # Calling _make_failed_load_tests(args, kwargs) (line 73)
        _make_failed_load_tests_call_result_189318 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), _make_failed_load_tests_189311, *[name___189313, e_189314, suiteClass_189316], **kwargs_189317)
        
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'stypy_return_type', _make_failed_load_tests_call_result_189318)
        # SSA join for try-except statement (line 70)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'tests' (line 75)
        tests_189319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'tests')
        # Assigning a type to the variable 'stypy_return_type' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type', tests_189319)
        
        # ################# End of 'loadTestsFromModule(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'loadTestsFromModule' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_189320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189320)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'loadTestsFromModule'
        return stypy_return_type_189320


    @norecursion
    def loadTestsFromName(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 77)
        None_189321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'None')
        defaults = [None_189321]
        # Create a new context for function 'loadTestsFromName'
        module_type_store = module_type_store.open_function_context('loadTestsFromName', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_localization', localization)
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_function_name', 'TestLoader.loadTestsFromName')
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_param_names_list', ['name', 'module'])
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader.loadTestsFromName.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader.loadTestsFromName', ['name', 'module'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'loadTestsFromName', localization, ['name', 'module'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'loadTestsFromName(...)' code ##################

        str_189322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', 'Return a suite of all tests cases given a string specifier.\n\n        The name may resolve either to a module, a test case class, a\n        test method within a test case class, or a callable object which\n        returns a TestCase or TestSuite instance.\n\n        The method optionally resolves the names relative to a given module.\n        ')
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to split(...): (line 86)
        # Processing the call arguments (line 86)
        str_189325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'str', '.')
        # Processing the call keyword arguments (line 86)
        kwargs_189326 = {}
        # Getting the type of 'name' (line 86)
        name_189323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'name', False)
        # Obtaining the member 'split' of a type (line 86)
        split_189324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), name_189323, 'split')
        # Calling split(args, kwargs) (line 86)
        split_call_result_189327 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), split_189324, *[str_189325], **kwargs_189326)
        
        # Assigning a type to the variable 'parts' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'parts', split_call_result_189327)
        
        # Type idiom detected: calculating its left and rigth part (line 87)
        # Getting the type of 'module' (line 87)
        module_189328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'module')
        # Getting the type of 'None' (line 87)
        None_189329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'None')
        
        (may_be_189330, more_types_in_union_189331) = may_be_none(module_189328, None_189329)

        if may_be_189330:

            if more_types_in_union_189331:
                # Runtime conditional SSA (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 88):
            
            # Assigning a Subscript to a Name (line 88):
            
            # Obtaining the type of the subscript
            slice_189332 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 88, 25), None, None, None)
            # Getting the type of 'parts' (line 88)
            parts_189333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'parts')
            # Obtaining the member '__getitem__' of a type (line 88)
            getitem___189334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 25), parts_189333, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 88)
            subscript_call_result_189335 = invoke(stypy.reporting.localization.Localization(__file__, 88, 25), getitem___189334, slice_189332)
            
            # Assigning a type to the variable 'parts_copy' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'parts_copy', subscript_call_result_189335)
            
            # Getting the type of 'parts_copy' (line 89)
            parts_copy_189336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'parts_copy')
            # Testing the type of an if condition (line 89)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 12), parts_copy_189336)
            # SSA begins for while statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            
            # SSA begins for try-except statement (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 91):
            
            # Assigning a Call to a Name (line 91):
            
            # Call to __import__(...): (line 91)
            # Processing the call arguments (line 91)
            
            # Call to join(...): (line 91)
            # Processing the call arguments (line 91)
            # Getting the type of 'parts_copy' (line 91)
            parts_copy_189340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'parts_copy', False)
            # Processing the call keyword arguments (line 91)
            kwargs_189341 = {}
            str_189338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 40), 'str', '.')
            # Obtaining the member 'join' of a type (line 91)
            join_189339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 40), str_189338, 'join')
            # Calling join(args, kwargs) (line 91)
            join_call_result_189342 = invoke(stypy.reporting.localization.Localization(__file__, 91, 40), join_189339, *[parts_copy_189340], **kwargs_189341)
            
            # Processing the call keyword arguments (line 91)
            kwargs_189343 = {}
            # Getting the type of '__import__' (line 91)
            import___189337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), '__import__', False)
            # Calling __import__(args, kwargs) (line 91)
            import___call_result_189344 = invoke(stypy.reporting.localization.Localization(__file__, 91, 29), import___189337, *[join_call_result_189342], **kwargs_189343)
            
            # Assigning a type to the variable 'module' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'module', import___call_result_189344)
            # SSA branch for the except part of a try statement (line 90)
            # SSA branch for the except 'ImportError' branch of a try statement (line 90)
            module_type_store.open_ssa_branch('except')
            # Deleting a member
            # Getting the type of 'parts_copy' (line 94)
            parts_copy_189345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'parts_copy')
            
            # Obtaining the type of the subscript
            int_189346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 35), 'int')
            # Getting the type of 'parts_copy' (line 94)
            parts_copy_189347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'parts_copy')
            # Obtaining the member '__getitem__' of a type (line 94)
            getitem___189348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 24), parts_copy_189347, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 94)
            subscript_call_result_189349 = invoke(stypy.reporting.localization.Localization(__file__, 94, 24), getitem___189348, int_189346)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 20), parts_copy_189345, subscript_call_result_189349)
            
            
            # Getting the type of 'parts_copy' (line 95)
            parts_copy_189350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 27), 'parts_copy')
            # Applying the 'not' unary operator (line 95)
            result_not__189351 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 23), 'not', parts_copy_189350)
            
            # Testing the type of an if condition (line 95)
            if_condition_189352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 20), result_not__189351)
            # Assigning a type to the variable 'if_condition_189352' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'if_condition_189352', if_condition_189352)
            # SSA begins for if statement (line 95)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 95)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for try-except statement (line 90)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for while statement (line 89)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Subscript to a Name (line 97):
            
            # Assigning a Subscript to a Name (line 97):
            
            # Obtaining the type of the subscript
            int_189353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 26), 'int')
            slice_189354 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 20), int_189353, None, None)
            # Getting the type of 'parts' (line 97)
            parts_189355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'parts')
            # Obtaining the member '__getitem__' of a type (line 97)
            getitem___189356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 20), parts_189355, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 97)
            subscript_call_result_189357 = invoke(stypy.reporting.localization.Localization(__file__, 97, 20), getitem___189356, slice_189354)
            
            # Assigning a type to the variable 'parts' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'parts', subscript_call_result_189357)

            if more_types_in_union_189331:
                # SSA join for if statement (line 87)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 98):
        
        # Assigning a Name to a Name (line 98):
        # Getting the type of 'module' (line 98)
        module_189358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'module')
        # Assigning a type to the variable 'obj' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'obj', module_189358)
        
        # Getting the type of 'parts' (line 99)
        parts_189359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'parts')
        # Testing the type of a for loop iterable (line 99)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 8), parts_189359)
        # Getting the type of the for loop variable (line 99)
        for_loop_var_189360 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 8), parts_189359)
        # Assigning a type to the variable 'part' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'part', for_loop_var_189360)
        # SSA begins for a for statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Tuple to a Tuple (line 100):
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'obj' (line 100)
        obj_189361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 26), 'obj')
        # Assigning a type to the variable 'tuple_assignment_189152' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_assignment_189152', obj_189361)
        
        # Assigning a Call to a Name (line 100):
        
        # Call to getattr(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'obj' (line 100)
        obj_189363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'obj', False)
        # Getting the type of 'part' (line 100)
        part_189364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), 'part', False)
        # Processing the call keyword arguments (line 100)
        kwargs_189365 = {}
        # Getting the type of 'getattr' (line 100)
        getattr_189362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'getattr', False)
        # Calling getattr(args, kwargs) (line 100)
        getattr_call_result_189366 = invoke(stypy.reporting.localization.Localization(__file__, 100, 31), getattr_189362, *[obj_189363, part_189364], **kwargs_189365)
        
        # Assigning a type to the variable 'tuple_assignment_189153' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_assignment_189153', getattr_call_result_189366)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'tuple_assignment_189152' (line 100)
        tuple_assignment_189152_189367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_assignment_189152')
        # Assigning a type to the variable 'parent' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'parent', tuple_assignment_189152_189367)
        
        # Assigning a Name to a Name (line 100):
        # Getting the type of 'tuple_assignment_189153' (line 100)
        tuple_assignment_189153_189368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'tuple_assignment_189153')
        # Assigning a type to the variable 'obj' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'obj', tuple_assignment_189153_189368)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'obj' (line 102)
        obj_189370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'obj', False)
        # Getting the type of 'types' (line 102)
        types_189371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'types', False)
        # Obtaining the member 'ModuleType' of a type (line 102)
        ModuleType_189372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 27), types_189371, 'ModuleType')
        # Processing the call keyword arguments (line 102)
        kwargs_189373 = {}
        # Getting the type of 'isinstance' (line 102)
        isinstance_189369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 102)
        isinstance_call_result_189374 = invoke(stypy.reporting.localization.Localization(__file__, 102, 11), isinstance_189369, *[obj_189370, ModuleType_189372], **kwargs_189373)
        
        # Testing the type of an if condition (line 102)
        if_condition_189375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), isinstance_call_result_189374)
        # Assigning a type to the variable 'if_condition_189375' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_189375', if_condition_189375)
        # SSA begins for if statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to loadTestsFromModule(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'obj' (line 103)
        obj_189378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 44), 'obj', False)
        # Processing the call keyword arguments (line 103)
        kwargs_189379 = {}
        # Getting the type of 'self' (line 103)
        self_189376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'self', False)
        # Obtaining the member 'loadTestsFromModule' of a type (line 103)
        loadTestsFromModule_189377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), self_189376, 'loadTestsFromModule')
        # Calling loadTestsFromModule(args, kwargs) (line 103)
        loadTestsFromModule_call_result_189380 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), loadTestsFromModule_189377, *[obj_189378], **kwargs_189379)
        
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', loadTestsFromModule_call_result_189380)
        # SSA branch for the else part of an if statement (line 102)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'obj' (line 104)
        obj_189382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'obj', False)
        # Getting the type of 'type' (line 104)
        type_189383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'type', False)
        # Processing the call keyword arguments (line 104)
        kwargs_189384 = {}
        # Getting the type of 'isinstance' (line 104)
        isinstance_189381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 104)
        isinstance_call_result_189385 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), isinstance_189381, *[obj_189382, type_189383], **kwargs_189384)
        
        
        # Call to issubclass(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'obj' (line 104)
        obj_189387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 50), 'obj', False)
        # Getting the type of 'case' (line 104)
        case_189388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 55), 'case', False)
        # Obtaining the member 'TestCase' of a type (line 104)
        TestCase_189389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 55), case_189388, 'TestCase')
        # Processing the call keyword arguments (line 104)
        kwargs_189390 = {}
        # Getting the type of 'issubclass' (line 104)
        issubclass_189386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 104)
        issubclass_call_result_189391 = invoke(stypy.reporting.localization.Localization(__file__, 104, 39), issubclass_189386, *[obj_189387, TestCase_189389], **kwargs_189390)
        
        # Applying the binary operator 'and' (line 104)
        result_and_keyword_189392 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 13), 'and', isinstance_call_result_189385, issubclass_call_result_189391)
        
        # Testing the type of an if condition (line 104)
        if_condition_189393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_and_keyword_189392)
        # Assigning a type to the variable 'if_condition_189393' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_189393', if_condition_189393)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to loadTestsFromTestCase(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'obj' (line 105)
        obj_189396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 46), 'obj', False)
        # Processing the call keyword arguments (line 105)
        kwargs_189397 = {}
        # Getting the type of 'self' (line 105)
        self_189394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'self', False)
        # Obtaining the member 'loadTestsFromTestCase' of a type (line 105)
        loadTestsFromTestCase_189395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), self_189394, 'loadTestsFromTestCase')
        # Calling loadTestsFromTestCase(args, kwargs) (line 105)
        loadTestsFromTestCase_call_result_189398 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), loadTestsFromTestCase_189395, *[obj_189396], **kwargs_189397)
        
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', loadTestsFromTestCase_call_result_189398)
        # SSA branch for the else part of an if statement (line 104)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'obj' (line 106)
        obj_189400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'obj', False)
        # Getting the type of 'types' (line 106)
        types_189401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'types', False)
        # Obtaining the member 'UnboundMethodType' of a type (line 106)
        UnboundMethodType_189402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 30), types_189401, 'UnboundMethodType')
        # Processing the call keyword arguments (line 106)
        kwargs_189403 = {}
        # Getting the type of 'isinstance' (line 106)
        isinstance_189399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 106)
        isinstance_call_result_189404 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), isinstance_189399, *[obj_189400, UnboundMethodType_189402], **kwargs_189403)
        
        
        # Call to isinstance(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'parent' (line 107)
        parent_189406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'parent', False)
        # Getting the type of 'type' (line 107)
        type_189407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'type', False)
        # Processing the call keyword arguments (line 107)
        kwargs_189408 = {}
        # Getting the type of 'isinstance' (line 107)
        isinstance_189405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 14), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 107)
        isinstance_call_result_189409 = invoke(stypy.reporting.localization.Localization(__file__, 107, 14), isinstance_189405, *[parent_189406, type_189407], **kwargs_189408)
        
        # Applying the binary operator 'and' (line 106)
        result_and_keyword_189410 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'and', isinstance_call_result_189404, isinstance_call_result_189409)
        
        # Call to issubclass(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'parent' (line 108)
        parent_189412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'parent', False)
        # Getting the type of 'case' (line 108)
        case_189413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'case', False)
        # Obtaining the member 'TestCase' of a type (line 108)
        TestCase_189414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 33), case_189413, 'TestCase')
        # Processing the call keyword arguments (line 108)
        kwargs_189415 = {}
        # Getting the type of 'issubclass' (line 108)
        issubclass_189411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 108)
        issubclass_call_result_189416 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), issubclass_189411, *[parent_189412, TestCase_189414], **kwargs_189415)
        
        # Applying the binary operator 'and' (line 106)
        result_and_keyword_189417 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 14), 'and', result_and_keyword_189410, issubclass_call_result_189416)
        
        # Testing the type of an if condition (line 106)
        if_condition_189418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_and_keyword_189417)
        # Assigning a type to the variable 'if_condition_189418' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_189418', if_condition_189418)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 109):
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        int_189419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'int')
        # Getting the type of 'parts' (line 109)
        parts_189420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'parts')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___189421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), parts_189420, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_189422 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), getitem___189421, int_189419)
        
        # Assigning a type to the variable 'name' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'name', subscript_call_result_189422)
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to parent(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'name' (line 110)
        name_189424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'name', False)
        # Processing the call keyword arguments (line 110)
        kwargs_189425 = {}
        # Getting the type of 'parent' (line 110)
        parent_189423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'parent', False)
        # Calling parent(args, kwargs) (line 110)
        parent_call_result_189426 = invoke(stypy.reporting.localization.Localization(__file__, 110, 19), parent_189423, *[name_189424], **kwargs_189425)
        
        # Assigning a type to the variable 'inst' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'inst', parent_call_result_189426)
        
        # Call to suiteClass(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_189429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        # Getting the type of 'inst' (line 111)
        inst_189430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'inst', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 35), list_189429, inst_189430)
        
        # Processing the call keyword arguments (line 111)
        kwargs_189431 = {}
        # Getting the type of 'self' (line 111)
        self_189427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'self', False)
        # Obtaining the member 'suiteClass' of a type (line 111)
        suiteClass_189428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 19), self_189427, 'suiteClass')
        # Calling suiteClass(args, kwargs) (line 111)
        suiteClass_call_result_189432 = invoke(stypy.reporting.localization.Localization(__file__, 111, 19), suiteClass_189428, *[list_189429], **kwargs_189431)
        
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', suiteClass_call_result_189432)
        # SSA branch for the else part of an if statement (line 106)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'obj' (line 112)
        obj_189434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'obj', False)
        # Getting the type of 'suite' (line 112)
        suite_189435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'suite', False)
        # Obtaining the member 'TestSuite' of a type (line 112)
        TestSuite_189436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 29), suite_189435, 'TestSuite')
        # Processing the call keyword arguments (line 112)
        kwargs_189437 = {}
        # Getting the type of 'isinstance' (line 112)
        isinstance_189433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 112)
        isinstance_call_result_189438 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), isinstance_189433, *[obj_189434, TestSuite_189436], **kwargs_189437)
        
        # Testing the type of an if condition (line 112)
        if_condition_189439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 13), isinstance_call_result_189438)
        # Assigning a type to the variable 'if_condition_189439' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'if_condition_189439', if_condition_189439)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'obj' (line 113)
        obj_189440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'stypy_return_type', obj_189440)
        # SSA branch for the else part of an if statement (line 112)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 114)
        str_189441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 26), 'str', '__call__')
        # Getting the type of 'obj' (line 114)
        obj_189442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'obj')
        
        (may_be_189443, more_types_in_union_189444) = may_provide_member(str_189441, obj_189442)

        if may_be_189443:

            if more_types_in_union_189444:
                # Runtime conditional SSA (line 114)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'obj' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'obj', remove_not_member_provider_from_union(obj_189442, '__call__'))
            
            # Assigning a Call to a Name (line 115):
            
            # Assigning a Call to a Name (line 115):
            
            # Call to obj(...): (line 115)
            # Processing the call keyword arguments (line 115)
            kwargs_189446 = {}
            # Getting the type of 'obj' (line 115)
            obj_189445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'obj', False)
            # Calling obj(args, kwargs) (line 115)
            obj_call_result_189447 = invoke(stypy.reporting.localization.Localization(__file__, 115, 19), obj_189445, *[], **kwargs_189446)
            
            # Assigning a type to the variable 'test' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'test', obj_call_result_189447)
            
            
            # Call to isinstance(...): (line 116)
            # Processing the call arguments (line 116)
            # Getting the type of 'test' (line 116)
            test_189449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'test', False)
            # Getting the type of 'suite' (line 116)
            suite_189450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 32), 'suite', False)
            # Obtaining the member 'TestSuite' of a type (line 116)
            TestSuite_189451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 32), suite_189450, 'TestSuite')
            # Processing the call keyword arguments (line 116)
            kwargs_189452 = {}
            # Getting the type of 'isinstance' (line 116)
            isinstance_189448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 116)
            isinstance_call_result_189453 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), isinstance_189448, *[test_189449, TestSuite_189451], **kwargs_189452)
            
            # Testing the type of an if condition (line 116)
            if_condition_189454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), isinstance_call_result_189453)
            # Assigning a type to the variable 'if_condition_189454' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_189454', if_condition_189454)
            # SSA begins for if statement (line 116)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'test' (line 117)
            test_189455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'test')
            # Assigning a type to the variable 'stypy_return_type' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'stypy_return_type', test_189455)
            # SSA branch for the else part of an if statement (line 116)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'test' (line 118)
            test_189457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'test', False)
            # Getting the type of 'case' (line 118)
            case_189458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), 'case', False)
            # Obtaining the member 'TestCase' of a type (line 118)
            TestCase_189459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 34), case_189458, 'TestCase')
            # Processing the call keyword arguments (line 118)
            kwargs_189460 = {}
            # Getting the type of 'isinstance' (line 118)
            isinstance_189456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 118)
            isinstance_call_result_189461 = invoke(stypy.reporting.localization.Localization(__file__, 118, 17), isinstance_189456, *[test_189457, TestCase_189459], **kwargs_189460)
            
            # Testing the type of an if condition (line 118)
            if_condition_189462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 17), isinstance_call_result_189461)
            # Assigning a type to the variable 'if_condition_189462' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'if_condition_189462', if_condition_189462)
            # SSA begins for if statement (line 118)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to suiteClass(...): (line 119)
            # Processing the call arguments (line 119)
            
            # Obtaining an instance of the builtin type 'list' (line 119)
            list_189465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 39), 'list')
            # Adding type elements to the builtin type 'list' instance (line 119)
            # Adding element type (line 119)
            # Getting the type of 'test' (line 119)
            test_189466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'test', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 39), list_189465, test_189466)
            
            # Processing the call keyword arguments (line 119)
            kwargs_189467 = {}
            # Getting the type of 'self' (line 119)
            self_189463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'self', False)
            # Obtaining the member 'suiteClass' of a type (line 119)
            suiteClass_189464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 23), self_189463, 'suiteClass')
            # Calling suiteClass(args, kwargs) (line 119)
            suiteClass_call_result_189468 = invoke(stypy.reporting.localization.Localization(__file__, 119, 23), suiteClass_189464, *[list_189465], **kwargs_189467)
            
            # Assigning a type to the variable 'stypy_return_type' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'stypy_return_type', suiteClass_call_result_189468)
            # SSA branch for the else part of an if statement (line 118)
            module_type_store.open_ssa_branch('else')
            
            # Call to TypeError(...): (line 121)
            # Processing the call arguments (line 121)
            str_189470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 32), 'str', 'calling %s returned %s, not a test')
            
            # Obtaining an instance of the builtin type 'tuple' (line 122)
            tuple_189471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 33), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 122)
            # Adding element type (line 122)
            # Getting the type of 'obj' (line 122)
            obj_189472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'obj', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 33), tuple_189471, obj_189472)
            # Adding element type (line 122)
            # Getting the type of 'test' (line 122)
            test_189473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'test', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 33), tuple_189471, test_189473)
            
            # Applying the binary operator '%' (line 121)
            result_mod_189474 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 32), '%', str_189470, tuple_189471)
            
            # Processing the call keyword arguments (line 121)
            kwargs_189475 = {}
            # Getting the type of 'TypeError' (line 121)
            TypeError_189469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 121)
            TypeError_call_result_189476 = invoke(stypy.reporting.localization.Localization(__file__, 121, 22), TypeError_189469, *[result_mod_189474], **kwargs_189475)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 121, 16), TypeError_call_result_189476, 'raise parameter', BaseException)
            # SSA join for if statement (line 118)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 116)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_189444:
                # Runtime conditional SSA for else branch (line 114)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_189443) or more_types_in_union_189444):
            # Assigning a type to the variable 'obj' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'obj', remove_member_provider_from_union(obj_189442, '__call__'))
            
            # Call to TypeError(...): (line 124)
            # Processing the call arguments (line 124)
            str_189478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 28), 'str', "don't know how to make test from: %s")
            # Getting the type of 'obj' (line 124)
            obj_189479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 69), 'obj', False)
            # Applying the binary operator '%' (line 124)
            result_mod_189480 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 28), '%', str_189478, obj_189479)
            
            # Processing the call keyword arguments (line 124)
            kwargs_189481 = {}
            # Getting the type of 'TypeError' (line 124)
            TypeError_189477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 124)
            TypeError_call_result_189482 = invoke(stypy.reporting.localization.Localization(__file__, 124, 18), TypeError_189477, *[result_mod_189480], **kwargs_189481)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 124, 12), TypeError_call_result_189482, 'raise parameter', BaseException)

            if (may_be_189443 and more_types_in_union_189444):
                # SSA join for if statement (line 114)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'loadTestsFromName(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'loadTestsFromName' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_189483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189483)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'loadTestsFromName'
        return stypy_return_type_189483


    @norecursion
    def loadTestsFromNames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 126)
        None_189484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 47), 'None')
        defaults = [None_189484]
        # Create a new context for function 'loadTestsFromNames'
        module_type_store = module_type_store.open_function_context('loadTestsFromNames', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_localization', localization)
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_function_name', 'TestLoader.loadTestsFromNames')
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_param_names_list', ['names', 'module'])
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader.loadTestsFromNames.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader.loadTestsFromNames', ['names', 'module'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'loadTestsFromNames', localization, ['names', 'module'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'loadTestsFromNames(...)' code ##################

        str_189485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', "Return a suite of all tests cases found using the given sequence\n        of string specifiers. See 'loadTestsFromName()'.\n        ")
        
        # Assigning a ListComp to a Name (line 130):
        
        # Assigning a ListComp to a Name (line 130):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'names' (line 130)
        names_189492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 67), 'names')
        comprehension_189493 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 18), names_189492)
        # Assigning a type to the variable 'name' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'name', comprehension_189493)
        
        # Call to loadTestsFromName(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'name' (line 130)
        name_189488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 41), 'name', False)
        # Getting the type of 'module' (line 130)
        module_189489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'module', False)
        # Processing the call keyword arguments (line 130)
        kwargs_189490 = {}
        # Getting the type of 'self' (line 130)
        self_189486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'self', False)
        # Obtaining the member 'loadTestsFromName' of a type (line 130)
        loadTestsFromName_189487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 18), self_189486, 'loadTestsFromName')
        # Calling loadTestsFromName(args, kwargs) (line 130)
        loadTestsFromName_call_result_189491 = invoke(stypy.reporting.localization.Localization(__file__, 130, 18), loadTestsFromName_189487, *[name_189488, module_189489], **kwargs_189490)
        
        list_189494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 18), list_189494, loadTestsFromName_call_result_189491)
        # Assigning a type to the variable 'suites' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'suites', list_189494)
        
        # Call to suiteClass(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'suites' (line 131)
        suites_189497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'suites', False)
        # Processing the call keyword arguments (line 131)
        kwargs_189498 = {}
        # Getting the type of 'self' (line 131)
        self_189495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'self', False)
        # Obtaining the member 'suiteClass' of a type (line 131)
        suiteClass_189496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 15), self_189495, 'suiteClass')
        # Calling suiteClass(args, kwargs) (line 131)
        suiteClass_call_result_189499 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), suiteClass_189496, *[suites_189497], **kwargs_189498)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', suiteClass_call_result_189499)
        
        # ################# End of 'loadTestsFromNames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'loadTestsFromNames' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_189500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189500)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'loadTestsFromNames'
        return stypy_return_type_189500


    @norecursion
    def getTestCaseNames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getTestCaseNames'
        module_type_store = module_type_store.open_function_context('getTestCaseNames', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_localization', localization)
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_function_name', 'TestLoader.getTestCaseNames')
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_param_names_list', ['testCaseClass'])
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader.getTestCaseNames.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader.getTestCaseNames', ['testCaseClass'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getTestCaseNames', localization, ['testCaseClass'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getTestCaseNames(...)' code ##################

        str_189501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, (-1)), 'str', 'Return a sorted sequence of method names found within testCaseClass\n        ')

        @norecursion
        def isTestMethod(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'testCaseClass' (line 136)
            testCaseClass_189502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'testCaseClass')
            # Getting the type of 'self' (line 137)
            self_189503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 32), 'self')
            # Obtaining the member 'testMethodPrefix' of a type (line 137)
            testMethodPrefix_189504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 32), self_189503, 'testMethodPrefix')
            defaults = [testCaseClass_189502, testMethodPrefix_189504]
            # Create a new context for function 'isTestMethod'
            module_type_store = module_type_store.open_function_context('isTestMethod', 136, 8, False)
            
            # Passed parameters checking function
            isTestMethod.stypy_localization = localization
            isTestMethod.stypy_type_of_self = None
            isTestMethod.stypy_type_store = module_type_store
            isTestMethod.stypy_function_name = 'isTestMethod'
            isTestMethod.stypy_param_names_list = ['attrname', 'testCaseClass', 'prefix']
            isTestMethod.stypy_varargs_param_name = None
            isTestMethod.stypy_kwargs_param_name = None
            isTestMethod.stypy_call_defaults = defaults
            isTestMethod.stypy_call_varargs = varargs
            isTestMethod.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'isTestMethod', ['attrname', 'testCaseClass', 'prefix'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'isTestMethod', localization, ['attrname', 'testCaseClass', 'prefix'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'isTestMethod(...)' code ##################

            
            # Evaluating a boolean operation
            
            # Call to startswith(...): (line 138)
            # Processing the call arguments (line 138)
            # Getting the type of 'prefix' (line 138)
            prefix_189507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 39), 'prefix', False)
            # Processing the call keyword arguments (line 138)
            kwargs_189508 = {}
            # Getting the type of 'attrname' (line 138)
            attrname_189505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'attrname', False)
            # Obtaining the member 'startswith' of a type (line 138)
            startswith_189506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), attrname_189505, 'startswith')
            # Calling startswith(args, kwargs) (line 138)
            startswith_call_result_189509 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), startswith_189506, *[prefix_189507], **kwargs_189508)
            
            
            # Call to hasattr(...): (line 139)
            # Processing the call arguments (line 139)
            
            # Call to getattr(...): (line 139)
            # Processing the call arguments (line 139)
            # Getting the type of 'testCaseClass' (line 139)
            testCaseClass_189512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 32), 'testCaseClass', False)
            # Getting the type of 'attrname' (line 139)
            attrname_189513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 47), 'attrname', False)
            # Processing the call keyword arguments (line 139)
            kwargs_189514 = {}
            # Getting the type of 'getattr' (line 139)
            getattr_189511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'getattr', False)
            # Calling getattr(args, kwargs) (line 139)
            getattr_call_result_189515 = invoke(stypy.reporting.localization.Localization(__file__, 139, 24), getattr_189511, *[testCaseClass_189512, attrname_189513], **kwargs_189514)
            
            str_189516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 58), 'str', '__call__')
            # Processing the call keyword arguments (line 139)
            kwargs_189517 = {}
            # Getting the type of 'hasattr' (line 139)
            hasattr_189510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 139)
            hasattr_call_result_189518 = invoke(stypy.reporting.localization.Localization(__file__, 139, 16), hasattr_189510, *[getattr_call_result_189515, str_189516], **kwargs_189517)
            
            # Applying the binary operator 'and' (line 138)
            result_and_keyword_189519 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 19), 'and', startswith_call_result_189509, hasattr_call_result_189518)
            
            # Assigning a type to the variable 'stypy_return_type' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'stypy_return_type', result_and_keyword_189519)
            
            # ################# End of 'isTestMethod(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'isTestMethod' in the type store
            # Getting the type of 'stypy_return_type' (line 136)
            stypy_return_type_189520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_189520)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'isTestMethod'
            return stypy_return_type_189520

        # Assigning a type to the variable 'isTestMethod' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'isTestMethod', isTestMethod)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to filter(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'isTestMethod' (line 140)
        isTestMethod_189522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'isTestMethod', False)
        
        # Call to dir(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'testCaseClass' (line 140)
        testCaseClass_189524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 47), 'testCaseClass', False)
        # Processing the call keyword arguments (line 140)
        kwargs_189525 = {}
        # Getting the type of 'dir' (line 140)
        dir_189523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'dir', False)
        # Calling dir(args, kwargs) (line 140)
        dir_call_result_189526 = invoke(stypy.reporting.localization.Localization(__file__, 140, 43), dir_189523, *[testCaseClass_189524], **kwargs_189525)
        
        # Processing the call keyword arguments (line 140)
        kwargs_189527 = {}
        # Getting the type of 'filter' (line 140)
        filter_189521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 22), 'filter', False)
        # Calling filter(args, kwargs) (line 140)
        filter_call_result_189528 = invoke(stypy.reporting.localization.Localization(__file__, 140, 22), filter_189521, *[isTestMethod_189522, dir_call_result_189526], **kwargs_189527)
        
        # Assigning a type to the variable 'testFnNames' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'testFnNames', filter_call_result_189528)
        
        # Getting the type of 'self' (line 141)
        self_189529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'self')
        # Obtaining the member 'sortTestMethodsUsing' of a type (line 141)
        sortTestMethodsUsing_189530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 11), self_189529, 'sortTestMethodsUsing')
        # Testing the type of an if condition (line 141)
        if_condition_189531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), sortTestMethodsUsing_189530)
        # Assigning a type to the variable 'if_condition_189531' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_189531', if_condition_189531)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to sort(...): (line 142)
        # Processing the call keyword arguments (line 142)
        
        # Call to _CmpToKey(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_189535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 43), 'self', False)
        # Obtaining the member 'sortTestMethodsUsing' of a type (line 142)
        sortTestMethodsUsing_189536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 43), self_189535, 'sortTestMethodsUsing')
        # Processing the call keyword arguments (line 142)
        kwargs_189537 = {}
        # Getting the type of '_CmpToKey' (line 142)
        _CmpToKey_189534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 33), '_CmpToKey', False)
        # Calling _CmpToKey(args, kwargs) (line 142)
        _CmpToKey_call_result_189538 = invoke(stypy.reporting.localization.Localization(__file__, 142, 33), _CmpToKey_189534, *[sortTestMethodsUsing_189536], **kwargs_189537)
        
        keyword_189539 = _CmpToKey_call_result_189538
        kwargs_189540 = {'key': keyword_189539}
        # Getting the type of 'testFnNames' (line 142)
        testFnNames_189532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'testFnNames', False)
        # Obtaining the member 'sort' of a type (line 142)
        sort_189533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), testFnNames_189532, 'sort')
        # Calling sort(args, kwargs) (line 142)
        sort_call_result_189541 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), sort_189533, *[], **kwargs_189540)
        
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'testFnNames' (line 143)
        testFnNames_189542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'testFnNames')
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', testFnNames_189542)
        
        # ################# End of 'getTestCaseNames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getTestCaseNames' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_189543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189543)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getTestCaseNames'
        return stypy_return_type_189543


    @norecursion
    def discover(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_189544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 42), 'str', 'test*.py')
        # Getting the type of 'None' (line 145)
        None_189545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 68), 'None')
        defaults = [str_189544, None_189545]
        # Create a new context for function 'discover'
        module_type_store = module_type_store.open_function_context('discover', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader.discover.__dict__.__setitem__('stypy_localization', localization)
        TestLoader.discover.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader.discover.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader.discover.__dict__.__setitem__('stypy_function_name', 'TestLoader.discover')
        TestLoader.discover.__dict__.__setitem__('stypy_param_names_list', ['start_dir', 'pattern', 'top_level_dir'])
        TestLoader.discover.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader.discover.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader.discover.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader.discover.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader.discover.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader.discover.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader.discover', ['start_dir', 'pattern', 'top_level_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'discover', localization, ['start_dir', 'pattern', 'top_level_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'discover(...)' code ##################

        str_189546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, (-1)), 'str', "Find and return all test modules from the specified start\n        directory, recursing into subdirectories to find them. Only test files\n        that match the pattern will be loaded. (Using shell style pattern\n        matching.)\n\n        All test modules must be importable from the top level of the project.\n        If the start directory is not the top level directory then the top\n        level directory must be specified separately.\n\n        If a test package name (directory with '__init__.py') matches the\n        pattern then the package will be checked for a 'load_tests' function. If\n        this exists then it will be called with loader, tests, pattern.\n\n        If load_tests exists then discovery does  *not* recurse into the package,\n        load_tests is responsible for loading all tests in the package.\n\n        The pattern is deliberately not stored as a loader attribute so that\n        packages can continue discovery themselves. top_level_dir is stored so\n        load_tests does not need to pass this argument in to loader.discover().\n        ")
        
        # Assigning a Name to a Name (line 166):
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'False' (line 166)
        False_189547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 27), 'False')
        # Assigning a type to the variable 'set_implicit_top' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'set_implicit_top', False_189547)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'top_level_dir' (line 167)
        top_level_dir_189548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'top_level_dir')
        # Getting the type of 'None' (line 167)
        None_189549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'None')
        # Applying the binary operator 'is' (line 167)
        result_is__189550 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 11), 'is', top_level_dir_189548, None_189549)
        
        
        # Getting the type of 'self' (line 167)
        self_189551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'self')
        # Obtaining the member '_top_level_dir' of a type (line 167)
        _top_level_dir_189552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 37), self_189551, '_top_level_dir')
        # Getting the type of 'None' (line 167)
        None_189553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 64), 'None')
        # Applying the binary operator 'isnot' (line 167)
        result_is_not_189554 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 37), 'isnot', _top_level_dir_189552, None_189553)
        
        # Applying the binary operator 'and' (line 167)
        result_and_keyword_189555 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 11), 'and', result_is__189550, result_is_not_189554)
        
        # Testing the type of an if condition (line 167)
        if_condition_189556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), result_and_keyword_189555)
        # Assigning a type to the variable 'if_condition_189556' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_189556', if_condition_189556)
        # SSA begins for if statement (line 167)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 169):
        
        # Assigning a Attribute to a Name (line 169):
        # Getting the type of 'self' (line 169)
        self_189557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'self')
        # Obtaining the member '_top_level_dir' of a type (line 169)
        _top_level_dir_189558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 28), self_189557, '_top_level_dir')
        # Assigning a type to the variable 'top_level_dir' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'top_level_dir', _top_level_dir_189558)
        # SSA branch for the else part of an if statement (line 167)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 170)
        # Getting the type of 'top_level_dir' (line 170)
        top_level_dir_189559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 13), 'top_level_dir')
        # Getting the type of 'None' (line 170)
        None_189560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'None')
        
        (may_be_189561, more_types_in_union_189562) = may_be_none(top_level_dir_189559, None_189560)

        if may_be_189561:

            if more_types_in_union_189562:
                # Runtime conditional SSA (line 170)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 171):
            
            # Assigning a Name to a Name (line 171):
            # Getting the type of 'True' (line 171)
            True_189563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'True')
            # Assigning a type to the variable 'set_implicit_top' (line 171)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'set_implicit_top', True_189563)
            
            # Assigning a Name to a Name (line 172):
            
            # Assigning a Name to a Name (line 172):
            # Getting the type of 'start_dir' (line 172)
            start_dir_189564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'start_dir')
            # Assigning a type to the variable 'top_level_dir' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'top_level_dir', start_dir_189564)

            if more_types_in_union_189562:
                # SSA join for if statement (line 170)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 167)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to abspath(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'top_level_dir' (line 174)
        top_level_dir_189568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 40), 'top_level_dir', False)
        # Processing the call keyword arguments (line 174)
        kwargs_189569 = {}
        # Getting the type of 'os' (line 174)
        os_189565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 174)
        path_189566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 24), os_189565, 'path')
        # Obtaining the member 'abspath' of a type (line 174)
        abspath_189567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 24), path_189566, 'abspath')
        # Calling abspath(args, kwargs) (line 174)
        abspath_call_result_189570 = invoke(stypy.reporting.localization.Localization(__file__, 174, 24), abspath_189567, *[top_level_dir_189568], **kwargs_189569)
        
        # Assigning a type to the variable 'top_level_dir' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'top_level_dir', abspath_call_result_189570)
        
        
        
        # Getting the type of 'top_level_dir' (line 176)
        top_level_dir_189571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'top_level_dir')
        # Getting the type of 'sys' (line 176)
        sys_189572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'sys')
        # Obtaining the member 'path' of a type (line 176)
        path_189573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 32), sys_189572, 'path')
        # Applying the binary operator 'in' (line 176)
        result_contains_189574 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 15), 'in', top_level_dir_189571, path_189573)
        
        # Applying the 'not' unary operator (line 176)
        result_not__189575 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 11), 'not', result_contains_189574)
        
        # Testing the type of an if condition (line 176)
        if_condition_189576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 8), result_not__189575)
        # Assigning a type to the variable 'if_condition_189576' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'if_condition_189576', if_condition_189576)
        # SSA begins for if statement (line 176)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to insert(...): (line 181)
        # Processing the call arguments (line 181)
        int_189580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 28), 'int')
        # Getting the type of 'top_level_dir' (line 181)
        top_level_dir_189581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'top_level_dir', False)
        # Processing the call keyword arguments (line 181)
        kwargs_189582 = {}
        # Getting the type of 'sys' (line 181)
        sys_189577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'sys', False)
        # Obtaining the member 'path' of a type (line 181)
        path_189578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 12), sys_189577, 'path')
        # Obtaining the member 'insert' of a type (line 181)
        insert_189579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 12), path_189578, 'insert')
        # Calling insert(args, kwargs) (line 181)
        insert_call_result_189583 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), insert_189579, *[int_189580, top_level_dir_189581], **kwargs_189582)
        
        # SSA join for if statement (line 176)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 182):
        
        # Assigning a Name to a Attribute (line 182):
        # Getting the type of 'top_level_dir' (line 182)
        top_level_dir_189584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'top_level_dir')
        # Getting the type of 'self' (line 182)
        self_189585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'self')
        # Setting the type of the member '_top_level_dir' of a type (line 182)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), self_189585, '_top_level_dir', top_level_dir_189584)
        
        # Assigning a Name to a Name (line 184):
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'False' (line 184)
        False_189586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'False')
        # Assigning a type to the variable 'is_not_importable' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'is_not_importable', False_189586)
        
        
        # Call to isdir(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Call to abspath(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'start_dir' (line 185)
        start_dir_189593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 41), 'start_dir', False)
        # Processing the call keyword arguments (line 185)
        kwargs_189594 = {}
        # Getting the type of 'os' (line 185)
        os_189590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 185)
        path_189591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 25), os_189590, 'path')
        # Obtaining the member 'abspath' of a type (line 185)
        abspath_189592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 25), path_189591, 'abspath')
        # Calling abspath(args, kwargs) (line 185)
        abspath_call_result_189595 = invoke(stypy.reporting.localization.Localization(__file__, 185, 25), abspath_189592, *[start_dir_189593], **kwargs_189594)
        
        # Processing the call keyword arguments (line 185)
        kwargs_189596 = {}
        # Getting the type of 'os' (line 185)
        os_189587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 185)
        path_189588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 11), os_189587, 'path')
        # Obtaining the member 'isdir' of a type (line 185)
        isdir_189589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 11), path_189588, 'isdir')
        # Calling isdir(args, kwargs) (line 185)
        isdir_call_result_189597 = invoke(stypy.reporting.localization.Localization(__file__, 185, 11), isdir_189589, *[abspath_call_result_189595], **kwargs_189596)
        
        # Testing the type of an if condition (line 185)
        if_condition_189598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 8), isdir_call_result_189597)
        # Assigning a type to the variable 'if_condition_189598' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'if_condition_189598', if_condition_189598)
        # SSA begins for if statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 186):
        
        # Assigning a Call to a Name (line 186):
        
        # Call to abspath(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'start_dir' (line 186)
        start_dir_189602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 40), 'start_dir', False)
        # Processing the call keyword arguments (line 186)
        kwargs_189603 = {}
        # Getting the type of 'os' (line 186)
        os_189599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 186)
        path_189600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 24), os_189599, 'path')
        # Obtaining the member 'abspath' of a type (line 186)
        abspath_189601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 24), path_189600, 'abspath')
        # Calling abspath(args, kwargs) (line 186)
        abspath_call_result_189604 = invoke(stypy.reporting.localization.Localization(__file__, 186, 24), abspath_189601, *[start_dir_189602], **kwargs_189603)
        
        # Assigning a type to the variable 'start_dir' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'start_dir', abspath_call_result_189604)
        
        
        # Getting the type of 'start_dir' (line 187)
        start_dir_189605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'start_dir')
        # Getting the type of 'top_level_dir' (line 187)
        top_level_dir_189606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'top_level_dir')
        # Applying the binary operator '!=' (line 187)
        result_ne_189607 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 15), '!=', start_dir_189605, top_level_dir_189606)
        
        # Testing the type of an if condition (line 187)
        if_condition_189608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 12), result_ne_189607)
        # Assigning a type to the variable 'if_condition_189608' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'if_condition_189608', if_condition_189608)
        # SSA begins for if statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a UnaryOp to a Name (line 188):
        
        # Assigning a UnaryOp to a Name (line 188):
        
        
        # Call to isfile(...): (line 188)
        # Processing the call arguments (line 188)
        
        # Call to join(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'start_dir' (line 188)
        start_dir_189615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 68), 'start_dir', False)
        str_189616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 79), 'str', '__init__.py')
        # Processing the call keyword arguments (line 188)
        kwargs_189617 = {}
        # Getting the type of 'os' (line 188)
        os_189612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 55), 'os', False)
        # Obtaining the member 'path' of a type (line 188)
        path_189613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 55), os_189612, 'path')
        # Obtaining the member 'join' of a type (line 188)
        join_189614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 55), path_189613, 'join')
        # Calling join(args, kwargs) (line 188)
        join_call_result_189618 = invoke(stypy.reporting.localization.Localization(__file__, 188, 55), join_189614, *[start_dir_189615, str_189616], **kwargs_189617)
        
        # Processing the call keyword arguments (line 188)
        kwargs_189619 = {}
        # Getting the type of 'os' (line 188)
        os_189609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 40), 'os', False)
        # Obtaining the member 'path' of a type (line 188)
        path_189610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 40), os_189609, 'path')
        # Obtaining the member 'isfile' of a type (line 188)
        isfile_189611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 40), path_189610, 'isfile')
        # Calling isfile(args, kwargs) (line 188)
        isfile_call_result_189620 = invoke(stypy.reporting.localization.Localization(__file__, 188, 40), isfile_189611, *[join_call_result_189618], **kwargs_189619)
        
        # Applying the 'not' unary operator (line 188)
        result_not__189621 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 36), 'not', isfile_call_result_189620)
        
        # Assigning a type to the variable 'is_not_importable' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'is_not_importable', result_not__189621)
        # SSA join for if statement (line 187)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 185)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to __import__(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'start_dir' (line 192)
        start_dir_189623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'start_dir', False)
        # Processing the call keyword arguments (line 192)
        kwargs_189624 = {}
        # Getting the type of '__import__' (line 192)
        import___189622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), '__import__', False)
        # Calling __import__(args, kwargs) (line 192)
        import___call_result_189625 = invoke(stypy.reporting.localization.Localization(__file__, 192, 16), import___189622, *[start_dir_189623], **kwargs_189624)
        
        # SSA branch for the except part of a try statement (line 191)
        # SSA branch for the except 'ImportError' branch of a try statement (line 191)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 194):
        
        # Assigning a Name to a Name (line 194):
        # Getting the type of 'True' (line 194)
        True_189626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 36), 'True')
        # Assigning a type to the variable 'is_not_importable' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'is_not_importable', True_189626)
        # SSA branch for the else branch of a try statement (line 191)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Subscript to a Name (line 196):
        
        # Assigning a Subscript to a Name (line 196):
        
        # Obtaining the type of the subscript
        # Getting the type of 'start_dir' (line 196)
        start_dir_189627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 41), 'start_dir')
        # Getting the type of 'sys' (line 196)
        sys_189628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 29), 'sys')
        # Obtaining the member 'modules' of a type (line 196)
        modules_189629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 29), sys_189628, 'modules')
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___189630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 29), modules_189629, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_189631 = invoke(stypy.reporting.localization.Localization(__file__, 196, 29), getitem___189630, start_dir_189627)
        
        # Assigning a type to the variable 'the_module' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'the_module', subscript_call_result_189631)
        
        # Assigning a Subscript to a Name (line 197):
        
        # Assigning a Subscript to a Name (line 197):
        
        # Obtaining the type of the subscript
        int_189632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 48), 'int')
        
        # Call to split(...): (line 197)
        # Processing the call arguments (line 197)
        str_189635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 43), 'str', '.')
        # Processing the call keyword arguments (line 197)
        kwargs_189636 = {}
        # Getting the type of 'start_dir' (line 197)
        start_dir_189633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'start_dir', False)
        # Obtaining the member 'split' of a type (line 197)
        split_189634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 27), start_dir_189633, 'split')
        # Calling split(args, kwargs) (line 197)
        split_call_result_189637 = invoke(stypy.reporting.localization.Localization(__file__, 197, 27), split_189634, *[str_189635], **kwargs_189636)
        
        # Obtaining the member '__getitem__' of a type (line 197)
        getitem___189638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 27), split_call_result_189637, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 197)
        subscript_call_result_189639 = invoke(stypy.reporting.localization.Localization(__file__, 197, 27), getitem___189638, int_189632)
        
        # Assigning a type to the variable 'top_part' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'top_part', subscript_call_result_189639)
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to abspath(...): (line 198)
        # Processing the call arguments (line 198)
        
        # Call to dirname(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'the_module' (line 198)
        the_module_189646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 61), 'the_module', False)
        # Obtaining the member '__file__' of a type (line 198)
        file___189647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 61), the_module_189646, '__file__')
        # Processing the call keyword arguments (line 198)
        kwargs_189648 = {}
        # Getting the type of 'os' (line 198)
        os_189643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 198)
        path_189644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 44), os_189643, 'path')
        # Obtaining the member 'dirname' of a type (line 198)
        dirname_189645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 44), path_189644, 'dirname')
        # Calling dirname(args, kwargs) (line 198)
        dirname_call_result_189649 = invoke(stypy.reporting.localization.Localization(__file__, 198, 44), dirname_189645, *[file___189647], **kwargs_189648)
        
        # Processing the call keyword arguments (line 198)
        kwargs_189650 = {}
        # Getting the type of 'os' (line 198)
        os_189640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 198)
        path_189641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 28), os_189640, 'path')
        # Obtaining the member 'abspath' of a type (line 198)
        abspath_189642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 28), path_189641, 'abspath')
        # Calling abspath(args, kwargs) (line 198)
        abspath_call_result_189651 = invoke(stypy.reporting.localization.Localization(__file__, 198, 28), abspath_189642, *[dirname_call_result_189649], **kwargs_189650)
        
        # Assigning a type to the variable 'start_dir' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'start_dir', abspath_call_result_189651)
        
        # Getting the type of 'set_implicit_top' (line 199)
        set_implicit_top_189652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'set_implicit_top')
        # Testing the type of an if condition (line 199)
        if_condition_189653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 16), set_implicit_top_189652)
        # Assigning a type to the variable 'if_condition_189653' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'if_condition_189653', if_condition_189653)
        # SSA begins for if statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 200):
        
        # Assigning a Call to a Attribute (line 200):
        
        # Call to _get_directory_containing_module(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'top_part' (line 200)
        top_part_189656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 80), 'top_part', False)
        # Processing the call keyword arguments (line 200)
        kwargs_189657 = {}
        # Getting the type of 'self' (line 200)
        self_189654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 42), 'self', False)
        # Obtaining the member '_get_directory_containing_module' of a type (line 200)
        _get_directory_containing_module_189655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 42), self_189654, '_get_directory_containing_module')
        # Calling _get_directory_containing_module(args, kwargs) (line 200)
        _get_directory_containing_module_call_result_189658 = invoke(stypy.reporting.localization.Localization(__file__, 200, 42), _get_directory_containing_module_189655, *[top_part_189656], **kwargs_189657)
        
        # Getting the type of 'self' (line 200)
        self_189659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'self')
        # Setting the type of the member '_top_level_dir' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 20), self_189659, '_top_level_dir', _get_directory_containing_module_call_result_189658)
        
        # Call to remove(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'top_level_dir' (line 201)
        top_level_dir_189663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 36), 'top_level_dir', False)
        # Processing the call keyword arguments (line 201)
        kwargs_189664 = {}
        # Getting the type of 'sys' (line 201)
        sys_189660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'sys', False)
        # Obtaining the member 'path' of a type (line 201)
        path_189661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), sys_189660, 'path')
        # Obtaining the member 'remove' of a type (line 201)
        remove_189662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), path_189661, 'remove')
        # Calling remove(args, kwargs) (line 201)
        remove_call_result_189665 = invoke(stypy.reporting.localization.Localization(__file__, 201, 20), remove_189662, *[top_level_dir_189663], **kwargs_189664)
        
        # SSA join for if statement (line 199)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 185)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'is_not_importable' (line 203)
        is_not_importable_189666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'is_not_importable')
        # Testing the type of an if condition (line 203)
        if_condition_189667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), is_not_importable_189666)
        # Assigning a type to the variable 'if_condition_189667' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_189667', if_condition_189667)
        # SSA begins for if statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ImportError(...): (line 204)
        # Processing the call arguments (line 204)
        str_189669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 30), 'str', 'Start directory is not importable: %r')
        # Getting the type of 'start_dir' (line 204)
        start_dir_189670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 72), 'start_dir', False)
        # Applying the binary operator '%' (line 204)
        result_mod_189671 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 30), '%', str_189669, start_dir_189670)
        
        # Processing the call keyword arguments (line 204)
        kwargs_189672 = {}
        # Getting the type of 'ImportError' (line 204)
        ImportError_189668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'ImportError', False)
        # Calling ImportError(args, kwargs) (line 204)
        ImportError_call_result_189673 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), ImportError_189668, *[result_mod_189671], **kwargs_189672)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 204, 12), ImportError_call_result_189673, 'raise parameter', BaseException)
        # SSA join for if statement (line 203)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to list(...): (line 206)
        # Processing the call arguments (line 206)
        
        # Call to _find_tests(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'start_dir' (line 206)
        start_dir_189677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 38), 'start_dir', False)
        # Getting the type of 'pattern' (line 206)
        pattern_189678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 49), 'pattern', False)
        # Processing the call keyword arguments (line 206)
        kwargs_189679 = {}
        # Getting the type of 'self' (line 206)
        self_189675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 21), 'self', False)
        # Obtaining the member '_find_tests' of a type (line 206)
        _find_tests_189676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 21), self_189675, '_find_tests')
        # Calling _find_tests(args, kwargs) (line 206)
        _find_tests_call_result_189680 = invoke(stypy.reporting.localization.Localization(__file__, 206, 21), _find_tests_189676, *[start_dir_189677, pattern_189678], **kwargs_189679)
        
        # Processing the call keyword arguments (line 206)
        kwargs_189681 = {}
        # Getting the type of 'list' (line 206)
        list_189674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'list', False)
        # Calling list(args, kwargs) (line 206)
        list_call_result_189682 = invoke(stypy.reporting.localization.Localization(__file__, 206, 16), list_189674, *[_find_tests_call_result_189680], **kwargs_189681)
        
        # Assigning a type to the variable 'tests' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tests', list_call_result_189682)
        
        # Call to suiteClass(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'tests' (line 207)
        tests_189685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 31), 'tests', False)
        # Processing the call keyword arguments (line 207)
        kwargs_189686 = {}
        # Getting the type of 'self' (line 207)
        self_189683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'self', False)
        # Obtaining the member 'suiteClass' of a type (line 207)
        suiteClass_189684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 15), self_189683, 'suiteClass')
        # Calling suiteClass(args, kwargs) (line 207)
        suiteClass_call_result_189687 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), suiteClass_189684, *[tests_189685], **kwargs_189686)
        
        # Assigning a type to the variable 'stypy_return_type' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', suiteClass_call_result_189687)
        
        # ################# End of 'discover(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'discover' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_189688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189688)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'discover'
        return stypy_return_type_189688


    @norecursion
    def _get_directory_containing_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_directory_containing_module'
        module_type_store = module_type_store.open_function_context('_get_directory_containing_module', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_localization', localization)
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_function_name', 'TestLoader._get_directory_containing_module')
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_param_names_list', ['module_name'])
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader._get_directory_containing_module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader._get_directory_containing_module', ['module_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_directory_containing_module', localization, ['module_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_directory_containing_module(...)' code ##################

        
        # Assigning a Subscript to a Name (line 210):
        
        # Assigning a Subscript to a Name (line 210):
        
        # Obtaining the type of the subscript
        # Getting the type of 'module_name' (line 210)
        module_name_189689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 29), 'module_name')
        # Getting the type of 'sys' (line 210)
        sys_189690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'sys')
        # Obtaining the member 'modules' of a type (line 210)
        modules_189691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 17), sys_189690, 'modules')
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___189692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 17), modules_189691, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_189693 = invoke(stypy.reporting.localization.Localization(__file__, 210, 17), getitem___189692, module_name_189689)
        
        # Assigning a type to the variable 'module' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'module', subscript_call_result_189693)
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to abspath(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'module' (line 211)
        module_189697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 36), 'module', False)
        # Obtaining the member '__file__' of a type (line 211)
        file___189698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 36), module_189697, '__file__')
        # Processing the call keyword arguments (line 211)
        kwargs_189699 = {}
        # Getting the type of 'os' (line 211)
        os_189694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 211)
        path_189695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 20), os_189694, 'path')
        # Obtaining the member 'abspath' of a type (line 211)
        abspath_189696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 20), path_189695, 'abspath')
        # Calling abspath(args, kwargs) (line 211)
        abspath_call_result_189700 = invoke(stypy.reporting.localization.Localization(__file__, 211, 20), abspath_189696, *[file___189698], **kwargs_189699)
        
        # Assigning a type to the variable 'full_path' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'full_path', abspath_call_result_189700)
        
        
        # Call to startswith(...): (line 213)
        # Processing the call arguments (line 213)
        str_189711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 58), 'str', '__init__.py')
        # Processing the call keyword arguments (line 213)
        kwargs_189712 = {}
        
        # Call to lower(...): (line 213)
        # Processing the call keyword arguments (line 213)
        kwargs_189708 = {}
        
        # Call to basename(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'full_path' (line 213)
        full_path_189704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'full_path', False)
        # Processing the call keyword arguments (line 213)
        kwargs_189705 = {}
        # Getting the type of 'os' (line 213)
        os_189701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 213)
        path_189702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), os_189701, 'path')
        # Obtaining the member 'basename' of a type (line 213)
        basename_189703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), path_189702, 'basename')
        # Calling basename(args, kwargs) (line 213)
        basename_call_result_189706 = invoke(stypy.reporting.localization.Localization(__file__, 213, 11), basename_189703, *[full_path_189704], **kwargs_189705)
        
        # Obtaining the member 'lower' of a type (line 213)
        lower_189707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), basename_call_result_189706, 'lower')
        # Calling lower(args, kwargs) (line 213)
        lower_call_result_189709 = invoke(stypy.reporting.localization.Localization(__file__, 213, 11), lower_189707, *[], **kwargs_189708)
        
        # Obtaining the member 'startswith' of a type (line 213)
        startswith_189710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), lower_call_result_189709, 'startswith')
        # Calling startswith(args, kwargs) (line 213)
        startswith_call_result_189713 = invoke(stypy.reporting.localization.Localization(__file__, 213, 11), startswith_189710, *[str_189711], **kwargs_189712)
        
        # Testing the type of an if condition (line 213)
        if_condition_189714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), startswith_call_result_189713)
        # Assigning a type to the variable 'if_condition_189714' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_189714', if_condition_189714)
        # SSA begins for if statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to dirname(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Call to dirname(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'full_path' (line 214)
        full_path_189721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 51), 'full_path', False)
        # Processing the call keyword arguments (line 214)
        kwargs_189722 = {}
        # Getting the type of 'os' (line 214)
        os_189718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 35), 'os', False)
        # Obtaining the member 'path' of a type (line 214)
        path_189719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 35), os_189718, 'path')
        # Obtaining the member 'dirname' of a type (line 214)
        dirname_189720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 35), path_189719, 'dirname')
        # Calling dirname(args, kwargs) (line 214)
        dirname_call_result_189723 = invoke(stypy.reporting.localization.Localization(__file__, 214, 35), dirname_189720, *[full_path_189721], **kwargs_189722)
        
        # Processing the call keyword arguments (line 214)
        kwargs_189724 = {}
        # Getting the type of 'os' (line 214)
        os_189715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 214)
        path_189716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 19), os_189715, 'path')
        # Obtaining the member 'dirname' of a type (line 214)
        dirname_189717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 19), path_189716, 'dirname')
        # Calling dirname(args, kwargs) (line 214)
        dirname_call_result_189725 = invoke(stypy.reporting.localization.Localization(__file__, 214, 19), dirname_189717, *[dirname_call_result_189723], **kwargs_189724)
        
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'stypy_return_type', dirname_call_result_189725)
        # SSA branch for the else part of an if statement (line 213)
        module_type_store.open_ssa_branch('else')
        
        # Call to dirname(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'full_path' (line 219)
        full_path_189729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 35), 'full_path', False)
        # Processing the call keyword arguments (line 219)
        kwargs_189730 = {}
        # Getting the type of 'os' (line 219)
        os_189726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 219)
        path_189727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 19), os_189726, 'path')
        # Obtaining the member 'dirname' of a type (line 219)
        dirname_189728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 19), path_189727, 'dirname')
        # Calling dirname(args, kwargs) (line 219)
        dirname_call_result_189731 = invoke(stypy.reporting.localization.Localization(__file__, 219, 19), dirname_189728, *[full_path_189729], **kwargs_189730)
        
        # Assigning a type to the variable 'stypy_return_type' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'stypy_return_type', dirname_call_result_189731)
        # SSA join for if statement (line 213)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_get_directory_containing_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_directory_containing_module' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_189732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_directory_containing_module'
        return stypy_return_type_189732


    @norecursion
    def _get_name_from_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_name_from_path'
        module_type_store = module_type_store.open_function_context('_get_name_from_path', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_localization', localization)
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_function_name', 'TestLoader._get_name_from_path')
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_param_names_list', ['path'])
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader._get_name_from_path.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader._get_name_from_path', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_name_from_path', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_name_from_path(...)' code ##################

        
        # Assigning a Subscript to a Name (line 222):
        
        # Assigning a Subscript to a Name (line 222):
        
        # Obtaining the type of the subscript
        int_189733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 56), 'int')
        
        # Call to splitext(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Call to normpath(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'path' (line 222)
        path_189740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 49), 'path', False)
        # Processing the call keyword arguments (line 222)
        kwargs_189741 = {}
        # Getting the type of 'os' (line 222)
        os_189737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 32), 'os', False)
        # Obtaining the member 'path' of a type (line 222)
        path_189738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 32), os_189737, 'path')
        # Obtaining the member 'normpath' of a type (line 222)
        normpath_189739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 32), path_189738, 'normpath')
        # Calling normpath(args, kwargs) (line 222)
        normpath_call_result_189742 = invoke(stypy.reporting.localization.Localization(__file__, 222, 32), normpath_189739, *[path_189740], **kwargs_189741)
        
        # Processing the call keyword arguments (line 222)
        kwargs_189743 = {}
        # Getting the type of 'os' (line 222)
        os_189734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 222)
        path_189735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 15), os_189734, 'path')
        # Obtaining the member 'splitext' of a type (line 222)
        splitext_189736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 15), path_189735, 'splitext')
        # Calling splitext(args, kwargs) (line 222)
        splitext_call_result_189744 = invoke(stypy.reporting.localization.Localization(__file__, 222, 15), splitext_189736, *[normpath_call_result_189742], **kwargs_189743)
        
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___189745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 15), splitext_call_result_189744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_189746 = invoke(stypy.reporting.localization.Localization(__file__, 222, 15), getitem___189745, int_189733)
        
        # Assigning a type to the variable 'path' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'path', subscript_call_result_189746)
        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to relpath(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'path' (line 224)
        path_189750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'path', False)
        # Getting the type of 'self' (line 224)
        self_189751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 41), 'self', False)
        # Obtaining the member '_top_level_dir' of a type (line 224)
        _top_level_dir_189752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 41), self_189751, '_top_level_dir')
        # Processing the call keyword arguments (line 224)
        kwargs_189753 = {}
        # Getting the type of 'os' (line 224)
        os_189747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 224)
        path_189748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 19), os_189747, 'path')
        # Obtaining the member 'relpath' of a type (line 224)
        relpath_189749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 19), path_189748, 'relpath')
        # Calling relpath(args, kwargs) (line 224)
        relpath_call_result_189754 = invoke(stypy.reporting.localization.Localization(__file__, 224, 19), relpath_189749, *[path_189750, _top_level_dir_189752], **kwargs_189753)
        
        # Assigning a type to the variable '_relpath' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), '_relpath', relpath_call_result_189754)
        # Evaluating assert statement condition
        
        
        # Call to isabs(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of '_relpath' (line 225)
        _relpath_189758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 33), '_relpath', False)
        # Processing the call keyword arguments (line 225)
        kwargs_189759 = {}
        # Getting the type of 'os' (line 225)
        os_189755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 225)
        path_189756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 19), os_189755, 'path')
        # Obtaining the member 'isabs' of a type (line 225)
        isabs_189757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 19), path_189756, 'isabs')
        # Calling isabs(args, kwargs) (line 225)
        isabs_call_result_189760 = invoke(stypy.reporting.localization.Localization(__file__, 225, 19), isabs_189757, *[_relpath_189758], **kwargs_189759)
        
        # Applying the 'not' unary operator (line 225)
        result_not__189761 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 15), 'not', isabs_call_result_189760)
        
        # Evaluating assert statement condition
        
        
        # Call to startswith(...): (line 226)
        # Processing the call arguments (line 226)
        str_189764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 39), 'str', '..')
        # Processing the call keyword arguments (line 226)
        kwargs_189765 = {}
        # Getting the type of '_relpath' (line 226)
        _relpath_189762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), '_relpath', False)
        # Obtaining the member 'startswith' of a type (line 226)
        startswith_189763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 19), _relpath_189762, 'startswith')
        # Calling startswith(args, kwargs) (line 226)
        startswith_call_result_189766 = invoke(stypy.reporting.localization.Localization(__file__, 226, 19), startswith_189763, *[str_189764], **kwargs_189765)
        
        # Applying the 'not' unary operator (line 226)
        result_not__189767 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 15), 'not', startswith_call_result_189766)
        
        
        # Assigning a Call to a Name (line 228):
        
        # Assigning a Call to a Name (line 228):
        
        # Call to replace(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'os' (line 228)
        os_189770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 32), 'os', False)
        # Obtaining the member 'path' of a type (line 228)
        path_189771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 32), os_189770, 'path')
        # Obtaining the member 'sep' of a type (line 228)
        sep_189772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 32), path_189771, 'sep')
        str_189773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 45), 'str', '.')
        # Processing the call keyword arguments (line 228)
        kwargs_189774 = {}
        # Getting the type of '_relpath' (line 228)
        _relpath_189768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), '_relpath', False)
        # Obtaining the member 'replace' of a type (line 228)
        replace_189769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 15), _relpath_189768, 'replace')
        # Calling replace(args, kwargs) (line 228)
        replace_call_result_189775 = invoke(stypy.reporting.localization.Localization(__file__, 228, 15), replace_189769, *[sep_189772, str_189773], **kwargs_189774)
        
        # Assigning a type to the variable 'name' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'name', replace_call_result_189775)
        # Getting the type of 'name' (line 229)
        name_189776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', name_189776)
        
        # ################# End of '_get_name_from_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_name_from_path' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_189777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189777)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_name_from_path'
        return stypy_return_type_189777


    @norecursion
    def _get_module_from_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_module_from_name'
        module_type_store = module_type_store.open_function_context('_get_module_from_name', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_localization', localization)
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_function_name', 'TestLoader._get_module_from_name')
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_param_names_list', ['name'])
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader._get_module_from_name.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader._get_module_from_name', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_module_from_name', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_module_from_name(...)' code ##################

        
        # Call to __import__(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'name' (line 232)
        name_189779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'name', False)
        # Processing the call keyword arguments (line 232)
        kwargs_189780 = {}
        # Getting the type of '__import__' (line 232)
        import___189778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), '__import__', False)
        # Calling __import__(args, kwargs) (line 232)
        import___call_result_189781 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), import___189778, *[name_189779], **kwargs_189780)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 233)
        name_189782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'name')
        # Getting the type of 'sys' (line 233)
        sys_189783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'sys')
        # Obtaining the member 'modules' of a type (line 233)
        modules_189784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 15), sys_189783, 'modules')
        # Obtaining the member '__getitem__' of a type (line 233)
        getitem___189785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 15), modules_189784, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 233)
        subscript_call_result_189786 = invoke(stypy.reporting.localization.Localization(__file__, 233, 15), getitem___189785, name_189782)
        
        # Assigning a type to the variable 'stypy_return_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'stypy_return_type', subscript_call_result_189786)
        
        # ################# End of '_get_module_from_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_module_from_name' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_189787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189787)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_module_from_name'
        return stypy_return_type_189787


    @norecursion
    def _match_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_match_path'
        module_type_store = module_type_store.open_function_context('_match_path', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader._match_path.__dict__.__setitem__('stypy_localization', localization)
        TestLoader._match_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader._match_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader._match_path.__dict__.__setitem__('stypy_function_name', 'TestLoader._match_path')
        TestLoader._match_path.__dict__.__setitem__('stypy_param_names_list', ['path', 'full_path', 'pattern'])
        TestLoader._match_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader._match_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader._match_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader._match_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader._match_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader._match_path.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader._match_path', ['path', 'full_path', 'pattern'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_match_path', localization, ['path', 'full_path', 'pattern'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_match_path(...)' code ##################

        
        # Call to fnmatch(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'path' (line 237)
        path_189789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 23), 'path', False)
        # Getting the type of 'pattern' (line 237)
        pattern_189790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 29), 'pattern', False)
        # Processing the call keyword arguments (line 237)
        kwargs_189791 = {}
        # Getting the type of 'fnmatch' (line 237)
        fnmatch_189788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'fnmatch', False)
        # Calling fnmatch(args, kwargs) (line 237)
        fnmatch_call_result_189792 = invoke(stypy.reporting.localization.Localization(__file__, 237, 15), fnmatch_189788, *[path_189789, pattern_189790], **kwargs_189791)
        
        # Assigning a type to the variable 'stypy_return_type' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type', fnmatch_call_result_189792)
        
        # ################# End of '_match_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_match_path' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_189793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_189793)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_match_path'
        return stypy_return_type_189793


    @norecursion
    def _find_tests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_find_tests'
        module_type_store = module_type_store.open_function_context('_find_tests', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLoader._find_tests.__dict__.__setitem__('stypy_localization', localization)
        TestLoader._find_tests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLoader._find_tests.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLoader._find_tests.__dict__.__setitem__('stypy_function_name', 'TestLoader._find_tests')
        TestLoader._find_tests.__dict__.__setitem__('stypy_param_names_list', ['start_dir', 'pattern'])
        TestLoader._find_tests.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLoader._find_tests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLoader._find_tests.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLoader._find_tests.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLoader._find_tests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLoader._find_tests.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader._find_tests', ['start_dir', 'pattern'], None, None, defaults, varargs, kwargs)

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

        str_189794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 8), 'str', 'Used by discovery. Yields test suites it loads.')
        
        # Assigning a Call to a Name (line 241):
        
        # Assigning a Call to a Name (line 241):
        
        # Call to listdir(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'start_dir' (line 241)
        start_dir_189797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 27), 'start_dir', False)
        # Processing the call keyword arguments (line 241)
        kwargs_189798 = {}
        # Getting the type of 'os' (line 241)
        os_189795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 16), 'os', False)
        # Obtaining the member 'listdir' of a type (line 241)
        listdir_189796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 16), os_189795, 'listdir')
        # Calling listdir(args, kwargs) (line 241)
        listdir_call_result_189799 = invoke(stypy.reporting.localization.Localization(__file__, 241, 16), listdir_189796, *[start_dir_189797], **kwargs_189798)
        
        # Assigning a type to the variable 'paths' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'paths', listdir_call_result_189799)
        
        # Getting the type of 'paths' (line 243)
        paths_189800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'paths')
        # Testing the type of a for loop iterable (line 243)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 243, 8), paths_189800)
        # Getting the type of the for loop variable (line 243)
        for_loop_var_189801 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 243, 8), paths_189800)
        # Assigning a type to the variable 'path' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'path', for_loop_var_189801)
        # SSA begins for a for statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Call to join(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'start_dir' (line 244)
        start_dir_189805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 37), 'start_dir', False)
        # Getting the type of 'path' (line 244)
        path_189806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 48), 'path', False)
        # Processing the call keyword arguments (line 244)
        kwargs_189807 = {}
        # Getting the type of 'os' (line 244)
        os_189802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 244)
        path_189803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), os_189802, 'path')
        # Obtaining the member 'join' of a type (line 244)
        join_189804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), path_189803, 'join')
        # Calling join(args, kwargs) (line 244)
        join_call_result_189808 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), join_189804, *[start_dir_189805, path_189806], **kwargs_189807)
        
        # Assigning a type to the variable 'full_path' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'full_path', join_call_result_189808)
        
        
        # Call to isfile(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'full_path' (line 245)
        full_path_189812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 30), 'full_path', False)
        # Processing the call keyword arguments (line 245)
        kwargs_189813 = {}
        # Getting the type of 'os' (line 245)
        os_189809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 245)
        path_189810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), os_189809, 'path')
        # Obtaining the member 'isfile' of a type (line 245)
        isfile_189811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), path_189810, 'isfile')
        # Calling isfile(args, kwargs) (line 245)
        isfile_call_result_189814 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), isfile_189811, *[full_path_189812], **kwargs_189813)
        
        # Testing the type of an if condition (line 245)
        if_condition_189815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 12), isfile_call_result_189814)
        # Assigning a type to the variable 'if_condition_189815' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'if_condition_189815', if_condition_189815)
        # SSA begins for if statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to match(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'path' (line 246)
        path_189818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 47), 'path', False)
        # Processing the call keyword arguments (line 246)
        kwargs_189819 = {}
        # Getting the type of 'VALID_MODULE_NAME' (line 246)
        VALID_MODULE_NAME_189816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'VALID_MODULE_NAME', False)
        # Obtaining the member 'match' of a type (line 246)
        match_189817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 23), VALID_MODULE_NAME_189816, 'match')
        # Calling match(args, kwargs) (line 246)
        match_call_result_189820 = invoke(stypy.reporting.localization.Localization(__file__, 246, 23), match_189817, *[path_189818], **kwargs_189819)
        
        # Applying the 'not' unary operator (line 246)
        result_not__189821 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 19), 'not', match_call_result_189820)
        
        # Testing the type of an if condition (line 246)
        if_condition_189822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 16), result_not__189821)
        # Assigning a type to the variable 'if_condition_189822' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'if_condition_189822', if_condition_189822)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to _match_path(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'path' (line 249)
        path_189825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 40), 'path', False)
        # Getting the type of 'full_path' (line 249)
        full_path_189826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 46), 'full_path', False)
        # Getting the type of 'pattern' (line 249)
        pattern_189827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 57), 'pattern', False)
        # Processing the call keyword arguments (line 249)
        kwargs_189828 = {}
        # Getting the type of 'self' (line 249)
        self_189823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 23), 'self', False)
        # Obtaining the member '_match_path' of a type (line 249)
        _match_path_189824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 23), self_189823, '_match_path')
        # Calling _match_path(args, kwargs) (line 249)
        _match_path_call_result_189829 = invoke(stypy.reporting.localization.Localization(__file__, 249, 23), _match_path_189824, *[path_189825, full_path_189826, pattern_189827], **kwargs_189828)
        
        # Applying the 'not' unary operator (line 249)
        result_not__189830 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 19), 'not', _match_path_call_result_189829)
        
        # Testing the type of an if condition (line 249)
        if_condition_189831 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 16), result_not__189830)
        # Assigning a type to the variable 'if_condition_189831' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'if_condition_189831', if_condition_189831)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to _get_name_from_path(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'full_path' (line 252)
        full_path_189834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 48), 'full_path', False)
        # Processing the call keyword arguments (line 252)
        kwargs_189835 = {}
        # Getting the type of 'self' (line 252)
        self_189832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 23), 'self', False)
        # Obtaining the member '_get_name_from_path' of a type (line 252)
        _get_name_from_path_189833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 23), self_189832, '_get_name_from_path')
        # Calling _get_name_from_path(args, kwargs) (line 252)
        _get_name_from_path_call_result_189836 = invoke(stypy.reporting.localization.Localization(__file__, 252, 23), _get_name_from_path_189833, *[full_path_189834], **kwargs_189835)
        
        # Assigning a type to the variable 'name' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'name', _get_name_from_path_call_result_189836)
        
        
        # SSA begins for try-except statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 254):
        
        # Assigning a Call to a Name (line 254):
        
        # Call to _get_module_from_name(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'name' (line 254)
        name_189839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 56), 'name', False)
        # Processing the call keyword arguments (line 254)
        kwargs_189840 = {}
        # Getting the type of 'self' (line 254)
        self_189837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 29), 'self', False)
        # Obtaining the member '_get_module_from_name' of a type (line 254)
        _get_module_from_name_189838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 29), self_189837, '_get_module_from_name')
        # Calling _get_module_from_name(args, kwargs) (line 254)
        _get_module_from_name_call_result_189841 = invoke(stypy.reporting.localization.Localization(__file__, 254, 29), _get_module_from_name_189838, *[name_189839], **kwargs_189840)
        
        # Assigning a type to the variable 'module' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'module', _get_module_from_name_call_result_189841)
        # SSA branch for the except part of a try statement (line 253)
        # SSA branch for the except '<any exception>' branch of a try statement (line 253)
        module_type_store.open_ssa_branch('except')
        # Creating a generator
        
        # Call to _make_failed_import_test(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'name' (line 256)
        name_189843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 51), 'name', False)
        # Getting the type of 'self' (line 256)
        self_189844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 57), 'self', False)
        # Obtaining the member 'suiteClass' of a type (line 256)
        suiteClass_189845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 57), self_189844, 'suiteClass')
        # Processing the call keyword arguments (line 256)
        kwargs_189846 = {}
        # Getting the type of '_make_failed_import_test' (line 256)
        _make_failed_import_test_189842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), '_make_failed_import_test', False)
        # Calling _make_failed_import_test(args, kwargs) (line 256)
        _make_failed_import_test_call_result_189847 = invoke(stypy.reporting.localization.Localization(__file__, 256, 26), _make_failed_import_test_189842, *[name_189843, suiteClass_189845], **kwargs_189846)
        
        GeneratorType_189848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 20), GeneratorType_189848, _make_failed_import_test_call_result_189847)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'stypy_return_type', GeneratorType_189848)
        # SSA branch for the else branch of a try statement (line 253)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Call to abspath(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Call to getattr(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'module' (line 258)
        module_189853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 55), 'module', False)
        str_189854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 63), 'str', '__file__')
        # Getting the type of 'full_path' (line 258)
        full_path_189855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 75), 'full_path', False)
        # Processing the call keyword arguments (line 258)
        kwargs_189856 = {}
        # Getting the type of 'getattr' (line 258)
        getattr_189852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'getattr', False)
        # Calling getattr(args, kwargs) (line 258)
        getattr_call_result_189857 = invoke(stypy.reporting.localization.Localization(__file__, 258, 47), getattr_189852, *[module_189853, str_189854, full_path_189855], **kwargs_189856)
        
        # Processing the call keyword arguments (line 258)
        kwargs_189858 = {}
        # Getting the type of 'os' (line 258)
        os_189849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 258)
        path_189850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 31), os_189849, 'path')
        # Obtaining the member 'abspath' of a type (line 258)
        abspath_189851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 31), path_189850, 'abspath')
        # Calling abspath(args, kwargs) (line 258)
        abspath_call_result_189859 = invoke(stypy.reporting.localization.Localization(__file__, 258, 31), abspath_189851, *[getattr_call_result_189857], **kwargs_189858)
        
        # Assigning a type to the variable 'mod_file' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'mod_file', abspath_call_result_189859)
        
        # Assigning a Subscript to a Name (line 259):
        
        # Assigning a Subscript to a Name (line 259):
        
        # Obtaining the type of the subscript
        int_189860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 76), 'int')
        
        # Call to splitext(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Call to realpath(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'mod_file' (line 259)
        mod_file_189867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 65), 'mod_file', False)
        # Processing the call keyword arguments (line 259)
        kwargs_189868 = {}
        # Getting the type of 'os' (line 259)
        os_189864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 48), 'os', False)
        # Obtaining the member 'path' of a type (line 259)
        path_189865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 48), os_189864, 'path')
        # Obtaining the member 'realpath' of a type (line 259)
        realpath_189866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 48), path_189865, 'realpath')
        # Calling realpath(args, kwargs) (line 259)
        realpath_call_result_189869 = invoke(stypy.reporting.localization.Localization(__file__, 259, 48), realpath_189866, *[mod_file_189867], **kwargs_189868)
        
        # Processing the call keyword arguments (line 259)
        kwargs_189870 = {}
        # Getting the type of 'os' (line 259)
        os_189861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 259)
        path_189862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 31), os_189861, 'path')
        # Obtaining the member 'splitext' of a type (line 259)
        splitext_189863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 31), path_189862, 'splitext')
        # Calling splitext(args, kwargs) (line 259)
        splitext_call_result_189871 = invoke(stypy.reporting.localization.Localization(__file__, 259, 31), splitext_189863, *[realpath_call_result_189869], **kwargs_189870)
        
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___189872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 31), splitext_call_result_189871, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_189873 = invoke(stypy.reporting.localization.Localization(__file__, 259, 31), getitem___189872, int_189860)
        
        # Assigning a type to the variable 'realpath' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'realpath', subscript_call_result_189873)
        
        # Assigning a Subscript to a Name (line 260):
        
        # Assigning a Subscript to a Name (line 260):
        
        # Obtaining the type of the subscript
        int_189874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 83), 'int')
        
        # Call to splitext(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Call to realpath(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'full_path' (line 260)
        full_path_189881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 71), 'full_path', False)
        # Processing the call keyword arguments (line 260)
        kwargs_189882 = {}
        # Getting the type of 'os' (line 260)
        os_189878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 54), 'os', False)
        # Obtaining the member 'path' of a type (line 260)
        path_189879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 54), os_189878, 'path')
        # Obtaining the member 'realpath' of a type (line 260)
        realpath_189880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 54), path_189879, 'realpath')
        # Calling realpath(args, kwargs) (line 260)
        realpath_call_result_189883 = invoke(stypy.reporting.localization.Localization(__file__, 260, 54), realpath_189880, *[full_path_189881], **kwargs_189882)
        
        # Processing the call keyword arguments (line 260)
        kwargs_189884 = {}
        # Getting the type of 'os' (line 260)
        os_189875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 260)
        path_189876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 37), os_189875, 'path')
        # Obtaining the member 'splitext' of a type (line 260)
        splitext_189877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 37), path_189876, 'splitext')
        # Calling splitext(args, kwargs) (line 260)
        splitext_call_result_189885 = invoke(stypy.reporting.localization.Localization(__file__, 260, 37), splitext_189877, *[realpath_call_result_189883], **kwargs_189884)
        
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___189886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 37), splitext_call_result_189885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_189887 = invoke(stypy.reporting.localization.Localization(__file__, 260, 37), getitem___189886, int_189874)
        
        # Assigning a type to the variable 'fullpath_noext' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'fullpath_noext', subscript_call_result_189887)
        
        
        
        # Call to lower(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_189890 = {}
        # Getting the type of 'realpath' (line 261)
        realpath_189888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'realpath', False)
        # Obtaining the member 'lower' of a type (line 261)
        lower_189889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 23), realpath_189888, 'lower')
        # Calling lower(args, kwargs) (line 261)
        lower_call_result_189891 = invoke(stypy.reporting.localization.Localization(__file__, 261, 23), lower_189889, *[], **kwargs_189890)
        
        
        # Call to lower(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_189894 = {}
        # Getting the type of 'fullpath_noext' (line 261)
        fullpath_noext_189892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 43), 'fullpath_noext', False)
        # Obtaining the member 'lower' of a type (line 261)
        lower_189893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 43), fullpath_noext_189892, 'lower')
        # Calling lower(args, kwargs) (line 261)
        lower_call_result_189895 = invoke(stypy.reporting.localization.Localization(__file__, 261, 43), lower_189893, *[], **kwargs_189894)
        
        # Applying the binary operator '!=' (line 261)
        result_ne_189896 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 23), '!=', lower_call_result_189891, lower_call_result_189895)
        
        # Testing the type of an if condition (line 261)
        if_condition_189897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 20), result_ne_189896)
        # Assigning a type to the variable 'if_condition_189897' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'if_condition_189897', if_condition_189897)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 262):
        
        # Assigning a Call to a Name (line 262):
        
        # Call to dirname(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'realpath' (line 262)
        realpath_189901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 53), 'realpath', False)
        # Processing the call keyword arguments (line 262)
        kwargs_189902 = {}
        # Getting the type of 'os' (line 262)
        os_189898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 262)
        path_189899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 37), os_189898, 'path')
        # Obtaining the member 'dirname' of a type (line 262)
        dirname_189900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 37), path_189899, 'dirname')
        # Calling dirname(args, kwargs) (line 262)
        dirname_call_result_189903 = invoke(stypy.reporting.localization.Localization(__file__, 262, 37), dirname_189900, *[realpath_189901], **kwargs_189902)
        
        # Assigning a type to the variable 'module_dir' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'module_dir', dirname_call_result_189903)
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_189904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 81), 'int')
        
        # Call to splitext(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Call to basename(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'full_path' (line 263)
        full_path_189911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 69), 'full_path', False)
        # Processing the call keyword arguments (line 263)
        kwargs_189912 = {}
        # Getting the type of 'os' (line 263)
        os_189908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 52), 'os', False)
        # Obtaining the member 'path' of a type (line 263)
        path_189909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 52), os_189908, 'path')
        # Obtaining the member 'basename' of a type (line 263)
        basename_189910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 52), path_189909, 'basename')
        # Calling basename(args, kwargs) (line 263)
        basename_call_result_189913 = invoke(stypy.reporting.localization.Localization(__file__, 263, 52), basename_189910, *[full_path_189911], **kwargs_189912)
        
        # Processing the call keyword arguments (line 263)
        kwargs_189914 = {}
        # Getting the type of 'os' (line 263)
        os_189905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 35), 'os', False)
        # Obtaining the member 'path' of a type (line 263)
        path_189906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 35), os_189905, 'path')
        # Obtaining the member 'splitext' of a type (line 263)
        splitext_189907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 35), path_189906, 'splitext')
        # Calling splitext(args, kwargs) (line 263)
        splitext_call_result_189915 = invoke(stypy.reporting.localization.Localization(__file__, 263, 35), splitext_189907, *[basename_call_result_189913], **kwargs_189914)
        
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___189916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 35), splitext_call_result_189915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_189917 = invoke(stypy.reporting.localization.Localization(__file__, 263, 35), getitem___189916, int_189904)
        
        # Assigning a type to the variable 'mod_name' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'mod_name', subscript_call_result_189917)
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to dirname(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'full_path' (line 264)
        full_path_189921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 55), 'full_path', False)
        # Processing the call keyword arguments (line 264)
        kwargs_189922 = {}
        # Getting the type of 'os' (line 264)
        os_189918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 264)
        path_189919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 39), os_189918, 'path')
        # Obtaining the member 'dirname' of a type (line 264)
        dirname_189920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 39), path_189919, 'dirname')
        # Calling dirname(args, kwargs) (line 264)
        dirname_call_result_189923 = invoke(stypy.reporting.localization.Localization(__file__, 264, 39), dirname_189920, *[full_path_189921], **kwargs_189922)
        
        # Assigning a type to the variable 'expected_dir' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'expected_dir', dirname_call_result_189923)
        
        # Assigning a Str to a Name (line 265):
        
        # Assigning a Str to a Name (line 265):
        str_189924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 31), 'str', '%r module incorrectly imported from %r. Expected %r. Is this module globally installed?')
        # Assigning a type to the variable 'msg' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'msg', str_189924)
        
        # Call to ImportError(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'msg' (line 267)
        msg_189926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 42), 'msg', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 267)
        tuple_189927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 267)
        # Adding element type (line 267)
        # Getting the type of 'mod_name' (line 267)
        mod_name_189928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 49), 'mod_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 49), tuple_189927, mod_name_189928)
        # Adding element type (line 267)
        # Getting the type of 'module_dir' (line 267)
        module_dir_189929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 59), 'module_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 49), tuple_189927, module_dir_189929)
        # Adding element type (line 267)
        # Getting the type of 'expected_dir' (line 267)
        expected_dir_189930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 71), 'expected_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 49), tuple_189927, expected_dir_189930)
        
        # Applying the binary operator '%' (line 267)
        result_mod_189931 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 42), '%', msg_189926, tuple_189927)
        
        # Processing the call keyword arguments (line 267)
        kwargs_189932 = {}
        # Getting the type of 'ImportError' (line 267)
        ImportError_189925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 30), 'ImportError', False)
        # Calling ImportError(args, kwargs) (line 267)
        ImportError_call_result_189933 = invoke(stypy.reporting.localization.Localization(__file__, 267, 30), ImportError_189925, *[result_mod_189931], **kwargs_189932)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 267, 24), ImportError_call_result_189933, 'raise parameter', BaseException)
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        # Creating a generator
        
        # Call to loadTestsFromModule(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'module' (line 268)
        module_189936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 51), 'module', False)
        # Processing the call keyword arguments (line 268)
        kwargs_189937 = {}
        # Getting the type of 'self' (line 268)
        self_189934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 26), 'self', False)
        # Obtaining the member 'loadTestsFromModule' of a type (line 268)
        loadTestsFromModule_189935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 26), self_189934, 'loadTestsFromModule')
        # Calling loadTestsFromModule(args, kwargs) (line 268)
        loadTestsFromModule_call_result_189938 = invoke(stypy.reporting.localization.Localization(__file__, 268, 26), loadTestsFromModule_189935, *[module_189936], **kwargs_189937)
        
        GeneratorType_189939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 20), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 20), GeneratorType_189939, loadTestsFromModule_call_result_189938)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'stypy_return_type', GeneratorType_189939)
        # SSA join for try-except statement (line 253)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 245)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isdir(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'full_path' (line 269)
        full_path_189943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 31), 'full_path', False)
        # Processing the call keyword arguments (line 269)
        kwargs_189944 = {}
        # Getting the type of 'os' (line 269)
        os_189940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 269)
        path_189941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 17), os_189940, 'path')
        # Obtaining the member 'isdir' of a type (line 269)
        isdir_189942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 17), path_189941, 'isdir')
        # Calling isdir(args, kwargs) (line 269)
        isdir_call_result_189945 = invoke(stypy.reporting.localization.Localization(__file__, 269, 17), isdir_189942, *[full_path_189943], **kwargs_189944)
        
        # Testing the type of an if condition (line 269)
        if_condition_189946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 17), isdir_call_result_189945)
        # Assigning a type to the variable 'if_condition_189946' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 17), 'if_condition_189946', if_condition_189946)
        # SSA begins for if statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to isfile(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Call to join(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'full_path' (line 270)
        full_path_189953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 51), 'full_path', False)
        str_189954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 62), 'str', '__init__.py')
        # Processing the call keyword arguments (line 270)
        kwargs_189955 = {}
        # Getting the type of 'os' (line 270)
        os_189950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 38), 'os', False)
        # Obtaining the member 'path' of a type (line 270)
        path_189951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 38), os_189950, 'path')
        # Obtaining the member 'join' of a type (line 270)
        join_189952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 38), path_189951, 'join')
        # Calling join(args, kwargs) (line 270)
        join_call_result_189956 = invoke(stypy.reporting.localization.Localization(__file__, 270, 38), join_189952, *[full_path_189953, str_189954], **kwargs_189955)
        
        # Processing the call keyword arguments (line 270)
        kwargs_189957 = {}
        # Getting the type of 'os' (line 270)
        os_189947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 270)
        path_189948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 23), os_189947, 'path')
        # Obtaining the member 'isfile' of a type (line 270)
        isfile_189949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 23), path_189948, 'isfile')
        # Calling isfile(args, kwargs) (line 270)
        isfile_call_result_189958 = invoke(stypy.reporting.localization.Localization(__file__, 270, 23), isfile_189949, *[join_call_result_189956], **kwargs_189957)
        
        # Applying the 'not' unary operator (line 270)
        result_not__189959 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 19), 'not', isfile_call_result_189958)
        
        # Testing the type of an if condition (line 270)
        if_condition_189960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 16), result_not__189959)
        # Assigning a type to the variable 'if_condition_189960' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'if_condition_189960', if_condition_189960)
        # SSA begins for if statement (line 270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 270)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 273):
        
        # Assigning a Name to a Name (line 273):
        # Getting the type of 'None' (line 273)
        None_189961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 29), 'None')
        # Assigning a type to the variable 'load_tests' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'load_tests', None_189961)
        
        # Assigning a Name to a Name (line 274):
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'None' (line 274)
        None_189962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'None')
        # Assigning a type to the variable 'tests' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'tests', None_189962)
        
        
        # Call to fnmatch(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'path' (line 275)
        path_189964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'path', False)
        # Getting the type of 'pattern' (line 275)
        pattern_189965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 33), 'pattern', False)
        # Processing the call keyword arguments (line 275)
        kwargs_189966 = {}
        # Getting the type of 'fnmatch' (line 275)
        fnmatch_189963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'fnmatch', False)
        # Calling fnmatch(args, kwargs) (line 275)
        fnmatch_call_result_189967 = invoke(stypy.reporting.localization.Localization(__file__, 275, 19), fnmatch_189963, *[path_189964, pattern_189965], **kwargs_189966)
        
        # Testing the type of an if condition (line 275)
        if_condition_189968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 16), fnmatch_call_result_189967)
        # Assigning a type to the variable 'if_condition_189968' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'if_condition_189968', if_condition_189968)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 277):
        
        # Assigning a Call to a Name (line 277):
        
        # Call to _get_name_from_path(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'full_path' (line 277)
        full_path_189971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 52), 'full_path', False)
        # Processing the call keyword arguments (line 277)
        kwargs_189972 = {}
        # Getting the type of 'self' (line 277)
        self_189969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 27), 'self', False)
        # Obtaining the member '_get_name_from_path' of a type (line 277)
        _get_name_from_path_189970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 27), self_189969, '_get_name_from_path')
        # Calling _get_name_from_path(args, kwargs) (line 277)
        _get_name_from_path_call_result_189973 = invoke(stypy.reporting.localization.Localization(__file__, 277, 27), _get_name_from_path_189970, *[full_path_189971], **kwargs_189972)
        
        # Assigning a type to the variable 'name' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'name', _get_name_from_path_call_result_189973)
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to _get_module_from_name(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'name' (line 278)
        name_189976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 57), 'name', False)
        # Processing the call keyword arguments (line 278)
        kwargs_189977 = {}
        # Getting the type of 'self' (line 278)
        self_189974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 30), 'self', False)
        # Obtaining the member '_get_module_from_name' of a type (line 278)
        _get_module_from_name_189975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 30), self_189974, '_get_module_from_name')
        # Calling _get_module_from_name(args, kwargs) (line 278)
        _get_module_from_name_call_result_189978 = invoke(stypy.reporting.localization.Localization(__file__, 278, 30), _get_module_from_name_189975, *[name_189976], **kwargs_189977)
        
        # Assigning a type to the variable 'package' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 20), 'package', _get_module_from_name_call_result_189978)
        
        # Assigning a Call to a Name (line 279):
        
        # Assigning a Call to a Name (line 279):
        
        # Call to getattr(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'package' (line 279)
        package_189980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 41), 'package', False)
        str_189981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 50), 'str', 'load_tests')
        # Getting the type of 'None' (line 279)
        None_189982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 64), 'None', False)
        # Processing the call keyword arguments (line 279)
        kwargs_189983 = {}
        # Getting the type of 'getattr' (line 279)
        getattr_189979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 33), 'getattr', False)
        # Calling getattr(args, kwargs) (line 279)
        getattr_call_result_189984 = invoke(stypy.reporting.localization.Localization(__file__, 279, 33), getattr_189979, *[package_189980, str_189981, None_189982], **kwargs_189983)
        
        # Assigning a type to the variable 'load_tests' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'load_tests', getattr_call_result_189984)
        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to loadTestsFromModule(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'package' (line 280)
        package_189987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 53), 'package', False)
        # Processing the call keyword arguments (line 280)
        # Getting the type of 'False' (line 280)
        False_189988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 77), 'False', False)
        keyword_189989 = False_189988
        kwargs_189990 = {'use_load_tests': keyword_189989}
        # Getting the type of 'self' (line 280)
        self_189985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 28), 'self', False)
        # Obtaining the member 'loadTestsFromModule' of a type (line 280)
        loadTestsFromModule_189986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 28), self_189985, 'loadTestsFromModule')
        # Calling loadTestsFromModule(args, kwargs) (line 280)
        loadTestsFromModule_call_result_189991 = invoke(stypy.reporting.localization.Localization(__file__, 280, 28), loadTestsFromModule_189986, *[package_189987], **kwargs_189990)
        
        # Assigning a type to the variable 'tests' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 20), 'tests', loadTestsFromModule_call_result_189991)
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 282)
        # Getting the type of 'load_tests' (line 282)
        load_tests_189992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'load_tests')
        # Getting the type of 'None' (line 282)
        None_189993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 33), 'None')
        
        (may_be_189994, more_types_in_union_189995) = may_be_none(load_tests_189992, None_189993)

        if may_be_189994:

            if more_types_in_union_189995:
                # Runtime conditional SSA (line 282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 283)
            # Getting the type of 'tests' (line 283)
            tests_189996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'tests')
            # Getting the type of 'None' (line 283)
            None_189997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 36), 'None')
            
            (may_be_189998, more_types_in_union_189999) = may_not_be_none(tests_189996, None_189997)

            if may_be_189998:

                if more_types_in_union_189999:
                    # Runtime conditional SSA (line 283)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Creating a generator
                # Getting the type of 'tests' (line 285)
                tests_190000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 30), 'tests')
                GeneratorType_190001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 24), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 24), GeneratorType_190001, tests_190000)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'stypy_return_type', GeneratorType_190001)

                if more_types_in_union_189999:
                    # SSA join for if statement (line 283)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Call to _find_tests(...): (line 287)
            # Processing the call arguments (line 287)
            # Getting the type of 'full_path' (line 287)
            full_path_190004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 49), 'full_path', False)
            # Getting the type of 'pattern' (line 287)
            pattern_190005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 60), 'pattern', False)
            # Processing the call keyword arguments (line 287)
            kwargs_190006 = {}
            # Getting the type of 'self' (line 287)
            self_190002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'self', False)
            # Obtaining the member '_find_tests' of a type (line 287)
            _find_tests_190003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 32), self_190002, '_find_tests')
            # Calling _find_tests(args, kwargs) (line 287)
            _find_tests_call_result_190007 = invoke(stypy.reporting.localization.Localization(__file__, 287, 32), _find_tests_190003, *[full_path_190004, pattern_190005], **kwargs_190006)
            
            # Testing the type of a for loop iterable (line 287)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 287, 20), _find_tests_call_result_190007)
            # Getting the type of the for loop variable (line 287)
            for_loop_var_190008 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 287, 20), _find_tests_call_result_190007)
            # Assigning a type to the variable 'test' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'test', for_loop_var_190008)
            # SSA begins for a for statement (line 287)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'test' (line 288)
            test_190009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 30), 'test')
            GeneratorType_190010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 24), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 24), GeneratorType_190010, test_190009)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'stypy_return_type', GeneratorType_190010)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_189995:
                # Runtime conditional SSA for else branch (line 282)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_189994) or more_types_in_union_189995):
            
            
            # SSA begins for try-except statement (line 290)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            # Creating a generator
            
            # Call to load_tests(...): (line 291)
            # Processing the call arguments (line 291)
            # Getting the type of 'self' (line 291)
            self_190012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 41), 'self', False)
            # Getting the type of 'tests' (line 291)
            tests_190013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 47), 'tests', False)
            # Getting the type of 'pattern' (line 291)
            pattern_190014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 54), 'pattern', False)
            # Processing the call keyword arguments (line 291)
            kwargs_190015 = {}
            # Getting the type of 'load_tests' (line 291)
            load_tests_190011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 30), 'load_tests', False)
            # Calling load_tests(args, kwargs) (line 291)
            load_tests_call_result_190016 = invoke(stypy.reporting.localization.Localization(__file__, 291, 30), load_tests_190011, *[self_190012, tests_190013, pattern_190014], **kwargs_190015)
            
            GeneratorType_190017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 24), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 24), GeneratorType_190017, load_tests_call_result_190016)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'stypy_return_type', GeneratorType_190017)
            # SSA branch for the except part of a try statement (line 290)
            # SSA branch for the except 'Exception' branch of a try statement (line 290)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'Exception' (line 292)
            Exception_190018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 27), 'Exception')
            # Assigning a type to the variable 'e' (line 292)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 20), 'e', Exception_190018)
            # Creating a generator
            
            # Call to _make_failed_load_tests(...): (line 293)
            # Processing the call arguments (line 293)
            # Getting the type of 'package' (line 293)
            package_190020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 54), 'package', False)
            # Obtaining the member '__name__' of a type (line 293)
            name___190021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 54), package_190020, '__name__')
            # Getting the type of 'e' (line 293)
            e_190022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 72), 'e', False)
            # Getting the type of 'self' (line 294)
            self_190023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 54), 'self', False)
            # Obtaining the member 'suiteClass' of a type (line 294)
            suiteClass_190024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 54), self_190023, 'suiteClass')
            # Processing the call keyword arguments (line 293)
            kwargs_190025 = {}
            # Getting the type of '_make_failed_load_tests' (line 293)
            _make_failed_load_tests_190019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), '_make_failed_load_tests', False)
            # Calling _make_failed_load_tests(args, kwargs) (line 293)
            _make_failed_load_tests_call_result_190026 = invoke(stypy.reporting.localization.Localization(__file__, 293, 30), _make_failed_load_tests_190019, *[name___190021, e_190022, suiteClass_190024], **kwargs_190025)
            
            GeneratorType_190027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 24), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 24), GeneratorType_190027, _make_failed_load_tests_call_result_190026)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'stypy_return_type', GeneratorType_190027)
            # SSA join for try-except statement (line 290)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_189994 and more_types_in_union_189995):
                # SSA join for if statement (line 282)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 269)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_find_tests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_find_tests' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_190028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_190028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_find_tests'
        return stypy_return_type_190028


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 0, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLoader.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLoader' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'TestLoader', TestLoader)

# Assigning a Str to a Name (line 43):
str_190029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'str', 'test')
# Getting the type of 'TestLoader'
TestLoader_190030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestLoader')
# Setting the type of the member 'testMethodPrefix' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestLoader_190030, 'testMethodPrefix', str_190029)

# Assigning a Name to a Name (line 44):
# Getting the type of 'cmp' (line 44)
cmp_190031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'cmp')
# Getting the type of 'TestLoader'
TestLoader_190032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestLoader')
# Setting the type of the member 'sortTestMethodsUsing' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestLoader_190032, 'sortTestMethodsUsing', cmp_190031)

# Assigning a Attribute to a Name (line 45):
# Getting the type of 'suite' (line 45)
suite_190033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'suite')
# Obtaining the member 'TestSuite' of a type (line 45)
TestSuite_190034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), suite_190033, 'TestSuite')
# Getting the type of 'TestLoader'
TestLoader_190035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestLoader')
# Setting the type of the member 'suiteClass' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestLoader_190035, 'suiteClass', TestSuite_190034)

# Assigning a Name to a Name (line 46):
# Getting the type of 'None' (line 46)
None_190036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'None')
# Getting the type of 'TestLoader'
TestLoader_190037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestLoader')
# Setting the type of the member '_top_level_dir' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestLoader_190037, '_top_level_dir', None_190036)

# Assigning a Call to a Name (line 296):

# Assigning a Call to a Name (line 296):

# Call to TestLoader(...): (line 296)
# Processing the call keyword arguments (line 296)
kwargs_190039 = {}
# Getting the type of 'TestLoader' (line 296)
TestLoader_190038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'TestLoader', False)
# Calling TestLoader(args, kwargs) (line 296)
TestLoader_call_result_190040 = invoke(stypy.reporting.localization.Localization(__file__, 296, 20), TestLoader_190038, *[], **kwargs_190039)

# Assigning a type to the variable 'defaultTestLoader' (line 296)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'defaultTestLoader', TestLoader_call_result_190040)

@norecursion
def _makeLoader(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 299)
    None_190041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 46), 'None')
    defaults = [None_190041]
    # Create a new context for function '_makeLoader'
    module_type_store = module_type_store.open_function_context('_makeLoader', 299, 0, False)
    
    # Passed parameters checking function
    _makeLoader.stypy_localization = localization
    _makeLoader.stypy_type_of_self = None
    _makeLoader.stypy_type_store = module_type_store
    _makeLoader.stypy_function_name = '_makeLoader'
    _makeLoader.stypy_param_names_list = ['prefix', 'sortUsing', 'suiteClass']
    _makeLoader.stypy_varargs_param_name = None
    _makeLoader.stypy_kwargs_param_name = None
    _makeLoader.stypy_call_defaults = defaults
    _makeLoader.stypy_call_varargs = varargs
    _makeLoader.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_makeLoader', ['prefix', 'sortUsing', 'suiteClass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_makeLoader', localization, ['prefix', 'sortUsing', 'suiteClass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_makeLoader(...)' code ##################

    
    # Assigning a Call to a Name (line 300):
    
    # Assigning a Call to a Name (line 300):
    
    # Call to TestLoader(...): (line 300)
    # Processing the call keyword arguments (line 300)
    kwargs_190043 = {}
    # Getting the type of 'TestLoader' (line 300)
    TestLoader_190042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 13), 'TestLoader', False)
    # Calling TestLoader(args, kwargs) (line 300)
    TestLoader_call_result_190044 = invoke(stypy.reporting.localization.Localization(__file__, 300, 13), TestLoader_190042, *[], **kwargs_190043)
    
    # Assigning a type to the variable 'loader' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'loader', TestLoader_call_result_190044)
    
    # Assigning a Name to a Attribute (line 301):
    
    # Assigning a Name to a Attribute (line 301):
    # Getting the type of 'sortUsing' (line 301)
    sortUsing_190045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 34), 'sortUsing')
    # Getting the type of 'loader' (line 301)
    loader_190046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'loader')
    # Setting the type of the member 'sortTestMethodsUsing' of a type (line 301)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 4), loader_190046, 'sortTestMethodsUsing', sortUsing_190045)
    
    # Assigning a Name to a Attribute (line 302):
    
    # Assigning a Name to a Attribute (line 302):
    # Getting the type of 'prefix' (line 302)
    prefix_190047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 30), 'prefix')
    # Getting the type of 'loader' (line 302)
    loader_190048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'loader')
    # Setting the type of the member 'testMethodPrefix' of a type (line 302)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 4), loader_190048, 'testMethodPrefix', prefix_190047)
    
    # Getting the type of 'suiteClass' (line 303)
    suiteClass_190049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 7), 'suiteClass')
    # Testing the type of an if condition (line 303)
    if_condition_190050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 4), suiteClass_190049)
    # Assigning a type to the variable 'if_condition_190050' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'if_condition_190050', if_condition_190050)
    # SSA begins for if statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 304):
    
    # Assigning a Name to a Attribute (line 304):
    # Getting the type of 'suiteClass' (line 304)
    suiteClass_190051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 28), 'suiteClass')
    # Getting the type of 'loader' (line 304)
    loader_190052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'loader')
    # Setting the type of the member 'suiteClass' of a type (line 304)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), loader_190052, 'suiteClass', suiteClass_190051)
    # SSA join for if statement (line 303)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'loader' (line 305)
    loader_190053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'loader')
    # Assigning a type to the variable 'stypy_return_type' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type', loader_190053)
    
    # ################# End of '_makeLoader(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_makeLoader' in the type store
    # Getting the type of 'stypy_return_type' (line 299)
    stypy_return_type_190054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190054)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_makeLoader'
    return stypy_return_type_190054

# Assigning a type to the variable '_makeLoader' (line 299)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 0), '_makeLoader', _makeLoader)

@norecursion
def getTestCaseNames(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'cmp' (line 307)
    cmp_190055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 54), 'cmp')
    defaults = [cmp_190055]
    # Create a new context for function 'getTestCaseNames'
    module_type_store = module_type_store.open_function_context('getTestCaseNames', 307, 0, False)
    
    # Passed parameters checking function
    getTestCaseNames.stypy_localization = localization
    getTestCaseNames.stypy_type_of_self = None
    getTestCaseNames.stypy_type_store = module_type_store
    getTestCaseNames.stypy_function_name = 'getTestCaseNames'
    getTestCaseNames.stypy_param_names_list = ['testCaseClass', 'prefix', 'sortUsing']
    getTestCaseNames.stypy_varargs_param_name = None
    getTestCaseNames.stypy_kwargs_param_name = None
    getTestCaseNames.stypy_call_defaults = defaults
    getTestCaseNames.stypy_call_varargs = varargs
    getTestCaseNames.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getTestCaseNames', ['testCaseClass', 'prefix', 'sortUsing'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getTestCaseNames', localization, ['testCaseClass', 'prefix', 'sortUsing'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getTestCaseNames(...)' code ##################

    
    # Call to getTestCaseNames(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'testCaseClass' (line 308)
    testCaseClass_190062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 59), 'testCaseClass', False)
    # Processing the call keyword arguments (line 308)
    kwargs_190063 = {}
    
    # Call to _makeLoader(...): (line 308)
    # Processing the call arguments (line 308)
    # Getting the type of 'prefix' (line 308)
    prefix_190057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 23), 'prefix', False)
    # Getting the type of 'sortUsing' (line 308)
    sortUsing_190058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 31), 'sortUsing', False)
    # Processing the call keyword arguments (line 308)
    kwargs_190059 = {}
    # Getting the type of '_makeLoader' (line 308)
    _makeLoader_190056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), '_makeLoader', False)
    # Calling _makeLoader(args, kwargs) (line 308)
    _makeLoader_call_result_190060 = invoke(stypy.reporting.localization.Localization(__file__, 308, 11), _makeLoader_190056, *[prefix_190057, sortUsing_190058], **kwargs_190059)
    
    # Obtaining the member 'getTestCaseNames' of a type (line 308)
    getTestCaseNames_190061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 11), _makeLoader_call_result_190060, 'getTestCaseNames')
    # Calling getTestCaseNames(args, kwargs) (line 308)
    getTestCaseNames_call_result_190064 = invoke(stypy.reporting.localization.Localization(__file__, 308, 11), getTestCaseNames_190061, *[testCaseClass_190062], **kwargs_190063)
    
    # Assigning a type to the variable 'stypy_return_type' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type', getTestCaseNames_call_result_190064)
    
    # ################# End of 'getTestCaseNames(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getTestCaseNames' in the type store
    # Getting the type of 'stypy_return_type' (line 307)
    stypy_return_type_190065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190065)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getTestCaseNames'
    return stypy_return_type_190065

# Assigning a type to the variable 'getTestCaseNames' (line 307)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'getTestCaseNames', getTestCaseNames)

@norecursion
def makeSuite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_190066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 36), 'str', 'test')
    # Getting the type of 'cmp' (line 310)
    cmp_190067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 54), 'cmp')
    # Getting the type of 'suite' (line 311)
    suite_190068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 25), 'suite')
    # Obtaining the member 'TestSuite' of a type (line 311)
    TestSuite_190069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 25), suite_190068, 'TestSuite')
    defaults = [str_190066, cmp_190067, TestSuite_190069]
    # Create a new context for function 'makeSuite'
    module_type_store = module_type_store.open_function_context('makeSuite', 310, 0, False)
    
    # Passed parameters checking function
    makeSuite.stypy_localization = localization
    makeSuite.stypy_type_of_self = None
    makeSuite.stypy_type_store = module_type_store
    makeSuite.stypy_function_name = 'makeSuite'
    makeSuite.stypy_param_names_list = ['testCaseClass', 'prefix', 'sortUsing', 'suiteClass']
    makeSuite.stypy_varargs_param_name = None
    makeSuite.stypy_kwargs_param_name = None
    makeSuite.stypy_call_defaults = defaults
    makeSuite.stypy_call_varargs = varargs
    makeSuite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'makeSuite', ['testCaseClass', 'prefix', 'sortUsing', 'suiteClass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'makeSuite', localization, ['testCaseClass', 'prefix', 'sortUsing', 'suiteClass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'makeSuite(...)' code ##################

    
    # Call to loadTestsFromTestCase(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'testCaseClass' (line 312)
    testCaseClass_190077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 76), 'testCaseClass', False)
    # Processing the call keyword arguments (line 312)
    kwargs_190078 = {}
    
    # Call to _makeLoader(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'prefix' (line 312)
    prefix_190071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'prefix', False)
    # Getting the type of 'sortUsing' (line 312)
    sortUsing_190072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 31), 'sortUsing', False)
    # Getting the type of 'suiteClass' (line 312)
    suiteClass_190073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 42), 'suiteClass', False)
    # Processing the call keyword arguments (line 312)
    kwargs_190074 = {}
    # Getting the type of '_makeLoader' (line 312)
    _makeLoader_190070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), '_makeLoader', False)
    # Calling _makeLoader(args, kwargs) (line 312)
    _makeLoader_call_result_190075 = invoke(stypy.reporting.localization.Localization(__file__, 312, 11), _makeLoader_190070, *[prefix_190071, sortUsing_190072, suiteClass_190073], **kwargs_190074)
    
    # Obtaining the member 'loadTestsFromTestCase' of a type (line 312)
    loadTestsFromTestCase_190076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 11), _makeLoader_call_result_190075, 'loadTestsFromTestCase')
    # Calling loadTestsFromTestCase(args, kwargs) (line 312)
    loadTestsFromTestCase_call_result_190079 = invoke(stypy.reporting.localization.Localization(__file__, 312, 11), loadTestsFromTestCase_190076, *[testCaseClass_190077], **kwargs_190078)
    
    # Assigning a type to the variable 'stypy_return_type' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type', loadTestsFromTestCase_call_result_190079)
    
    # ################# End of 'makeSuite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'makeSuite' in the type store
    # Getting the type of 'stypy_return_type' (line 310)
    stypy_return_type_190080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190080)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'makeSuite'
    return stypy_return_type_190080

# Assigning a type to the variable 'makeSuite' (line 310)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'makeSuite', makeSuite)

@norecursion
def findTestCases(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_190081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 33), 'str', 'test')
    # Getting the type of 'cmp' (line 314)
    cmp_190082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 51), 'cmp')
    # Getting the type of 'suite' (line 315)
    suite_190083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 29), 'suite')
    # Obtaining the member 'TestSuite' of a type (line 315)
    TestSuite_190084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 29), suite_190083, 'TestSuite')
    defaults = [str_190081, cmp_190082, TestSuite_190084]
    # Create a new context for function 'findTestCases'
    module_type_store = module_type_store.open_function_context('findTestCases', 314, 0, False)
    
    # Passed parameters checking function
    findTestCases.stypy_localization = localization
    findTestCases.stypy_type_of_self = None
    findTestCases.stypy_type_store = module_type_store
    findTestCases.stypy_function_name = 'findTestCases'
    findTestCases.stypy_param_names_list = ['module', 'prefix', 'sortUsing', 'suiteClass']
    findTestCases.stypy_varargs_param_name = None
    findTestCases.stypy_kwargs_param_name = None
    findTestCases.stypy_call_defaults = defaults
    findTestCases.stypy_call_varargs = varargs
    findTestCases.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findTestCases', ['module', 'prefix', 'sortUsing', 'suiteClass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findTestCases', localization, ['module', 'prefix', 'sortUsing', 'suiteClass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findTestCases(...)' code ##################

    
    # Call to loadTestsFromModule(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'module' (line 316)
    module_190092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 74), 'module', False)
    # Processing the call keyword arguments (line 316)
    kwargs_190093 = {}
    
    # Call to _makeLoader(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'prefix' (line 316)
    prefix_190086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'prefix', False)
    # Getting the type of 'sortUsing' (line 316)
    sortUsing_190087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 31), 'sortUsing', False)
    # Getting the type of 'suiteClass' (line 316)
    suiteClass_190088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 42), 'suiteClass', False)
    # Processing the call keyword arguments (line 316)
    kwargs_190089 = {}
    # Getting the type of '_makeLoader' (line 316)
    _makeLoader_190085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 11), '_makeLoader', False)
    # Calling _makeLoader(args, kwargs) (line 316)
    _makeLoader_call_result_190090 = invoke(stypy.reporting.localization.Localization(__file__, 316, 11), _makeLoader_190085, *[prefix_190086, sortUsing_190087, suiteClass_190088], **kwargs_190089)
    
    # Obtaining the member 'loadTestsFromModule' of a type (line 316)
    loadTestsFromModule_190091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 11), _makeLoader_call_result_190090, 'loadTestsFromModule')
    # Calling loadTestsFromModule(args, kwargs) (line 316)
    loadTestsFromModule_call_result_190094 = invoke(stypy.reporting.localization.Localization(__file__, 316, 11), loadTestsFromModule_190091, *[module_190092], **kwargs_190093)
    
    # Assigning a type to the variable 'stypy_return_type' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'stypy_return_type', loadTestsFromModule_call_result_190094)
    
    # ################# End of 'findTestCases(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findTestCases' in the type store
    # Getting the type of 'stypy_return_type' (line 314)
    stypy_return_type_190095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_190095)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findTestCases'
    return stypy_return_type_190095

# Assigning a type to the variable 'findTestCases' (line 314)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 0), 'findTestCases', findTestCases)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
