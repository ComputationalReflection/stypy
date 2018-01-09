
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: import re
3: import sys
4: 
5: import unittest
6: import unittest.test
7: 
8: 
9: class TestDiscovery(unittest.TestCase):
10: 
11:     # Heavily mocked tests so I can avoid hitting the filesystem
12:     def test_get_name_from_path(self):
13:         loader = unittest.TestLoader()
14: 
15:         loader._top_level_dir = '/foo'
16:         name = loader._get_name_from_path('/foo/bar/baz.py')
17:         self.assertEqual(name, 'bar.baz')
18: 
19:         if not __debug__:
20:             # asserts are off
21:             return
22: 
23:         with self.assertRaises(AssertionError):
24:             loader._get_name_from_path('/bar/baz.py')
25: 
26:     def test_find_tests(self):
27:         loader = unittest.TestLoader()
28: 
29:         original_listdir = os.listdir
30:         def restore_listdir():
31:             os.listdir = original_listdir
32:         original_isfile = os.path.isfile
33:         def restore_isfile():
34:             os.path.isfile = original_isfile
35:         original_isdir = os.path.isdir
36:         def restore_isdir():
37:             os.path.isdir = original_isdir
38: 
39:         path_lists = [['test1.py', 'test2.py', 'not_a_test.py', 'test_dir',
40:                        'test.foo', 'test-not-a-module.py', 'another_dir'],
41:                       ['test3.py', 'test4.py', ]]
42:         os.listdir = lambda path: path_lists.pop(0)
43:         self.addCleanup(restore_listdir)
44: 
45:         def isdir(path):
46:             return path.endswith('dir')
47:         os.path.isdir = isdir
48:         self.addCleanup(restore_isdir)
49: 
50:         def isfile(path):
51:             # another_dir is not a package and so shouldn't be recursed into
52:             return not path.endswith('dir') and not 'another_dir' in path
53:         os.path.isfile = isfile
54:         self.addCleanup(restore_isfile)
55: 
56:         loader._get_module_from_name = lambda path: path + ' module'
57:         loader.loadTestsFromModule = lambda module: module + ' tests'
58: 
59:         top_level = os.path.abspath('/foo')
60:         loader._top_level_dir = top_level
61:         suite = list(loader._find_tests(top_level, 'test*.py'))
62: 
63:         expected = [name + ' module tests' for name in
64:                     ('test1', 'test2')]
65:         expected.extend([('test_dir.%s' % name) + ' module tests' for name in
66:                     ('test3', 'test4')])
67:         self.assertEqual(suite, expected)
68: 
69:     def test_find_tests_with_package(self):
70:         loader = unittest.TestLoader()
71: 
72:         original_listdir = os.listdir
73:         def restore_listdir():
74:             os.listdir = original_listdir
75:         original_isfile = os.path.isfile
76:         def restore_isfile():
77:             os.path.isfile = original_isfile
78:         original_isdir = os.path.isdir
79:         def restore_isdir():
80:             os.path.isdir = original_isdir
81: 
82:         directories = ['a_directory', 'test_directory', 'test_directory2']
83:         path_lists = [directories, [], [], []]
84:         os.listdir = lambda path: path_lists.pop(0)
85:         self.addCleanup(restore_listdir)
86: 
87:         os.path.isdir = lambda path: True
88:         self.addCleanup(restore_isdir)
89: 
90:         os.path.isfile = lambda path: os.path.basename(path) not in directories
91:         self.addCleanup(restore_isfile)
92: 
93:         class Module(object):
94:             paths = []
95:             load_tests_args = []
96: 
97:             def __init__(self, path):
98:                 self.path = path
99:                 self.paths.append(path)
100:                 if os.path.basename(path) == 'test_directory':
101:                     def load_tests(loader, tests, pattern):
102:                         self.load_tests_args.append((loader, tests, pattern))
103:                         return 'load_tests'
104:                     self.load_tests = load_tests
105: 
106:             def __eq__(self, other):
107:                 return self.path == other.path
108: 
109:             # Silence py3k warning
110:             __hash__ = None
111: 
112:         loader._get_module_from_name = lambda name: Module(name)
113:         def loadTestsFromModule(module, use_load_tests):
114:             if use_load_tests:
115:                 raise self.failureException('use_load_tests should be False for packages')
116:             return module.path + ' module tests'
117:         loader.loadTestsFromModule = loadTestsFromModule
118: 
119:         loader._top_level_dir = '/foo'
120:         # this time no '.py' on the pattern so that it can match
121:         # a test package
122:         suite = list(loader._find_tests('/foo', 'test*'))
123: 
124:         # We should have loaded tests from the test_directory package by calling load_tests
125:         # and directly from the test_directory2 package
126:         self.assertEqual(suite,
127:                          ['load_tests', 'test_directory2' + ' module tests'])
128:         self.assertEqual(Module.paths, ['test_directory', 'test_directory2'])
129: 
130:         # load_tests should have been called once with loader, tests and pattern
131:         self.assertEqual(Module.load_tests_args,
132:                          [(loader, 'test_directory' + ' module tests', 'test*')])
133: 
134:     def test_discover(self):
135:         loader = unittest.TestLoader()
136: 
137:         original_isfile = os.path.isfile
138:         original_isdir = os.path.isdir
139:         def restore_isfile():
140:             os.path.isfile = original_isfile
141: 
142:         os.path.isfile = lambda path: False
143:         self.addCleanup(restore_isfile)
144: 
145:         orig_sys_path = sys.path[:]
146:         def restore_path():
147:             sys.path[:] = orig_sys_path
148:         self.addCleanup(restore_path)
149: 
150:         full_path = os.path.abspath(os.path.normpath('/foo'))
151:         with self.assertRaises(ImportError):
152:             loader.discover('/foo/bar', top_level_dir='/foo')
153: 
154:         self.assertEqual(loader._top_level_dir, full_path)
155:         self.assertIn(full_path, sys.path)
156: 
157:         os.path.isfile = lambda path: True
158:         os.path.isdir = lambda path: True
159: 
160:         def restore_isdir():
161:             os.path.isdir = original_isdir
162:         self.addCleanup(restore_isdir)
163: 
164:         _find_tests_args = []
165:         def _find_tests(start_dir, pattern):
166:             _find_tests_args.append((start_dir, pattern))
167:             return ['tests']
168:         loader._find_tests = _find_tests
169:         loader.suiteClass = str
170: 
171:         suite = loader.discover('/foo/bar/baz', 'pattern', '/foo/bar')
172: 
173:         top_level_dir = os.path.abspath('/foo/bar')
174:         start_dir = os.path.abspath('/foo/bar/baz')
175:         self.assertEqual(suite, "['tests']")
176:         self.assertEqual(loader._top_level_dir, top_level_dir)
177:         self.assertEqual(_find_tests_args, [(start_dir, 'pattern')])
178:         self.assertIn(top_level_dir, sys.path)
179: 
180:     def test_discover_with_modules_that_fail_to_import(self):
181:         loader = unittest.TestLoader()
182: 
183:         listdir = os.listdir
184:         os.listdir = lambda _: ['test_this_does_not_exist.py']
185:         isfile = os.path.isfile
186:         os.path.isfile = lambda _: True
187:         orig_sys_path = sys.path[:]
188:         def restore():
189:             os.path.isfile = isfile
190:             os.listdir = listdir
191:             sys.path[:] = orig_sys_path
192:         self.addCleanup(restore)
193: 
194:         suite = loader.discover('.')
195:         self.assertIn(os.getcwd(), sys.path)
196:         self.assertEqual(suite.countTestCases(), 1)
197:         test = list(list(suite)[0])[0] # extract test from suite
198: 
199:         with self.assertRaises(ImportError):
200:             test.test_this_does_not_exist()
201: 
202:     def test_command_line_handling_parseArgs(self):
203:         # Haha - take that uninstantiable class
204:         program = object.__new__(unittest.TestProgram)
205: 
206:         args = []
207:         def do_discovery(argv):
208:             args.extend(argv)
209:         program._do_discovery = do_discovery
210:         program.parseArgs(['something', 'discover'])
211:         self.assertEqual(args, [])
212: 
213:         program.parseArgs(['something', 'discover', 'foo', 'bar'])
214:         self.assertEqual(args, ['foo', 'bar'])
215: 
216:     def test_command_line_handling_do_discovery_too_many_arguments(self):
217:         class Stop(Exception):
218:             pass
219:         def usageExit():
220:             raise Stop
221: 
222:         program = object.__new__(unittest.TestProgram)
223:         program.usageExit = usageExit
224:         program.testLoader = None
225: 
226:         with self.assertRaises(Stop):
227:             # too many args
228:             program._do_discovery(['one', 'two', 'three', 'four'])
229: 
230: 
231:     def test_command_line_handling_do_discovery_uses_default_loader(self):
232:         program = object.__new__(unittest.TestProgram)
233: 
234:         class Loader(object):
235:             args = []
236:             def discover(self, start_dir, pattern, top_level_dir):
237:                 self.args.append((start_dir, pattern, top_level_dir))
238:                 return 'tests'
239: 
240:         program.testLoader = Loader()
241:         program._do_discovery(['-v'])
242:         self.assertEqual(Loader.args, [('.', 'test*.py', None)])
243: 
244:     def test_command_line_handling_do_discovery_calls_loader(self):
245:         program = object.__new__(unittest.TestProgram)
246: 
247:         class Loader(object):
248:             args = []
249:             def discover(self, start_dir, pattern, top_level_dir):
250:                 self.args.append((start_dir, pattern, top_level_dir))
251:                 return 'tests'
252: 
253:         program._do_discovery(['-v'], Loader=Loader)
254:         self.assertEqual(program.verbosity, 2)
255:         self.assertEqual(program.test, 'tests')
256:         self.assertEqual(Loader.args, [('.', 'test*.py', None)])
257: 
258:         Loader.args = []
259:         program = object.__new__(unittest.TestProgram)
260:         program._do_discovery(['--verbose'], Loader=Loader)
261:         self.assertEqual(program.test, 'tests')
262:         self.assertEqual(Loader.args, [('.', 'test*.py', None)])
263: 
264:         Loader.args = []
265:         program = object.__new__(unittest.TestProgram)
266:         program._do_discovery([], Loader=Loader)
267:         self.assertEqual(program.test, 'tests')
268:         self.assertEqual(Loader.args, [('.', 'test*.py', None)])
269: 
270:         Loader.args = []
271:         program = object.__new__(unittest.TestProgram)
272:         program._do_discovery(['fish'], Loader=Loader)
273:         self.assertEqual(program.test, 'tests')
274:         self.assertEqual(Loader.args, [('fish', 'test*.py', None)])
275: 
276:         Loader.args = []
277:         program = object.__new__(unittest.TestProgram)
278:         program._do_discovery(['fish', 'eggs'], Loader=Loader)
279:         self.assertEqual(program.test, 'tests')
280:         self.assertEqual(Loader.args, [('fish', 'eggs', None)])
281: 
282:         Loader.args = []
283:         program = object.__new__(unittest.TestProgram)
284:         program._do_discovery(['fish', 'eggs', 'ham'], Loader=Loader)
285:         self.assertEqual(program.test, 'tests')
286:         self.assertEqual(Loader.args, [('fish', 'eggs', 'ham')])
287: 
288:         Loader.args = []
289:         program = object.__new__(unittest.TestProgram)
290:         program._do_discovery(['-s', 'fish'], Loader=Loader)
291:         self.assertEqual(program.test, 'tests')
292:         self.assertEqual(Loader.args, [('fish', 'test*.py', None)])
293: 
294:         Loader.args = []
295:         program = object.__new__(unittest.TestProgram)
296:         program._do_discovery(['-t', 'fish'], Loader=Loader)
297:         self.assertEqual(program.test, 'tests')
298:         self.assertEqual(Loader.args, [('.', 'test*.py', 'fish')])
299: 
300:         Loader.args = []
301:         program = object.__new__(unittest.TestProgram)
302:         program._do_discovery(['-p', 'fish'], Loader=Loader)
303:         self.assertEqual(program.test, 'tests')
304:         self.assertEqual(Loader.args, [('.', 'fish', None)])
305:         self.assertFalse(program.failfast)
306:         self.assertFalse(program.catchbreak)
307: 
308:         Loader.args = []
309:         program = object.__new__(unittest.TestProgram)
310:         program._do_discovery(['-p', 'eggs', '-s', 'fish', '-v', '-f', '-c'],
311:                               Loader=Loader)
312:         self.assertEqual(program.test, 'tests')
313:         self.assertEqual(Loader.args, [('fish', 'eggs', None)])
314:         self.assertEqual(program.verbosity, 2)
315:         self.assertTrue(program.failfast)
316:         self.assertTrue(program.catchbreak)
317: 
318:     def setup_module_clash(self):
319:         class Module(object):
320:             __file__ = 'bar/foo.py'
321:         sys.modules['foo'] = Module
322:         full_path = os.path.abspath('foo')
323:         original_listdir = os.listdir
324:         original_isfile = os.path.isfile
325:         original_isdir = os.path.isdir
326: 
327:         def cleanup():
328:             os.listdir = original_listdir
329:             os.path.isfile = original_isfile
330:             os.path.isdir = original_isdir
331:             del sys.modules['foo']
332:             if full_path in sys.path:
333:                 sys.path.remove(full_path)
334:         self.addCleanup(cleanup)
335: 
336:         def listdir(_):
337:             return ['foo.py']
338:         def isfile(_):
339:             return True
340:         def isdir(_):
341:             return True
342:         os.listdir = listdir
343:         os.path.isfile = isfile
344:         os.path.isdir = isdir
345:         return full_path
346: 
347:     def test_detect_module_clash(self):
348:         full_path = self.setup_module_clash()
349:         loader = unittest.TestLoader()
350: 
351:         mod_dir = os.path.abspath('bar')
352:         expected_dir = os.path.abspath('foo')
353:         msg = re.escape(r"'foo' module incorrectly imported from %r. Expected %r. "
354:                 "Is this module globally installed?" % (mod_dir, expected_dir))
355:         self.assertRaisesRegexp(
356:             ImportError, '^%s$' % msg, loader.discover,
357:             start_dir='foo', pattern='foo.py'
358:         )
359:         self.assertEqual(sys.path[0], full_path)
360: 
361:     def test_module_symlink_ok(self):
362:         full_path = self.setup_module_clash()
363: 
364:         original_realpath = os.path.realpath
365: 
366:         mod_dir = os.path.abspath('bar')
367:         expected_dir = os.path.abspath('foo')
368: 
369:         def cleanup():
370:             os.path.realpath = original_realpath
371:         self.addCleanup(cleanup)
372: 
373:         def realpath(path):
374:             if path == os.path.join(mod_dir, 'foo.py'):
375:                 return os.path.join(expected_dir, 'foo.py')
376:             return path
377:         os.path.realpath = realpath
378:         loader = unittest.TestLoader()
379:         loader.discover(start_dir='foo', pattern='foo.py')
380: 
381:     def test_discovery_from_dotted_path(self):
382:         loader = unittest.TestLoader()
383: 
384:         tests = [self]
385:         expectedPath = os.path.abspath(os.path.dirname(unittest.test.__file__))
386: 
387:         self.wasRun = False
388:         def _find_tests(start_dir, pattern):
389:             self.wasRun = True
390:             self.assertEqual(start_dir, expectedPath)
391:             return tests
392:         loader._find_tests = _find_tests
393:         suite = loader.discover('unittest.test')
394:         self.assertTrue(self.wasRun)
395:         self.assertEqual(suite._tests, tests)
396: 
397: 
398: if __name__ == '__main__':
399:     unittest.main()
400: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import os' statement (line 1)
import os

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import re' statement (line 2)
import re

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import unittest' statement (line 5)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import unittest.test' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/unittest/test/')
import_199098 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test')

if (type(import_199098) is not StypyTypeError):

    if (import_199098 != 'pyd_module'):
        __import__(import_199098)
        sys_modules_199099 = sys.modules[import_199098]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test', sys_modules_199099.module_type_store, module_type_store)
    else:
        import unittest.test

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test', unittest.test, module_type_store)

else:
    # Assigning a type to the variable 'unittest.test' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest.test', import_199098)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/test/')

# Declaration of the 'TestDiscovery' class
# Getting the type of 'unittest' (line 9)
unittest_199100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'unittest')
# Obtaining the member 'TestCase' of a type (line 9)
TestCase_199101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 20), unittest_199100, 'TestCase')

class TestDiscovery(TestCase_199101, ):

    @norecursion
    def test_get_name_from_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_name_from_path'
        module_type_store = module_type_store.open_function_context('test_get_name_from_path', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_get_name_from_path')
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_get_name_from_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_get_name_from_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_name_from_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_name_from_path(...)' code ##################

        
        # Assigning a Call to a Name (line 13):
        
        # Call to TestLoader(...): (line 13)
        # Processing the call keyword arguments (line 13)
        kwargs_199104 = {}
        # Getting the type of 'unittest' (line 13)
        unittest_199102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'unittest', False)
        # Obtaining the member 'TestLoader' of a type (line 13)
        TestLoader_199103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 17), unittest_199102, 'TestLoader')
        # Calling TestLoader(args, kwargs) (line 13)
        TestLoader_call_result_199105 = invoke(stypy.reporting.localization.Localization(__file__, 13, 17), TestLoader_199103, *[], **kwargs_199104)
        
        # Assigning a type to the variable 'loader' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'loader', TestLoader_call_result_199105)
        
        # Assigning a Str to a Attribute (line 15):
        str_199106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'str', '/foo')
        # Getting the type of 'loader' (line 15)
        loader_199107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'loader')
        # Setting the type of the member '_top_level_dir' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), loader_199107, '_top_level_dir', str_199106)
        
        # Assigning a Call to a Name (line 16):
        
        # Call to _get_name_from_path(...): (line 16)
        # Processing the call arguments (line 16)
        str_199110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 42), 'str', '/foo/bar/baz.py')
        # Processing the call keyword arguments (line 16)
        kwargs_199111 = {}
        # Getting the type of 'loader' (line 16)
        loader_199108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'loader', False)
        # Obtaining the member '_get_name_from_path' of a type (line 16)
        _get_name_from_path_199109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 15), loader_199108, '_get_name_from_path')
        # Calling _get_name_from_path(args, kwargs) (line 16)
        _get_name_from_path_call_result_199112 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), _get_name_from_path_199109, *[str_199110], **kwargs_199111)
        
        # Assigning a type to the variable 'name' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'name', _get_name_from_path_call_result_199112)
        
        # Call to assertEqual(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'name' (line 17)
        name_199115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'name', False)
        str_199116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'str', 'bar.baz')
        # Processing the call keyword arguments (line 17)
        kwargs_199117 = {}
        # Getting the type of 'self' (line 17)
        self_199113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 17)
        assertEqual_199114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_199113, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 17)
        assertEqual_call_result_199118 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), assertEqual_199114, *[name_199115, str_199116], **kwargs_199117)
        
        
        
        # Getting the type of '__debug__' (line 19)
        debug___199119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), '__debug__')
        # Applying the 'not' unary operator (line 19)
        result_not__199120 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 11), 'not', debug___199119)
        
        # Testing the type of an if condition (line 19)
        if_condition_199121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 8), result_not__199120)
        # Assigning a type to the variable 'if_condition_199121' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'if_condition_199121', if_condition_199121)
        # SSA begins for if statement (line 19)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 19)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertRaises(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'AssertionError' (line 23)
        AssertionError_199124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'AssertionError', False)
        # Processing the call keyword arguments (line 23)
        kwargs_199125 = {}
        # Getting the type of 'self' (line 23)
        self_199122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 23)
        assertRaises_199123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), self_199122, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 23)
        assertRaises_call_result_199126 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), assertRaises_199123, *[AssertionError_199124], **kwargs_199125)
        
        with_199127 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 23, 13), assertRaises_call_result_199126, 'with parameter', '__enter__', '__exit__')

        if with_199127:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 23)
            enter___199128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), assertRaises_call_result_199126, '__enter__')
            with_enter_199129 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), enter___199128)
            
            # Call to _get_name_from_path(...): (line 24)
            # Processing the call arguments (line 24)
            str_199132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 39), 'str', '/bar/baz.py')
            # Processing the call keyword arguments (line 24)
            kwargs_199133 = {}
            # Getting the type of 'loader' (line 24)
            loader_199130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'loader', False)
            # Obtaining the member '_get_name_from_path' of a type (line 24)
            _get_name_from_path_199131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), loader_199130, '_get_name_from_path')
            # Calling _get_name_from_path(args, kwargs) (line 24)
            _get_name_from_path_call_result_199134 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), _get_name_from_path_199131, *[str_199132], **kwargs_199133)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 23)
            exit___199135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), assertRaises_call_result_199126, '__exit__')
            with_exit_199136 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), exit___199135, None, None, None)

        
        # ################# End of 'test_get_name_from_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_name_from_path' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_199137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_199137)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_name_from_path'
        return stypy_return_type_199137


    @norecursion
    def test_find_tests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_find_tests'
        module_type_store = module_type_store.open_function_context('test_find_tests', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_find_tests')
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_find_tests.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_find_tests', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_find_tests', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_find_tests(...)' code ##################

        
        # Assigning a Call to a Name (line 27):
        
        # Call to TestLoader(...): (line 27)
        # Processing the call keyword arguments (line 27)
        kwargs_199140 = {}
        # Getting the type of 'unittest' (line 27)
        unittest_199138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'unittest', False)
        # Obtaining the member 'TestLoader' of a type (line 27)
        TestLoader_199139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 17), unittest_199138, 'TestLoader')
        # Calling TestLoader(args, kwargs) (line 27)
        TestLoader_call_result_199141 = invoke(stypy.reporting.localization.Localization(__file__, 27, 17), TestLoader_199139, *[], **kwargs_199140)
        
        # Assigning a type to the variable 'loader' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'loader', TestLoader_call_result_199141)
        
        # Assigning a Attribute to a Name (line 29):
        # Getting the type of 'os' (line 29)
        os_199142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'os')
        # Obtaining the member 'listdir' of a type (line 29)
        listdir_199143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 27), os_199142, 'listdir')
        # Assigning a type to the variable 'original_listdir' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'original_listdir', listdir_199143)

        @norecursion
        def restore_listdir(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore_listdir'
            module_type_store = module_type_store.open_function_context('restore_listdir', 30, 8, False)
            
            # Passed parameters checking function
            restore_listdir.stypy_localization = localization
            restore_listdir.stypy_type_of_self = None
            restore_listdir.stypy_type_store = module_type_store
            restore_listdir.stypy_function_name = 'restore_listdir'
            restore_listdir.stypy_param_names_list = []
            restore_listdir.stypy_varargs_param_name = None
            restore_listdir.stypy_kwargs_param_name = None
            restore_listdir.stypy_call_defaults = defaults
            restore_listdir.stypy_call_varargs = varargs
            restore_listdir.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore_listdir', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore_listdir', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore_listdir(...)' code ##################

            
            # Assigning a Name to a Attribute (line 31):
            # Getting the type of 'original_listdir' (line 31)
            original_listdir_199144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'original_listdir')
            # Getting the type of 'os' (line 31)
            os_199145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'os')
            # Setting the type of the member 'listdir' of a type (line 31)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), os_199145, 'listdir', original_listdir_199144)
            
            # ################# End of 'restore_listdir(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore_listdir' in the type store
            # Getting the type of 'stypy_return_type' (line 30)
            stypy_return_type_199146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199146)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore_listdir'
            return stypy_return_type_199146

        # Assigning a type to the variable 'restore_listdir' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'restore_listdir', restore_listdir)
        
        # Assigning a Attribute to a Name (line 32):
        # Getting the type of 'os' (line 32)
        os_199147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'os')
        # Obtaining the member 'path' of a type (line 32)
        path_199148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 26), os_199147, 'path')
        # Obtaining the member 'isfile' of a type (line 32)
        isfile_199149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 26), path_199148, 'isfile')
        # Assigning a type to the variable 'original_isfile' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'original_isfile', isfile_199149)

        @norecursion
        def restore_isfile(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore_isfile'
            module_type_store = module_type_store.open_function_context('restore_isfile', 33, 8, False)
            
            # Passed parameters checking function
            restore_isfile.stypy_localization = localization
            restore_isfile.stypy_type_of_self = None
            restore_isfile.stypy_type_store = module_type_store
            restore_isfile.stypy_function_name = 'restore_isfile'
            restore_isfile.stypy_param_names_list = []
            restore_isfile.stypy_varargs_param_name = None
            restore_isfile.stypy_kwargs_param_name = None
            restore_isfile.stypy_call_defaults = defaults
            restore_isfile.stypy_call_varargs = varargs
            restore_isfile.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore_isfile', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore_isfile', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore_isfile(...)' code ##################

            
            # Assigning a Name to a Attribute (line 34):
            # Getting the type of 'original_isfile' (line 34)
            original_isfile_199150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 29), 'original_isfile')
            # Getting the type of 'os' (line 34)
            os_199151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'os')
            # Obtaining the member 'path' of a type (line 34)
            path_199152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), os_199151, 'path')
            # Setting the type of the member 'isfile' of a type (line 34)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), path_199152, 'isfile', original_isfile_199150)
            
            # ################# End of 'restore_isfile(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore_isfile' in the type store
            # Getting the type of 'stypy_return_type' (line 33)
            stypy_return_type_199153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199153)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore_isfile'
            return stypy_return_type_199153

        # Assigning a type to the variable 'restore_isfile' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'restore_isfile', restore_isfile)
        
        # Assigning a Attribute to a Name (line 35):
        # Getting the type of 'os' (line 35)
        os_199154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'os')
        # Obtaining the member 'path' of a type (line 35)
        path_199155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 25), os_199154, 'path')
        # Obtaining the member 'isdir' of a type (line 35)
        isdir_199156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 25), path_199155, 'isdir')
        # Assigning a type to the variable 'original_isdir' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'original_isdir', isdir_199156)

        @norecursion
        def restore_isdir(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore_isdir'
            module_type_store = module_type_store.open_function_context('restore_isdir', 36, 8, False)
            
            # Passed parameters checking function
            restore_isdir.stypy_localization = localization
            restore_isdir.stypy_type_of_self = None
            restore_isdir.stypy_type_store = module_type_store
            restore_isdir.stypy_function_name = 'restore_isdir'
            restore_isdir.stypy_param_names_list = []
            restore_isdir.stypy_varargs_param_name = None
            restore_isdir.stypy_kwargs_param_name = None
            restore_isdir.stypy_call_defaults = defaults
            restore_isdir.stypy_call_varargs = varargs
            restore_isdir.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore_isdir', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore_isdir', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore_isdir(...)' code ##################

            
            # Assigning a Name to a Attribute (line 37):
            # Getting the type of 'original_isdir' (line 37)
            original_isdir_199157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 28), 'original_isdir')
            # Getting the type of 'os' (line 37)
            os_199158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'os')
            # Obtaining the member 'path' of a type (line 37)
            path_199159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), os_199158, 'path')
            # Setting the type of the member 'isdir' of a type (line 37)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), path_199159, 'isdir', original_isdir_199157)
            
            # ################# End of 'restore_isdir(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore_isdir' in the type store
            # Getting the type of 'stypy_return_type' (line 36)
            stypy_return_type_199160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199160)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore_isdir'
            return stypy_return_type_199160

        # Assigning a type to the variable 'restore_isdir' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'restore_isdir', restore_isdir)
        
        # Assigning a List to a Name (line 39):
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_199161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_199162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        str_199163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'str', 'test1.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_199162, str_199163)
        # Adding element type (line 39)
        str_199164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'str', 'test2.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_199162, str_199164)
        # Adding element type (line 39)
        str_199165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 47), 'str', 'not_a_test.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_199162, str_199165)
        # Adding element type (line 39)
        str_199166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 64), 'str', 'test_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_199162, str_199166)
        # Adding element type (line 39)
        str_199167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'str', 'test.foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_199162, str_199167)
        # Adding element type (line 39)
        str_199168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'str', 'test-not-a-module.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_199162, str_199168)
        # Adding element type (line 39)
        str_199169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 59), 'str', 'another_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_199162, str_199169)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 21), list_199161, list_199162)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_199170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        str_199171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'str', 'test3.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), list_199170, str_199171)
        # Adding element type (line 41)
        str_199172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 35), 'str', 'test4.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), list_199170, str_199172)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 21), list_199161, list_199170)
        
        # Assigning a type to the variable 'path_lists' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'path_lists', list_199161)
        
        # Assigning a Lambda to a Attribute (line 42):

        @norecursion
        def _stypy_temp_lambda_65(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_65'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_65', 42, 21, True)
            # Passed parameters checking function
            _stypy_temp_lambda_65.stypy_localization = localization
            _stypy_temp_lambda_65.stypy_type_of_self = None
            _stypy_temp_lambda_65.stypy_type_store = module_type_store
            _stypy_temp_lambda_65.stypy_function_name = '_stypy_temp_lambda_65'
            _stypy_temp_lambda_65.stypy_param_names_list = ['path']
            _stypy_temp_lambda_65.stypy_varargs_param_name = None
            _stypy_temp_lambda_65.stypy_kwargs_param_name = None
            _stypy_temp_lambda_65.stypy_call_defaults = defaults
            _stypy_temp_lambda_65.stypy_call_varargs = varargs
            _stypy_temp_lambda_65.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_65', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_65', ['path'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to pop(...): (line 42)
            # Processing the call arguments (line 42)
            int_199175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 49), 'int')
            # Processing the call keyword arguments (line 42)
            kwargs_199176 = {}
            # Getting the type of 'path_lists' (line 42)
            path_lists_199173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 34), 'path_lists', False)
            # Obtaining the member 'pop' of a type (line 42)
            pop_199174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 34), path_lists_199173, 'pop')
            # Calling pop(args, kwargs) (line 42)
            pop_call_result_199177 = invoke(stypy.reporting.localization.Localization(__file__, 42, 34), pop_199174, *[int_199175], **kwargs_199176)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'stypy_return_type', pop_call_result_199177)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_65' in the type store
            # Getting the type of 'stypy_return_type' (line 42)
            stypy_return_type_199178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199178)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_65'
            return stypy_return_type_199178

        # Assigning a type to the variable '_stypy_temp_lambda_65' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), '_stypy_temp_lambda_65', _stypy_temp_lambda_65)
        # Getting the type of '_stypy_temp_lambda_65' (line 42)
        _stypy_temp_lambda_65_199179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), '_stypy_temp_lambda_65')
        # Getting the type of 'os' (line 42)
        os_199180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'os')
        # Setting the type of the member 'listdir' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), os_199180, 'listdir', _stypy_temp_lambda_65_199179)
        
        # Call to addCleanup(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'restore_listdir' (line 43)
        restore_listdir_199183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'restore_listdir', False)
        # Processing the call keyword arguments (line 43)
        kwargs_199184 = {}
        # Getting the type of 'self' (line 43)
        self_199181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 43)
        addCleanup_199182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_199181, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 43)
        addCleanup_call_result_199185 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), addCleanup_199182, *[restore_listdir_199183], **kwargs_199184)
        

        @norecursion
        def isdir(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'isdir'
            module_type_store = module_type_store.open_function_context('isdir', 45, 8, False)
            
            # Passed parameters checking function
            isdir.stypy_localization = localization
            isdir.stypy_type_of_self = None
            isdir.stypy_type_store = module_type_store
            isdir.stypy_function_name = 'isdir'
            isdir.stypy_param_names_list = ['path']
            isdir.stypy_varargs_param_name = None
            isdir.stypy_kwargs_param_name = None
            isdir.stypy_call_defaults = defaults
            isdir.stypy_call_varargs = varargs
            isdir.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'isdir', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'isdir', localization, ['path'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'isdir(...)' code ##################

            
            # Call to endswith(...): (line 46)
            # Processing the call arguments (line 46)
            str_199188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'str', 'dir')
            # Processing the call keyword arguments (line 46)
            kwargs_199189 = {}
            # Getting the type of 'path' (line 46)
            path_199186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'path', False)
            # Obtaining the member 'endswith' of a type (line 46)
            endswith_199187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), path_199186, 'endswith')
            # Calling endswith(args, kwargs) (line 46)
            endswith_call_result_199190 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), endswith_199187, *[str_199188], **kwargs_199189)
            
            # Assigning a type to the variable 'stypy_return_type' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', endswith_call_result_199190)
            
            # ################# End of 'isdir(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'isdir' in the type store
            # Getting the type of 'stypy_return_type' (line 45)
            stypy_return_type_199191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199191)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'isdir'
            return stypy_return_type_199191

        # Assigning a type to the variable 'isdir' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'isdir', isdir)
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'isdir' (line 47)
        isdir_199192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'isdir')
        # Getting the type of 'os' (line 47)
        os_199193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'os')
        # Obtaining the member 'path' of a type (line 47)
        path_199194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), os_199193, 'path')
        # Setting the type of the member 'isdir' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), path_199194, 'isdir', isdir_199192)
        
        # Call to addCleanup(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'restore_isdir' (line 48)
        restore_isdir_199197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'restore_isdir', False)
        # Processing the call keyword arguments (line 48)
        kwargs_199198 = {}
        # Getting the type of 'self' (line 48)
        self_199195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 48)
        addCleanup_199196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_199195, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 48)
        addCleanup_call_result_199199 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), addCleanup_199196, *[restore_isdir_199197], **kwargs_199198)
        

        @norecursion
        def isfile(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'isfile'
            module_type_store = module_type_store.open_function_context('isfile', 50, 8, False)
            
            # Passed parameters checking function
            isfile.stypy_localization = localization
            isfile.stypy_type_of_self = None
            isfile.stypy_type_store = module_type_store
            isfile.stypy_function_name = 'isfile'
            isfile.stypy_param_names_list = ['path']
            isfile.stypy_varargs_param_name = None
            isfile.stypy_kwargs_param_name = None
            isfile.stypy_call_defaults = defaults
            isfile.stypy_call_varargs = varargs
            isfile.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'isfile', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'isfile', localization, ['path'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'isfile(...)' code ##################

            
            # Evaluating a boolean operation
            
            
            # Call to endswith(...): (line 52)
            # Processing the call arguments (line 52)
            str_199202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 37), 'str', 'dir')
            # Processing the call keyword arguments (line 52)
            kwargs_199203 = {}
            # Getting the type of 'path' (line 52)
            path_199200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'path', False)
            # Obtaining the member 'endswith' of a type (line 52)
            endswith_199201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 23), path_199200, 'endswith')
            # Calling endswith(args, kwargs) (line 52)
            endswith_call_result_199204 = invoke(stypy.reporting.localization.Localization(__file__, 52, 23), endswith_199201, *[str_199202], **kwargs_199203)
            
            # Applying the 'not' unary operator (line 52)
            result_not__199205 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 19), 'not', endswith_call_result_199204)
            
            
            
            str_199206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 52), 'str', 'another_dir')
            # Getting the type of 'path' (line 52)
            path_199207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 69), 'path')
            # Applying the binary operator 'in' (line 52)
            result_contains_199208 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 52), 'in', str_199206, path_199207)
            
            # Applying the 'not' unary operator (line 52)
            result_not__199209 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 48), 'not', result_contains_199208)
            
            # Applying the binary operator 'and' (line 52)
            result_and_keyword_199210 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 19), 'and', result_not__199205, result_not__199209)
            
            # Assigning a type to the variable 'stypy_return_type' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'stypy_return_type', result_and_keyword_199210)
            
            # ################# End of 'isfile(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'isfile' in the type store
            # Getting the type of 'stypy_return_type' (line 50)
            stypy_return_type_199211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199211)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'isfile'
            return stypy_return_type_199211

        # Assigning a type to the variable 'isfile' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'isfile', isfile)
        
        # Assigning a Name to a Attribute (line 53):
        # Getting the type of 'isfile' (line 53)
        isfile_199212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'isfile')
        # Getting the type of 'os' (line 53)
        os_199213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'os')
        # Obtaining the member 'path' of a type (line 53)
        path_199214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), os_199213, 'path')
        # Setting the type of the member 'isfile' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), path_199214, 'isfile', isfile_199212)
        
        # Call to addCleanup(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'restore_isfile' (line 54)
        restore_isfile_199217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'restore_isfile', False)
        # Processing the call keyword arguments (line 54)
        kwargs_199218 = {}
        # Getting the type of 'self' (line 54)
        self_199215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 54)
        addCleanup_199216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_199215, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 54)
        addCleanup_call_result_199219 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), addCleanup_199216, *[restore_isfile_199217], **kwargs_199218)
        
        
        # Assigning a Lambda to a Attribute (line 56):

        @norecursion
        def _stypy_temp_lambda_66(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_66'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_66', 56, 39, True)
            # Passed parameters checking function
            _stypy_temp_lambda_66.stypy_localization = localization
            _stypy_temp_lambda_66.stypy_type_of_self = None
            _stypy_temp_lambda_66.stypy_type_store = module_type_store
            _stypy_temp_lambda_66.stypy_function_name = '_stypy_temp_lambda_66'
            _stypy_temp_lambda_66.stypy_param_names_list = ['path']
            _stypy_temp_lambda_66.stypy_varargs_param_name = None
            _stypy_temp_lambda_66.stypy_kwargs_param_name = None
            _stypy_temp_lambda_66.stypy_call_defaults = defaults
            _stypy_temp_lambda_66.stypy_call_varargs = varargs
            _stypy_temp_lambda_66.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_66', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_66', ['path'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'path' (line 56)
            path_199220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 52), 'path')
            str_199221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 59), 'str', ' module')
            # Applying the binary operator '+' (line 56)
            result_add_199222 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 52), '+', path_199220, str_199221)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'stypy_return_type', result_add_199222)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_66' in the type store
            # Getting the type of 'stypy_return_type' (line 56)
            stypy_return_type_199223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199223)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_66'
            return stypy_return_type_199223

        # Assigning a type to the variable '_stypy_temp_lambda_66' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), '_stypy_temp_lambda_66', _stypy_temp_lambda_66)
        # Getting the type of '_stypy_temp_lambda_66' (line 56)
        _stypy_temp_lambda_66_199224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), '_stypy_temp_lambda_66')
        # Getting the type of 'loader' (line 56)
        loader_199225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'loader')
        # Setting the type of the member '_get_module_from_name' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), loader_199225, '_get_module_from_name', _stypy_temp_lambda_66_199224)
        
        # Assigning a Lambda to a Attribute (line 57):

        @norecursion
        def _stypy_temp_lambda_67(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_67'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_67', 57, 37, True)
            # Passed parameters checking function
            _stypy_temp_lambda_67.stypy_localization = localization
            _stypy_temp_lambda_67.stypy_type_of_self = None
            _stypy_temp_lambda_67.stypy_type_store = module_type_store
            _stypy_temp_lambda_67.stypy_function_name = '_stypy_temp_lambda_67'
            _stypy_temp_lambda_67.stypy_param_names_list = ['module']
            _stypy_temp_lambda_67.stypy_varargs_param_name = None
            _stypy_temp_lambda_67.stypy_kwargs_param_name = None
            _stypy_temp_lambda_67.stypy_call_defaults = defaults
            _stypy_temp_lambda_67.stypy_call_varargs = varargs
            _stypy_temp_lambda_67.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_67', ['module'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_67', ['module'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'module' (line 57)
            module_199226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 52), 'module')
            str_199227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 61), 'str', ' tests')
            # Applying the binary operator '+' (line 57)
            result_add_199228 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 52), '+', module_199226, str_199227)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'stypy_return_type', result_add_199228)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_67' in the type store
            # Getting the type of 'stypy_return_type' (line 57)
            stypy_return_type_199229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199229)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_67'
            return stypy_return_type_199229

        # Assigning a type to the variable '_stypy_temp_lambda_67' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), '_stypy_temp_lambda_67', _stypy_temp_lambda_67)
        # Getting the type of '_stypy_temp_lambda_67' (line 57)
        _stypy_temp_lambda_67_199230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), '_stypy_temp_lambda_67')
        # Getting the type of 'loader' (line 57)
        loader_199231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'loader')
        # Setting the type of the member 'loadTestsFromModule' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), loader_199231, 'loadTestsFromModule', _stypy_temp_lambda_67_199230)
        
        # Assigning a Call to a Name (line 59):
        
        # Call to abspath(...): (line 59)
        # Processing the call arguments (line 59)
        str_199235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'str', '/foo')
        # Processing the call keyword arguments (line 59)
        kwargs_199236 = {}
        # Getting the type of 'os' (line 59)
        os_199232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 59)
        path_199233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), os_199232, 'path')
        # Obtaining the member 'abspath' of a type (line 59)
        abspath_199234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), path_199233, 'abspath')
        # Calling abspath(args, kwargs) (line 59)
        abspath_call_result_199237 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), abspath_199234, *[str_199235], **kwargs_199236)
        
        # Assigning a type to the variable 'top_level' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'top_level', abspath_call_result_199237)
        
        # Assigning a Name to a Attribute (line 60):
        # Getting the type of 'top_level' (line 60)
        top_level_199238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'top_level')
        # Getting the type of 'loader' (line 60)
        loader_199239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'loader')
        # Setting the type of the member '_top_level_dir' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), loader_199239, '_top_level_dir', top_level_199238)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to list(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to _find_tests(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'top_level' (line 61)
        top_level_199243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 40), 'top_level', False)
        str_199244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 51), 'str', 'test*.py')
        # Processing the call keyword arguments (line 61)
        kwargs_199245 = {}
        # Getting the type of 'loader' (line 61)
        loader_199241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'loader', False)
        # Obtaining the member '_find_tests' of a type (line 61)
        _find_tests_199242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 21), loader_199241, '_find_tests')
        # Calling _find_tests(args, kwargs) (line 61)
        _find_tests_call_result_199246 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), _find_tests_199242, *[top_level_199243, str_199244], **kwargs_199245)
        
        # Processing the call keyword arguments (line 61)
        kwargs_199247 = {}
        # Getting the type of 'list' (line 61)
        list_199240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'list', False)
        # Calling list(args, kwargs) (line 61)
        list_call_result_199248 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), list_199240, *[_find_tests_call_result_199246], **kwargs_199247)
        
        # Assigning a type to the variable 'suite' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'suite', list_call_result_199248)
        
        # Assigning a ListComp to a Name (line 63):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 64)
        tuple_199252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 64)
        # Adding element type (line 64)
        str_199253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 21), 'str', 'test1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 21), tuple_199252, str_199253)
        # Adding element type (line 64)
        str_199254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 30), 'str', 'test2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 21), tuple_199252, str_199254)
        
        comprehension_199255 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), tuple_199252)
        # Assigning a type to the variable 'name' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'name', comprehension_199255)
        # Getting the type of 'name' (line 63)
        name_199249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'name')
        str_199250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'str', ' module tests')
        # Applying the binary operator '+' (line 63)
        result_add_199251 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 20), '+', name_199249, str_199250)
        
        list_199256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), list_199256, result_add_199251)
        # Assigning a type to the variable 'expected' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'expected', list_199256)
        
        # Call to extend(...): (line 65)
        # Processing the call arguments (line 65)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_199264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        str_199265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'str', 'test3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 21), tuple_199264, str_199265)
        # Adding element type (line 66)
        str_199266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 30), 'str', 'test4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 21), tuple_199264, str_199266)
        
        comprehension_199267 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), tuple_199264)
        # Assigning a type to the variable 'name' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'name', comprehension_199267)
        str_199259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'str', 'test_dir.%s')
        # Getting the type of 'name' (line 65)
        name_199260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'name', False)
        # Applying the binary operator '%' (line 65)
        result_mod_199261 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 26), '%', str_199259, name_199260)
        
        str_199262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 50), 'str', ' module tests')
        # Applying the binary operator '+' (line 65)
        result_add_199263 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 25), '+', result_mod_199261, str_199262)
        
        list_199268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 25), list_199268, result_add_199263)
        # Processing the call keyword arguments (line 65)
        kwargs_199269 = {}
        # Getting the type of 'expected' (line 65)
        expected_199257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'expected', False)
        # Obtaining the member 'extend' of a type (line 65)
        extend_199258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), expected_199257, 'extend')
        # Calling extend(args, kwargs) (line 65)
        extend_call_result_199270 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), extend_199258, *[list_199268], **kwargs_199269)
        
        
        # Call to assertEqual(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'suite' (line 67)
        suite_199273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'suite', False)
        # Getting the type of 'expected' (line 67)
        expected_199274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 32), 'expected', False)
        # Processing the call keyword arguments (line 67)
        kwargs_199275 = {}
        # Getting the type of 'self' (line 67)
        self_199271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 67)
        assertEqual_199272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_199271, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 67)
        assertEqual_call_result_199276 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assertEqual_199272, *[suite_199273, expected_199274], **kwargs_199275)
        
        
        # ################# End of 'test_find_tests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_find_tests' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_199277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_199277)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_find_tests'
        return stypy_return_type_199277


    @norecursion
    def test_find_tests_with_package(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_find_tests_with_package'
        module_type_store = module_type_store.open_function_context('test_find_tests_with_package', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_find_tests_with_package')
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_find_tests_with_package.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_find_tests_with_package', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_find_tests_with_package', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_find_tests_with_package(...)' code ##################

        
        # Assigning a Call to a Name (line 70):
        
        # Call to TestLoader(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_199280 = {}
        # Getting the type of 'unittest' (line 70)
        unittest_199278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'unittest', False)
        # Obtaining the member 'TestLoader' of a type (line 70)
        TestLoader_199279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 17), unittest_199278, 'TestLoader')
        # Calling TestLoader(args, kwargs) (line 70)
        TestLoader_call_result_199281 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), TestLoader_199279, *[], **kwargs_199280)
        
        # Assigning a type to the variable 'loader' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'loader', TestLoader_call_result_199281)
        
        # Assigning a Attribute to a Name (line 72):
        # Getting the type of 'os' (line 72)
        os_199282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'os')
        # Obtaining the member 'listdir' of a type (line 72)
        listdir_199283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 27), os_199282, 'listdir')
        # Assigning a type to the variable 'original_listdir' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'original_listdir', listdir_199283)

        @norecursion
        def restore_listdir(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore_listdir'
            module_type_store = module_type_store.open_function_context('restore_listdir', 73, 8, False)
            
            # Passed parameters checking function
            restore_listdir.stypy_localization = localization
            restore_listdir.stypy_type_of_self = None
            restore_listdir.stypy_type_store = module_type_store
            restore_listdir.stypy_function_name = 'restore_listdir'
            restore_listdir.stypy_param_names_list = []
            restore_listdir.stypy_varargs_param_name = None
            restore_listdir.stypy_kwargs_param_name = None
            restore_listdir.stypy_call_defaults = defaults
            restore_listdir.stypy_call_varargs = varargs
            restore_listdir.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore_listdir', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore_listdir', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore_listdir(...)' code ##################

            
            # Assigning a Name to a Attribute (line 74):
            # Getting the type of 'original_listdir' (line 74)
            original_listdir_199284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'original_listdir')
            # Getting the type of 'os' (line 74)
            os_199285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'os')
            # Setting the type of the member 'listdir' of a type (line 74)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), os_199285, 'listdir', original_listdir_199284)
            
            # ################# End of 'restore_listdir(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore_listdir' in the type store
            # Getting the type of 'stypy_return_type' (line 73)
            stypy_return_type_199286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199286)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore_listdir'
            return stypy_return_type_199286

        # Assigning a type to the variable 'restore_listdir' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'restore_listdir', restore_listdir)
        
        # Assigning a Attribute to a Name (line 75):
        # Getting the type of 'os' (line 75)
        os_199287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'os')
        # Obtaining the member 'path' of a type (line 75)
        path_199288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 26), os_199287, 'path')
        # Obtaining the member 'isfile' of a type (line 75)
        isfile_199289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 26), path_199288, 'isfile')
        # Assigning a type to the variable 'original_isfile' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'original_isfile', isfile_199289)

        @norecursion
        def restore_isfile(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore_isfile'
            module_type_store = module_type_store.open_function_context('restore_isfile', 76, 8, False)
            
            # Passed parameters checking function
            restore_isfile.stypy_localization = localization
            restore_isfile.stypy_type_of_self = None
            restore_isfile.stypy_type_store = module_type_store
            restore_isfile.stypy_function_name = 'restore_isfile'
            restore_isfile.stypy_param_names_list = []
            restore_isfile.stypy_varargs_param_name = None
            restore_isfile.stypy_kwargs_param_name = None
            restore_isfile.stypy_call_defaults = defaults
            restore_isfile.stypy_call_varargs = varargs
            restore_isfile.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore_isfile', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore_isfile', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore_isfile(...)' code ##################

            
            # Assigning a Name to a Attribute (line 77):
            # Getting the type of 'original_isfile' (line 77)
            original_isfile_199290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'original_isfile')
            # Getting the type of 'os' (line 77)
            os_199291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'os')
            # Obtaining the member 'path' of a type (line 77)
            path_199292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), os_199291, 'path')
            # Setting the type of the member 'isfile' of a type (line 77)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), path_199292, 'isfile', original_isfile_199290)
            
            # ################# End of 'restore_isfile(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore_isfile' in the type store
            # Getting the type of 'stypy_return_type' (line 76)
            stypy_return_type_199293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199293)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore_isfile'
            return stypy_return_type_199293

        # Assigning a type to the variable 'restore_isfile' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'restore_isfile', restore_isfile)
        
        # Assigning a Attribute to a Name (line 78):
        # Getting the type of 'os' (line 78)
        os_199294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'os')
        # Obtaining the member 'path' of a type (line 78)
        path_199295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 25), os_199294, 'path')
        # Obtaining the member 'isdir' of a type (line 78)
        isdir_199296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 25), path_199295, 'isdir')
        # Assigning a type to the variable 'original_isdir' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'original_isdir', isdir_199296)

        @norecursion
        def restore_isdir(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore_isdir'
            module_type_store = module_type_store.open_function_context('restore_isdir', 79, 8, False)
            
            # Passed parameters checking function
            restore_isdir.stypy_localization = localization
            restore_isdir.stypy_type_of_self = None
            restore_isdir.stypy_type_store = module_type_store
            restore_isdir.stypy_function_name = 'restore_isdir'
            restore_isdir.stypy_param_names_list = []
            restore_isdir.stypy_varargs_param_name = None
            restore_isdir.stypy_kwargs_param_name = None
            restore_isdir.stypy_call_defaults = defaults
            restore_isdir.stypy_call_varargs = varargs
            restore_isdir.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore_isdir', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore_isdir', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore_isdir(...)' code ##################

            
            # Assigning a Name to a Attribute (line 80):
            # Getting the type of 'original_isdir' (line 80)
            original_isdir_199297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'original_isdir')
            # Getting the type of 'os' (line 80)
            os_199298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'os')
            # Obtaining the member 'path' of a type (line 80)
            path_199299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), os_199298, 'path')
            # Setting the type of the member 'isdir' of a type (line 80)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), path_199299, 'isdir', original_isdir_199297)
            
            # ################# End of 'restore_isdir(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore_isdir' in the type store
            # Getting the type of 'stypy_return_type' (line 79)
            stypy_return_type_199300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199300)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore_isdir'
            return stypy_return_type_199300

        # Assigning a type to the variable 'restore_isdir' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'restore_isdir', restore_isdir)
        
        # Assigning a List to a Name (line 82):
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_199301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        str_199302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'str', 'a_directory')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_199301, str_199302)
        # Adding element type (line 82)
        str_199303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 38), 'str', 'test_directory')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_199301, str_199303)
        # Adding element type (line 82)
        str_199304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 56), 'str', 'test_directory2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_199301, str_199304)
        
        # Assigning a type to the variable 'directories' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'directories', list_199301)
        
        # Assigning a List to a Name (line 83):
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_199305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        # Getting the type of 'directories' (line 83)
        directories_199306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'directories')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_199305, directories_199306)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_199307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_199305, list_199307)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_199308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_199305, list_199308)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_199309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_199305, list_199309)
        
        # Assigning a type to the variable 'path_lists' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'path_lists', list_199305)
        
        # Assigning a Lambda to a Attribute (line 84):

        @norecursion
        def _stypy_temp_lambda_68(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_68'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_68', 84, 21, True)
            # Passed parameters checking function
            _stypy_temp_lambda_68.stypy_localization = localization
            _stypy_temp_lambda_68.stypy_type_of_self = None
            _stypy_temp_lambda_68.stypy_type_store = module_type_store
            _stypy_temp_lambda_68.stypy_function_name = '_stypy_temp_lambda_68'
            _stypy_temp_lambda_68.stypy_param_names_list = ['path']
            _stypy_temp_lambda_68.stypy_varargs_param_name = None
            _stypy_temp_lambda_68.stypy_kwargs_param_name = None
            _stypy_temp_lambda_68.stypy_call_defaults = defaults
            _stypy_temp_lambda_68.stypy_call_varargs = varargs
            _stypy_temp_lambda_68.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_68', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_68', ['path'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to pop(...): (line 84)
            # Processing the call arguments (line 84)
            int_199312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 49), 'int')
            # Processing the call keyword arguments (line 84)
            kwargs_199313 = {}
            # Getting the type of 'path_lists' (line 84)
            path_lists_199310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 34), 'path_lists', False)
            # Obtaining the member 'pop' of a type (line 84)
            pop_199311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 34), path_lists_199310, 'pop')
            # Calling pop(args, kwargs) (line 84)
            pop_call_result_199314 = invoke(stypy.reporting.localization.Localization(__file__, 84, 34), pop_199311, *[int_199312], **kwargs_199313)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'stypy_return_type', pop_call_result_199314)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_68' in the type store
            # Getting the type of 'stypy_return_type' (line 84)
            stypy_return_type_199315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199315)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_68'
            return stypy_return_type_199315

        # Assigning a type to the variable '_stypy_temp_lambda_68' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), '_stypy_temp_lambda_68', _stypy_temp_lambda_68)
        # Getting the type of '_stypy_temp_lambda_68' (line 84)
        _stypy_temp_lambda_68_199316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), '_stypy_temp_lambda_68')
        # Getting the type of 'os' (line 84)
        os_199317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'os')
        # Setting the type of the member 'listdir' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), os_199317, 'listdir', _stypy_temp_lambda_68_199316)
        
        # Call to addCleanup(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'restore_listdir' (line 85)
        restore_listdir_199320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'restore_listdir', False)
        # Processing the call keyword arguments (line 85)
        kwargs_199321 = {}
        # Getting the type of 'self' (line 85)
        self_199318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 85)
        addCleanup_199319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_199318, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 85)
        addCleanup_call_result_199322 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), addCleanup_199319, *[restore_listdir_199320], **kwargs_199321)
        
        
        # Assigning a Lambda to a Attribute (line 87):

        @norecursion
        def _stypy_temp_lambda_69(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_69'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_69', 87, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_69.stypy_localization = localization
            _stypy_temp_lambda_69.stypy_type_of_self = None
            _stypy_temp_lambda_69.stypy_type_store = module_type_store
            _stypy_temp_lambda_69.stypy_function_name = '_stypy_temp_lambda_69'
            _stypy_temp_lambda_69.stypy_param_names_list = ['path']
            _stypy_temp_lambda_69.stypy_varargs_param_name = None
            _stypy_temp_lambda_69.stypy_kwargs_param_name = None
            _stypy_temp_lambda_69.stypy_call_defaults = defaults
            _stypy_temp_lambda_69.stypy_call_varargs = varargs
            _stypy_temp_lambda_69.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_69', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_69', ['path'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'True' (line 87)
            True_199323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'True')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'stypy_return_type', True_199323)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_69' in the type store
            # Getting the type of 'stypy_return_type' (line 87)
            stypy_return_type_199324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199324)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_69'
            return stypy_return_type_199324

        # Assigning a type to the variable '_stypy_temp_lambda_69' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), '_stypy_temp_lambda_69', _stypy_temp_lambda_69)
        # Getting the type of '_stypy_temp_lambda_69' (line 87)
        _stypy_temp_lambda_69_199325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), '_stypy_temp_lambda_69')
        # Getting the type of 'os' (line 87)
        os_199326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'os')
        # Obtaining the member 'path' of a type (line 87)
        path_199327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), os_199326, 'path')
        # Setting the type of the member 'isdir' of a type (line 87)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), path_199327, 'isdir', _stypy_temp_lambda_69_199325)
        
        # Call to addCleanup(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'restore_isdir' (line 88)
        restore_isdir_199330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'restore_isdir', False)
        # Processing the call keyword arguments (line 88)
        kwargs_199331 = {}
        # Getting the type of 'self' (line 88)
        self_199328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 88)
        addCleanup_199329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_199328, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 88)
        addCleanup_call_result_199332 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), addCleanup_199329, *[restore_isdir_199330], **kwargs_199331)
        
        
        # Assigning a Lambda to a Attribute (line 90):

        @norecursion
        def _stypy_temp_lambda_70(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_70'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_70', 90, 25, True)
            # Passed parameters checking function
            _stypy_temp_lambda_70.stypy_localization = localization
            _stypy_temp_lambda_70.stypy_type_of_self = None
            _stypy_temp_lambda_70.stypy_type_store = module_type_store
            _stypy_temp_lambda_70.stypy_function_name = '_stypy_temp_lambda_70'
            _stypy_temp_lambda_70.stypy_param_names_list = ['path']
            _stypy_temp_lambda_70.stypy_varargs_param_name = None
            _stypy_temp_lambda_70.stypy_kwargs_param_name = None
            _stypy_temp_lambda_70.stypy_call_defaults = defaults
            _stypy_temp_lambda_70.stypy_call_varargs = varargs
            _stypy_temp_lambda_70.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_70', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_70', ['path'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Call to basename(...): (line 90)
            # Processing the call arguments (line 90)
            # Getting the type of 'path' (line 90)
            path_199336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 55), 'path', False)
            # Processing the call keyword arguments (line 90)
            kwargs_199337 = {}
            # Getting the type of 'os' (line 90)
            os_199333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 38), 'os', False)
            # Obtaining the member 'path' of a type (line 90)
            path_199334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 38), os_199333, 'path')
            # Obtaining the member 'basename' of a type (line 90)
            basename_199335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 38), path_199334, 'basename')
            # Calling basename(args, kwargs) (line 90)
            basename_call_result_199338 = invoke(stypy.reporting.localization.Localization(__file__, 90, 38), basename_199335, *[path_199336], **kwargs_199337)
            
            # Getting the type of 'directories' (line 90)
            directories_199339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 68), 'directories')
            # Applying the binary operator 'notin' (line 90)
            result_contains_199340 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 38), 'notin', basename_call_result_199338, directories_199339)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'stypy_return_type', result_contains_199340)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_70' in the type store
            # Getting the type of 'stypy_return_type' (line 90)
            stypy_return_type_199341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199341)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_70'
            return stypy_return_type_199341

        # Assigning a type to the variable '_stypy_temp_lambda_70' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), '_stypy_temp_lambda_70', _stypy_temp_lambda_70)
        # Getting the type of '_stypy_temp_lambda_70' (line 90)
        _stypy_temp_lambda_70_199342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), '_stypy_temp_lambda_70')
        # Getting the type of 'os' (line 90)
        os_199343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'os')
        # Obtaining the member 'path' of a type (line 90)
        path_199344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), os_199343, 'path')
        # Setting the type of the member 'isfile' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), path_199344, 'isfile', _stypy_temp_lambda_70_199342)
        
        # Call to addCleanup(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'restore_isfile' (line 91)
        restore_isfile_199347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'restore_isfile', False)
        # Processing the call keyword arguments (line 91)
        kwargs_199348 = {}
        # Getting the type of 'self' (line 91)
        self_199345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 91)
        addCleanup_199346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_199345, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 91)
        addCleanup_call_result_199349 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), addCleanup_199346, *[restore_isfile_199347], **kwargs_199348)
        
        # Declaration of the 'Module' class

        class Module(object, ):
            
            # Assigning a List to a Name (line 94):
            
            # Obtaining an instance of the builtin type 'list' (line 94)
            list_199350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 94)
            
            # Assigning a type to the variable 'paths' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'paths', list_199350)
            
            # Assigning a List to a Name (line 95):
            
            # Obtaining an instance of the builtin type 'list' (line 95)
            list_199351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 30), 'list')
            # Adding type elements to the builtin type 'list' instance (line 95)
            
            # Assigning a type to the variable 'load_tests_args' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'load_tests_args', list_199351)

            @norecursion
            def __init__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__init__'
                module_type_store = module_type_store.open_function_context('__init__', 97, 12, False)
                # Assigning a type to the variable 'self' (line 98)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Module.__init__', ['path'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return

                # Initialize method data
                init_call_information(module_type_store, '__init__', localization, ['path'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of '__init__(...)' code ##################

                
                # Assigning a Name to a Attribute (line 98):
                # Getting the type of 'path' (line 98)
                path_199352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'path')
                # Getting the type of 'self' (line 98)
                self_199353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'self')
                # Setting the type of the member 'path' of a type (line 98)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), self_199353, 'path', path_199352)
                
                # Call to append(...): (line 99)
                # Processing the call arguments (line 99)
                # Getting the type of 'path' (line 99)
                path_199357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'path', False)
                # Processing the call keyword arguments (line 99)
                kwargs_199358 = {}
                # Getting the type of 'self' (line 99)
                self_199354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'self', False)
                # Obtaining the member 'paths' of a type (line 99)
                paths_199355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), self_199354, 'paths')
                # Obtaining the member 'append' of a type (line 99)
                append_199356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), paths_199355, 'append')
                # Calling append(args, kwargs) (line 99)
                append_call_result_199359 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), append_199356, *[path_199357], **kwargs_199358)
                
                
                
                
                # Call to basename(...): (line 100)
                # Processing the call arguments (line 100)
                # Getting the type of 'path' (line 100)
                path_199363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), 'path', False)
                # Processing the call keyword arguments (line 100)
                kwargs_199364 = {}
                # Getting the type of 'os' (line 100)
                os_199360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'os', False)
                # Obtaining the member 'path' of a type (line 100)
                path_199361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 19), os_199360, 'path')
                # Obtaining the member 'basename' of a type (line 100)
                basename_199362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 19), path_199361, 'basename')
                # Calling basename(args, kwargs) (line 100)
                basename_call_result_199365 = invoke(stypy.reporting.localization.Localization(__file__, 100, 19), basename_199362, *[path_199363], **kwargs_199364)
                
                str_199366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 45), 'str', 'test_directory')
                # Applying the binary operator '==' (line 100)
                result_eq_199367 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 19), '==', basename_call_result_199365, str_199366)
                
                # Testing the type of an if condition (line 100)
                if_condition_199368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 16), result_eq_199367)
                # Assigning a type to the variable 'if_condition_199368' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'if_condition_199368', if_condition_199368)
                # SSA begins for if statement (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

                @norecursion
                def load_tests(localization, *varargs, **kwargs):
                    global module_type_store
                    # Assign values to the parameters with defaults
                    defaults = []
                    # Create a new context for function 'load_tests'
                    module_type_store = module_type_store.open_function_context('load_tests', 101, 20, False)
                    
                    # Passed parameters checking function
                    load_tests.stypy_localization = localization
                    load_tests.stypy_type_of_self = None
                    load_tests.stypy_type_store = module_type_store
                    load_tests.stypy_function_name = 'load_tests'
                    load_tests.stypy_param_names_list = ['loader', 'tests', 'pattern']
                    load_tests.stypy_varargs_param_name = None
                    load_tests.stypy_kwargs_param_name = None
                    load_tests.stypy_call_defaults = defaults
                    load_tests.stypy_call_varargs = varargs
                    load_tests.stypy_call_kwargs = kwargs
                    arguments = process_argument_values(localization, None, module_type_store, 'load_tests', ['loader', 'tests', 'pattern'], None, None, defaults, varargs, kwargs)

                    if is_error_type(arguments):
                        # Destroy the current context
                        module_type_store = module_type_store.close_function_context()
                        return arguments

                    # Initialize method data
                    init_call_information(module_type_store, 'load_tests', localization, ['loader', 'tests', 'pattern'], arguments)
                    
                    # Default return type storage variable (SSA)
                    # Assigning a type to the variable 'stypy_return_type'
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                    
                    
                    # ################# Begin of 'load_tests(...)' code ##################

                    
                    # Call to append(...): (line 102)
                    # Processing the call arguments (line 102)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 102)
                    tuple_199372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 53), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 102)
                    # Adding element type (line 102)
                    # Getting the type of 'loader' (line 102)
                    loader_199373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 53), 'loader', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 53), tuple_199372, loader_199373)
                    # Adding element type (line 102)
                    # Getting the type of 'tests' (line 102)
                    tests_199374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 61), 'tests', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 53), tuple_199372, tests_199374)
                    # Adding element type (line 102)
                    # Getting the type of 'pattern' (line 102)
                    pattern_199375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 68), 'pattern', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 53), tuple_199372, pattern_199375)
                    
                    # Processing the call keyword arguments (line 102)
                    kwargs_199376 = {}
                    # Getting the type of 'self' (line 102)
                    self_199369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'self', False)
                    # Obtaining the member 'load_tests_args' of a type (line 102)
                    load_tests_args_199370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), self_199369, 'load_tests_args')
                    # Obtaining the member 'append' of a type (line 102)
                    append_199371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), load_tests_args_199370, 'append')
                    # Calling append(args, kwargs) (line 102)
                    append_call_result_199377 = invoke(stypy.reporting.localization.Localization(__file__, 102, 24), append_199371, *[tuple_199372], **kwargs_199376)
                    
                    str_199378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 31), 'str', 'load_tests')
                    # Assigning a type to the variable 'stypy_return_type' (line 103)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'stypy_return_type', str_199378)
                    
                    # ################# End of 'load_tests(...)' code ##################

                    # Teardown call information
                    teardown_call_information(localization, arguments)
                    
                    # Storing the return type of function 'load_tests' in the type store
                    # Getting the type of 'stypy_return_type' (line 101)
                    stypy_return_type_199379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'stypy_return_type')
                    module_type_store.store_return_type_of_current_context(stypy_return_type_199379)
                    
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    
                    # Return type of the function 'load_tests'
                    return stypy_return_type_199379

                # Assigning a type to the variable 'load_tests' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'load_tests', load_tests)
                
                # Assigning a Name to a Attribute (line 104):
                # Getting the type of 'load_tests' (line 104)
                load_tests_199380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 38), 'load_tests')
                # Getting the type of 'self' (line 104)
                self_199381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'self')
                # Setting the type of the member 'load_tests' of a type (line 104)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 20), self_199381, 'load_tests', load_tests_199380)
                # SSA join for if statement (line 100)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # ################# End of '__init__(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()


            @norecursion
            def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__eq__'
                module_type_store = module_type_store.open_function_context('__eq__', 106, 12, False)
                # Assigning a type to the variable 'self' (line 107)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Module.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
                Module.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Module.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
                Module.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'Module.__eq__')
                Module.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
                Module.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
                Module.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Module.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
                Module.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
                Module.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Module.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Module.__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

                
                # Getting the type of 'self' (line 107)
                self_199382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'self')
                # Obtaining the member 'path' of a type (line 107)
                path_199383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 23), self_199382, 'path')
                # Getting the type of 'other' (line 107)
                other_199384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'other')
                # Obtaining the member 'path' of a type (line 107)
                path_199385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 36), other_199384, 'path')
                # Applying the binary operator '==' (line 107)
                result_eq_199386 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 23), '==', path_199383, path_199385)
                
                # Assigning a type to the variable 'stypy_return_type' (line 107)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'stypy_return_type', result_eq_199386)
                
                # ################# End of '__eq__(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function '__eq__' in the type store
                # Getting the type of 'stypy_return_type' (line 106)
                stypy_return_type_199387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_199387)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '__eq__'
                return stypy_return_type_199387

            
            # Assigning a Name to a Name (line 110):
            # Getting the type of 'None' (line 110)
            None_199388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'None')
            # Assigning a type to the variable '__hash__' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), '__hash__', None_199388)
        
        # Assigning a type to the variable 'Module' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'Module', Module)
        
        # Assigning a Lambda to a Attribute (line 112):

        @norecursion
        def _stypy_temp_lambda_71(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_71'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_71', 112, 39, True)
            # Passed parameters checking function
            _stypy_temp_lambda_71.stypy_localization = localization
            _stypy_temp_lambda_71.stypy_type_of_self = None
            _stypy_temp_lambda_71.stypy_type_store = module_type_store
            _stypy_temp_lambda_71.stypy_function_name = '_stypy_temp_lambda_71'
            _stypy_temp_lambda_71.stypy_param_names_list = ['name']
            _stypy_temp_lambda_71.stypy_varargs_param_name = None
            _stypy_temp_lambda_71.stypy_kwargs_param_name = None
            _stypy_temp_lambda_71.stypy_call_defaults = defaults
            _stypy_temp_lambda_71.stypy_call_varargs = varargs
            _stypy_temp_lambda_71.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_71', ['name'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_71', ['name'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to Module(...): (line 112)
            # Processing the call arguments (line 112)
            # Getting the type of 'name' (line 112)
            name_199390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 59), 'name', False)
            # Processing the call keyword arguments (line 112)
            kwargs_199391 = {}
            # Getting the type of 'Module' (line 112)
            Module_199389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 52), 'Module', False)
            # Calling Module(args, kwargs) (line 112)
            Module_call_result_199392 = invoke(stypy.reporting.localization.Localization(__file__, 112, 52), Module_199389, *[name_199390], **kwargs_199391)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 39), 'stypy_return_type', Module_call_result_199392)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_71' in the type store
            # Getting the type of 'stypy_return_type' (line 112)
            stypy_return_type_199393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 39), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199393)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_71'
            return stypy_return_type_199393

        # Assigning a type to the variable '_stypy_temp_lambda_71' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 39), '_stypy_temp_lambda_71', _stypy_temp_lambda_71)
        # Getting the type of '_stypy_temp_lambda_71' (line 112)
        _stypy_temp_lambda_71_199394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 39), '_stypy_temp_lambda_71')
        # Getting the type of 'loader' (line 112)
        loader_199395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'loader')
        # Setting the type of the member '_get_module_from_name' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), loader_199395, '_get_module_from_name', _stypy_temp_lambda_71_199394)

        @norecursion
        def loadTestsFromModule(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'loadTestsFromModule'
            module_type_store = module_type_store.open_function_context('loadTestsFromModule', 113, 8, False)
            
            # Passed parameters checking function
            loadTestsFromModule.stypy_localization = localization
            loadTestsFromModule.stypy_type_of_self = None
            loadTestsFromModule.stypy_type_store = module_type_store
            loadTestsFromModule.stypy_function_name = 'loadTestsFromModule'
            loadTestsFromModule.stypy_param_names_list = ['module', 'use_load_tests']
            loadTestsFromModule.stypy_varargs_param_name = None
            loadTestsFromModule.stypy_kwargs_param_name = None
            loadTestsFromModule.stypy_call_defaults = defaults
            loadTestsFromModule.stypy_call_varargs = varargs
            loadTestsFromModule.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'loadTestsFromModule', ['module', 'use_load_tests'], None, None, defaults, varargs, kwargs)

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

            
            # Getting the type of 'use_load_tests' (line 114)
            use_load_tests_199396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'use_load_tests')
            # Testing the type of an if condition (line 114)
            if_condition_199397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 12), use_load_tests_199396)
            # Assigning a type to the variable 'if_condition_199397' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'if_condition_199397', if_condition_199397)
            # SSA begins for if statement (line 114)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to failureException(...): (line 115)
            # Processing the call arguments (line 115)
            str_199400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 44), 'str', 'use_load_tests should be False for packages')
            # Processing the call keyword arguments (line 115)
            kwargs_199401 = {}
            # Getting the type of 'self' (line 115)
            self_199398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'self', False)
            # Obtaining the member 'failureException' of a type (line 115)
            failureException_199399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 22), self_199398, 'failureException')
            # Calling failureException(args, kwargs) (line 115)
            failureException_call_result_199402 = invoke(stypy.reporting.localization.Localization(__file__, 115, 22), failureException_199399, *[str_199400], **kwargs_199401)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 115, 16), failureException_call_result_199402, 'raise parameter', BaseException)
            # SSA join for if statement (line 114)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'module' (line 116)
            module_199403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'module')
            # Obtaining the member 'path' of a type (line 116)
            path_199404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 19), module_199403, 'path')
            str_199405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 33), 'str', ' module tests')
            # Applying the binary operator '+' (line 116)
            result_add_199406 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 19), '+', path_199404, str_199405)
            
            # Assigning a type to the variable 'stypy_return_type' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'stypy_return_type', result_add_199406)
            
            # ################# End of 'loadTestsFromModule(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'loadTestsFromModule' in the type store
            # Getting the type of 'stypy_return_type' (line 113)
            stypy_return_type_199407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199407)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'loadTestsFromModule'
            return stypy_return_type_199407

        # Assigning a type to the variable 'loadTestsFromModule' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'loadTestsFromModule', loadTestsFromModule)
        
        # Assigning a Name to a Attribute (line 117):
        # Getting the type of 'loadTestsFromModule' (line 117)
        loadTestsFromModule_199408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 37), 'loadTestsFromModule')
        # Getting the type of 'loader' (line 117)
        loader_199409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'loader')
        # Setting the type of the member 'loadTestsFromModule' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), loader_199409, 'loadTestsFromModule', loadTestsFromModule_199408)
        
        # Assigning a Str to a Attribute (line 119):
        str_199410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'str', '/foo')
        # Getting the type of 'loader' (line 119)
        loader_199411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'loader')
        # Setting the type of the member '_top_level_dir' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), loader_199411, '_top_level_dir', str_199410)
        
        # Assigning a Call to a Name (line 122):
        
        # Call to list(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to _find_tests(...): (line 122)
        # Processing the call arguments (line 122)
        str_199415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 40), 'str', '/foo')
        str_199416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 48), 'str', 'test*')
        # Processing the call keyword arguments (line 122)
        kwargs_199417 = {}
        # Getting the type of 'loader' (line 122)
        loader_199413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'loader', False)
        # Obtaining the member '_find_tests' of a type (line 122)
        _find_tests_199414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 21), loader_199413, '_find_tests')
        # Calling _find_tests(args, kwargs) (line 122)
        _find_tests_call_result_199418 = invoke(stypy.reporting.localization.Localization(__file__, 122, 21), _find_tests_199414, *[str_199415, str_199416], **kwargs_199417)
        
        # Processing the call keyword arguments (line 122)
        kwargs_199419 = {}
        # Getting the type of 'list' (line 122)
        list_199412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'list', False)
        # Calling list(args, kwargs) (line 122)
        list_call_result_199420 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), list_199412, *[_find_tests_call_result_199418], **kwargs_199419)
        
        # Assigning a type to the variable 'suite' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'suite', list_call_result_199420)
        
        # Call to assertEqual(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'suite' (line 126)
        suite_199423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'suite', False)
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_199424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        str_199425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 26), 'str', 'load_tests')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 25), list_199424, str_199425)
        # Adding element type (line 127)
        str_199426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 40), 'str', 'test_directory2')
        str_199427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 60), 'str', ' module tests')
        # Applying the binary operator '+' (line 127)
        result_add_199428 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 40), '+', str_199426, str_199427)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 25), list_199424, result_add_199428)
        
        # Processing the call keyword arguments (line 126)
        kwargs_199429 = {}
        # Getting the type of 'self' (line 126)
        self_199421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 126)
        assertEqual_199422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), self_199421, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 126)
        assertEqual_call_result_199430 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), assertEqual_199422, *[suite_199423, list_199424], **kwargs_199429)
        
        
        # Call to assertEqual(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'Module' (line 128)
        Module_199433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'Module', False)
        # Obtaining the member 'paths' of a type (line 128)
        paths_199434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 25), Module_199433, 'paths')
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_199435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        # Adding element type (line 128)
        str_199436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 40), 'str', 'test_directory')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 39), list_199435, str_199436)
        # Adding element type (line 128)
        str_199437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 58), 'str', 'test_directory2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 39), list_199435, str_199437)
        
        # Processing the call keyword arguments (line 128)
        kwargs_199438 = {}
        # Getting the type of 'self' (line 128)
        self_199431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 128)
        assertEqual_199432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_199431, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 128)
        assertEqual_call_result_199439 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), assertEqual_199432, *[paths_199434, list_199435], **kwargs_199438)
        
        
        # Call to assertEqual(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'Module' (line 131)
        Module_199442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'Module', False)
        # Obtaining the member 'load_tests_args' of a type (line 131)
        load_tests_args_199443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 25), Module_199442, 'load_tests_args')
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_199444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        # Adding element type (line 132)
        
        # Obtaining an instance of the builtin type 'tuple' (line 132)
        tuple_199445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 132)
        # Adding element type (line 132)
        # Getting the type of 'loader' (line 132)
        loader_199446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'loader', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 27), tuple_199445, loader_199446)
        # Adding element type (line 132)
        str_199447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 35), 'str', 'test_directory')
        str_199448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 54), 'str', ' module tests')
        # Applying the binary operator '+' (line 132)
        result_add_199449 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 35), '+', str_199447, str_199448)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 27), tuple_199445, result_add_199449)
        # Adding element type (line 132)
        str_199450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 71), 'str', 'test*')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 27), tuple_199445, str_199450)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 25), list_199444, tuple_199445)
        
        # Processing the call keyword arguments (line 131)
        kwargs_199451 = {}
        # Getting the type of 'self' (line 131)
        self_199440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 131)
        assertEqual_199441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_199440, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 131)
        assertEqual_call_result_199452 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assertEqual_199441, *[load_tests_args_199443, list_199444], **kwargs_199451)
        
        
        # ################# End of 'test_find_tests_with_package(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_find_tests_with_package' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_199453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_199453)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_find_tests_with_package'
        return stypy_return_type_199453


    @norecursion
    def test_discover(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_discover'
        module_type_store = module_type_store.open_function_context('test_discover', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_discover')
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_discover.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_discover', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_discover', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_discover(...)' code ##################

        
        # Assigning a Call to a Name (line 135):
        
        # Call to TestLoader(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_199456 = {}
        # Getting the type of 'unittest' (line 135)
        unittest_199454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'unittest', False)
        # Obtaining the member 'TestLoader' of a type (line 135)
        TestLoader_199455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), unittest_199454, 'TestLoader')
        # Calling TestLoader(args, kwargs) (line 135)
        TestLoader_call_result_199457 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), TestLoader_199455, *[], **kwargs_199456)
        
        # Assigning a type to the variable 'loader' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'loader', TestLoader_call_result_199457)
        
        # Assigning a Attribute to a Name (line 137):
        # Getting the type of 'os' (line 137)
        os_199458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'os')
        # Obtaining the member 'path' of a type (line 137)
        path_199459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 26), os_199458, 'path')
        # Obtaining the member 'isfile' of a type (line 137)
        isfile_199460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 26), path_199459, 'isfile')
        # Assigning a type to the variable 'original_isfile' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'original_isfile', isfile_199460)
        
        # Assigning a Attribute to a Name (line 138):
        # Getting the type of 'os' (line 138)
        os_199461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'os')
        # Obtaining the member 'path' of a type (line 138)
        path_199462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 25), os_199461, 'path')
        # Obtaining the member 'isdir' of a type (line 138)
        isdir_199463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 25), path_199462, 'isdir')
        # Assigning a type to the variable 'original_isdir' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'original_isdir', isdir_199463)

        @norecursion
        def restore_isfile(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore_isfile'
            module_type_store = module_type_store.open_function_context('restore_isfile', 139, 8, False)
            
            # Passed parameters checking function
            restore_isfile.stypy_localization = localization
            restore_isfile.stypy_type_of_self = None
            restore_isfile.stypy_type_store = module_type_store
            restore_isfile.stypy_function_name = 'restore_isfile'
            restore_isfile.stypy_param_names_list = []
            restore_isfile.stypy_varargs_param_name = None
            restore_isfile.stypy_kwargs_param_name = None
            restore_isfile.stypy_call_defaults = defaults
            restore_isfile.stypy_call_varargs = varargs
            restore_isfile.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore_isfile', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore_isfile', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore_isfile(...)' code ##################

            
            # Assigning a Name to a Attribute (line 140):
            # Getting the type of 'original_isfile' (line 140)
            original_isfile_199464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'original_isfile')
            # Getting the type of 'os' (line 140)
            os_199465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'os')
            # Obtaining the member 'path' of a type (line 140)
            path_199466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), os_199465, 'path')
            # Setting the type of the member 'isfile' of a type (line 140)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), path_199466, 'isfile', original_isfile_199464)
            
            # ################# End of 'restore_isfile(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore_isfile' in the type store
            # Getting the type of 'stypy_return_type' (line 139)
            stypy_return_type_199467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199467)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore_isfile'
            return stypy_return_type_199467

        # Assigning a type to the variable 'restore_isfile' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'restore_isfile', restore_isfile)
        
        # Assigning a Lambda to a Attribute (line 142):

        @norecursion
        def _stypy_temp_lambda_72(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_72'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_72', 142, 25, True)
            # Passed parameters checking function
            _stypy_temp_lambda_72.stypy_localization = localization
            _stypy_temp_lambda_72.stypy_type_of_self = None
            _stypy_temp_lambda_72.stypy_type_store = module_type_store
            _stypy_temp_lambda_72.stypy_function_name = '_stypy_temp_lambda_72'
            _stypy_temp_lambda_72.stypy_param_names_list = ['path']
            _stypy_temp_lambda_72.stypy_varargs_param_name = None
            _stypy_temp_lambda_72.stypy_kwargs_param_name = None
            _stypy_temp_lambda_72.stypy_call_defaults = defaults
            _stypy_temp_lambda_72.stypy_call_varargs = varargs
            _stypy_temp_lambda_72.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_72', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_72', ['path'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'False' (line 142)
            False_199468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), 'False')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'stypy_return_type', False_199468)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_72' in the type store
            # Getting the type of 'stypy_return_type' (line 142)
            stypy_return_type_199469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199469)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_72'
            return stypy_return_type_199469

        # Assigning a type to the variable '_stypy_temp_lambda_72' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), '_stypy_temp_lambda_72', _stypy_temp_lambda_72)
        # Getting the type of '_stypy_temp_lambda_72' (line 142)
        _stypy_temp_lambda_72_199470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), '_stypy_temp_lambda_72')
        # Getting the type of 'os' (line 142)
        os_199471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'os')
        # Obtaining the member 'path' of a type (line 142)
        path_199472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), os_199471, 'path')
        # Setting the type of the member 'isfile' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), path_199472, 'isfile', _stypy_temp_lambda_72_199470)
        
        # Call to addCleanup(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'restore_isfile' (line 143)
        restore_isfile_199475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'restore_isfile', False)
        # Processing the call keyword arguments (line 143)
        kwargs_199476 = {}
        # Getting the type of 'self' (line 143)
        self_199473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 143)
        addCleanup_199474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_199473, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 143)
        addCleanup_call_result_199477 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), addCleanup_199474, *[restore_isfile_199475], **kwargs_199476)
        
        
        # Assigning a Subscript to a Name (line 145):
        
        # Obtaining the type of the subscript
        slice_199478 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 145, 24), None, None, None)
        # Getting the type of 'sys' (line 145)
        sys_199479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'sys')
        # Obtaining the member 'path' of a type (line 145)
        path_199480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 24), sys_199479, 'path')
        # Obtaining the member '__getitem__' of a type (line 145)
        getitem___199481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 24), path_199480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 145)
        subscript_call_result_199482 = invoke(stypy.reporting.localization.Localization(__file__, 145, 24), getitem___199481, slice_199478)
        
        # Assigning a type to the variable 'orig_sys_path' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'orig_sys_path', subscript_call_result_199482)

        @norecursion
        def restore_path(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore_path'
            module_type_store = module_type_store.open_function_context('restore_path', 146, 8, False)
            
            # Passed parameters checking function
            restore_path.stypy_localization = localization
            restore_path.stypy_type_of_self = None
            restore_path.stypy_type_store = module_type_store
            restore_path.stypy_function_name = 'restore_path'
            restore_path.stypy_param_names_list = []
            restore_path.stypy_varargs_param_name = None
            restore_path.stypy_kwargs_param_name = None
            restore_path.stypy_call_defaults = defaults
            restore_path.stypy_call_varargs = varargs
            restore_path.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore_path', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore_path', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore_path(...)' code ##################

            
            # Assigning a Name to a Subscript (line 147):
            # Getting the type of 'orig_sys_path' (line 147)
            orig_sys_path_199483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'orig_sys_path')
            # Getting the type of 'sys' (line 147)
            sys_199484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'sys')
            # Obtaining the member 'path' of a type (line 147)
            path_199485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), sys_199484, 'path')
            slice_199486 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 147, 12), None, None, None)
            # Storing an element on a container (line 147)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 12), path_199485, (slice_199486, orig_sys_path_199483))
            
            # ################# End of 'restore_path(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore_path' in the type store
            # Getting the type of 'stypy_return_type' (line 146)
            stypy_return_type_199487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199487)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore_path'
            return stypy_return_type_199487

        # Assigning a type to the variable 'restore_path' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'restore_path', restore_path)
        
        # Call to addCleanup(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'restore_path' (line 148)
        restore_path_199490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'restore_path', False)
        # Processing the call keyword arguments (line 148)
        kwargs_199491 = {}
        # Getting the type of 'self' (line 148)
        self_199488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 148)
        addCleanup_199489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_199488, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 148)
        addCleanup_call_result_199492 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), addCleanup_199489, *[restore_path_199490], **kwargs_199491)
        
        
        # Assigning a Call to a Name (line 150):
        
        # Call to abspath(...): (line 150)
        # Processing the call arguments (line 150)
        
        # Call to normpath(...): (line 150)
        # Processing the call arguments (line 150)
        str_199499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 53), 'str', '/foo')
        # Processing the call keyword arguments (line 150)
        kwargs_199500 = {}
        # Getting the type of 'os' (line 150)
        os_199496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 36), 'os', False)
        # Obtaining the member 'path' of a type (line 150)
        path_199497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 36), os_199496, 'path')
        # Obtaining the member 'normpath' of a type (line 150)
        normpath_199498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 36), path_199497, 'normpath')
        # Calling normpath(args, kwargs) (line 150)
        normpath_call_result_199501 = invoke(stypy.reporting.localization.Localization(__file__, 150, 36), normpath_199498, *[str_199499], **kwargs_199500)
        
        # Processing the call keyword arguments (line 150)
        kwargs_199502 = {}
        # Getting the type of 'os' (line 150)
        os_199493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 150)
        path_199494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 20), os_199493, 'path')
        # Obtaining the member 'abspath' of a type (line 150)
        abspath_199495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 20), path_199494, 'abspath')
        # Calling abspath(args, kwargs) (line 150)
        abspath_call_result_199503 = invoke(stypy.reporting.localization.Localization(__file__, 150, 20), abspath_199495, *[normpath_call_result_199501], **kwargs_199502)
        
        # Assigning a type to the variable 'full_path' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'full_path', abspath_call_result_199503)
        
        # Call to assertRaises(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'ImportError' (line 151)
        ImportError_199506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'ImportError', False)
        # Processing the call keyword arguments (line 151)
        kwargs_199507 = {}
        # Getting the type of 'self' (line 151)
        self_199504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 151)
        assertRaises_199505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 13), self_199504, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 151)
        assertRaises_call_result_199508 = invoke(stypy.reporting.localization.Localization(__file__, 151, 13), assertRaises_199505, *[ImportError_199506], **kwargs_199507)
        
        with_199509 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 151, 13), assertRaises_call_result_199508, 'with parameter', '__enter__', '__exit__')

        if with_199509:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 151)
            enter___199510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 13), assertRaises_call_result_199508, '__enter__')
            with_enter_199511 = invoke(stypy.reporting.localization.Localization(__file__, 151, 13), enter___199510)
            
            # Call to discover(...): (line 152)
            # Processing the call arguments (line 152)
            str_199514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'str', '/foo/bar')
            # Processing the call keyword arguments (line 152)
            str_199515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 54), 'str', '/foo')
            keyword_199516 = str_199515
            kwargs_199517 = {'top_level_dir': keyword_199516}
            # Getting the type of 'loader' (line 152)
            loader_199512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'loader', False)
            # Obtaining the member 'discover' of a type (line 152)
            discover_199513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), loader_199512, 'discover')
            # Calling discover(args, kwargs) (line 152)
            discover_call_result_199518 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), discover_199513, *[str_199514], **kwargs_199517)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 151)
            exit___199519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 13), assertRaises_call_result_199508, '__exit__')
            with_exit_199520 = invoke(stypy.reporting.localization.Localization(__file__, 151, 13), exit___199519, None, None, None)

        
        # Call to assertEqual(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'loader' (line 154)
        loader_199523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'loader', False)
        # Obtaining the member '_top_level_dir' of a type (line 154)
        _top_level_dir_199524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 25), loader_199523, '_top_level_dir')
        # Getting the type of 'full_path' (line 154)
        full_path_199525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 48), 'full_path', False)
        # Processing the call keyword arguments (line 154)
        kwargs_199526 = {}
        # Getting the type of 'self' (line 154)
        self_199521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 154)
        assertEqual_199522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_199521, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 154)
        assertEqual_call_result_199527 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), assertEqual_199522, *[_top_level_dir_199524, full_path_199525], **kwargs_199526)
        
        
        # Call to assertIn(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'full_path' (line 155)
        full_path_199530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'full_path', False)
        # Getting the type of 'sys' (line 155)
        sys_199531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 33), 'sys', False)
        # Obtaining the member 'path' of a type (line 155)
        path_199532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 33), sys_199531, 'path')
        # Processing the call keyword arguments (line 155)
        kwargs_199533 = {}
        # Getting the type of 'self' (line 155)
        self_199528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 155)
        assertIn_199529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_199528, 'assertIn')
        # Calling assertIn(args, kwargs) (line 155)
        assertIn_call_result_199534 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), assertIn_199529, *[full_path_199530, path_199532], **kwargs_199533)
        
        
        # Assigning a Lambda to a Attribute (line 157):

        @norecursion
        def _stypy_temp_lambda_73(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_73'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_73', 157, 25, True)
            # Passed parameters checking function
            _stypy_temp_lambda_73.stypy_localization = localization
            _stypy_temp_lambda_73.stypy_type_of_self = None
            _stypy_temp_lambda_73.stypy_type_store = module_type_store
            _stypy_temp_lambda_73.stypy_function_name = '_stypy_temp_lambda_73'
            _stypy_temp_lambda_73.stypy_param_names_list = ['path']
            _stypy_temp_lambda_73.stypy_varargs_param_name = None
            _stypy_temp_lambda_73.stypy_kwargs_param_name = None
            _stypy_temp_lambda_73.stypy_call_defaults = defaults
            _stypy_temp_lambda_73.stypy_call_varargs = varargs
            _stypy_temp_lambda_73.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_73', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_73', ['path'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'True' (line 157)
            True_199535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 38), 'True')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), 'stypy_return_type', True_199535)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_73' in the type store
            # Getting the type of 'stypy_return_type' (line 157)
            stypy_return_type_199536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199536)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_73'
            return stypy_return_type_199536

        # Assigning a type to the variable '_stypy_temp_lambda_73' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), '_stypy_temp_lambda_73', _stypy_temp_lambda_73)
        # Getting the type of '_stypy_temp_lambda_73' (line 157)
        _stypy_temp_lambda_73_199537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), '_stypy_temp_lambda_73')
        # Getting the type of 'os' (line 157)
        os_199538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'os')
        # Obtaining the member 'path' of a type (line 157)
        path_199539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), os_199538, 'path')
        # Setting the type of the member 'isfile' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), path_199539, 'isfile', _stypy_temp_lambda_73_199537)
        
        # Assigning a Lambda to a Attribute (line 158):

        @norecursion
        def _stypy_temp_lambda_74(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_74'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_74', 158, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_74.stypy_localization = localization
            _stypy_temp_lambda_74.stypy_type_of_self = None
            _stypy_temp_lambda_74.stypy_type_store = module_type_store
            _stypy_temp_lambda_74.stypy_function_name = '_stypy_temp_lambda_74'
            _stypy_temp_lambda_74.stypy_param_names_list = ['path']
            _stypy_temp_lambda_74.stypy_varargs_param_name = None
            _stypy_temp_lambda_74.stypy_kwargs_param_name = None
            _stypy_temp_lambda_74.stypy_call_defaults = defaults
            _stypy_temp_lambda_74.stypy_call_varargs = varargs
            _stypy_temp_lambda_74.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_74', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_74', ['path'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'True' (line 158)
            True_199540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 37), 'True')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'stypy_return_type', True_199540)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_74' in the type store
            # Getting the type of 'stypy_return_type' (line 158)
            stypy_return_type_199541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199541)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_74'
            return stypy_return_type_199541

        # Assigning a type to the variable '_stypy_temp_lambda_74' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), '_stypy_temp_lambda_74', _stypy_temp_lambda_74)
        # Getting the type of '_stypy_temp_lambda_74' (line 158)
        _stypy_temp_lambda_74_199542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), '_stypy_temp_lambda_74')
        # Getting the type of 'os' (line 158)
        os_199543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'os')
        # Obtaining the member 'path' of a type (line 158)
        path_199544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), os_199543, 'path')
        # Setting the type of the member 'isdir' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), path_199544, 'isdir', _stypy_temp_lambda_74_199542)

        @norecursion
        def restore_isdir(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore_isdir'
            module_type_store = module_type_store.open_function_context('restore_isdir', 160, 8, False)
            
            # Passed parameters checking function
            restore_isdir.stypy_localization = localization
            restore_isdir.stypy_type_of_self = None
            restore_isdir.stypy_type_store = module_type_store
            restore_isdir.stypy_function_name = 'restore_isdir'
            restore_isdir.stypy_param_names_list = []
            restore_isdir.stypy_varargs_param_name = None
            restore_isdir.stypy_kwargs_param_name = None
            restore_isdir.stypy_call_defaults = defaults
            restore_isdir.stypy_call_varargs = varargs
            restore_isdir.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'restore_isdir', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'restore_isdir', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'restore_isdir(...)' code ##################

            
            # Assigning a Name to a Attribute (line 161):
            # Getting the type of 'original_isdir' (line 161)
            original_isdir_199545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'original_isdir')
            # Getting the type of 'os' (line 161)
            os_199546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'os')
            # Obtaining the member 'path' of a type (line 161)
            path_199547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), os_199546, 'path')
            # Setting the type of the member 'isdir' of a type (line 161)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), path_199547, 'isdir', original_isdir_199545)
            
            # ################# End of 'restore_isdir(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore_isdir' in the type store
            # Getting the type of 'stypy_return_type' (line 160)
            stypy_return_type_199548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199548)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore_isdir'
            return stypy_return_type_199548

        # Assigning a type to the variable 'restore_isdir' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'restore_isdir', restore_isdir)
        
        # Call to addCleanup(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'restore_isdir' (line 162)
        restore_isdir_199551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'restore_isdir', False)
        # Processing the call keyword arguments (line 162)
        kwargs_199552 = {}
        # Getting the type of 'self' (line 162)
        self_199549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 162)
        addCleanup_199550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_199549, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 162)
        addCleanup_call_result_199553 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), addCleanup_199550, *[restore_isdir_199551], **kwargs_199552)
        
        
        # Assigning a List to a Name (line 164):
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_199554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        
        # Assigning a type to the variable '_find_tests_args' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), '_find_tests_args', list_199554)

        @norecursion
        def _find_tests(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_find_tests'
            module_type_store = module_type_store.open_function_context('_find_tests', 165, 8, False)
            
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

            
            # Call to append(...): (line 166)
            # Processing the call arguments (line 166)
            
            # Obtaining an instance of the builtin type 'tuple' (line 166)
            tuple_199557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 37), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 166)
            # Adding element type (line 166)
            # Getting the type of 'start_dir' (line 166)
            start_dir_199558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'start_dir', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 37), tuple_199557, start_dir_199558)
            # Adding element type (line 166)
            # Getting the type of 'pattern' (line 166)
            pattern_199559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 48), 'pattern', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 37), tuple_199557, pattern_199559)
            
            # Processing the call keyword arguments (line 166)
            kwargs_199560 = {}
            # Getting the type of '_find_tests_args' (line 166)
            _find_tests_args_199555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), '_find_tests_args', False)
            # Obtaining the member 'append' of a type (line 166)
            append_199556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), _find_tests_args_199555, 'append')
            # Calling append(args, kwargs) (line 166)
            append_call_result_199561 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), append_199556, *[tuple_199557], **kwargs_199560)
            
            
            # Obtaining an instance of the builtin type 'list' (line 167)
            list_199562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 167)
            # Adding element type (line 167)
            str_199563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'str', 'tests')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 19), list_199562, str_199563)
            
            # Assigning a type to the variable 'stypy_return_type' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'stypy_return_type', list_199562)
            
            # ################# End of '_find_tests(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_find_tests' in the type store
            # Getting the type of 'stypy_return_type' (line 165)
            stypy_return_type_199564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199564)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_find_tests'
            return stypy_return_type_199564

        # Assigning a type to the variable '_find_tests' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), '_find_tests', _find_tests)
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of '_find_tests' (line 168)
        _find_tests_199565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 29), '_find_tests')
        # Getting the type of 'loader' (line 168)
        loader_199566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'loader')
        # Setting the type of the member '_find_tests' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), loader_199566, '_find_tests', _find_tests_199565)
        
        # Assigning a Name to a Attribute (line 169):
        # Getting the type of 'str' (line 169)
        str_199567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'str')
        # Getting the type of 'loader' (line 169)
        loader_199568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'loader')
        # Setting the type of the member 'suiteClass' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), loader_199568, 'suiteClass', str_199567)
        
        # Assigning a Call to a Name (line 171):
        
        # Call to discover(...): (line 171)
        # Processing the call arguments (line 171)
        str_199571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 32), 'str', '/foo/bar/baz')
        str_199572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 48), 'str', 'pattern')
        str_199573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 59), 'str', '/foo/bar')
        # Processing the call keyword arguments (line 171)
        kwargs_199574 = {}
        # Getting the type of 'loader' (line 171)
        loader_199569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'loader', False)
        # Obtaining the member 'discover' of a type (line 171)
        discover_199570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), loader_199569, 'discover')
        # Calling discover(args, kwargs) (line 171)
        discover_call_result_199575 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), discover_199570, *[str_199571, str_199572, str_199573], **kwargs_199574)
        
        # Assigning a type to the variable 'suite' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'suite', discover_call_result_199575)
        
        # Assigning a Call to a Name (line 173):
        
        # Call to abspath(...): (line 173)
        # Processing the call arguments (line 173)
        str_199579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 40), 'str', '/foo/bar')
        # Processing the call keyword arguments (line 173)
        kwargs_199580 = {}
        # Getting the type of 'os' (line 173)
        os_199576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 173)
        path_199577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 24), os_199576, 'path')
        # Obtaining the member 'abspath' of a type (line 173)
        abspath_199578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 24), path_199577, 'abspath')
        # Calling abspath(args, kwargs) (line 173)
        abspath_call_result_199581 = invoke(stypy.reporting.localization.Localization(__file__, 173, 24), abspath_199578, *[str_199579], **kwargs_199580)
        
        # Assigning a type to the variable 'top_level_dir' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'top_level_dir', abspath_call_result_199581)
        
        # Assigning a Call to a Name (line 174):
        
        # Call to abspath(...): (line 174)
        # Processing the call arguments (line 174)
        str_199585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 36), 'str', '/foo/bar/baz')
        # Processing the call keyword arguments (line 174)
        kwargs_199586 = {}
        # Getting the type of 'os' (line 174)
        os_199582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 174)
        path_199583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 20), os_199582, 'path')
        # Obtaining the member 'abspath' of a type (line 174)
        abspath_199584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 20), path_199583, 'abspath')
        # Calling abspath(args, kwargs) (line 174)
        abspath_call_result_199587 = invoke(stypy.reporting.localization.Localization(__file__, 174, 20), abspath_199584, *[str_199585], **kwargs_199586)
        
        # Assigning a type to the variable 'start_dir' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'start_dir', abspath_call_result_199587)
        
        # Call to assertEqual(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'suite' (line 175)
        suite_199590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'suite', False)
        str_199591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 32), 'str', "['tests']")
        # Processing the call keyword arguments (line 175)
        kwargs_199592 = {}
        # Getting the type of 'self' (line 175)
        self_199588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 175)
        assertEqual_199589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_199588, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 175)
        assertEqual_call_result_199593 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), assertEqual_199589, *[suite_199590, str_199591], **kwargs_199592)
        
        
        # Call to assertEqual(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'loader' (line 176)
        loader_199596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'loader', False)
        # Obtaining the member '_top_level_dir' of a type (line 176)
        _top_level_dir_199597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), loader_199596, '_top_level_dir')
        # Getting the type of 'top_level_dir' (line 176)
        top_level_dir_199598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 48), 'top_level_dir', False)
        # Processing the call keyword arguments (line 176)
        kwargs_199599 = {}
        # Getting the type of 'self' (line 176)
        self_199594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 176)
        assertEqual_199595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_199594, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 176)
        assertEqual_call_result_199600 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assertEqual_199595, *[_top_level_dir_199597, top_level_dir_199598], **kwargs_199599)
        
        
        # Call to assertEqual(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of '_find_tests_args' (line 177)
        _find_tests_args_199603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), '_find_tests_args', False)
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_199604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'tuple' (line 177)
        tuple_199605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 177)
        # Adding element type (line 177)
        # Getting the type of 'start_dir' (line 177)
        start_dir_199606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 45), 'start_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 45), tuple_199605, start_dir_199606)
        # Adding element type (line 177)
        str_199607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 56), 'str', 'pattern')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 45), tuple_199605, str_199607)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 43), list_199604, tuple_199605)
        
        # Processing the call keyword arguments (line 177)
        kwargs_199608 = {}
        # Getting the type of 'self' (line 177)
        self_199601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 177)
        assertEqual_199602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_199601, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 177)
        assertEqual_call_result_199609 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assertEqual_199602, *[_find_tests_args_199603, list_199604], **kwargs_199608)
        
        
        # Call to assertIn(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'top_level_dir' (line 178)
        top_level_dir_199612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'top_level_dir', False)
        # Getting the type of 'sys' (line 178)
        sys_199613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 37), 'sys', False)
        # Obtaining the member 'path' of a type (line 178)
        path_199614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 37), sys_199613, 'path')
        # Processing the call keyword arguments (line 178)
        kwargs_199615 = {}
        # Getting the type of 'self' (line 178)
        self_199610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 178)
        assertIn_199611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_199610, 'assertIn')
        # Calling assertIn(args, kwargs) (line 178)
        assertIn_call_result_199616 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), assertIn_199611, *[top_level_dir_199612, path_199614], **kwargs_199615)
        
        
        # ################# End of 'test_discover(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_discover' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_199617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_199617)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_discover'
        return stypy_return_type_199617


    @norecursion
    def test_discover_with_modules_that_fail_to_import(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_discover_with_modules_that_fail_to_import'
        module_type_store = module_type_store.open_function_context('test_discover_with_modules_that_fail_to_import', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_discover_with_modules_that_fail_to_import')
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_discover_with_modules_that_fail_to_import.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_discover_with_modules_that_fail_to_import', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_discover_with_modules_that_fail_to_import', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_discover_with_modules_that_fail_to_import(...)' code ##################

        
        # Assigning a Call to a Name (line 181):
        
        # Call to TestLoader(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_199620 = {}
        # Getting the type of 'unittest' (line 181)
        unittest_199618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 17), 'unittest', False)
        # Obtaining the member 'TestLoader' of a type (line 181)
        TestLoader_199619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 17), unittest_199618, 'TestLoader')
        # Calling TestLoader(args, kwargs) (line 181)
        TestLoader_call_result_199621 = invoke(stypy.reporting.localization.Localization(__file__, 181, 17), TestLoader_199619, *[], **kwargs_199620)
        
        # Assigning a type to the variable 'loader' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'loader', TestLoader_call_result_199621)
        
        # Assigning a Attribute to a Name (line 183):
        # Getting the type of 'os' (line 183)
        os_199622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 18), 'os')
        # Obtaining the member 'listdir' of a type (line 183)
        listdir_199623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 18), os_199622, 'listdir')
        # Assigning a type to the variable 'listdir' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'listdir', listdir_199623)
        
        # Assigning a Lambda to a Attribute (line 184):

        @norecursion
        def _stypy_temp_lambda_75(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_75'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_75', 184, 21, True)
            # Passed parameters checking function
            _stypy_temp_lambda_75.stypy_localization = localization
            _stypy_temp_lambda_75.stypy_type_of_self = None
            _stypy_temp_lambda_75.stypy_type_store = module_type_store
            _stypy_temp_lambda_75.stypy_function_name = '_stypy_temp_lambda_75'
            _stypy_temp_lambda_75.stypy_param_names_list = ['_']
            _stypy_temp_lambda_75.stypy_varargs_param_name = None
            _stypy_temp_lambda_75.stypy_kwargs_param_name = None
            _stypy_temp_lambda_75.stypy_call_defaults = defaults
            _stypy_temp_lambda_75.stypy_call_varargs = varargs
            _stypy_temp_lambda_75.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_75', ['_'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_75', ['_'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 184)
            list_199624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 184)
            # Adding element type (line 184)
            str_199625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 32), 'str', 'test_this_does_not_exist.py')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 31), list_199624, str_199625)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'stypy_return_type', list_199624)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_75' in the type store
            # Getting the type of 'stypy_return_type' (line 184)
            stypy_return_type_199626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199626)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_75'
            return stypy_return_type_199626

        # Assigning a type to the variable '_stypy_temp_lambda_75' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), '_stypy_temp_lambda_75', _stypy_temp_lambda_75)
        # Getting the type of '_stypy_temp_lambda_75' (line 184)
        _stypy_temp_lambda_75_199627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), '_stypy_temp_lambda_75')
        # Getting the type of 'os' (line 184)
        os_199628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'os')
        # Setting the type of the member 'listdir' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), os_199628, 'listdir', _stypy_temp_lambda_75_199627)
        
        # Assigning a Attribute to a Name (line 185):
        # Getting the type of 'os' (line 185)
        os_199629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'os')
        # Obtaining the member 'path' of a type (line 185)
        path_199630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 17), os_199629, 'path')
        # Obtaining the member 'isfile' of a type (line 185)
        isfile_199631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 17), path_199630, 'isfile')
        # Assigning a type to the variable 'isfile' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'isfile', isfile_199631)
        
        # Assigning a Lambda to a Attribute (line 186):

        @norecursion
        def _stypy_temp_lambda_76(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_76'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_76', 186, 25, True)
            # Passed parameters checking function
            _stypy_temp_lambda_76.stypy_localization = localization
            _stypy_temp_lambda_76.stypy_type_of_self = None
            _stypy_temp_lambda_76.stypy_type_store = module_type_store
            _stypy_temp_lambda_76.stypy_function_name = '_stypy_temp_lambda_76'
            _stypy_temp_lambda_76.stypy_param_names_list = ['_']
            _stypy_temp_lambda_76.stypy_varargs_param_name = None
            _stypy_temp_lambda_76.stypy_kwargs_param_name = None
            _stypy_temp_lambda_76.stypy_call_defaults = defaults
            _stypy_temp_lambda_76.stypy_call_varargs = varargs
            _stypy_temp_lambda_76.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_76', ['_'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_76', ['_'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'True' (line 186)
            True_199632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 35), 'True')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'stypy_return_type', True_199632)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_76' in the type store
            # Getting the type of 'stypy_return_type' (line 186)
            stypy_return_type_199633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199633)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_76'
            return stypy_return_type_199633

        # Assigning a type to the variable '_stypy_temp_lambda_76' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), '_stypy_temp_lambda_76', _stypy_temp_lambda_76)
        # Getting the type of '_stypy_temp_lambda_76' (line 186)
        _stypy_temp_lambda_76_199634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), '_stypy_temp_lambda_76')
        # Getting the type of 'os' (line 186)
        os_199635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'os')
        # Obtaining the member 'path' of a type (line 186)
        path_199636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), os_199635, 'path')
        # Setting the type of the member 'isfile' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), path_199636, 'isfile', _stypy_temp_lambda_76_199634)
        
        # Assigning a Subscript to a Name (line 187):
        
        # Obtaining the type of the subscript
        slice_199637 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 24), None, None, None)
        # Getting the type of 'sys' (line 187)
        sys_199638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'sys')
        # Obtaining the member 'path' of a type (line 187)
        path_199639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 24), sys_199638, 'path')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___199640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 24), path_199639, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_199641 = invoke(stypy.reporting.localization.Localization(__file__, 187, 24), getitem___199640, slice_199637)
        
        # Assigning a type to the variable 'orig_sys_path' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'orig_sys_path', subscript_call_result_199641)

        @norecursion
        def restore(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'restore'
            module_type_store = module_type_store.open_function_context('restore', 188, 8, False)
            
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

            
            # Assigning a Name to a Attribute (line 189):
            # Getting the type of 'isfile' (line 189)
            isfile_199642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'isfile')
            # Getting the type of 'os' (line 189)
            os_199643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'os')
            # Obtaining the member 'path' of a type (line 189)
            path_199644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), os_199643, 'path')
            # Setting the type of the member 'isfile' of a type (line 189)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), path_199644, 'isfile', isfile_199642)
            
            # Assigning a Name to a Attribute (line 190):
            # Getting the type of 'listdir' (line 190)
            listdir_199645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'listdir')
            # Getting the type of 'os' (line 190)
            os_199646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'os')
            # Setting the type of the member 'listdir' of a type (line 190)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 12), os_199646, 'listdir', listdir_199645)
            
            # Assigning a Name to a Subscript (line 191):
            # Getting the type of 'orig_sys_path' (line 191)
            orig_sys_path_199647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 26), 'orig_sys_path')
            # Getting the type of 'sys' (line 191)
            sys_199648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'sys')
            # Obtaining the member 'path' of a type (line 191)
            path_199649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), sys_199648, 'path')
            slice_199650 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 191, 12), None, None, None)
            # Storing an element on a container (line 191)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), path_199649, (slice_199650, orig_sys_path_199647))
            
            # ################# End of 'restore(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'restore' in the type store
            # Getting the type of 'stypy_return_type' (line 188)
            stypy_return_type_199651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199651)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'restore'
            return stypy_return_type_199651

        # Assigning a type to the variable 'restore' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'restore', restore)
        
        # Call to addCleanup(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'restore' (line 192)
        restore_199654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'restore', False)
        # Processing the call keyword arguments (line 192)
        kwargs_199655 = {}
        # Getting the type of 'self' (line 192)
        self_199652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 192)
        addCleanup_199653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_199652, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 192)
        addCleanup_call_result_199656 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), addCleanup_199653, *[restore_199654], **kwargs_199655)
        
        
        # Assigning a Call to a Name (line 194):
        
        # Call to discover(...): (line 194)
        # Processing the call arguments (line 194)
        str_199659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 32), 'str', '.')
        # Processing the call keyword arguments (line 194)
        kwargs_199660 = {}
        # Getting the type of 'loader' (line 194)
        loader_199657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'loader', False)
        # Obtaining the member 'discover' of a type (line 194)
        discover_199658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 16), loader_199657, 'discover')
        # Calling discover(args, kwargs) (line 194)
        discover_call_result_199661 = invoke(stypy.reporting.localization.Localization(__file__, 194, 16), discover_199658, *[str_199659], **kwargs_199660)
        
        # Assigning a type to the variable 'suite' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'suite', discover_call_result_199661)
        
        # Call to assertIn(...): (line 195)
        # Processing the call arguments (line 195)
        
        # Call to getcwd(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_199666 = {}
        # Getting the type of 'os' (line 195)
        os_199664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 195)
        getcwd_199665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 22), os_199664, 'getcwd')
        # Calling getcwd(args, kwargs) (line 195)
        getcwd_call_result_199667 = invoke(stypy.reporting.localization.Localization(__file__, 195, 22), getcwd_199665, *[], **kwargs_199666)
        
        # Getting the type of 'sys' (line 195)
        sys_199668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 35), 'sys', False)
        # Obtaining the member 'path' of a type (line 195)
        path_199669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 35), sys_199668, 'path')
        # Processing the call keyword arguments (line 195)
        kwargs_199670 = {}
        # Getting the type of 'self' (line 195)
        self_199662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 195)
        assertIn_199663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), self_199662, 'assertIn')
        # Calling assertIn(args, kwargs) (line 195)
        assertIn_call_result_199671 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), assertIn_199663, *[getcwd_call_result_199667, path_199669], **kwargs_199670)
        
        
        # Call to assertEqual(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Call to countTestCases(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_199676 = {}
        # Getting the type of 'suite' (line 196)
        suite_199674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 25), 'suite', False)
        # Obtaining the member 'countTestCases' of a type (line 196)
        countTestCases_199675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 25), suite_199674, 'countTestCases')
        # Calling countTestCases(args, kwargs) (line 196)
        countTestCases_call_result_199677 = invoke(stypy.reporting.localization.Localization(__file__, 196, 25), countTestCases_199675, *[], **kwargs_199676)
        
        int_199678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 49), 'int')
        # Processing the call keyword arguments (line 196)
        kwargs_199679 = {}
        # Getting the type of 'self' (line 196)
        self_199672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 196)
        assertEqual_199673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_199672, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 196)
        assertEqual_call_result_199680 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), assertEqual_199673, *[countTestCases_call_result_199677, int_199678], **kwargs_199679)
        
        
        # Assigning a Subscript to a Name (line 197):
        
        # Obtaining the type of the subscript
        int_199681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 36), 'int')
        
        # Call to list(...): (line 197)
        # Processing the call arguments (line 197)
        
        # Obtaining the type of the subscript
        int_199683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 32), 'int')
        
        # Call to list(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'suite' (line 197)
        suite_199685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 25), 'suite', False)
        # Processing the call keyword arguments (line 197)
        kwargs_199686 = {}
        # Getting the type of 'list' (line 197)
        list_199684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'list', False)
        # Calling list(args, kwargs) (line 197)
        list_call_result_199687 = invoke(stypy.reporting.localization.Localization(__file__, 197, 20), list_199684, *[suite_199685], **kwargs_199686)
        
        # Obtaining the member '__getitem__' of a type (line 197)
        getitem___199688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 20), list_call_result_199687, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 197)
        subscript_call_result_199689 = invoke(stypy.reporting.localization.Localization(__file__, 197, 20), getitem___199688, int_199683)
        
        # Processing the call keyword arguments (line 197)
        kwargs_199690 = {}
        # Getting the type of 'list' (line 197)
        list_199682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'list', False)
        # Calling list(args, kwargs) (line 197)
        list_call_result_199691 = invoke(stypy.reporting.localization.Localization(__file__, 197, 15), list_199682, *[subscript_call_result_199689], **kwargs_199690)
        
        # Obtaining the member '__getitem__' of a type (line 197)
        getitem___199692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 15), list_call_result_199691, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 197)
        subscript_call_result_199693 = invoke(stypy.reporting.localization.Localization(__file__, 197, 15), getitem___199692, int_199681)
        
        # Assigning a type to the variable 'test' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'test', subscript_call_result_199693)
        
        # Call to assertRaises(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'ImportError' (line 199)
        ImportError_199696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'ImportError', False)
        # Processing the call keyword arguments (line 199)
        kwargs_199697 = {}
        # Getting the type of 'self' (line 199)
        self_199694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 13), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 199)
        assertRaises_199695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 13), self_199694, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 199)
        assertRaises_call_result_199698 = invoke(stypy.reporting.localization.Localization(__file__, 199, 13), assertRaises_199695, *[ImportError_199696], **kwargs_199697)
        
        with_199699 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 199, 13), assertRaises_call_result_199698, 'with parameter', '__enter__', '__exit__')

        if with_199699:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 199)
            enter___199700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 13), assertRaises_call_result_199698, '__enter__')
            with_enter_199701 = invoke(stypy.reporting.localization.Localization(__file__, 199, 13), enter___199700)
            
            # Call to test_this_does_not_exist(...): (line 200)
            # Processing the call keyword arguments (line 200)
            kwargs_199704 = {}
            # Getting the type of 'test' (line 200)
            test_199702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'test', False)
            # Obtaining the member 'test_this_does_not_exist' of a type (line 200)
            test_this_does_not_exist_199703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), test_199702, 'test_this_does_not_exist')
            # Calling test_this_does_not_exist(args, kwargs) (line 200)
            test_this_does_not_exist_call_result_199705 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), test_this_does_not_exist_199703, *[], **kwargs_199704)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 199)
            exit___199706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 13), assertRaises_call_result_199698, '__exit__')
            with_exit_199707 = invoke(stypy.reporting.localization.Localization(__file__, 199, 13), exit___199706, None, None, None)

        
        # ################# End of 'test_discover_with_modules_that_fail_to_import(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_discover_with_modules_that_fail_to_import' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_199708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_199708)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_discover_with_modules_that_fail_to_import'
        return stypy_return_type_199708


    @norecursion
    def test_command_line_handling_parseArgs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_command_line_handling_parseArgs'
        module_type_store = module_type_store.open_function_context('test_command_line_handling_parseArgs', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_command_line_handling_parseArgs')
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_command_line_handling_parseArgs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_command_line_handling_parseArgs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_command_line_handling_parseArgs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_command_line_handling_parseArgs(...)' code ##################

        
        # Assigning a Call to a Name (line 204):
        
        # Call to __new__(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'unittest' (line 204)
        unittest_199711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 204)
        TestProgram_199712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 33), unittest_199711, 'TestProgram')
        # Processing the call keyword arguments (line 204)
        kwargs_199713 = {}
        # Getting the type of 'object' (line 204)
        object_199709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 204)
        new___199710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 18), object_199709, '__new__')
        # Calling __new__(args, kwargs) (line 204)
        new___call_result_199714 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), new___199710, *[TestProgram_199712], **kwargs_199713)
        
        # Assigning a type to the variable 'program' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'program', new___call_result_199714)
        
        # Assigning a List to a Name (line 206):
        
        # Obtaining an instance of the builtin type 'list' (line 206)
        list_199715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 206)
        
        # Assigning a type to the variable 'args' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'args', list_199715)

        @norecursion
        def do_discovery(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'do_discovery'
            module_type_store = module_type_store.open_function_context('do_discovery', 207, 8, False)
            
            # Passed parameters checking function
            do_discovery.stypy_localization = localization
            do_discovery.stypy_type_of_self = None
            do_discovery.stypy_type_store = module_type_store
            do_discovery.stypy_function_name = 'do_discovery'
            do_discovery.stypy_param_names_list = ['argv']
            do_discovery.stypy_varargs_param_name = None
            do_discovery.stypy_kwargs_param_name = None
            do_discovery.stypy_call_defaults = defaults
            do_discovery.stypy_call_varargs = varargs
            do_discovery.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'do_discovery', ['argv'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'do_discovery', localization, ['argv'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'do_discovery(...)' code ##################

            
            # Call to extend(...): (line 208)
            # Processing the call arguments (line 208)
            # Getting the type of 'argv' (line 208)
            argv_199718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 24), 'argv', False)
            # Processing the call keyword arguments (line 208)
            kwargs_199719 = {}
            # Getting the type of 'args' (line 208)
            args_199716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'args', False)
            # Obtaining the member 'extend' of a type (line 208)
            extend_199717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), args_199716, 'extend')
            # Calling extend(args, kwargs) (line 208)
            extend_call_result_199720 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), extend_199717, *[argv_199718], **kwargs_199719)
            
            
            # ################# End of 'do_discovery(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'do_discovery' in the type store
            # Getting the type of 'stypy_return_type' (line 207)
            stypy_return_type_199721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199721)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'do_discovery'
            return stypy_return_type_199721

        # Assigning a type to the variable 'do_discovery' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'do_discovery', do_discovery)
        
        # Assigning a Name to a Attribute (line 209):
        # Getting the type of 'do_discovery' (line 209)
        do_discovery_199722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 32), 'do_discovery')
        # Getting the type of 'program' (line 209)
        program_199723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'program')
        # Setting the type of the member '_do_discovery' of a type (line 209)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), program_199723, '_do_discovery', do_discovery_199722)
        
        # Call to parseArgs(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Obtaining an instance of the builtin type 'list' (line 210)
        list_199726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 210)
        # Adding element type (line 210)
        str_199727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 27), 'str', 'something')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 26), list_199726, str_199727)
        # Adding element type (line 210)
        str_199728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 40), 'str', 'discover')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 26), list_199726, str_199728)
        
        # Processing the call keyword arguments (line 210)
        kwargs_199729 = {}
        # Getting the type of 'program' (line 210)
        program_199724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'program', False)
        # Obtaining the member 'parseArgs' of a type (line 210)
        parseArgs_199725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), program_199724, 'parseArgs')
        # Calling parseArgs(args, kwargs) (line 210)
        parseArgs_call_result_199730 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), parseArgs_199725, *[list_199726], **kwargs_199729)
        
        
        # Call to assertEqual(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'args' (line 211)
        args_199733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'args', False)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_199734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        
        # Processing the call keyword arguments (line 211)
        kwargs_199735 = {}
        # Getting the type of 'self' (line 211)
        self_199731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 211)
        assertEqual_199732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_199731, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 211)
        assertEqual_call_result_199736 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assertEqual_199732, *[args_199733, list_199734], **kwargs_199735)
        
        
        # Call to parseArgs(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_199739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        str_199740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 27), 'str', 'something')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 26), list_199739, str_199740)
        # Adding element type (line 213)
        str_199741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 40), 'str', 'discover')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 26), list_199739, str_199741)
        # Adding element type (line 213)
        str_199742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 52), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 26), list_199739, str_199742)
        # Adding element type (line 213)
        str_199743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 59), 'str', 'bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 26), list_199739, str_199743)
        
        # Processing the call keyword arguments (line 213)
        kwargs_199744 = {}
        # Getting the type of 'program' (line 213)
        program_199737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'program', False)
        # Obtaining the member 'parseArgs' of a type (line 213)
        parseArgs_199738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), program_199737, 'parseArgs')
        # Calling parseArgs(args, kwargs) (line 213)
        parseArgs_call_result_199745 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), parseArgs_199738, *[list_199739], **kwargs_199744)
        
        
        # Call to assertEqual(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'args' (line 214)
        args_199748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 25), 'args', False)
        
        # Obtaining an instance of the builtin type 'list' (line 214)
        list_199749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 214)
        # Adding element type (line 214)
        str_199750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 32), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 31), list_199749, str_199750)
        # Adding element type (line 214)
        str_199751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 39), 'str', 'bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 31), list_199749, str_199751)
        
        # Processing the call keyword arguments (line 214)
        kwargs_199752 = {}
        # Getting the type of 'self' (line 214)
        self_199746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 214)
        assertEqual_199747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_199746, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 214)
        assertEqual_call_result_199753 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assertEqual_199747, *[args_199748, list_199749], **kwargs_199752)
        
        
        # ################# End of 'test_command_line_handling_parseArgs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_command_line_handling_parseArgs' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_199754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_199754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_command_line_handling_parseArgs'
        return stypy_return_type_199754


    @norecursion
    def test_command_line_handling_do_discovery_too_many_arguments(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_command_line_handling_do_discovery_too_many_arguments'
        module_type_store = module_type_store.open_function_context('test_command_line_handling_do_discovery_too_many_arguments', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments')
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_command_line_handling_do_discovery_too_many_arguments', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_command_line_handling_do_discovery_too_many_arguments', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_command_line_handling_do_discovery_too_many_arguments(...)' code ##################

        # Declaration of the 'Stop' class
        # Getting the type of 'Exception' (line 217)
        Exception_199755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'Exception')

        class Stop(Exception_199755, ):
            pass
        
        # Assigning a type to the variable 'Stop' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'Stop', Stop)

        @norecursion
        def usageExit(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'usageExit'
            module_type_store = module_type_store.open_function_context('usageExit', 219, 8, False)
            
            # Passed parameters checking function
            usageExit.stypy_localization = localization
            usageExit.stypy_type_of_self = None
            usageExit.stypy_type_store = module_type_store
            usageExit.stypy_function_name = 'usageExit'
            usageExit.stypy_param_names_list = []
            usageExit.stypy_varargs_param_name = None
            usageExit.stypy_kwargs_param_name = None
            usageExit.stypy_call_defaults = defaults
            usageExit.stypy_call_varargs = varargs
            usageExit.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'usageExit', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'usageExit', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'usageExit(...)' code ##################

            # Getting the type of 'Stop' (line 220)
            Stop_199756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 18), 'Stop')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 220, 12), Stop_199756, 'raise parameter', BaseException)
            
            # ################# End of 'usageExit(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'usageExit' in the type store
            # Getting the type of 'stypy_return_type' (line 219)
            stypy_return_type_199757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_199757)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'usageExit'
            return stypy_return_type_199757

        # Assigning a type to the variable 'usageExit' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'usageExit', usageExit)
        
        # Assigning a Call to a Name (line 222):
        
        # Call to __new__(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'unittest' (line 222)
        unittest_199760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 222)
        TestProgram_199761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 33), unittest_199760, 'TestProgram')
        # Processing the call keyword arguments (line 222)
        kwargs_199762 = {}
        # Getting the type of 'object' (line 222)
        object_199758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 222)
        new___199759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 18), object_199758, '__new__')
        # Calling __new__(args, kwargs) (line 222)
        new___call_result_199763 = invoke(stypy.reporting.localization.Localization(__file__, 222, 18), new___199759, *[TestProgram_199761], **kwargs_199762)
        
        # Assigning a type to the variable 'program' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'program', new___call_result_199763)
        
        # Assigning a Name to a Attribute (line 223):
        # Getting the type of 'usageExit' (line 223)
        usageExit_199764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 28), 'usageExit')
        # Getting the type of 'program' (line 223)
        program_199765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'program')
        # Setting the type of the member 'usageExit' of a type (line 223)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), program_199765, 'usageExit', usageExit_199764)
        
        # Assigning a Name to a Attribute (line 224):
        # Getting the type of 'None' (line 224)
        None_199766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 29), 'None')
        # Getting the type of 'program' (line 224)
        program_199767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'program')
        # Setting the type of the member 'testLoader' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), program_199767, 'testLoader', None_199766)
        
        # Call to assertRaises(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'Stop' (line 226)
        Stop_199770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 31), 'Stop', False)
        # Processing the call keyword arguments (line 226)
        kwargs_199771 = {}
        # Getting the type of 'self' (line 226)
        self_199768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 13), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 226)
        assertRaises_199769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 13), self_199768, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 226)
        assertRaises_call_result_199772 = invoke(stypy.reporting.localization.Localization(__file__, 226, 13), assertRaises_199769, *[Stop_199770], **kwargs_199771)
        
        with_199773 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 226, 13), assertRaises_call_result_199772, 'with parameter', '__enter__', '__exit__')

        if with_199773:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 226)
            enter___199774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 13), assertRaises_call_result_199772, '__enter__')
            with_enter_199775 = invoke(stypy.reporting.localization.Localization(__file__, 226, 13), enter___199774)
            
            # Call to _do_discovery(...): (line 228)
            # Processing the call arguments (line 228)
            
            # Obtaining an instance of the builtin type 'list' (line 228)
            list_199778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 34), 'list')
            # Adding type elements to the builtin type 'list' instance (line 228)
            # Adding element type (line 228)
            str_199779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 35), 'str', 'one')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 34), list_199778, str_199779)
            # Adding element type (line 228)
            str_199780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 42), 'str', 'two')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 34), list_199778, str_199780)
            # Adding element type (line 228)
            str_199781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 49), 'str', 'three')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 34), list_199778, str_199781)
            # Adding element type (line 228)
            str_199782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 58), 'str', 'four')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 34), list_199778, str_199782)
            
            # Processing the call keyword arguments (line 228)
            kwargs_199783 = {}
            # Getting the type of 'program' (line 228)
            program_199776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'program', False)
            # Obtaining the member '_do_discovery' of a type (line 228)
            _do_discovery_199777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), program_199776, '_do_discovery')
            # Calling _do_discovery(args, kwargs) (line 228)
            _do_discovery_call_result_199784 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), _do_discovery_199777, *[list_199778], **kwargs_199783)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 226)
            exit___199785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 13), assertRaises_call_result_199772, '__exit__')
            with_exit_199786 = invoke(stypy.reporting.localization.Localization(__file__, 226, 13), exit___199785, None, None, None)

        
        # ################# End of 'test_command_line_handling_do_discovery_too_many_arguments(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_command_line_handling_do_discovery_too_many_arguments' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_199787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_199787)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_command_line_handling_do_discovery_too_many_arguments'
        return stypy_return_type_199787


    @norecursion
    def test_command_line_handling_do_discovery_uses_default_loader(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_command_line_handling_do_discovery_uses_default_loader'
        module_type_store = module_type_store.open_function_context('test_command_line_handling_do_discovery_uses_default_loader', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader')
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_command_line_handling_do_discovery_uses_default_loader', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_command_line_handling_do_discovery_uses_default_loader', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_command_line_handling_do_discovery_uses_default_loader(...)' code ##################

        
        # Assigning a Call to a Name (line 232):
        
        # Call to __new__(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'unittest' (line 232)
        unittest_199790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 232)
        TestProgram_199791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 33), unittest_199790, 'TestProgram')
        # Processing the call keyword arguments (line 232)
        kwargs_199792 = {}
        # Getting the type of 'object' (line 232)
        object_199788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 232)
        new___199789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 18), object_199788, '__new__')
        # Calling __new__(args, kwargs) (line 232)
        new___call_result_199793 = invoke(stypy.reporting.localization.Localization(__file__, 232, 18), new___199789, *[TestProgram_199791], **kwargs_199792)
        
        # Assigning a type to the variable 'program' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'program', new___call_result_199793)
        # Declaration of the 'Loader' class

        class Loader(object, ):
            
            # Assigning a List to a Name (line 235):
            
            # Obtaining an instance of the builtin type 'list' (line 235)
            list_199794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 235)
            
            # Assigning a type to the variable 'args' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'args', list_199794)

            @norecursion
            def discover(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'discover'
                module_type_store = module_type_store.open_function_context('discover', 236, 12, False)
                # Assigning a type to the variable 'self' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Loader.discover.__dict__.__setitem__('stypy_localization', localization)
                Loader.discover.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Loader.discover.__dict__.__setitem__('stypy_type_store', module_type_store)
                Loader.discover.__dict__.__setitem__('stypy_function_name', 'Loader.discover')
                Loader.discover.__dict__.__setitem__('stypy_param_names_list', ['start_dir', 'pattern', 'top_level_dir'])
                Loader.discover.__dict__.__setitem__('stypy_varargs_param_name', None)
                Loader.discover.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Loader.discover.__dict__.__setitem__('stypy_call_defaults', defaults)
                Loader.discover.__dict__.__setitem__('stypy_call_varargs', varargs)
                Loader.discover.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Loader.discover.__dict__.__setitem__('stypy_declared_arg_number', 4)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Loader.discover', ['start_dir', 'pattern', 'top_level_dir'], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 237)
                # Processing the call arguments (line 237)
                
                # Obtaining an instance of the builtin type 'tuple' (line 237)
                tuple_199798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 237)
                # Adding element type (line 237)
                # Getting the type of 'start_dir' (line 237)
                start_dir_199799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 34), 'start_dir', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 34), tuple_199798, start_dir_199799)
                # Adding element type (line 237)
                # Getting the type of 'pattern' (line 237)
                pattern_199800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 45), 'pattern', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 34), tuple_199798, pattern_199800)
                # Adding element type (line 237)
                # Getting the type of 'top_level_dir' (line 237)
                top_level_dir_199801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 54), 'top_level_dir', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 34), tuple_199798, top_level_dir_199801)
                
                # Processing the call keyword arguments (line 237)
                kwargs_199802 = {}
                # Getting the type of 'self' (line 237)
                self_199795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'self', False)
                # Obtaining the member 'args' of a type (line 237)
                args_199796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), self_199795, 'args')
                # Obtaining the member 'append' of a type (line 237)
                append_199797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), args_199796, 'append')
                # Calling append(args, kwargs) (line 237)
                append_call_result_199803 = invoke(stypy.reporting.localization.Localization(__file__, 237, 16), append_199797, *[tuple_199798], **kwargs_199802)
                
                str_199804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 23), 'str', 'tests')
                # Assigning a type to the variable 'stypy_return_type' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'stypy_return_type', str_199804)
                
                # ################# End of 'discover(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'discover' in the type store
                # Getting the type of 'stypy_return_type' (line 236)
                stypy_return_type_199805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_199805)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'discover'
                return stypy_return_type_199805

        
        # Assigning a type to the variable 'Loader' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'Loader', Loader)
        
        # Assigning a Call to a Attribute (line 240):
        
        # Call to Loader(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_199807 = {}
        # Getting the type of 'Loader' (line 240)
        Loader_199806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 29), 'Loader', False)
        # Calling Loader(args, kwargs) (line 240)
        Loader_call_result_199808 = invoke(stypy.reporting.localization.Localization(__file__, 240, 29), Loader_199806, *[], **kwargs_199807)
        
        # Getting the type of 'program' (line 240)
        program_199809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'program')
        # Setting the type of the member 'testLoader' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), program_199809, 'testLoader', Loader_call_result_199808)
        
        # Call to _do_discovery(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_199812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        # Adding element type (line 241)
        str_199813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 31), 'str', '-v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 30), list_199812, str_199813)
        
        # Processing the call keyword arguments (line 241)
        kwargs_199814 = {}
        # Getting the type of 'program' (line 241)
        program_199810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 241)
        _do_discovery_199811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), program_199810, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 241)
        _do_discovery_call_result_199815 = invoke(stypy.reporting.localization.Localization(__file__, 241, 8), _do_discovery_199811, *[list_199812], **kwargs_199814)
        
        
        # Call to assertEqual(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'Loader' (line 242)
        Loader_199818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 242)
        args_199819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 25), Loader_199818, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 242)
        list_199820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 242)
        # Adding element type (line 242)
        
        # Obtaining an instance of the builtin type 'tuple' (line 242)
        tuple_199821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 242)
        # Adding element type (line 242)
        str_199822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 40), 'str', '.')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 40), tuple_199821, str_199822)
        # Adding element type (line 242)
        str_199823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 45), 'str', 'test*.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 40), tuple_199821, str_199823)
        # Adding element type (line 242)
        # Getting the type of 'None' (line 242)
        None_199824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 57), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 40), tuple_199821, None_199824)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 38), list_199820, tuple_199821)
        
        # Processing the call keyword arguments (line 242)
        kwargs_199825 = {}
        # Getting the type of 'self' (line 242)
        self_199816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 242)
        assertEqual_199817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_199816, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 242)
        assertEqual_call_result_199826 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assertEqual_199817, *[args_199819, list_199820], **kwargs_199825)
        
        
        # ################# End of 'test_command_line_handling_do_discovery_uses_default_loader(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_command_line_handling_do_discovery_uses_default_loader' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_199827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_199827)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_command_line_handling_do_discovery_uses_default_loader'
        return stypy_return_type_199827


    @norecursion
    def test_command_line_handling_do_discovery_calls_loader(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_command_line_handling_do_discovery_calls_loader'
        module_type_store = module_type_store.open_function_context('test_command_line_handling_do_discovery_calls_loader', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_command_line_handling_do_discovery_calls_loader')
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_command_line_handling_do_discovery_calls_loader.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_command_line_handling_do_discovery_calls_loader', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_command_line_handling_do_discovery_calls_loader', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_command_line_handling_do_discovery_calls_loader(...)' code ##################

        
        # Assigning a Call to a Name (line 245):
        
        # Call to __new__(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'unittest' (line 245)
        unittest_199830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 245)
        TestProgram_199831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 33), unittest_199830, 'TestProgram')
        # Processing the call keyword arguments (line 245)
        kwargs_199832 = {}
        # Getting the type of 'object' (line 245)
        object_199828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 245)
        new___199829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 18), object_199828, '__new__')
        # Calling __new__(args, kwargs) (line 245)
        new___call_result_199833 = invoke(stypy.reporting.localization.Localization(__file__, 245, 18), new___199829, *[TestProgram_199831], **kwargs_199832)
        
        # Assigning a type to the variable 'program' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'program', new___call_result_199833)
        # Declaration of the 'Loader' class

        class Loader(object, ):
            
            # Assigning a List to a Name (line 248):
            
            # Obtaining an instance of the builtin type 'list' (line 248)
            list_199834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 248)
            
            # Assigning a type to the variable 'args' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'args', list_199834)

            @norecursion
            def discover(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'discover'
                module_type_store = module_type_store.open_function_context('discover', 249, 12, False)
                # Assigning a type to the variable 'self' (line 250)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                Loader.discover.__dict__.__setitem__('stypy_localization', localization)
                Loader.discover.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                Loader.discover.__dict__.__setitem__('stypy_type_store', module_type_store)
                Loader.discover.__dict__.__setitem__('stypy_function_name', 'Loader.discover')
                Loader.discover.__dict__.__setitem__('stypy_param_names_list', ['start_dir', 'pattern', 'top_level_dir'])
                Loader.discover.__dict__.__setitem__('stypy_varargs_param_name', None)
                Loader.discover.__dict__.__setitem__('stypy_kwargs_param_name', None)
                Loader.discover.__dict__.__setitem__('stypy_call_defaults', defaults)
                Loader.discover.__dict__.__setitem__('stypy_call_varargs', varargs)
                Loader.discover.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                Loader.discover.__dict__.__setitem__('stypy_declared_arg_number', 4)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'Loader.discover', ['start_dir', 'pattern', 'top_level_dir'], None, None, defaults, varargs, kwargs)

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

                
                # Call to append(...): (line 250)
                # Processing the call arguments (line 250)
                
                # Obtaining an instance of the builtin type 'tuple' (line 250)
                tuple_199838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 250)
                # Adding element type (line 250)
                # Getting the type of 'start_dir' (line 250)
                start_dir_199839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 34), 'start_dir', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 34), tuple_199838, start_dir_199839)
                # Adding element type (line 250)
                # Getting the type of 'pattern' (line 250)
                pattern_199840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 45), 'pattern', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 34), tuple_199838, pattern_199840)
                # Adding element type (line 250)
                # Getting the type of 'top_level_dir' (line 250)
                top_level_dir_199841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 54), 'top_level_dir', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 34), tuple_199838, top_level_dir_199841)
                
                # Processing the call keyword arguments (line 250)
                kwargs_199842 = {}
                # Getting the type of 'self' (line 250)
                self_199835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'self', False)
                # Obtaining the member 'args' of a type (line 250)
                args_199836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), self_199835, 'args')
                # Obtaining the member 'append' of a type (line 250)
                append_199837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), args_199836, 'append')
                # Calling append(args, kwargs) (line 250)
                append_call_result_199843 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), append_199837, *[tuple_199838], **kwargs_199842)
                
                str_199844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 23), 'str', 'tests')
                # Assigning a type to the variable 'stypy_return_type' (line 251)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'stypy_return_type', str_199844)
                
                # ################# End of 'discover(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'discover' in the type store
                # Getting the type of 'stypy_return_type' (line 249)
                stypy_return_type_199845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_199845)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'discover'
                return stypy_return_type_199845

        
        # Assigning a type to the variable 'Loader' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'Loader', Loader)
        
        # Call to _do_discovery(...): (line 253)
        # Processing the call arguments (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_199848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        str_199849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 31), 'str', '-v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 30), list_199848, str_199849)
        
        # Processing the call keyword arguments (line 253)
        # Getting the type of 'Loader' (line 253)
        Loader_199850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 45), 'Loader', False)
        keyword_199851 = Loader_199850
        kwargs_199852 = {'Loader': keyword_199851}
        # Getting the type of 'program' (line 253)
        program_199846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 253)
        _do_discovery_199847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), program_199846, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 253)
        _do_discovery_call_result_199853 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), _do_discovery_199847, *[list_199848], **kwargs_199852)
        
        
        # Call to assertEqual(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'program' (line 254)
        program_199856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 25), 'program', False)
        # Obtaining the member 'verbosity' of a type (line 254)
        verbosity_199857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 25), program_199856, 'verbosity')
        int_199858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 44), 'int')
        # Processing the call keyword arguments (line 254)
        kwargs_199859 = {}
        # Getting the type of 'self' (line 254)
        self_199854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 254)
        assertEqual_199855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), self_199854, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 254)
        assertEqual_call_result_199860 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), assertEqual_199855, *[verbosity_199857, int_199858], **kwargs_199859)
        
        
        # Call to assertEqual(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'program' (line 255)
        program_199863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 255)
        test_199864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 25), program_199863, 'test')
        str_199865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 255)
        kwargs_199866 = {}
        # Getting the type of 'self' (line 255)
        self_199861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 255)
        assertEqual_199862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_199861, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 255)
        assertEqual_call_result_199867 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assertEqual_199862, *[test_199864, str_199865], **kwargs_199866)
        
        
        # Call to assertEqual(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'Loader' (line 256)
        Loader_199870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 256)
        args_199871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 25), Loader_199870, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 256)
        list_199872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 256)
        # Adding element type (line 256)
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_199873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        str_199874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'str', '.')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 40), tuple_199873, str_199874)
        # Adding element type (line 256)
        str_199875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 45), 'str', 'test*.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 40), tuple_199873, str_199875)
        # Adding element type (line 256)
        # Getting the type of 'None' (line 256)
        None_199876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 57), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 40), tuple_199873, None_199876)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 38), list_199872, tuple_199873)
        
        # Processing the call keyword arguments (line 256)
        kwargs_199877 = {}
        # Getting the type of 'self' (line 256)
        self_199868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 256)
        assertEqual_199869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), self_199868, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 256)
        assertEqual_call_result_199878 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), assertEqual_199869, *[args_199871, list_199872], **kwargs_199877)
        
        
        # Assigning a List to a Attribute (line 258):
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_199879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        
        # Getting the type of 'Loader' (line 258)
        Loader_199880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'Loader')
        # Setting the type of the member 'args' of a type (line 258)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), Loader_199880, 'args', list_199879)
        
        # Assigning a Call to a Name (line 259):
        
        # Call to __new__(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'unittest' (line 259)
        unittest_199883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 259)
        TestProgram_199884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 33), unittest_199883, 'TestProgram')
        # Processing the call keyword arguments (line 259)
        kwargs_199885 = {}
        # Getting the type of 'object' (line 259)
        object_199881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 259)
        new___199882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 18), object_199881, '__new__')
        # Calling __new__(args, kwargs) (line 259)
        new___call_result_199886 = invoke(stypy.reporting.localization.Localization(__file__, 259, 18), new___199882, *[TestProgram_199884], **kwargs_199885)
        
        # Assigning a type to the variable 'program' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'program', new___call_result_199886)
        
        # Call to _do_discovery(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_199889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        # Adding element type (line 260)
        str_199890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 31), 'str', '--verbose')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 30), list_199889, str_199890)
        
        # Processing the call keyword arguments (line 260)
        # Getting the type of 'Loader' (line 260)
        Loader_199891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 52), 'Loader', False)
        keyword_199892 = Loader_199891
        kwargs_199893 = {'Loader': keyword_199892}
        # Getting the type of 'program' (line 260)
        program_199887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 260)
        _do_discovery_199888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), program_199887, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 260)
        _do_discovery_call_result_199894 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), _do_discovery_199888, *[list_199889], **kwargs_199893)
        
        
        # Call to assertEqual(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'program' (line 261)
        program_199897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 261)
        test_199898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 25), program_199897, 'test')
        str_199899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 261)
        kwargs_199900 = {}
        # Getting the type of 'self' (line 261)
        self_199895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 261)
        assertEqual_199896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_199895, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 261)
        assertEqual_call_result_199901 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), assertEqual_199896, *[test_199898, str_199899], **kwargs_199900)
        
        
        # Call to assertEqual(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'Loader' (line 262)
        Loader_199904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 262)
        args_199905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 25), Loader_199904, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 262)
        list_199906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 262)
        # Adding element type (line 262)
        
        # Obtaining an instance of the builtin type 'tuple' (line 262)
        tuple_199907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 262)
        # Adding element type (line 262)
        str_199908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 40), 'str', '.')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 40), tuple_199907, str_199908)
        # Adding element type (line 262)
        str_199909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 45), 'str', 'test*.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 40), tuple_199907, str_199909)
        # Adding element type (line 262)
        # Getting the type of 'None' (line 262)
        None_199910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 57), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 40), tuple_199907, None_199910)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 38), list_199906, tuple_199907)
        
        # Processing the call keyword arguments (line 262)
        kwargs_199911 = {}
        # Getting the type of 'self' (line 262)
        self_199902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 262)
        assertEqual_199903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_199902, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 262)
        assertEqual_call_result_199912 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), assertEqual_199903, *[args_199905, list_199906], **kwargs_199911)
        
        
        # Assigning a List to a Attribute (line 264):
        
        # Obtaining an instance of the builtin type 'list' (line 264)
        list_199913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 264)
        
        # Getting the type of 'Loader' (line 264)
        Loader_199914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'Loader')
        # Setting the type of the member 'args' of a type (line 264)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), Loader_199914, 'args', list_199913)
        
        # Assigning a Call to a Name (line 265):
        
        # Call to __new__(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'unittest' (line 265)
        unittest_199917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 265)
        TestProgram_199918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 33), unittest_199917, 'TestProgram')
        # Processing the call keyword arguments (line 265)
        kwargs_199919 = {}
        # Getting the type of 'object' (line 265)
        object_199915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 265)
        new___199916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 18), object_199915, '__new__')
        # Calling __new__(args, kwargs) (line 265)
        new___call_result_199920 = invoke(stypy.reporting.localization.Localization(__file__, 265, 18), new___199916, *[TestProgram_199918], **kwargs_199919)
        
        # Assigning a type to the variable 'program' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'program', new___call_result_199920)
        
        # Call to _do_discovery(...): (line 266)
        # Processing the call arguments (line 266)
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_199923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        
        # Processing the call keyword arguments (line 266)
        # Getting the type of 'Loader' (line 266)
        Loader_199924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 41), 'Loader', False)
        keyword_199925 = Loader_199924
        kwargs_199926 = {'Loader': keyword_199925}
        # Getting the type of 'program' (line 266)
        program_199921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 266)
        _do_discovery_199922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), program_199921, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 266)
        _do_discovery_call_result_199927 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), _do_discovery_199922, *[list_199923], **kwargs_199926)
        
        
        # Call to assertEqual(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'program' (line 267)
        program_199930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 267)
        test_199931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 25), program_199930, 'test')
        str_199932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 267)
        kwargs_199933 = {}
        # Getting the type of 'self' (line 267)
        self_199928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 267)
        assertEqual_199929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_199928, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 267)
        assertEqual_call_result_199934 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), assertEqual_199929, *[test_199931, str_199932], **kwargs_199933)
        
        
        # Call to assertEqual(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'Loader' (line 268)
        Loader_199937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 268)
        args_199938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 25), Loader_199937, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 268)
        list_199939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 268)
        # Adding element type (line 268)
        
        # Obtaining an instance of the builtin type 'tuple' (line 268)
        tuple_199940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 268)
        # Adding element type (line 268)
        str_199941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 40), 'str', '.')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 40), tuple_199940, str_199941)
        # Adding element type (line 268)
        str_199942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 45), 'str', 'test*.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 40), tuple_199940, str_199942)
        # Adding element type (line 268)
        # Getting the type of 'None' (line 268)
        None_199943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 57), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 40), tuple_199940, None_199943)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 38), list_199939, tuple_199940)
        
        # Processing the call keyword arguments (line 268)
        kwargs_199944 = {}
        # Getting the type of 'self' (line 268)
        self_199935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 268)
        assertEqual_199936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_199935, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 268)
        assertEqual_call_result_199945 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), assertEqual_199936, *[args_199938, list_199939], **kwargs_199944)
        
        
        # Assigning a List to a Attribute (line 270):
        
        # Obtaining an instance of the builtin type 'list' (line 270)
        list_199946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 270)
        
        # Getting the type of 'Loader' (line 270)
        Loader_199947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'Loader')
        # Setting the type of the member 'args' of a type (line 270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), Loader_199947, 'args', list_199946)
        
        # Assigning a Call to a Name (line 271):
        
        # Call to __new__(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'unittest' (line 271)
        unittest_199950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 271)
        TestProgram_199951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 33), unittest_199950, 'TestProgram')
        # Processing the call keyword arguments (line 271)
        kwargs_199952 = {}
        # Getting the type of 'object' (line 271)
        object_199948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 271)
        new___199949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 18), object_199948, '__new__')
        # Calling __new__(args, kwargs) (line 271)
        new___call_result_199953 = invoke(stypy.reporting.localization.Localization(__file__, 271, 18), new___199949, *[TestProgram_199951], **kwargs_199952)
        
        # Assigning a type to the variable 'program' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'program', new___call_result_199953)
        
        # Call to _do_discovery(...): (line 272)
        # Processing the call arguments (line 272)
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_199956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        # Adding element type (line 272)
        str_199957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 31), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 30), list_199956, str_199957)
        
        # Processing the call keyword arguments (line 272)
        # Getting the type of 'Loader' (line 272)
        Loader_199958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 47), 'Loader', False)
        keyword_199959 = Loader_199958
        kwargs_199960 = {'Loader': keyword_199959}
        # Getting the type of 'program' (line 272)
        program_199954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 272)
        _do_discovery_199955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), program_199954, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 272)
        _do_discovery_call_result_199961 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), _do_discovery_199955, *[list_199956], **kwargs_199960)
        
        
        # Call to assertEqual(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'program' (line 273)
        program_199964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 273)
        test_199965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 25), program_199964, 'test')
        str_199966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 273)
        kwargs_199967 = {}
        # Getting the type of 'self' (line 273)
        self_199962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 273)
        assertEqual_199963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_199962, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 273)
        assertEqual_call_result_199968 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), assertEqual_199963, *[test_199965, str_199966], **kwargs_199967)
        
        
        # Call to assertEqual(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'Loader' (line 274)
        Loader_199971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 274)
        args_199972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 25), Loader_199971, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_199973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        
        # Obtaining an instance of the builtin type 'tuple' (line 274)
        tuple_199974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 274)
        # Adding element type (line 274)
        str_199975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 40), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 40), tuple_199974, str_199975)
        # Adding element type (line 274)
        str_199976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 48), 'str', 'test*.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 40), tuple_199974, str_199976)
        # Adding element type (line 274)
        # Getting the type of 'None' (line 274)
        None_199977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 60), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 40), tuple_199974, None_199977)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 38), list_199973, tuple_199974)
        
        # Processing the call keyword arguments (line 274)
        kwargs_199978 = {}
        # Getting the type of 'self' (line 274)
        self_199969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 274)
        assertEqual_199970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_199969, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 274)
        assertEqual_call_result_199979 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), assertEqual_199970, *[args_199972, list_199973], **kwargs_199978)
        
        
        # Assigning a List to a Attribute (line 276):
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_199980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        
        # Getting the type of 'Loader' (line 276)
        Loader_199981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'Loader')
        # Setting the type of the member 'args' of a type (line 276)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), Loader_199981, 'args', list_199980)
        
        # Assigning a Call to a Name (line 277):
        
        # Call to __new__(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'unittest' (line 277)
        unittest_199984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 277)
        TestProgram_199985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 33), unittest_199984, 'TestProgram')
        # Processing the call keyword arguments (line 277)
        kwargs_199986 = {}
        # Getting the type of 'object' (line 277)
        object_199982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 277)
        new___199983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 18), object_199982, '__new__')
        # Calling __new__(args, kwargs) (line 277)
        new___call_result_199987 = invoke(stypy.reporting.localization.Localization(__file__, 277, 18), new___199983, *[TestProgram_199985], **kwargs_199986)
        
        # Assigning a type to the variable 'program' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'program', new___call_result_199987)
        
        # Call to _do_discovery(...): (line 278)
        # Processing the call arguments (line 278)
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_199990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        str_199991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 30), list_199990, str_199991)
        # Adding element type (line 278)
        str_199992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 39), 'str', 'eggs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 30), list_199990, str_199992)
        
        # Processing the call keyword arguments (line 278)
        # Getting the type of 'Loader' (line 278)
        Loader_199993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 55), 'Loader', False)
        keyword_199994 = Loader_199993
        kwargs_199995 = {'Loader': keyword_199994}
        # Getting the type of 'program' (line 278)
        program_199988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 278)
        _do_discovery_199989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), program_199988, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 278)
        _do_discovery_call_result_199996 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), _do_discovery_199989, *[list_199990], **kwargs_199995)
        
        
        # Call to assertEqual(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'program' (line 279)
        program_199999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 279)
        test_200000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 25), program_199999, 'test')
        str_200001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 279)
        kwargs_200002 = {}
        # Getting the type of 'self' (line 279)
        self_199997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 279)
        assertEqual_199998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), self_199997, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 279)
        assertEqual_call_result_200003 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), assertEqual_199998, *[test_200000, str_200001], **kwargs_200002)
        
        
        # Call to assertEqual(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'Loader' (line 280)
        Loader_200006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 280)
        args_200007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 25), Loader_200006, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_200008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        
        # Obtaining an instance of the builtin type 'tuple' (line 280)
        tuple_200009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 280)
        # Adding element type (line 280)
        str_200010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 40), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 40), tuple_200009, str_200010)
        # Adding element type (line 280)
        str_200011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 48), 'str', 'eggs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 40), tuple_200009, str_200011)
        # Adding element type (line 280)
        # Getting the type of 'None' (line 280)
        None_200012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 56), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 40), tuple_200009, None_200012)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 38), list_200008, tuple_200009)
        
        # Processing the call keyword arguments (line 280)
        kwargs_200013 = {}
        # Getting the type of 'self' (line 280)
        self_200004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 280)
        assertEqual_200005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), self_200004, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 280)
        assertEqual_call_result_200014 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), assertEqual_200005, *[args_200007, list_200008], **kwargs_200013)
        
        
        # Assigning a List to a Attribute (line 282):
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_200015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        
        # Getting the type of 'Loader' (line 282)
        Loader_200016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'Loader')
        # Setting the type of the member 'args' of a type (line 282)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), Loader_200016, 'args', list_200015)
        
        # Assigning a Call to a Name (line 283):
        
        # Call to __new__(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'unittest' (line 283)
        unittest_200019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 283)
        TestProgram_200020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 33), unittest_200019, 'TestProgram')
        # Processing the call keyword arguments (line 283)
        kwargs_200021 = {}
        # Getting the type of 'object' (line 283)
        object_200017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 283)
        new___200018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 18), object_200017, '__new__')
        # Calling __new__(args, kwargs) (line 283)
        new___call_result_200022 = invoke(stypy.reporting.localization.Localization(__file__, 283, 18), new___200018, *[TestProgram_200020], **kwargs_200021)
        
        # Assigning a type to the variable 'program' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'program', new___call_result_200022)
        
        # Call to _do_discovery(...): (line 284)
        # Processing the call arguments (line 284)
        
        # Obtaining an instance of the builtin type 'list' (line 284)
        list_200025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 284)
        # Adding element type (line 284)
        str_200026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 31), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 30), list_200025, str_200026)
        # Adding element type (line 284)
        str_200027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 39), 'str', 'eggs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 30), list_200025, str_200027)
        # Adding element type (line 284)
        str_200028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 47), 'str', 'ham')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 30), list_200025, str_200028)
        
        # Processing the call keyword arguments (line 284)
        # Getting the type of 'Loader' (line 284)
        Loader_200029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 62), 'Loader', False)
        keyword_200030 = Loader_200029
        kwargs_200031 = {'Loader': keyword_200030}
        # Getting the type of 'program' (line 284)
        program_200023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 284)
        _do_discovery_200024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), program_200023, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 284)
        _do_discovery_call_result_200032 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), _do_discovery_200024, *[list_200025], **kwargs_200031)
        
        
        # Call to assertEqual(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'program' (line 285)
        program_200035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 285)
        test_200036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 25), program_200035, 'test')
        str_200037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 285)
        kwargs_200038 = {}
        # Getting the type of 'self' (line 285)
        self_200033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 285)
        assertEqual_200034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_200033, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 285)
        assertEqual_call_result_200039 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), assertEqual_200034, *[test_200036, str_200037], **kwargs_200038)
        
        
        # Call to assertEqual(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'Loader' (line 286)
        Loader_200042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 286)
        args_200043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 25), Loader_200042, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_200044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        
        # Obtaining an instance of the builtin type 'tuple' (line 286)
        tuple_200045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 286)
        # Adding element type (line 286)
        str_200046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 40), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 40), tuple_200045, str_200046)
        # Adding element type (line 286)
        str_200047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 48), 'str', 'eggs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 40), tuple_200045, str_200047)
        # Adding element type (line 286)
        str_200048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 56), 'str', 'ham')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 40), tuple_200045, str_200048)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 38), list_200044, tuple_200045)
        
        # Processing the call keyword arguments (line 286)
        kwargs_200049 = {}
        # Getting the type of 'self' (line 286)
        self_200040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 286)
        assertEqual_200041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), self_200040, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 286)
        assertEqual_call_result_200050 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), assertEqual_200041, *[args_200043, list_200044], **kwargs_200049)
        
        
        # Assigning a List to a Attribute (line 288):
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_200051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        
        # Getting the type of 'Loader' (line 288)
        Loader_200052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'Loader')
        # Setting the type of the member 'args' of a type (line 288)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), Loader_200052, 'args', list_200051)
        
        # Assigning a Call to a Name (line 289):
        
        # Call to __new__(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'unittest' (line 289)
        unittest_200055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 289)
        TestProgram_200056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 33), unittest_200055, 'TestProgram')
        # Processing the call keyword arguments (line 289)
        kwargs_200057 = {}
        # Getting the type of 'object' (line 289)
        object_200053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 289)
        new___200054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 18), object_200053, '__new__')
        # Calling __new__(args, kwargs) (line 289)
        new___call_result_200058 = invoke(stypy.reporting.localization.Localization(__file__, 289, 18), new___200054, *[TestProgram_200056], **kwargs_200057)
        
        # Assigning a type to the variable 'program' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'program', new___call_result_200058)
        
        # Call to _do_discovery(...): (line 290)
        # Processing the call arguments (line 290)
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_200061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        # Adding element type (line 290)
        str_200062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 31), 'str', '-s')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), list_200061, str_200062)
        # Adding element type (line 290)
        str_200063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 37), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 30), list_200061, str_200063)
        
        # Processing the call keyword arguments (line 290)
        # Getting the type of 'Loader' (line 290)
        Loader_200064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 53), 'Loader', False)
        keyword_200065 = Loader_200064
        kwargs_200066 = {'Loader': keyword_200065}
        # Getting the type of 'program' (line 290)
        program_200059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 290)
        _do_discovery_200060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), program_200059, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 290)
        _do_discovery_call_result_200067 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), _do_discovery_200060, *[list_200061], **kwargs_200066)
        
        
        # Call to assertEqual(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'program' (line 291)
        program_200070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 291)
        test_200071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 25), program_200070, 'test')
        str_200072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 291)
        kwargs_200073 = {}
        # Getting the type of 'self' (line 291)
        self_200068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 291)
        assertEqual_200069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), self_200068, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 291)
        assertEqual_call_result_200074 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), assertEqual_200069, *[test_200071, str_200072], **kwargs_200073)
        
        
        # Call to assertEqual(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'Loader' (line 292)
        Loader_200077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 292)
        args_200078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 25), Loader_200077, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_200079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        
        # Obtaining an instance of the builtin type 'tuple' (line 292)
        tuple_200080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 292)
        # Adding element type (line 292)
        str_200081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 40), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 40), tuple_200080, str_200081)
        # Adding element type (line 292)
        str_200082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 48), 'str', 'test*.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 40), tuple_200080, str_200082)
        # Adding element type (line 292)
        # Getting the type of 'None' (line 292)
        None_200083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 60), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 40), tuple_200080, None_200083)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 38), list_200079, tuple_200080)
        
        # Processing the call keyword arguments (line 292)
        kwargs_200084 = {}
        # Getting the type of 'self' (line 292)
        self_200075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 292)
        assertEqual_200076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), self_200075, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 292)
        assertEqual_call_result_200085 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), assertEqual_200076, *[args_200078, list_200079], **kwargs_200084)
        
        
        # Assigning a List to a Attribute (line 294):
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_200086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        
        # Getting the type of 'Loader' (line 294)
        Loader_200087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'Loader')
        # Setting the type of the member 'args' of a type (line 294)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), Loader_200087, 'args', list_200086)
        
        # Assigning a Call to a Name (line 295):
        
        # Call to __new__(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'unittest' (line 295)
        unittest_200090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 295)
        TestProgram_200091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 33), unittest_200090, 'TestProgram')
        # Processing the call keyword arguments (line 295)
        kwargs_200092 = {}
        # Getting the type of 'object' (line 295)
        object_200088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 295)
        new___200089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 18), object_200088, '__new__')
        # Calling __new__(args, kwargs) (line 295)
        new___call_result_200093 = invoke(stypy.reporting.localization.Localization(__file__, 295, 18), new___200089, *[TestProgram_200091], **kwargs_200092)
        
        # Assigning a type to the variable 'program' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'program', new___call_result_200093)
        
        # Call to _do_discovery(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_200096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        # Adding element type (line 296)
        str_200097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 31), 'str', '-t')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 30), list_200096, str_200097)
        # Adding element type (line 296)
        str_200098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 37), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 30), list_200096, str_200098)
        
        # Processing the call keyword arguments (line 296)
        # Getting the type of 'Loader' (line 296)
        Loader_200099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 53), 'Loader', False)
        keyword_200100 = Loader_200099
        kwargs_200101 = {'Loader': keyword_200100}
        # Getting the type of 'program' (line 296)
        program_200094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 296)
        _do_discovery_200095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), program_200094, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 296)
        _do_discovery_call_result_200102 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), _do_discovery_200095, *[list_200096], **kwargs_200101)
        
        
        # Call to assertEqual(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'program' (line 297)
        program_200105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 297)
        test_200106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), program_200105, 'test')
        str_200107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 297)
        kwargs_200108 = {}
        # Getting the type of 'self' (line 297)
        self_200103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 297)
        assertEqual_200104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), self_200103, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 297)
        assertEqual_call_result_200109 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), assertEqual_200104, *[test_200106, str_200107], **kwargs_200108)
        
        
        # Call to assertEqual(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'Loader' (line 298)
        Loader_200112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 298)
        args_200113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 25), Loader_200112, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 298)
        list_200114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 298)
        # Adding element type (line 298)
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_200115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        str_200116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 40), 'str', '.')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 40), tuple_200115, str_200116)
        # Adding element type (line 298)
        str_200117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 45), 'str', 'test*.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 40), tuple_200115, str_200117)
        # Adding element type (line 298)
        str_200118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 57), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 40), tuple_200115, str_200118)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 38), list_200114, tuple_200115)
        
        # Processing the call keyword arguments (line 298)
        kwargs_200119 = {}
        # Getting the type of 'self' (line 298)
        self_200110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 298)
        assertEqual_200111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), self_200110, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 298)
        assertEqual_call_result_200120 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), assertEqual_200111, *[args_200113, list_200114], **kwargs_200119)
        
        
        # Assigning a List to a Attribute (line 300):
        
        # Obtaining an instance of the builtin type 'list' (line 300)
        list_200121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 300)
        
        # Getting the type of 'Loader' (line 300)
        Loader_200122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'Loader')
        # Setting the type of the member 'args' of a type (line 300)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), Loader_200122, 'args', list_200121)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to __new__(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'unittest' (line 301)
        unittest_200125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 301)
        TestProgram_200126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 33), unittest_200125, 'TestProgram')
        # Processing the call keyword arguments (line 301)
        kwargs_200127 = {}
        # Getting the type of 'object' (line 301)
        object_200123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 301)
        new___200124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 18), object_200123, '__new__')
        # Calling __new__(args, kwargs) (line 301)
        new___call_result_200128 = invoke(stypy.reporting.localization.Localization(__file__, 301, 18), new___200124, *[TestProgram_200126], **kwargs_200127)
        
        # Assigning a type to the variable 'program' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'program', new___call_result_200128)
        
        # Call to _do_discovery(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Obtaining an instance of the builtin type 'list' (line 302)
        list_200131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 302)
        # Adding element type (line 302)
        str_200132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 31), 'str', '-p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 30), list_200131, str_200132)
        # Adding element type (line 302)
        str_200133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 37), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 30), list_200131, str_200133)
        
        # Processing the call keyword arguments (line 302)
        # Getting the type of 'Loader' (line 302)
        Loader_200134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 53), 'Loader', False)
        keyword_200135 = Loader_200134
        kwargs_200136 = {'Loader': keyword_200135}
        # Getting the type of 'program' (line 302)
        program_200129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 302)
        _do_discovery_200130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), program_200129, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 302)
        _do_discovery_call_result_200137 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), _do_discovery_200130, *[list_200131], **kwargs_200136)
        
        
        # Call to assertEqual(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'program' (line 303)
        program_200140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 303)
        test_200141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 25), program_200140, 'test')
        str_200142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 303)
        kwargs_200143 = {}
        # Getting the type of 'self' (line 303)
        self_200138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 303)
        assertEqual_200139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), self_200138, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 303)
        assertEqual_call_result_200144 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), assertEqual_200139, *[test_200141, str_200142], **kwargs_200143)
        
        
        # Call to assertEqual(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'Loader' (line 304)
        Loader_200147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 304)
        args_200148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 25), Loader_200147, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 304)
        list_200149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 304)
        # Adding element type (line 304)
        
        # Obtaining an instance of the builtin type 'tuple' (line 304)
        tuple_200150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 304)
        # Adding element type (line 304)
        str_200151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 40), 'str', '.')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 40), tuple_200150, str_200151)
        # Adding element type (line 304)
        str_200152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 45), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 40), tuple_200150, str_200152)
        # Adding element type (line 304)
        # Getting the type of 'None' (line 304)
        None_200153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 53), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 40), tuple_200150, None_200153)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 38), list_200149, tuple_200150)
        
        # Processing the call keyword arguments (line 304)
        kwargs_200154 = {}
        # Getting the type of 'self' (line 304)
        self_200145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 304)
        assertEqual_200146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), self_200145, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 304)
        assertEqual_call_result_200155 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), assertEqual_200146, *[args_200148, list_200149], **kwargs_200154)
        
        
        # Call to assertFalse(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'program' (line 305)
        program_200158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 25), 'program', False)
        # Obtaining the member 'failfast' of a type (line 305)
        failfast_200159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 25), program_200158, 'failfast')
        # Processing the call keyword arguments (line 305)
        kwargs_200160 = {}
        # Getting the type of 'self' (line 305)
        self_200156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 305)
        assertFalse_200157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), self_200156, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 305)
        assertFalse_call_result_200161 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), assertFalse_200157, *[failfast_200159], **kwargs_200160)
        
        
        # Call to assertFalse(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'program' (line 306)
        program_200164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 25), 'program', False)
        # Obtaining the member 'catchbreak' of a type (line 306)
        catchbreak_200165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 25), program_200164, 'catchbreak')
        # Processing the call keyword arguments (line 306)
        kwargs_200166 = {}
        # Getting the type of 'self' (line 306)
        self_200162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 306)
        assertFalse_200163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), self_200162, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 306)
        assertFalse_call_result_200167 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), assertFalse_200163, *[catchbreak_200165], **kwargs_200166)
        
        
        # Assigning a List to a Attribute (line 308):
        
        # Obtaining an instance of the builtin type 'list' (line 308)
        list_200168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 308)
        
        # Getting the type of 'Loader' (line 308)
        Loader_200169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'Loader')
        # Setting the type of the member 'args' of a type (line 308)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), Loader_200169, 'args', list_200168)
        
        # Assigning a Call to a Name (line 309):
        
        # Call to __new__(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'unittest' (line 309)
        unittest_200172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 33), 'unittest', False)
        # Obtaining the member 'TestProgram' of a type (line 309)
        TestProgram_200173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 33), unittest_200172, 'TestProgram')
        # Processing the call keyword arguments (line 309)
        kwargs_200174 = {}
        # Getting the type of 'object' (line 309)
        object_200170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 18), 'object', False)
        # Obtaining the member '__new__' of a type (line 309)
        new___200171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 18), object_200170, '__new__')
        # Calling __new__(args, kwargs) (line 309)
        new___call_result_200175 = invoke(stypy.reporting.localization.Localization(__file__, 309, 18), new___200171, *[TestProgram_200173], **kwargs_200174)
        
        # Assigning a type to the variable 'program' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'program', new___call_result_200175)
        
        # Call to _do_discovery(...): (line 310)
        # Processing the call arguments (line 310)
        
        # Obtaining an instance of the builtin type 'list' (line 310)
        list_200178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 310)
        # Adding element type (line 310)
        str_200179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 31), 'str', '-p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 30), list_200178, str_200179)
        # Adding element type (line 310)
        str_200180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 37), 'str', 'eggs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 30), list_200178, str_200180)
        # Adding element type (line 310)
        str_200181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 45), 'str', '-s')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 30), list_200178, str_200181)
        # Adding element type (line 310)
        str_200182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 51), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 30), list_200178, str_200182)
        # Adding element type (line 310)
        str_200183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 59), 'str', '-v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 30), list_200178, str_200183)
        # Adding element type (line 310)
        str_200184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 65), 'str', '-f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 30), list_200178, str_200184)
        # Adding element type (line 310)
        str_200185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 71), 'str', '-c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 30), list_200178, str_200185)
        
        # Processing the call keyword arguments (line 310)
        # Getting the type of 'Loader' (line 311)
        Loader_200186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 37), 'Loader', False)
        keyword_200187 = Loader_200186
        kwargs_200188 = {'Loader': keyword_200187}
        # Getting the type of 'program' (line 310)
        program_200176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'program', False)
        # Obtaining the member '_do_discovery' of a type (line 310)
        _do_discovery_200177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), program_200176, '_do_discovery')
        # Calling _do_discovery(args, kwargs) (line 310)
        _do_discovery_call_result_200189 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), _do_discovery_200177, *[list_200178], **kwargs_200188)
        
        
        # Call to assertEqual(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'program' (line 312)
        program_200192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 25), 'program', False)
        # Obtaining the member 'test' of a type (line 312)
        test_200193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 25), program_200192, 'test')
        str_200194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 39), 'str', 'tests')
        # Processing the call keyword arguments (line 312)
        kwargs_200195 = {}
        # Getting the type of 'self' (line 312)
        self_200190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 312)
        assertEqual_200191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), self_200190, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 312)
        assertEqual_call_result_200196 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), assertEqual_200191, *[test_200193, str_200194], **kwargs_200195)
        
        
        # Call to assertEqual(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'Loader' (line 313)
        Loader_200199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 25), 'Loader', False)
        # Obtaining the member 'args' of a type (line 313)
        args_200200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 25), Loader_200199, 'args')
        
        # Obtaining an instance of the builtin type 'list' (line 313)
        list_200201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 313)
        # Adding element type (line 313)
        
        # Obtaining an instance of the builtin type 'tuple' (line 313)
        tuple_200202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 313)
        # Adding element type (line 313)
        str_200203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 40), 'str', 'fish')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 40), tuple_200202, str_200203)
        # Adding element type (line 313)
        str_200204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 48), 'str', 'eggs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 40), tuple_200202, str_200204)
        # Adding element type (line 313)
        # Getting the type of 'None' (line 313)
        None_200205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 56), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 40), tuple_200202, None_200205)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 38), list_200201, tuple_200202)
        
        # Processing the call keyword arguments (line 313)
        kwargs_200206 = {}
        # Getting the type of 'self' (line 313)
        self_200197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 313)
        assertEqual_200198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), self_200197, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 313)
        assertEqual_call_result_200207 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), assertEqual_200198, *[args_200200, list_200201], **kwargs_200206)
        
        
        # Call to assertEqual(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'program' (line 314)
        program_200210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 25), 'program', False)
        # Obtaining the member 'verbosity' of a type (line 314)
        verbosity_200211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 25), program_200210, 'verbosity')
        int_200212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 44), 'int')
        # Processing the call keyword arguments (line 314)
        kwargs_200213 = {}
        # Getting the type of 'self' (line 314)
        self_200208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 314)
        assertEqual_200209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), self_200208, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 314)
        assertEqual_call_result_200214 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), assertEqual_200209, *[verbosity_200211, int_200212], **kwargs_200213)
        
        
        # Call to assertTrue(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'program' (line 315)
        program_200217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 24), 'program', False)
        # Obtaining the member 'failfast' of a type (line 315)
        failfast_200218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 24), program_200217, 'failfast')
        # Processing the call keyword arguments (line 315)
        kwargs_200219 = {}
        # Getting the type of 'self' (line 315)
        self_200215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 315)
        assertTrue_200216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_200215, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 315)
        assertTrue_call_result_200220 = invoke(stypy.reporting.localization.Localization(__file__, 315, 8), assertTrue_200216, *[failfast_200218], **kwargs_200219)
        
        
        # Call to assertTrue(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'program' (line 316)
        program_200223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 24), 'program', False)
        # Obtaining the member 'catchbreak' of a type (line 316)
        catchbreak_200224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 24), program_200223, 'catchbreak')
        # Processing the call keyword arguments (line 316)
        kwargs_200225 = {}
        # Getting the type of 'self' (line 316)
        self_200221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 316)
        assertTrue_200222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_200221, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 316)
        assertTrue_call_result_200226 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), assertTrue_200222, *[catchbreak_200224], **kwargs_200225)
        
        
        # ################# End of 'test_command_line_handling_do_discovery_calls_loader(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_command_line_handling_do_discovery_calls_loader' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_200227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_command_line_handling_do_discovery_calls_loader'
        return stypy_return_type_200227


    @norecursion
    def setup_module_clash(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_module_clash'
        module_type_store = module_type_store.open_function_context('setup_module_clash', 318, 4, False)
        # Assigning a type to the variable 'self' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.setup_module_clash')
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.setup_module_clash.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.setup_module_clash', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_module_clash', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_module_clash(...)' code ##################

        # Declaration of the 'Module' class

        class Module(object, ):
            
            # Assigning a Str to a Name (line 320):
            str_200228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 23), 'str', 'bar/foo.py')
            # Assigning a type to the variable '__file__' (line 320)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), '__file__', str_200228)
        
        # Assigning a type to the variable 'Module' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'Module', Module)
        
        # Assigning a Name to a Subscript (line 321):
        # Getting the type of 'Module' (line 321)
        Module_200229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 29), 'Module')
        # Getting the type of 'sys' (line 321)
        sys_200230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'sys')
        # Obtaining the member 'modules' of a type (line 321)
        modules_200231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), sys_200230, 'modules')
        str_200232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 20), 'str', 'foo')
        # Storing an element on a container (line 321)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 8), modules_200231, (str_200232, Module_200229))
        
        # Assigning a Call to a Name (line 322):
        
        # Call to abspath(...): (line 322)
        # Processing the call arguments (line 322)
        str_200236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 36), 'str', 'foo')
        # Processing the call keyword arguments (line 322)
        kwargs_200237 = {}
        # Getting the type of 'os' (line 322)
        os_200233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 322)
        path_200234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 20), os_200233, 'path')
        # Obtaining the member 'abspath' of a type (line 322)
        abspath_200235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 20), path_200234, 'abspath')
        # Calling abspath(args, kwargs) (line 322)
        abspath_call_result_200238 = invoke(stypy.reporting.localization.Localization(__file__, 322, 20), abspath_200235, *[str_200236], **kwargs_200237)
        
        # Assigning a type to the variable 'full_path' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'full_path', abspath_call_result_200238)
        
        # Assigning a Attribute to a Name (line 323):
        # Getting the type of 'os' (line 323)
        os_200239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 27), 'os')
        # Obtaining the member 'listdir' of a type (line 323)
        listdir_200240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 27), os_200239, 'listdir')
        # Assigning a type to the variable 'original_listdir' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'original_listdir', listdir_200240)
        
        # Assigning a Attribute to a Name (line 324):
        # Getting the type of 'os' (line 324)
        os_200241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 26), 'os')
        # Obtaining the member 'path' of a type (line 324)
        path_200242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 26), os_200241, 'path')
        # Obtaining the member 'isfile' of a type (line 324)
        isfile_200243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 26), path_200242, 'isfile')
        # Assigning a type to the variable 'original_isfile' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'original_isfile', isfile_200243)
        
        # Assigning a Attribute to a Name (line 325):
        # Getting the type of 'os' (line 325)
        os_200244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 25), 'os')
        # Obtaining the member 'path' of a type (line 325)
        path_200245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 25), os_200244, 'path')
        # Obtaining the member 'isdir' of a type (line 325)
        isdir_200246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 25), path_200245, 'isdir')
        # Assigning a type to the variable 'original_isdir' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'original_isdir', isdir_200246)

        @norecursion
        def cleanup(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup'
            module_type_store = module_type_store.open_function_context('cleanup', 327, 8, False)
            
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

            
            # Assigning a Name to a Attribute (line 328):
            # Getting the type of 'original_listdir' (line 328)
            original_listdir_200247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 25), 'original_listdir')
            # Getting the type of 'os' (line 328)
            os_200248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'os')
            # Setting the type of the member 'listdir' of a type (line 328)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), os_200248, 'listdir', original_listdir_200247)
            
            # Assigning a Name to a Attribute (line 329):
            # Getting the type of 'original_isfile' (line 329)
            original_isfile_200249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 29), 'original_isfile')
            # Getting the type of 'os' (line 329)
            os_200250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'os')
            # Obtaining the member 'path' of a type (line 329)
            path_200251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 12), os_200250, 'path')
            # Setting the type of the member 'isfile' of a type (line 329)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 12), path_200251, 'isfile', original_isfile_200249)
            
            # Assigning a Name to a Attribute (line 330):
            # Getting the type of 'original_isdir' (line 330)
            original_isdir_200252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 28), 'original_isdir')
            # Getting the type of 'os' (line 330)
            os_200253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'os')
            # Obtaining the member 'path' of a type (line 330)
            path_200254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), os_200253, 'path')
            # Setting the type of the member 'isdir' of a type (line 330)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), path_200254, 'isdir', original_isdir_200252)
            # Deleting a member
            # Getting the type of 'sys' (line 331)
            sys_200255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'sys')
            # Obtaining the member 'modules' of a type (line 331)
            modules_200256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 16), sys_200255, 'modules')
            
            # Obtaining the type of the subscript
            str_200257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 28), 'str', 'foo')
            # Getting the type of 'sys' (line 331)
            sys_200258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'sys')
            # Obtaining the member 'modules' of a type (line 331)
            modules_200259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 16), sys_200258, 'modules')
            # Obtaining the member '__getitem__' of a type (line 331)
            getitem___200260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 16), modules_200259, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 331)
            subscript_call_result_200261 = invoke(stypy.reporting.localization.Localization(__file__, 331, 16), getitem___200260, str_200257)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 12), modules_200256, subscript_call_result_200261)
            
            
            # Getting the type of 'full_path' (line 332)
            full_path_200262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'full_path')
            # Getting the type of 'sys' (line 332)
            sys_200263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 28), 'sys')
            # Obtaining the member 'path' of a type (line 332)
            path_200264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 28), sys_200263, 'path')
            # Applying the binary operator 'in' (line 332)
            result_contains_200265 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 15), 'in', full_path_200262, path_200264)
            
            # Testing the type of an if condition (line 332)
            if_condition_200266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 12), result_contains_200265)
            # Assigning a type to the variable 'if_condition_200266' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'if_condition_200266', if_condition_200266)
            # SSA begins for if statement (line 332)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 333)
            # Processing the call arguments (line 333)
            # Getting the type of 'full_path' (line 333)
            full_path_200270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 32), 'full_path', False)
            # Processing the call keyword arguments (line 333)
            kwargs_200271 = {}
            # Getting the type of 'sys' (line 333)
            sys_200267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'sys', False)
            # Obtaining the member 'path' of a type (line 333)
            path_200268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), sys_200267, 'path')
            # Obtaining the member 'remove' of a type (line 333)
            remove_200269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), path_200268, 'remove')
            # Calling remove(args, kwargs) (line 333)
            remove_call_result_200272 = invoke(stypy.reporting.localization.Localization(__file__, 333, 16), remove_200269, *[full_path_200270], **kwargs_200271)
            
            # SSA join for if statement (line 332)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'cleanup(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup' in the type store
            # Getting the type of 'stypy_return_type' (line 327)
            stypy_return_type_200273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200273)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup'
            return stypy_return_type_200273

        # Assigning a type to the variable 'cleanup' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'cleanup', cleanup)
        
        # Call to addCleanup(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'cleanup' (line 334)
        cleanup_200276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 24), 'cleanup', False)
        # Processing the call keyword arguments (line 334)
        kwargs_200277 = {}
        # Getting the type of 'self' (line 334)
        self_200274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 334)
        addCleanup_200275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), self_200274, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 334)
        addCleanup_call_result_200278 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), addCleanup_200275, *[cleanup_200276], **kwargs_200277)
        

        @norecursion
        def listdir(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'listdir'
            module_type_store = module_type_store.open_function_context('listdir', 336, 8, False)
            
            # Passed parameters checking function
            listdir.stypy_localization = localization
            listdir.stypy_type_of_self = None
            listdir.stypy_type_store = module_type_store
            listdir.stypy_function_name = 'listdir'
            listdir.stypy_param_names_list = ['_']
            listdir.stypy_varargs_param_name = None
            listdir.stypy_kwargs_param_name = None
            listdir.stypy_call_defaults = defaults
            listdir.stypy_call_varargs = varargs
            listdir.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'listdir', ['_'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'listdir', localization, ['_'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'listdir(...)' code ##################

            
            # Obtaining an instance of the builtin type 'list' (line 337)
            list_200279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 337)
            # Adding element type (line 337)
            str_200280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 20), 'str', 'foo.py')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 19), list_200279, str_200280)
            
            # Assigning a type to the variable 'stypy_return_type' (line 337)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'stypy_return_type', list_200279)
            
            # ################# End of 'listdir(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'listdir' in the type store
            # Getting the type of 'stypy_return_type' (line 336)
            stypy_return_type_200281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200281)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'listdir'
            return stypy_return_type_200281

        # Assigning a type to the variable 'listdir' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'listdir', listdir)

        @norecursion
        def isfile(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'isfile'
            module_type_store = module_type_store.open_function_context('isfile', 338, 8, False)
            
            # Passed parameters checking function
            isfile.stypy_localization = localization
            isfile.stypy_type_of_self = None
            isfile.stypy_type_store = module_type_store
            isfile.stypy_function_name = 'isfile'
            isfile.stypy_param_names_list = ['_']
            isfile.stypy_varargs_param_name = None
            isfile.stypy_kwargs_param_name = None
            isfile.stypy_call_defaults = defaults
            isfile.stypy_call_varargs = varargs
            isfile.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'isfile', ['_'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'isfile', localization, ['_'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'isfile(...)' code ##################

            # Getting the type of 'True' (line 339)
            True_200282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 339)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'stypy_return_type', True_200282)
            
            # ################# End of 'isfile(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'isfile' in the type store
            # Getting the type of 'stypy_return_type' (line 338)
            stypy_return_type_200283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200283)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'isfile'
            return stypy_return_type_200283

        # Assigning a type to the variable 'isfile' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'isfile', isfile)

        @norecursion
        def isdir(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'isdir'
            module_type_store = module_type_store.open_function_context('isdir', 340, 8, False)
            
            # Passed parameters checking function
            isdir.stypy_localization = localization
            isdir.stypy_type_of_self = None
            isdir.stypy_type_store = module_type_store
            isdir.stypy_function_name = 'isdir'
            isdir.stypy_param_names_list = ['_']
            isdir.stypy_varargs_param_name = None
            isdir.stypy_kwargs_param_name = None
            isdir.stypy_call_defaults = defaults
            isdir.stypy_call_varargs = varargs
            isdir.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'isdir', ['_'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'isdir', localization, ['_'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'isdir(...)' code ##################

            # Getting the type of 'True' (line 341)
            True_200284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'stypy_return_type', True_200284)
            
            # ################# End of 'isdir(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'isdir' in the type store
            # Getting the type of 'stypy_return_type' (line 340)
            stypy_return_type_200285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200285)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'isdir'
            return stypy_return_type_200285

        # Assigning a type to the variable 'isdir' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'isdir', isdir)
        
        # Assigning a Name to a Attribute (line 342):
        # Getting the type of 'listdir' (line 342)
        listdir_200286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 21), 'listdir')
        # Getting the type of 'os' (line 342)
        os_200287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'os')
        # Setting the type of the member 'listdir' of a type (line 342)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), os_200287, 'listdir', listdir_200286)
        
        # Assigning a Name to a Attribute (line 343):
        # Getting the type of 'isfile' (line 343)
        isfile_200288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'isfile')
        # Getting the type of 'os' (line 343)
        os_200289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'os')
        # Obtaining the member 'path' of a type (line 343)
        path_200290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), os_200289, 'path')
        # Setting the type of the member 'isfile' of a type (line 343)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), path_200290, 'isfile', isfile_200288)
        
        # Assigning a Name to a Attribute (line 344):
        # Getting the type of 'isdir' (line 344)
        isdir_200291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 24), 'isdir')
        # Getting the type of 'os' (line 344)
        os_200292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'os')
        # Obtaining the member 'path' of a type (line 344)
        path_200293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), os_200292, 'path')
        # Setting the type of the member 'isdir' of a type (line 344)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), path_200293, 'isdir', isdir_200291)
        # Getting the type of 'full_path' (line 345)
        full_path_200294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 15), 'full_path')
        # Assigning a type to the variable 'stypy_return_type' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'stypy_return_type', full_path_200294)
        
        # ################# End of 'setup_module_clash(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_module_clash' in the type store
        # Getting the type of 'stypy_return_type' (line 318)
        stypy_return_type_200295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200295)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_module_clash'
        return stypy_return_type_200295


    @norecursion
    def test_detect_module_clash(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_detect_module_clash'
        module_type_store = module_type_store.open_function_context('test_detect_module_clash', 347, 4, False)
        # Assigning a type to the variable 'self' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_detect_module_clash')
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_detect_module_clash.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_detect_module_clash', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_detect_module_clash', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_detect_module_clash(...)' code ##################

        
        # Assigning a Call to a Name (line 348):
        
        # Call to setup_module_clash(...): (line 348)
        # Processing the call keyword arguments (line 348)
        kwargs_200298 = {}
        # Getting the type of 'self' (line 348)
        self_200296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'self', False)
        # Obtaining the member 'setup_module_clash' of a type (line 348)
        setup_module_clash_200297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 20), self_200296, 'setup_module_clash')
        # Calling setup_module_clash(args, kwargs) (line 348)
        setup_module_clash_call_result_200299 = invoke(stypy.reporting.localization.Localization(__file__, 348, 20), setup_module_clash_200297, *[], **kwargs_200298)
        
        # Assigning a type to the variable 'full_path' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'full_path', setup_module_clash_call_result_200299)
        
        # Assigning a Call to a Name (line 349):
        
        # Call to TestLoader(...): (line 349)
        # Processing the call keyword arguments (line 349)
        kwargs_200302 = {}
        # Getting the type of 'unittest' (line 349)
        unittest_200300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 17), 'unittest', False)
        # Obtaining the member 'TestLoader' of a type (line 349)
        TestLoader_200301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 17), unittest_200300, 'TestLoader')
        # Calling TestLoader(args, kwargs) (line 349)
        TestLoader_call_result_200303 = invoke(stypy.reporting.localization.Localization(__file__, 349, 17), TestLoader_200301, *[], **kwargs_200302)
        
        # Assigning a type to the variable 'loader' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'loader', TestLoader_call_result_200303)
        
        # Assigning a Call to a Name (line 351):
        
        # Call to abspath(...): (line 351)
        # Processing the call arguments (line 351)
        str_200307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 34), 'str', 'bar')
        # Processing the call keyword arguments (line 351)
        kwargs_200308 = {}
        # Getting the type of 'os' (line 351)
        os_200304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 351)
        path_200305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 18), os_200304, 'path')
        # Obtaining the member 'abspath' of a type (line 351)
        abspath_200306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 18), path_200305, 'abspath')
        # Calling abspath(args, kwargs) (line 351)
        abspath_call_result_200309 = invoke(stypy.reporting.localization.Localization(__file__, 351, 18), abspath_200306, *[str_200307], **kwargs_200308)
        
        # Assigning a type to the variable 'mod_dir' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'mod_dir', abspath_call_result_200309)
        
        # Assigning a Call to a Name (line 352):
        
        # Call to abspath(...): (line 352)
        # Processing the call arguments (line 352)
        str_200313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 39), 'str', 'foo')
        # Processing the call keyword arguments (line 352)
        kwargs_200314 = {}
        # Getting the type of 'os' (line 352)
        os_200310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 352)
        path_200311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 23), os_200310, 'path')
        # Obtaining the member 'abspath' of a type (line 352)
        abspath_200312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 23), path_200311, 'abspath')
        # Calling abspath(args, kwargs) (line 352)
        abspath_call_result_200315 = invoke(stypy.reporting.localization.Localization(__file__, 352, 23), abspath_200312, *[str_200313], **kwargs_200314)
        
        # Assigning a type to the variable 'expected_dir' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'expected_dir', abspath_call_result_200315)
        
        # Assigning a Call to a Name (line 353):
        
        # Call to escape(...): (line 353)
        # Processing the call arguments (line 353)
        str_200318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 24), 'str', "'foo' module incorrectly imported from %r. Expected %r. Is this module globally installed?")
        
        # Obtaining an instance of the builtin type 'tuple' (line 354)
        tuple_200319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 354)
        # Adding element type (line 354)
        # Getting the type of 'mod_dir' (line 354)
        mod_dir_200320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 56), 'mod_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 56), tuple_200319, mod_dir_200320)
        # Adding element type (line 354)
        # Getting the type of 'expected_dir' (line 354)
        expected_dir_200321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 65), 'expected_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 56), tuple_200319, expected_dir_200321)
        
        # Applying the binary operator '%' (line 353)
        result_mod_200322 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 24), '%', str_200318, tuple_200319)
        
        # Processing the call keyword arguments (line 353)
        kwargs_200323 = {}
        # Getting the type of 're' (line 353)
        re_200316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 're', False)
        # Obtaining the member 'escape' of a type (line 353)
        escape_200317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 14), re_200316, 'escape')
        # Calling escape(args, kwargs) (line 353)
        escape_call_result_200324 = invoke(stypy.reporting.localization.Localization(__file__, 353, 14), escape_200317, *[result_mod_200322], **kwargs_200323)
        
        # Assigning a type to the variable 'msg' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'msg', escape_call_result_200324)
        
        # Call to assertRaisesRegexp(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'ImportError' (line 356)
        ImportError_200327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'ImportError', False)
        str_200328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 25), 'str', '^%s$')
        # Getting the type of 'msg' (line 356)
        msg_200329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 34), 'msg', False)
        # Applying the binary operator '%' (line 356)
        result_mod_200330 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 25), '%', str_200328, msg_200329)
        
        # Getting the type of 'loader' (line 356)
        loader_200331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 39), 'loader', False)
        # Obtaining the member 'discover' of a type (line 356)
        discover_200332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 39), loader_200331, 'discover')
        # Processing the call keyword arguments (line 355)
        str_200333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 22), 'str', 'foo')
        keyword_200334 = str_200333
        str_200335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 37), 'str', 'foo.py')
        keyword_200336 = str_200335
        kwargs_200337 = {'start_dir': keyword_200334, 'pattern': keyword_200336}
        # Getting the type of 'self' (line 355)
        self_200325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'self', False)
        # Obtaining the member 'assertRaisesRegexp' of a type (line 355)
        assertRaisesRegexp_200326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), self_200325, 'assertRaisesRegexp')
        # Calling assertRaisesRegexp(args, kwargs) (line 355)
        assertRaisesRegexp_call_result_200338 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), assertRaisesRegexp_200326, *[ImportError_200327, result_mod_200330, discover_200332], **kwargs_200337)
        
        
        # Call to assertEqual(...): (line 359)
        # Processing the call arguments (line 359)
        
        # Obtaining the type of the subscript
        int_200341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 34), 'int')
        # Getting the type of 'sys' (line 359)
        sys_200342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 25), 'sys', False)
        # Obtaining the member 'path' of a type (line 359)
        path_200343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 25), sys_200342, 'path')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___200344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 25), path_200343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_200345 = invoke(stypy.reporting.localization.Localization(__file__, 359, 25), getitem___200344, int_200341)
        
        # Getting the type of 'full_path' (line 359)
        full_path_200346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 38), 'full_path', False)
        # Processing the call keyword arguments (line 359)
        kwargs_200347 = {}
        # Getting the type of 'self' (line 359)
        self_200339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 359)
        assertEqual_200340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), self_200339, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 359)
        assertEqual_call_result_200348 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), assertEqual_200340, *[subscript_call_result_200345, full_path_200346], **kwargs_200347)
        
        
        # ################# End of 'test_detect_module_clash(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_detect_module_clash' in the type store
        # Getting the type of 'stypy_return_type' (line 347)
        stypy_return_type_200349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200349)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_detect_module_clash'
        return stypy_return_type_200349


    @norecursion
    def test_module_symlink_ok(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_module_symlink_ok'
        module_type_store = module_type_store.open_function_context('test_module_symlink_ok', 361, 4, False)
        # Assigning a type to the variable 'self' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_module_symlink_ok')
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_module_symlink_ok.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_module_symlink_ok', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_module_symlink_ok', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_module_symlink_ok(...)' code ##################

        
        # Assigning a Call to a Name (line 362):
        
        # Call to setup_module_clash(...): (line 362)
        # Processing the call keyword arguments (line 362)
        kwargs_200352 = {}
        # Getting the type of 'self' (line 362)
        self_200350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 20), 'self', False)
        # Obtaining the member 'setup_module_clash' of a type (line 362)
        setup_module_clash_200351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 20), self_200350, 'setup_module_clash')
        # Calling setup_module_clash(args, kwargs) (line 362)
        setup_module_clash_call_result_200353 = invoke(stypy.reporting.localization.Localization(__file__, 362, 20), setup_module_clash_200351, *[], **kwargs_200352)
        
        # Assigning a type to the variable 'full_path' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'full_path', setup_module_clash_call_result_200353)
        
        # Assigning a Attribute to a Name (line 364):
        # Getting the type of 'os' (line 364)
        os_200354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 28), 'os')
        # Obtaining the member 'path' of a type (line 364)
        path_200355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 28), os_200354, 'path')
        # Obtaining the member 'realpath' of a type (line 364)
        realpath_200356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 28), path_200355, 'realpath')
        # Assigning a type to the variable 'original_realpath' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'original_realpath', realpath_200356)
        
        # Assigning a Call to a Name (line 366):
        
        # Call to abspath(...): (line 366)
        # Processing the call arguments (line 366)
        str_200360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 34), 'str', 'bar')
        # Processing the call keyword arguments (line 366)
        kwargs_200361 = {}
        # Getting the type of 'os' (line 366)
        os_200357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 366)
        path_200358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 18), os_200357, 'path')
        # Obtaining the member 'abspath' of a type (line 366)
        abspath_200359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 18), path_200358, 'abspath')
        # Calling abspath(args, kwargs) (line 366)
        abspath_call_result_200362 = invoke(stypy.reporting.localization.Localization(__file__, 366, 18), abspath_200359, *[str_200360], **kwargs_200361)
        
        # Assigning a type to the variable 'mod_dir' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'mod_dir', abspath_call_result_200362)
        
        # Assigning a Call to a Name (line 367):
        
        # Call to abspath(...): (line 367)
        # Processing the call arguments (line 367)
        str_200366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 39), 'str', 'foo')
        # Processing the call keyword arguments (line 367)
        kwargs_200367 = {}
        # Getting the type of 'os' (line 367)
        os_200363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 367)
        path_200364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 23), os_200363, 'path')
        # Obtaining the member 'abspath' of a type (line 367)
        abspath_200365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 23), path_200364, 'abspath')
        # Calling abspath(args, kwargs) (line 367)
        abspath_call_result_200368 = invoke(stypy.reporting.localization.Localization(__file__, 367, 23), abspath_200365, *[str_200366], **kwargs_200367)
        
        # Assigning a type to the variable 'expected_dir' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'expected_dir', abspath_call_result_200368)

        @norecursion
        def cleanup(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup'
            module_type_store = module_type_store.open_function_context('cleanup', 369, 8, False)
            
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

            
            # Assigning a Name to a Attribute (line 370):
            # Getting the type of 'original_realpath' (line 370)
            original_realpath_200369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 31), 'original_realpath')
            # Getting the type of 'os' (line 370)
            os_200370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'os')
            # Obtaining the member 'path' of a type (line 370)
            path_200371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), os_200370, 'path')
            # Setting the type of the member 'realpath' of a type (line 370)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), path_200371, 'realpath', original_realpath_200369)
            
            # ################# End of 'cleanup(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup' in the type store
            # Getting the type of 'stypy_return_type' (line 369)
            stypy_return_type_200372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200372)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup'
            return stypy_return_type_200372

        # Assigning a type to the variable 'cleanup' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'cleanup', cleanup)
        
        # Call to addCleanup(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'cleanup' (line 371)
        cleanup_200375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'cleanup', False)
        # Processing the call keyword arguments (line 371)
        kwargs_200376 = {}
        # Getting the type of 'self' (line 371)
        self_200373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 371)
        addCleanup_200374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), self_200373, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 371)
        addCleanup_call_result_200377 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), addCleanup_200374, *[cleanup_200375], **kwargs_200376)
        

        @norecursion
        def realpath(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'realpath'
            module_type_store = module_type_store.open_function_context('realpath', 373, 8, False)
            
            # Passed parameters checking function
            realpath.stypy_localization = localization
            realpath.stypy_type_of_self = None
            realpath.stypy_type_store = module_type_store
            realpath.stypy_function_name = 'realpath'
            realpath.stypy_param_names_list = ['path']
            realpath.stypy_varargs_param_name = None
            realpath.stypy_kwargs_param_name = None
            realpath.stypy_call_defaults = defaults
            realpath.stypy_call_varargs = varargs
            realpath.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'realpath', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'realpath', localization, ['path'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'realpath(...)' code ##################

            
            
            # Getting the type of 'path' (line 374)
            path_200378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'path')
            
            # Call to join(...): (line 374)
            # Processing the call arguments (line 374)
            # Getting the type of 'mod_dir' (line 374)
            mod_dir_200382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 36), 'mod_dir', False)
            str_200383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 45), 'str', 'foo.py')
            # Processing the call keyword arguments (line 374)
            kwargs_200384 = {}
            # Getting the type of 'os' (line 374)
            os_200379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 374)
            path_200380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), os_200379, 'path')
            # Obtaining the member 'join' of a type (line 374)
            join_200381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), path_200380, 'join')
            # Calling join(args, kwargs) (line 374)
            join_call_result_200385 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), join_200381, *[mod_dir_200382, str_200383], **kwargs_200384)
            
            # Applying the binary operator '==' (line 374)
            result_eq_200386 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 15), '==', path_200378, join_call_result_200385)
            
            # Testing the type of an if condition (line 374)
            if_condition_200387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 12), result_eq_200386)
            # Assigning a type to the variable 'if_condition_200387' (line 374)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'if_condition_200387', if_condition_200387)
            # SSA begins for if statement (line 374)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to join(...): (line 375)
            # Processing the call arguments (line 375)
            # Getting the type of 'expected_dir' (line 375)
            expected_dir_200391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 36), 'expected_dir', False)
            str_200392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 50), 'str', 'foo.py')
            # Processing the call keyword arguments (line 375)
            kwargs_200393 = {}
            # Getting the type of 'os' (line 375)
            os_200388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 375)
            path_200389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 23), os_200388, 'path')
            # Obtaining the member 'join' of a type (line 375)
            join_200390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 23), path_200389, 'join')
            # Calling join(args, kwargs) (line 375)
            join_call_result_200394 = invoke(stypy.reporting.localization.Localization(__file__, 375, 23), join_200390, *[expected_dir_200391, str_200392], **kwargs_200393)
            
            # Assigning a type to the variable 'stypy_return_type' (line 375)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'stypy_return_type', join_call_result_200394)
            # SSA join for if statement (line 374)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'path' (line 376)
            path_200395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'path')
            # Assigning a type to the variable 'stypy_return_type' (line 376)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', path_200395)
            
            # ################# End of 'realpath(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'realpath' in the type store
            # Getting the type of 'stypy_return_type' (line 373)
            stypy_return_type_200396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200396)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'realpath'
            return stypy_return_type_200396

        # Assigning a type to the variable 'realpath' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'realpath', realpath)
        
        # Assigning a Name to a Attribute (line 377):
        # Getting the type of 'realpath' (line 377)
        realpath_200397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 27), 'realpath')
        # Getting the type of 'os' (line 377)
        os_200398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'os')
        # Obtaining the member 'path' of a type (line 377)
        path_200399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), os_200398, 'path')
        # Setting the type of the member 'realpath' of a type (line 377)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), path_200399, 'realpath', realpath_200397)
        
        # Assigning a Call to a Name (line 378):
        
        # Call to TestLoader(...): (line 378)
        # Processing the call keyword arguments (line 378)
        kwargs_200402 = {}
        # Getting the type of 'unittest' (line 378)
        unittest_200400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'unittest', False)
        # Obtaining the member 'TestLoader' of a type (line 378)
        TestLoader_200401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 17), unittest_200400, 'TestLoader')
        # Calling TestLoader(args, kwargs) (line 378)
        TestLoader_call_result_200403 = invoke(stypy.reporting.localization.Localization(__file__, 378, 17), TestLoader_200401, *[], **kwargs_200402)
        
        # Assigning a type to the variable 'loader' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'loader', TestLoader_call_result_200403)
        
        # Call to discover(...): (line 379)
        # Processing the call keyword arguments (line 379)
        str_200406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 34), 'str', 'foo')
        keyword_200407 = str_200406
        str_200408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 49), 'str', 'foo.py')
        keyword_200409 = str_200408
        kwargs_200410 = {'start_dir': keyword_200407, 'pattern': keyword_200409}
        # Getting the type of 'loader' (line 379)
        loader_200404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'loader', False)
        # Obtaining the member 'discover' of a type (line 379)
        discover_200405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), loader_200404, 'discover')
        # Calling discover(args, kwargs) (line 379)
        discover_call_result_200411 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), discover_200405, *[], **kwargs_200410)
        
        
        # ################# End of 'test_module_symlink_ok(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_module_symlink_ok' in the type store
        # Getting the type of 'stypy_return_type' (line 361)
        stypy_return_type_200412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200412)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_module_symlink_ok'
        return stypy_return_type_200412


    @norecursion
    def test_discovery_from_dotted_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_discovery_from_dotted_path'
        module_type_store = module_type_store.open_function_context('test_discovery_from_dotted_path', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_localization', localization)
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_function_name', 'TestDiscovery.test_discovery_from_dotted_path')
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiscovery.test_discovery_from_dotted_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.test_discovery_from_dotted_path', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 382):
        
        # Call to TestLoader(...): (line 382)
        # Processing the call keyword arguments (line 382)
        kwargs_200415 = {}
        # Getting the type of 'unittest' (line 382)
        unittest_200413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 17), 'unittest', False)
        # Obtaining the member 'TestLoader' of a type (line 382)
        TestLoader_200414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 17), unittest_200413, 'TestLoader')
        # Calling TestLoader(args, kwargs) (line 382)
        TestLoader_call_result_200416 = invoke(stypy.reporting.localization.Localization(__file__, 382, 17), TestLoader_200414, *[], **kwargs_200415)
        
        # Assigning a type to the variable 'loader' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'loader', TestLoader_call_result_200416)
        
        # Assigning a List to a Name (line 384):
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_200417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        # Getting the type of 'self' (line 384)
        self_200418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 17), 'self')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 16), list_200417, self_200418)
        
        # Assigning a type to the variable 'tests' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'tests', list_200417)
        
        # Assigning a Call to a Name (line 385):
        
        # Call to abspath(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Call to dirname(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'unittest' (line 385)
        unittest_200425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 55), 'unittest', False)
        # Obtaining the member 'test' of a type (line 385)
        test_200426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 55), unittest_200425, 'test')
        # Obtaining the member '__file__' of a type (line 385)
        file___200427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 55), test_200426, '__file__')
        # Processing the call keyword arguments (line 385)
        kwargs_200428 = {}
        # Getting the type of 'os' (line 385)
        os_200422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 385)
        path_200423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 39), os_200422, 'path')
        # Obtaining the member 'dirname' of a type (line 385)
        dirname_200424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 39), path_200423, 'dirname')
        # Calling dirname(args, kwargs) (line 385)
        dirname_call_result_200429 = invoke(stypy.reporting.localization.Localization(__file__, 385, 39), dirname_200424, *[file___200427], **kwargs_200428)
        
        # Processing the call keyword arguments (line 385)
        kwargs_200430 = {}
        # Getting the type of 'os' (line 385)
        os_200419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 385)
        path_200420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 23), os_200419, 'path')
        # Obtaining the member 'abspath' of a type (line 385)
        abspath_200421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 23), path_200420, 'abspath')
        # Calling abspath(args, kwargs) (line 385)
        abspath_call_result_200431 = invoke(stypy.reporting.localization.Localization(__file__, 385, 23), abspath_200421, *[dirname_call_result_200429], **kwargs_200430)
        
        # Assigning a type to the variable 'expectedPath' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'expectedPath', abspath_call_result_200431)
        
        # Assigning a Name to a Attribute (line 387):
        # Getting the type of 'False' (line 387)
        False_200432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 22), 'False')
        # Getting the type of 'self' (line 387)
        self_200433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'self')
        # Setting the type of the member 'wasRun' of a type (line 387)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 8), self_200433, 'wasRun', False_200432)

        @norecursion
        def _find_tests(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_find_tests'
            module_type_store = module_type_store.open_function_context('_find_tests', 388, 8, False)
            
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

            
            # Assigning a Name to a Attribute (line 389):
            # Getting the type of 'True' (line 389)
            True_200434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 26), 'True')
            # Getting the type of 'self' (line 389)
            self_200435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'self')
            # Setting the type of the member 'wasRun' of a type (line 389)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), self_200435, 'wasRun', True_200434)
            
            # Call to assertEqual(...): (line 390)
            # Processing the call arguments (line 390)
            # Getting the type of 'start_dir' (line 390)
            start_dir_200438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 'start_dir', False)
            # Getting the type of 'expectedPath' (line 390)
            expectedPath_200439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 40), 'expectedPath', False)
            # Processing the call keyword arguments (line 390)
            kwargs_200440 = {}
            # Getting the type of 'self' (line 390)
            self_200436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 390)
            assertEqual_200437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), self_200436, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 390)
            assertEqual_call_result_200441 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), assertEqual_200437, *[start_dir_200438, expectedPath_200439], **kwargs_200440)
            
            # Getting the type of 'tests' (line 391)
            tests_200442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 19), 'tests')
            # Assigning a type to the variable 'stypy_return_type' (line 391)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'stypy_return_type', tests_200442)
            
            # ################# End of '_find_tests(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_find_tests' in the type store
            # Getting the type of 'stypy_return_type' (line 388)
            stypy_return_type_200443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_200443)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_find_tests'
            return stypy_return_type_200443

        # Assigning a type to the variable '_find_tests' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), '_find_tests', _find_tests)
        
        # Assigning a Name to a Attribute (line 392):
        # Getting the type of '_find_tests' (line 392)
        _find_tests_200444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 29), '_find_tests')
        # Getting the type of 'loader' (line 392)
        loader_200445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'loader')
        # Setting the type of the member '_find_tests' of a type (line 392)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), loader_200445, '_find_tests', _find_tests_200444)
        
        # Assigning a Call to a Name (line 393):
        
        # Call to discover(...): (line 393)
        # Processing the call arguments (line 393)
        str_200448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 32), 'str', 'unittest.test')
        # Processing the call keyword arguments (line 393)
        kwargs_200449 = {}
        # Getting the type of 'loader' (line 393)
        loader_200446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'loader', False)
        # Obtaining the member 'discover' of a type (line 393)
        discover_200447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 16), loader_200446, 'discover')
        # Calling discover(args, kwargs) (line 393)
        discover_call_result_200450 = invoke(stypy.reporting.localization.Localization(__file__, 393, 16), discover_200447, *[str_200448], **kwargs_200449)
        
        # Assigning a type to the variable 'suite' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'suite', discover_call_result_200450)
        
        # Call to assertTrue(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'self' (line 394)
        self_200453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 24), 'self', False)
        # Obtaining the member 'wasRun' of a type (line 394)
        wasRun_200454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 24), self_200453, 'wasRun')
        # Processing the call keyword arguments (line 394)
        kwargs_200455 = {}
        # Getting the type of 'self' (line 394)
        self_200451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 394)
        assertTrue_200452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), self_200451, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 394)
        assertTrue_call_result_200456 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), assertTrue_200452, *[wasRun_200454], **kwargs_200455)
        
        
        # Call to assertEqual(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'suite' (line 395)
        suite_200459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 25), 'suite', False)
        # Obtaining the member '_tests' of a type (line 395)
        _tests_200460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 25), suite_200459, '_tests')
        # Getting the type of 'tests' (line 395)
        tests_200461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 39), 'tests', False)
        # Processing the call keyword arguments (line 395)
        kwargs_200462 = {}
        # Getting the type of 'self' (line 395)
        self_200457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 395)
        assertEqual_200458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), self_200457, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 395)
        assertEqual_call_result_200463 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), assertEqual_200458, *[_tests_200460, tests_200461], **kwargs_200462)
        
        
        # ################# End of 'test_discovery_from_dotted_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_discovery_from_dotted_path' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_200464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200464)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_discovery_from_dotted_path'
        return stypy_return_type_200464


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiscovery.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDiscovery' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'TestDiscovery', TestDiscovery)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 399)
    # Processing the call keyword arguments (line 399)
    kwargs_200467 = {}
    # Getting the type of 'unittest' (line 399)
    unittest_200465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 399)
    main_200466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 4), unittest_200465, 'main')
    # Calling main(args, kwargs) (line 399)
    main_call_result_200468 = invoke(stypy.reporting.localization.Localization(__file__, 399, 4), main_200466, *[], **kwargs_200467)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
