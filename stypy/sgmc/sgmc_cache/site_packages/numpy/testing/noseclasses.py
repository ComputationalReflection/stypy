
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # These classes implement a doctest runner plugin for nose, a "known failure"
2: # error class, and a customized TestProgram for NumPy.
3: 
4: # Because this module imports nose directly, it should not
5: # be used except by nosetester.py to avoid a general NumPy
6: # dependency on nose.
7: from __future__ import division, absolute_import, print_function
8: 
9: import os
10: import doctest
11: import inspect
12: 
13: import nose
14: from nose.plugins import doctests as npd
15: from nose.plugins.errorclass import ErrorClass, ErrorClassPlugin
16: from nose.plugins.base import Plugin
17: from nose.util import src
18: import numpy
19: from .nosetester import get_package_name
20: from .utils import KnownFailureException, KnownFailureTest
21: 
22: 
23: # Some of the classes in this module begin with 'Numpy' to clearly distinguish
24: # them from the plethora of very similar names from nose/unittest/doctest
25: 
26: #-----------------------------------------------------------------------------
27: # Modified version of the one in the stdlib, that fixes a python bug (doctests
28: # not found in extension modules, http://bugs.python.org/issue3158)
29: class NumpyDocTestFinder(doctest.DocTestFinder):
30: 
31:     def _from_module(self, module, object):
32:         '''
33:         Return true if the given object is defined in the given
34:         module.
35:         '''
36:         if module is None:
37:             return True
38:         elif inspect.isfunction(object):
39:             return module.__dict__ is object.__globals__
40:         elif inspect.isbuiltin(object):
41:             return module.__name__ == object.__module__
42:         elif inspect.isclass(object):
43:             return module.__name__ == object.__module__
44:         elif inspect.ismethod(object):
45:             # This one may be a bug in cython that fails to correctly set the
46:             # __module__ attribute of methods, but since the same error is easy
47:             # to make by extension code writers, having this safety in place
48:             # isn't such a bad idea
49:             return module.__name__ == object.__self__.__class__.__module__
50:         elif inspect.getmodule(object) is not None:
51:             return module is inspect.getmodule(object)
52:         elif hasattr(object, '__module__'):
53:             return module.__name__ == object.__module__
54:         elif isinstance(object, property):
55:             return True  # [XX] no way not be sure.
56:         else:
57:             raise ValueError("object must be a class or function")
58: 
59:     def _find(self, tests, obj, name, module, source_lines, globs, seen):
60:         '''
61:         Find tests for the given object and any contained objects, and
62:         add them to `tests`.
63:         '''
64: 
65:         doctest.DocTestFinder._find(self, tests, obj, name, module,
66:                                     source_lines, globs, seen)
67: 
68:         # Below we re-run pieces of the above method with manual modifications,
69:         # because the original code is buggy and fails to correctly identify
70:         # doctests in extension modules.
71: 
72:         # Local shorthands
73:         from inspect import (
74:             isroutine, isclass, ismodule, isfunction, ismethod
75:             )
76: 
77:         # Look for tests in a module's contained objects.
78:         if ismodule(obj) and self._recurse:
79:             for valname, val in obj.__dict__.items():
80:                 valname1 = '%s.%s' % (name, valname)
81:                 if ( (isroutine(val) or isclass(val))
82:                      and self._from_module(module, val)):
83: 
84:                     self._find(tests, val, valname1, module, source_lines,
85:                                globs, seen)
86: 
87:         # Look for tests in a class's contained objects.
88:         if isclass(obj) and self._recurse:
89:             for valname, val in obj.__dict__.items():
90:                 # Special handling for staticmethod/classmethod.
91:                 if isinstance(val, staticmethod):
92:                     val = getattr(obj, valname)
93:                 if isinstance(val, classmethod):
94:                     val = getattr(obj, valname).__func__
95: 
96:                 # Recurse to methods, properties, and nested classes.
97:                 if ((isfunction(val) or isclass(val) or
98:                      ismethod(val) or isinstance(val, property)) and
99:                       self._from_module(module, val)):
100:                     valname = '%s.%s' % (name, valname)
101:                     self._find(tests, val, valname, module, source_lines,
102:                                globs, seen)
103: 
104: 
105: # second-chance checker; if the default comparison doesn't
106: # pass, then see if the expected output string contains flags that
107: # tell us to ignore the output
108: class NumpyOutputChecker(doctest.OutputChecker):
109:     def check_output(self, want, got, optionflags):
110:         ret = doctest.OutputChecker.check_output(self, want, got,
111:                                                  optionflags)
112:         if not ret:
113:             if "#random" in want:
114:                 return True
115: 
116:             # it would be useful to normalize endianness so that
117:             # bigendian machines don't fail all the tests (and there are
118:             # actually some bigendian examples in the doctests). Let's try
119:             # making them all little endian
120:             got = got.replace("'>", "'<")
121:             want = want.replace("'>", "'<")
122: 
123:             # try to normalize out 32 and 64 bit default int sizes
124:             for sz in [4, 8]:
125:                 got = got.replace("'<i%d'" % sz, "int")
126:                 want = want.replace("'<i%d'" % sz, "int")
127: 
128:             ret = doctest.OutputChecker.check_output(self, want,
129:                     got, optionflags)
130: 
131:         return ret
132: 
133: 
134: # Subclass nose.plugins.doctests.DocTestCase to work around a bug in
135: # its constructor that blocks non-default arguments from being passed
136: # down into doctest.DocTestCase
137: class NumpyDocTestCase(npd.DocTestCase):
138:     def __init__(self, test, optionflags=0, setUp=None, tearDown=None,
139:                  checker=None, obj=None, result_var='_'):
140:         self._result_var = result_var
141:         self._nose_obj = obj
142:         doctest.DocTestCase.__init__(self, test,
143:                                      optionflags=optionflags,
144:                                      setUp=setUp, tearDown=tearDown,
145:                                      checker=checker)
146: 
147: 
148: print_state = numpy.get_printoptions()
149: 
150: class NumpyDoctest(npd.Doctest):
151:     name = 'numpydoctest'   # call nosetests with --with-numpydoctest
152:     score = 1000  # load late, after doctest builtin
153: 
154:     # always use whitespace and ellipsis options for doctests
155:     doctest_optflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
156: 
157:     # files that should be ignored for doctests
158:     doctest_ignore = ['generate_numpy_api.py',
159:                       'setup.py']
160: 
161:     # Custom classes; class variables to allow subclassing
162:     doctest_case_class = NumpyDocTestCase
163:     out_check_class = NumpyOutputChecker
164:     test_finder_class = NumpyDocTestFinder
165: 
166:     # Don't use the standard doctest option handler; hard-code the option values
167:     def options(self, parser, env=os.environ):
168:         Plugin.options(self, parser, env)
169:         # Test doctests in 'test' files / directories. Standard plugin default
170:         # is False
171:         self.doctest_tests = True
172:         # Variable name; if defined, doctest results stored in this variable in
173:         # the top-level namespace.  None is the standard default
174:         self.doctest_result_var = None
175: 
176:     def configure(self, options, config):
177:         # parent method sets enabled flag from command line --with-numpydoctest
178:         Plugin.configure(self, options, config)
179:         self.finder = self.test_finder_class()
180:         self.parser = doctest.DocTestParser()
181:         if self.enabled:
182:             # Pull standard doctest out of plugin list; there's no reason to run
183:             # both.  In practice the Unplugger plugin above would cover us when
184:             # run from a standard numpy.test() call; this is just in case
185:             # someone wants to run our plugin outside the numpy.test() machinery
186:             config.plugins.plugins = [p for p in config.plugins.plugins
187:                                       if p.name != 'doctest']
188: 
189:     def set_test_context(self, test):
190:         ''' Configure `test` object to set test context
191: 
192:         We set the numpy / scipy standard doctest namespace
193: 
194:         Parameters
195:         ----------
196:         test : test object
197:             with ``globs`` dictionary defining namespace
198: 
199:         Returns
200:         -------
201:         None
202: 
203:         Notes
204:         -----
205:         `test` object modified in place
206:         '''
207:         # set the namespace for tests
208:         pkg_name = get_package_name(os.path.dirname(test.filename))
209: 
210:         # Each doctest should execute in an environment equivalent to
211:         # starting Python and executing "import numpy as np", and,
212:         # for SciPy packages, an additional import of the local
213:         # package (so that scipy.linalg.basic.py's doctests have an
214:         # implicit "from scipy import linalg" as well.
215:         #
216:         # Note: __file__ allows the doctest in NoseTester to run
217:         # without producing an error
218:         test.globs = {'__builtins__':__builtins__,
219:                       '__file__':'__main__',
220:                       '__name__':'__main__',
221:                       'np':numpy}
222:         # add appropriate scipy import for SciPy tests
223:         if 'scipy' in pkg_name:
224:             p = pkg_name.split('.')
225:             p2 = p[-1]
226:             test.globs[p2] = __import__(pkg_name, test.globs, {}, [p2])
227: 
228:     # Override test loading to customize test context (with set_test_context
229:     # method), set standard docstring options, and install our own test output
230:     # checker
231:     def loadTestsFromModule(self, module):
232:         if not self.matches(module.__name__):
233:             npd.log.debug("Doctest doesn't want module %s", module)
234:             return
235:         try:
236:             tests = self.finder.find(module)
237:         except AttributeError:
238:             # nose allows module.__test__ = False; doctest does not and
239:             # throws AttributeError
240:             return
241:         if not tests:
242:             return
243:         tests.sort()
244:         module_file = src(module.__file__)
245:         for test in tests:
246:             if not test.examples:
247:                 continue
248:             if not test.filename:
249:                 test.filename = module_file
250:             # Set test namespace; test altered in place
251:             self.set_test_context(test)
252:             yield self.doctest_case_class(test,
253:                                           optionflags=self.doctest_optflags,
254:                                           checker=self.out_check_class(),
255:                                           result_var=self.doctest_result_var)
256: 
257:     # Add an afterContext method to nose.plugins.doctests.Doctest in order
258:     # to restore print options to the original state after each doctest
259:     def afterContext(self):
260:         numpy.set_printoptions(**print_state)
261: 
262:     # Ignore NumPy-specific build files that shouldn't be searched for tests
263:     def wantFile(self, file):
264:         bn = os.path.basename(file)
265:         if bn in self.doctest_ignore:
266:             return False
267:         return npd.Doctest.wantFile(self, file)
268: 
269: 
270: class Unplugger(object):
271:     ''' Nose plugin to remove named plugin late in loading
272: 
273:     By default it removes the "doctest" plugin.
274:     '''
275:     name = 'unplugger'
276:     enabled = True  # always enabled
277:     score = 4000  # load late in order to be after builtins
278: 
279:     def __init__(self, to_unplug='doctest'):
280:         self.to_unplug = to_unplug
281: 
282:     def options(self, parser, env):
283:         pass
284: 
285:     def configure(self, options, config):
286:         # Pull named plugin out of plugins list
287:         config.plugins.plugins = [p for p in config.plugins.plugins
288:                                   if p.name != self.to_unplug]
289: 
290: 
291: class KnownFailurePlugin(ErrorClassPlugin):
292:     '''Plugin that installs a KNOWNFAIL error class for the
293:     KnownFailureClass exception.  When KnownFailure is raised,
294:     the exception will be logged in the knownfail attribute of the
295:     result, 'K' or 'KNOWNFAIL' (verbose) will be output, and the
296:     exception will not be counted as an error or failure.'''
297:     enabled = True
298:     knownfail = ErrorClass(KnownFailureException,
299:                            label='KNOWNFAIL',
300:                            isfailure=False)
301: 
302:     def options(self, parser, env=os.environ):
303:         env_opt = 'NOSE_WITHOUT_KNOWNFAIL'
304:         parser.add_option('--no-knownfail', action='store_true',
305:                           dest='noKnownFail', default=env.get(env_opt, False),
306:                           help='Disable special handling of KnownFailure '
307:                                'exceptions')
308: 
309:     def configure(self, options, conf):
310:         if not self.can_configure:
311:             return
312:         self.conf = conf
313:         disable = getattr(options, 'noKnownFail', False)
314:         if disable:
315:             self.enabled = False
316: 
317: KnownFailure = KnownFailurePlugin   # backwards compat
318: 
319: 
320: # Class allows us to save the results of the tests in runTests - see runTests
321: # method docstring for details
322: class NumpyTestProgram(nose.core.TestProgram):
323:     def runTests(self):
324:         '''Run Tests. Returns true on success, false on failure, and
325:         sets self.success to the same value.
326: 
327:         Because nose currently discards the test result object, but we need
328:         to return it to the user, override TestProgram.runTests to retain
329:         the result
330:         '''
331:         if self.testRunner is None:
332:             self.testRunner = nose.core.TextTestRunner(stream=self.config.stream,
333:                                                        verbosity=self.config.verbosity,
334:                                                        config=self.config)
335:         plug_runner = self.config.plugins.prepareTestRunner(self.testRunner)
336:         if plug_runner is not None:
337:             self.testRunner = plug_runner
338:         self.result = self.testRunner.run(self.test)
339:         self.success = self.result.wasSuccessful()
340:         return self.success
341: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import doctest' statement (line 10)
import doctest

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'doctest', doctest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import inspect' statement (line 11)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import nose' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181068 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'nose')

if (type(import_181068) is not StypyTypeError):

    if (import_181068 != 'pyd_module'):
        __import__(import_181068)
        sys_modules_181069 = sys.modules[import_181068]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'nose', sys_modules_181069.module_type_store, module_type_store)
    else:
        import nose

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'nose', nose, module_type_store)

else:
    # Assigning a type to the variable 'nose' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'nose', import_181068)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from nose.plugins import npd' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181070 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'nose.plugins')

if (type(import_181070) is not StypyTypeError):

    if (import_181070 != 'pyd_module'):
        __import__(import_181070)
        sys_modules_181071 = sys.modules[import_181070]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'nose.plugins', sys_modules_181071.module_type_store, module_type_store, ['doctests'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_181071, sys_modules_181071.module_type_store, module_type_store)
    else:
        from nose.plugins import doctests as npd

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'nose.plugins', None, module_type_store, ['doctests'], [npd])

else:
    # Assigning a type to the variable 'nose.plugins' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'nose.plugins', import_181070)

# Adding an alias
module_type_store.add_alias('npd', 'doctests')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from nose.plugins.errorclass import ErrorClass, ErrorClassPlugin' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'nose.plugins.errorclass')

if (type(import_181072) is not StypyTypeError):

    if (import_181072 != 'pyd_module'):
        __import__(import_181072)
        sys_modules_181073 = sys.modules[import_181072]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'nose.plugins.errorclass', sys_modules_181073.module_type_store, module_type_store, ['ErrorClass', 'ErrorClassPlugin'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_181073, sys_modules_181073.module_type_store, module_type_store)
    else:
        from nose.plugins.errorclass import ErrorClass, ErrorClassPlugin

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'nose.plugins.errorclass', None, module_type_store, ['ErrorClass', 'ErrorClassPlugin'], [ErrorClass, ErrorClassPlugin])

else:
    # Assigning a type to the variable 'nose.plugins.errorclass' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'nose.plugins.errorclass', import_181072)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from nose.plugins.base import Plugin' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'nose.plugins.base')

if (type(import_181074) is not StypyTypeError):

    if (import_181074 != 'pyd_module'):
        __import__(import_181074)
        sys_modules_181075 = sys.modules[import_181074]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'nose.plugins.base', sys_modules_181075.module_type_store, module_type_store, ['Plugin'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_181075, sys_modules_181075.module_type_store, module_type_store)
    else:
        from nose.plugins.base import Plugin

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'nose.plugins.base', None, module_type_store, ['Plugin'], [Plugin])

else:
    # Assigning a type to the variable 'nose.plugins.base' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'nose.plugins.base', import_181074)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from nose.util import src' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181076 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'nose.util')

if (type(import_181076) is not StypyTypeError):

    if (import_181076 != 'pyd_module'):
        __import__(import_181076)
        sys_modules_181077 = sys.modules[import_181076]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'nose.util', sys_modules_181077.module_type_store, module_type_store, ['src'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_181077, sys_modules_181077.module_type_store, module_type_store)
    else:
        from nose.util import src

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'nose.util', None, module_type_store, ['src'], [src])

else:
    # Assigning a type to the variable 'nose.util' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'nose.util', import_181076)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import numpy' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181078 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy')

if (type(import_181078) is not StypyTypeError):

    if (import_181078 != 'pyd_module'):
        __import__(import_181078)
        sys_modules_181079 = sys.modules[import_181078]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy', sys_modules_181079.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy', import_181078)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from numpy.testing.nosetester import get_package_name' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181080 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.testing.nosetester')

if (type(import_181080) is not StypyTypeError):

    if (import_181080 != 'pyd_module'):
        __import__(import_181080)
        sys_modules_181081 = sys.modules[import_181080]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.testing.nosetester', sys_modules_181081.module_type_store, module_type_store, ['get_package_name'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_181081, sys_modules_181081.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import get_package_name

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.testing.nosetester', None, module_type_store, ['get_package_name'], [get_package_name])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.testing.nosetester', import_181080)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from numpy.testing.utils import KnownFailureException, KnownFailureTest' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181082 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.testing.utils')

if (type(import_181082) is not StypyTypeError):

    if (import_181082 != 'pyd_module'):
        __import__(import_181082)
        sys_modules_181083 = sys.modules[import_181082]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.testing.utils', sys_modules_181083.module_type_store, module_type_store, ['KnownFailureException', 'KnownFailureTest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_181083, sys_modules_181083.module_type_store, module_type_store)
    else:
        from numpy.testing.utils import KnownFailureException, KnownFailureTest

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.testing.utils', None, module_type_store, ['KnownFailureException', 'KnownFailureTest'], [KnownFailureException, KnownFailureTest])

else:
    # Assigning a type to the variable 'numpy.testing.utils' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.testing.utils', import_181082)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

# Declaration of the 'NumpyDocTestFinder' class
# Getting the type of 'doctest' (line 29)
doctest_181084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 25), 'doctest')
# Obtaining the member 'DocTestFinder' of a type (line 29)
DocTestFinder_181085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 25), doctest_181084, 'DocTestFinder')

class NumpyDocTestFinder(DocTestFinder_181085, ):

    @norecursion
    def _from_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_from_module'
        module_type_store = module_type_store.open_function_context('_from_module', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_localization', localization)
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_function_name', 'NumpyDocTestFinder._from_module')
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_param_names_list', ['module', 'object'])
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyDocTestFinder._from_module.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDocTestFinder._from_module', ['module', 'object'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_from_module', localization, ['module', 'object'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_from_module(...)' code ##################

        str_181086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', '\n        Return true if the given object is defined in the given\n        module.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 36)
        # Getting the type of 'module' (line 36)
        module_181087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'module')
        # Getting the type of 'None' (line 36)
        None_181088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'None')
        
        (may_be_181089, more_types_in_union_181090) = may_be_none(module_181087, None_181088)

        if may_be_181089:

            if more_types_in_union_181090:
                # Runtime conditional SSA (line 36)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'True' (line 37)
            True_181091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'stypy_return_type', True_181091)

            if more_types_in_union_181090:
                # Runtime conditional SSA for else branch (line 36)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_181089) or more_types_in_union_181090):
            
            
            # Call to isfunction(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'object' (line 38)
            object_181094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), 'object', False)
            # Processing the call keyword arguments (line 38)
            kwargs_181095 = {}
            # Getting the type of 'inspect' (line 38)
            inspect_181092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 38)
            isfunction_181093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), inspect_181092, 'isfunction')
            # Calling isfunction(args, kwargs) (line 38)
            isfunction_call_result_181096 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), isfunction_181093, *[object_181094], **kwargs_181095)
            
            # Testing the type of an if condition (line 38)
            if_condition_181097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 13), isfunction_call_result_181096)
            # Assigning a type to the variable 'if_condition_181097' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'if_condition_181097', if_condition_181097)
            # SSA begins for if statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'module' (line 39)
            module_181098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'module')
            # Obtaining the member '__dict__' of a type (line 39)
            dict___181099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 19), module_181098, '__dict__')
            # Getting the type of 'object' (line 39)
            object_181100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 38), 'object')
            # Obtaining the member '__globals__' of a type (line 39)
            globals___181101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 38), object_181100, '__globals__')
            # Applying the binary operator 'is' (line 39)
            result_is__181102 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 19), 'is', dict___181099, globals___181101)
            
            # Assigning a type to the variable 'stypy_return_type' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'stypy_return_type', result_is__181102)
            # SSA branch for the else part of an if statement (line 38)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isbuiltin(...): (line 40)
            # Processing the call arguments (line 40)
            # Getting the type of 'object' (line 40)
            object_181105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'object', False)
            # Processing the call keyword arguments (line 40)
            kwargs_181106 = {}
            # Getting the type of 'inspect' (line 40)
            inspect_181103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'inspect', False)
            # Obtaining the member 'isbuiltin' of a type (line 40)
            isbuiltin_181104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 13), inspect_181103, 'isbuiltin')
            # Calling isbuiltin(args, kwargs) (line 40)
            isbuiltin_call_result_181107 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), isbuiltin_181104, *[object_181105], **kwargs_181106)
            
            # Testing the type of an if condition (line 40)
            if_condition_181108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 13), isbuiltin_call_result_181107)
            # Assigning a type to the variable 'if_condition_181108' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'if_condition_181108', if_condition_181108)
            # SSA begins for if statement (line 40)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'module' (line 41)
            module_181109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'module')
            # Obtaining the member '__name__' of a type (line 41)
            name___181110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), module_181109, '__name__')
            # Getting the type of 'object' (line 41)
            object_181111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 38), 'object')
            # Obtaining the member '__module__' of a type (line 41)
            module___181112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 38), object_181111, '__module__')
            # Applying the binary operator '==' (line 41)
            result_eq_181113 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 19), '==', name___181110, module___181112)
            
            # Assigning a type to the variable 'stypy_return_type' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'stypy_return_type', result_eq_181113)
            # SSA branch for the else part of an if statement (line 40)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isclass(...): (line 42)
            # Processing the call arguments (line 42)
            # Getting the type of 'object' (line 42)
            object_181116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'object', False)
            # Processing the call keyword arguments (line 42)
            kwargs_181117 = {}
            # Getting the type of 'inspect' (line 42)
            inspect_181114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 42)
            isclass_181115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), inspect_181114, 'isclass')
            # Calling isclass(args, kwargs) (line 42)
            isclass_call_result_181118 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), isclass_181115, *[object_181116], **kwargs_181117)
            
            # Testing the type of an if condition (line 42)
            if_condition_181119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 13), isclass_call_result_181118)
            # Assigning a type to the variable 'if_condition_181119' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'if_condition_181119', if_condition_181119)
            # SSA begins for if statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'module' (line 43)
            module_181120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'module')
            # Obtaining the member '__name__' of a type (line 43)
            name___181121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 19), module_181120, '__name__')
            # Getting the type of 'object' (line 43)
            object_181122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 38), 'object')
            # Obtaining the member '__module__' of a type (line 43)
            module___181123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 38), object_181122, '__module__')
            # Applying the binary operator '==' (line 43)
            result_eq_181124 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 19), '==', name___181121, module___181123)
            
            # Assigning a type to the variable 'stypy_return_type' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'stypy_return_type', result_eq_181124)
            # SSA branch for the else part of an if statement (line 42)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to ismethod(...): (line 44)
            # Processing the call arguments (line 44)
            # Getting the type of 'object' (line 44)
            object_181127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'object', False)
            # Processing the call keyword arguments (line 44)
            kwargs_181128 = {}
            # Getting the type of 'inspect' (line 44)
            inspect_181125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'inspect', False)
            # Obtaining the member 'ismethod' of a type (line 44)
            ismethod_181126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), inspect_181125, 'ismethod')
            # Calling ismethod(args, kwargs) (line 44)
            ismethod_call_result_181129 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), ismethod_181126, *[object_181127], **kwargs_181128)
            
            # Testing the type of an if condition (line 44)
            if_condition_181130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 13), ismethod_call_result_181129)
            # Assigning a type to the variable 'if_condition_181130' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'if_condition_181130', if_condition_181130)
            # SSA begins for if statement (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'module' (line 49)
            module_181131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'module')
            # Obtaining the member '__name__' of a type (line 49)
            name___181132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), module_181131, '__name__')
            # Getting the type of 'object' (line 49)
            object_181133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'object')
            # Obtaining the member '__self__' of a type (line 49)
            self___181134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 38), object_181133, '__self__')
            # Obtaining the member '__class__' of a type (line 49)
            class___181135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 38), self___181134, '__class__')
            # Obtaining the member '__module__' of a type (line 49)
            module___181136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 38), class___181135, '__module__')
            # Applying the binary operator '==' (line 49)
            result_eq_181137 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 19), '==', name___181132, module___181136)
            
            # Assigning a type to the variable 'stypy_return_type' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'stypy_return_type', result_eq_181137)
            # SSA branch for the else part of an if statement (line 44)
            module_type_store.open_ssa_branch('else')
            
            
            
            # Call to getmodule(...): (line 50)
            # Processing the call arguments (line 50)
            # Getting the type of 'object' (line 50)
            object_181140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'object', False)
            # Processing the call keyword arguments (line 50)
            kwargs_181141 = {}
            # Getting the type of 'inspect' (line 50)
            inspect_181138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'inspect', False)
            # Obtaining the member 'getmodule' of a type (line 50)
            getmodule_181139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 13), inspect_181138, 'getmodule')
            # Calling getmodule(args, kwargs) (line 50)
            getmodule_call_result_181142 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), getmodule_181139, *[object_181140], **kwargs_181141)
            
            # Getting the type of 'None' (line 50)
            None_181143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 46), 'None')
            # Applying the binary operator 'isnot' (line 50)
            result_is_not_181144 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 13), 'isnot', getmodule_call_result_181142, None_181143)
            
            # Testing the type of an if condition (line 50)
            if_condition_181145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 13), result_is_not_181144)
            # Assigning a type to the variable 'if_condition_181145' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'if_condition_181145', if_condition_181145)
            # SSA begins for if statement (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'module' (line 51)
            module_181146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'module')
            
            # Call to getmodule(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'object' (line 51)
            object_181149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 47), 'object', False)
            # Processing the call keyword arguments (line 51)
            kwargs_181150 = {}
            # Getting the type of 'inspect' (line 51)
            inspect_181147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'inspect', False)
            # Obtaining the member 'getmodule' of a type (line 51)
            getmodule_181148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 29), inspect_181147, 'getmodule')
            # Calling getmodule(args, kwargs) (line 51)
            getmodule_call_result_181151 = invoke(stypy.reporting.localization.Localization(__file__, 51, 29), getmodule_181148, *[object_181149], **kwargs_181150)
            
            # Applying the binary operator 'is' (line 51)
            result_is__181152 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 19), 'is', module_181146, getmodule_call_result_181151)
            
            # Assigning a type to the variable 'stypy_return_type' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'stypy_return_type', result_is__181152)
            # SSA branch for the else part of an if statement (line 50)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 52)
            str_181153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'str', '__module__')
            # Getting the type of 'object' (line 52)
            object_181154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'object')
            
            (may_be_181155, more_types_in_union_181156) = may_provide_member(str_181153, object_181154)

            if may_be_181155:

                if more_types_in_union_181156:
                    # Runtime conditional SSA (line 52)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'object' (line 52)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 13), 'object', remove_not_member_provider_from_union(object_181154, '__module__'))
                
                # Getting the type of 'module' (line 53)
                module_181157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'module')
                # Obtaining the member '__name__' of a type (line 53)
                name___181158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 19), module_181157, '__name__')
                # Getting the type of 'object' (line 53)
                object_181159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 38), 'object')
                # Obtaining the member '__module__' of a type (line 53)
                module___181160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 38), object_181159, '__module__')
                # Applying the binary operator '==' (line 53)
                result_eq_181161 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 19), '==', name___181158, module___181160)
                
                # Assigning a type to the variable 'stypy_return_type' (line 53)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'stypy_return_type', result_eq_181161)

                if more_types_in_union_181156:
                    # Runtime conditional SSA for else branch (line 52)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_181155) or more_types_in_union_181156):
                # Assigning a type to the variable 'object' (line 52)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 13), 'object', remove_member_provider_from_union(object_181154, '__module__'))
                
                # Type idiom detected: calculating its left and rigth part (line 54)
                # Getting the type of 'property' (line 54)
                property_181162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 32), 'property')
                # Getting the type of 'object' (line 54)
                object_181163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'object')
                
                (may_be_181164, more_types_in_union_181165) = may_be_subtype(property_181162, object_181163)

                if may_be_181164:

                    if more_types_in_union_181165:
                        # Runtime conditional SSA (line 54)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'object' (line 54)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'object', remove_not_subtype_from_union(object_181163, property))
                    # Getting the type of 'True' (line 55)
                    True_181166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'True')
                    # Assigning a type to the variable 'stypy_return_type' (line 55)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stypy_return_type', True_181166)

                    if more_types_in_union_181165:
                        # Runtime conditional SSA for else branch (line 54)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_181164) or more_types_in_union_181165):
                    # Assigning a type to the variable 'object' (line 54)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'object', remove_subtype_from_union(object_181163, property))
                    
                    # Call to ValueError(...): (line 57)
                    # Processing the call arguments (line 57)
                    str_181168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 29), 'str', 'object must be a class or function')
                    # Processing the call keyword arguments (line 57)
                    kwargs_181169 = {}
                    # Getting the type of 'ValueError' (line 57)
                    ValueError_181167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'ValueError', False)
                    # Calling ValueError(args, kwargs) (line 57)
                    ValueError_call_result_181170 = invoke(stypy.reporting.localization.Localization(__file__, 57, 18), ValueError_181167, *[str_181168], **kwargs_181169)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 57, 12), ValueError_call_result_181170, 'raise parameter', BaseException)

                    if (may_be_181164 and more_types_in_union_181165):
                        # SSA join for if statement (line 54)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_181155 and more_types_in_union_181156):
                    # SSA join for if statement (line 52)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 40)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 38)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_181089 and more_types_in_union_181090):
                # SSA join for if statement (line 36)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_from_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_from_module' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_181171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181171)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_from_module'
        return stypy_return_type_181171


    @norecursion
    def _find(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_find'
        module_type_store = module_type_store.open_function_context('_find', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_localization', localization)
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_function_name', 'NumpyDocTestFinder._find')
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_param_names_list', ['tests', 'obj', 'name', 'module', 'source_lines', 'globs', 'seen'])
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyDocTestFinder._find.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDocTestFinder._find', ['tests', 'obj', 'name', 'module', 'source_lines', 'globs', 'seen'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_find', localization, ['tests', 'obj', 'name', 'module', 'source_lines', 'globs', 'seen'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_find(...)' code ##################

        str_181172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', '\n        Find tests for the given object and any contained objects, and\n        add them to `tests`.\n        ')
        
        # Call to _find(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_181176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'self', False)
        # Getting the type of 'tests' (line 65)
        tests_181177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'tests', False)
        # Getting the type of 'obj' (line 65)
        obj_181178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 49), 'obj', False)
        # Getting the type of 'name' (line 65)
        name_181179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 54), 'name', False)
        # Getting the type of 'module' (line 65)
        module_181180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 60), 'module', False)
        # Getting the type of 'source_lines' (line 66)
        source_lines_181181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 36), 'source_lines', False)
        # Getting the type of 'globs' (line 66)
        globs_181182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 50), 'globs', False)
        # Getting the type of 'seen' (line 66)
        seen_181183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 57), 'seen', False)
        # Processing the call keyword arguments (line 65)
        kwargs_181184 = {}
        # Getting the type of 'doctest' (line 65)
        doctest_181173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'doctest', False)
        # Obtaining the member 'DocTestFinder' of a type (line 65)
        DocTestFinder_181174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), doctest_181173, 'DocTestFinder')
        # Obtaining the member '_find' of a type (line 65)
        _find_181175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), DocTestFinder_181174, '_find')
        # Calling _find(args, kwargs) (line 65)
        _find_call_result_181185 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), _find_181175, *[self_181176, tests_181177, obj_181178, name_181179, module_181180, source_lines_181181, globs_181182, seen_181183], **kwargs_181184)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 73, 8))
        
        # 'from inspect import isroutine, isclass, ismodule, isfunction, ismethod' statement (line 73)
        from inspect import isroutine, isclass, ismodule, isfunction, ismethod

        import_from_module(stypy.reporting.localization.Localization(__file__, 73, 8), 'inspect', None, module_type_store, ['isroutine', 'isclass', 'ismodule', 'isfunction', 'ismethod'], [isroutine, isclass, ismodule, isfunction, ismethod])
        
        
        
        # Evaluating a boolean operation
        
        # Call to ismodule(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'obj' (line 78)
        obj_181187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'obj', False)
        # Processing the call keyword arguments (line 78)
        kwargs_181188 = {}
        # Getting the type of 'ismodule' (line 78)
        ismodule_181186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'ismodule', False)
        # Calling ismodule(args, kwargs) (line 78)
        ismodule_call_result_181189 = invoke(stypy.reporting.localization.Localization(__file__, 78, 11), ismodule_181186, *[obj_181187], **kwargs_181188)
        
        # Getting the type of 'self' (line 78)
        self_181190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 29), 'self')
        # Obtaining the member '_recurse' of a type (line 78)
        _recurse_181191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 29), self_181190, '_recurse')
        # Applying the binary operator 'and' (line 78)
        result_and_keyword_181192 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), 'and', ismodule_call_result_181189, _recurse_181191)
        
        # Testing the type of an if condition (line 78)
        if_condition_181193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_and_keyword_181192)
        # Assigning a type to the variable 'if_condition_181193' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_181193', if_condition_181193)
        # SSA begins for if statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to items(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_181197 = {}
        # Getting the type of 'obj' (line 79)
        obj_181194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'obj', False)
        # Obtaining the member '__dict__' of a type (line 79)
        dict___181195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 32), obj_181194, '__dict__')
        # Obtaining the member 'items' of a type (line 79)
        items_181196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 32), dict___181195, 'items')
        # Calling items(args, kwargs) (line 79)
        items_call_result_181198 = invoke(stypy.reporting.localization.Localization(__file__, 79, 32), items_181196, *[], **kwargs_181197)
        
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 12), items_call_result_181198)
        # Getting the type of the for loop variable (line 79)
        for_loop_var_181199 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 12), items_call_result_181198)
        # Assigning a type to the variable 'valname' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'valname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 12), for_loop_var_181199))
        # Assigning a type to the variable 'val' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 12), for_loop_var_181199))
        # SSA begins for a for statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 80):
        str_181200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 27), 'str', '%s.%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_181201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        # Getting the type of 'name' (line 80)
        name_181202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 38), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 38), tuple_181201, name_181202)
        # Adding element type (line 80)
        # Getting the type of 'valname' (line 80)
        valname_181203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'valname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 38), tuple_181201, valname_181203)
        
        # Applying the binary operator '%' (line 80)
        result_mod_181204 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 27), '%', str_181200, tuple_181201)
        
        # Assigning a type to the variable 'valname1' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'valname1', result_mod_181204)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Call to isroutine(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'val' (line 81)
        val_181206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 32), 'val', False)
        # Processing the call keyword arguments (line 81)
        kwargs_181207 = {}
        # Getting the type of 'isroutine' (line 81)
        isroutine_181205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'isroutine', False)
        # Calling isroutine(args, kwargs) (line 81)
        isroutine_call_result_181208 = invoke(stypy.reporting.localization.Localization(__file__, 81, 22), isroutine_181205, *[val_181206], **kwargs_181207)
        
        
        # Call to isclass(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'val' (line 81)
        val_181210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 48), 'val', False)
        # Processing the call keyword arguments (line 81)
        kwargs_181211 = {}
        # Getting the type of 'isclass' (line 81)
        isclass_181209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'isclass', False)
        # Calling isclass(args, kwargs) (line 81)
        isclass_call_result_181212 = invoke(stypy.reporting.localization.Localization(__file__, 81, 40), isclass_181209, *[val_181210], **kwargs_181211)
        
        # Applying the binary operator 'or' (line 81)
        result_or_keyword_181213 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 22), 'or', isroutine_call_result_181208, isclass_call_result_181212)
        
        
        # Call to _from_module(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'module' (line 82)
        module_181216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), 'module', False)
        # Getting the type of 'val' (line 82)
        val_181217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'val', False)
        # Processing the call keyword arguments (line 82)
        kwargs_181218 = {}
        # Getting the type of 'self' (line 82)
        self_181214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'self', False)
        # Obtaining the member '_from_module' of a type (line 82)
        _from_module_181215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 25), self_181214, '_from_module')
        # Calling _from_module(args, kwargs) (line 82)
        _from_module_call_result_181219 = invoke(stypy.reporting.localization.Localization(__file__, 82, 25), _from_module_181215, *[module_181216, val_181217], **kwargs_181218)
        
        # Applying the binary operator 'and' (line 81)
        result_and_keyword_181220 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 21), 'and', result_or_keyword_181213, _from_module_call_result_181219)
        
        # Testing the type of an if condition (line 81)
        if_condition_181221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 16), result_and_keyword_181220)
        # Assigning a type to the variable 'if_condition_181221' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'if_condition_181221', if_condition_181221)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _find(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'tests' (line 84)
        tests_181224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'tests', False)
        # Getting the type of 'val' (line 84)
        val_181225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 38), 'val', False)
        # Getting the type of 'valname1' (line 84)
        valname1_181226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 43), 'valname1', False)
        # Getting the type of 'module' (line 84)
        module_181227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 53), 'module', False)
        # Getting the type of 'source_lines' (line 84)
        source_lines_181228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 61), 'source_lines', False)
        # Getting the type of 'globs' (line 85)
        globs_181229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'globs', False)
        # Getting the type of 'seen' (line 85)
        seen_181230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 38), 'seen', False)
        # Processing the call keyword arguments (line 84)
        kwargs_181231 = {}
        # Getting the type of 'self' (line 84)
        self_181222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'self', False)
        # Obtaining the member '_find' of a type (line 84)
        _find_181223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), self_181222, '_find')
        # Calling _find(args, kwargs) (line 84)
        _find_call_result_181232 = invoke(stypy.reporting.localization.Localization(__file__, 84, 20), _find_181223, *[tests_181224, val_181225, valname1_181226, module_181227, source_lines_181228, globs_181229, seen_181230], **kwargs_181231)
        
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to isclass(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'obj' (line 88)
        obj_181234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'obj', False)
        # Processing the call keyword arguments (line 88)
        kwargs_181235 = {}
        # Getting the type of 'isclass' (line 88)
        isclass_181233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'isclass', False)
        # Calling isclass(args, kwargs) (line 88)
        isclass_call_result_181236 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), isclass_181233, *[obj_181234], **kwargs_181235)
        
        # Getting the type of 'self' (line 88)
        self_181237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 'self')
        # Obtaining the member '_recurse' of a type (line 88)
        _recurse_181238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 28), self_181237, '_recurse')
        # Applying the binary operator 'and' (line 88)
        result_and_keyword_181239 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 11), 'and', isclass_call_result_181236, _recurse_181238)
        
        # Testing the type of an if condition (line 88)
        if_condition_181240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 8), result_and_keyword_181239)
        # Assigning a type to the variable 'if_condition_181240' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'if_condition_181240', if_condition_181240)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to items(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_181244 = {}
        # Getting the type of 'obj' (line 89)
        obj_181241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 32), 'obj', False)
        # Obtaining the member '__dict__' of a type (line 89)
        dict___181242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 32), obj_181241, '__dict__')
        # Obtaining the member 'items' of a type (line 89)
        items_181243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 32), dict___181242, 'items')
        # Calling items(args, kwargs) (line 89)
        items_call_result_181245 = invoke(stypy.reporting.localization.Localization(__file__, 89, 32), items_181243, *[], **kwargs_181244)
        
        # Testing the type of a for loop iterable (line 89)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 12), items_call_result_181245)
        # Getting the type of the for loop variable (line 89)
        for_loop_var_181246 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 12), items_call_result_181245)
        # Assigning a type to the variable 'valname' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'valname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 12), for_loop_var_181246))
        # Assigning a type to the variable 'val' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 12), for_loop_var_181246))
        # SSA begins for a for statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 91)
        # Getting the type of 'staticmethod' (line 91)
        staticmethod_181247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 35), 'staticmethod')
        # Getting the type of 'val' (line 91)
        val_181248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'val')
        
        (may_be_181249, more_types_in_union_181250) = may_be_subtype(staticmethod_181247, val_181248)

        if may_be_181249:

            if more_types_in_union_181250:
                # Runtime conditional SSA (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'val' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'val', remove_not_subtype_from_union(val_181248, staticmethod))
            
            # Assigning a Call to a Name (line 92):
            
            # Call to getattr(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'obj' (line 92)
            obj_181252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 34), 'obj', False)
            # Getting the type of 'valname' (line 92)
            valname_181253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 39), 'valname', False)
            # Processing the call keyword arguments (line 92)
            kwargs_181254 = {}
            # Getting the type of 'getattr' (line 92)
            getattr_181251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 26), 'getattr', False)
            # Calling getattr(args, kwargs) (line 92)
            getattr_call_result_181255 = invoke(stypy.reporting.localization.Localization(__file__, 92, 26), getattr_181251, *[obj_181252, valname_181253], **kwargs_181254)
            
            # Assigning a type to the variable 'val' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'val', getattr_call_result_181255)

            if more_types_in_union_181250:
                # SSA join for if statement (line 91)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 93)
        # Getting the type of 'classmethod' (line 93)
        classmethod_181256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'classmethod')
        # Getting the type of 'val' (line 93)
        val_181257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'val')
        
        (may_be_181258, more_types_in_union_181259) = may_be_subtype(classmethod_181256, val_181257)

        if may_be_181258:

            if more_types_in_union_181259:
                # Runtime conditional SSA (line 93)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'val' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'val', remove_not_subtype_from_union(val_181257, classmethod))
            
            # Assigning a Attribute to a Name (line 94):
            
            # Call to getattr(...): (line 94)
            # Processing the call arguments (line 94)
            # Getting the type of 'obj' (line 94)
            obj_181261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 34), 'obj', False)
            # Getting the type of 'valname' (line 94)
            valname_181262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 39), 'valname', False)
            # Processing the call keyword arguments (line 94)
            kwargs_181263 = {}
            # Getting the type of 'getattr' (line 94)
            getattr_181260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'getattr', False)
            # Calling getattr(args, kwargs) (line 94)
            getattr_call_result_181264 = invoke(stypy.reporting.localization.Localization(__file__, 94, 26), getattr_181260, *[obj_181261, valname_181262], **kwargs_181263)
            
            # Obtaining the member '__func__' of a type (line 94)
            func___181265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 26), getattr_call_result_181264, '__func__')
            # Assigning a type to the variable 'val' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'val', func___181265)

            if more_types_in_union_181259:
                # SSA join for if statement (line 93)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Call to isfunction(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'val' (line 97)
        val_181267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'val', False)
        # Processing the call keyword arguments (line 97)
        kwargs_181268 = {}
        # Getting the type of 'isfunction' (line 97)
        isfunction_181266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'isfunction', False)
        # Calling isfunction(args, kwargs) (line 97)
        isfunction_call_result_181269 = invoke(stypy.reporting.localization.Localization(__file__, 97, 21), isfunction_181266, *[val_181267], **kwargs_181268)
        
        
        # Call to isclass(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'val' (line 97)
        val_181271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 48), 'val', False)
        # Processing the call keyword arguments (line 97)
        kwargs_181272 = {}
        # Getting the type of 'isclass' (line 97)
        isclass_181270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'isclass', False)
        # Calling isclass(args, kwargs) (line 97)
        isclass_call_result_181273 = invoke(stypy.reporting.localization.Localization(__file__, 97, 40), isclass_181270, *[val_181271], **kwargs_181272)
        
        # Applying the binary operator 'or' (line 97)
        result_or_keyword_181274 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 21), 'or', isfunction_call_result_181269, isclass_call_result_181273)
        
        # Call to ismethod(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'val' (line 98)
        val_181276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 'val', False)
        # Processing the call keyword arguments (line 98)
        kwargs_181277 = {}
        # Getting the type of 'ismethod' (line 98)
        ismethod_181275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'ismethod', False)
        # Calling ismethod(args, kwargs) (line 98)
        ismethod_call_result_181278 = invoke(stypy.reporting.localization.Localization(__file__, 98, 21), ismethod_181275, *[val_181276], **kwargs_181277)
        
        # Applying the binary operator 'or' (line 97)
        result_or_keyword_181279 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 21), 'or', result_or_keyword_181274, ismethod_call_result_181278)
        
        # Call to isinstance(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'val' (line 98)
        val_181281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 49), 'val', False)
        # Getting the type of 'property' (line 98)
        property_181282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 54), 'property', False)
        # Processing the call keyword arguments (line 98)
        kwargs_181283 = {}
        # Getting the type of 'isinstance' (line 98)
        isinstance_181280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 98)
        isinstance_call_result_181284 = invoke(stypy.reporting.localization.Localization(__file__, 98, 38), isinstance_181280, *[val_181281, property_181282], **kwargs_181283)
        
        # Applying the binary operator 'or' (line 97)
        result_or_keyword_181285 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 21), 'or', result_or_keyword_181279, isinstance_call_result_181284)
        
        
        # Call to _from_module(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'module' (line 99)
        module_181288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'module', False)
        # Getting the type of 'val' (line 99)
        val_181289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 48), 'val', False)
        # Processing the call keyword arguments (line 99)
        kwargs_181290 = {}
        # Getting the type of 'self' (line 99)
        self_181286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'self', False)
        # Obtaining the member '_from_module' of a type (line 99)
        _from_module_181287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 22), self_181286, '_from_module')
        # Calling _from_module(args, kwargs) (line 99)
        _from_module_call_result_181291 = invoke(stypy.reporting.localization.Localization(__file__, 99, 22), _from_module_181287, *[module_181288, val_181289], **kwargs_181290)
        
        # Applying the binary operator 'and' (line 97)
        result_and_keyword_181292 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 20), 'and', result_or_keyword_181285, _from_module_call_result_181291)
        
        # Testing the type of an if condition (line 97)
        if_condition_181293 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 16), result_and_keyword_181292)
        # Assigning a type to the variable 'if_condition_181293' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'if_condition_181293', if_condition_181293)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 100):
        str_181294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 30), 'str', '%s.%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 100)
        tuple_181295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 100)
        # Adding element type (line 100)
        # Getting the type of 'name' (line 100)
        name_181296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 41), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 41), tuple_181295, name_181296)
        # Adding element type (line 100)
        # Getting the type of 'valname' (line 100)
        valname_181297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 47), 'valname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 41), tuple_181295, valname_181297)
        
        # Applying the binary operator '%' (line 100)
        result_mod_181298 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 30), '%', str_181294, tuple_181295)
        
        # Assigning a type to the variable 'valname' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'valname', result_mod_181298)
        
        # Call to _find(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'tests' (line 101)
        tests_181301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'tests', False)
        # Getting the type of 'val' (line 101)
        val_181302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'val', False)
        # Getting the type of 'valname' (line 101)
        valname_181303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 43), 'valname', False)
        # Getting the type of 'module' (line 101)
        module_181304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 52), 'module', False)
        # Getting the type of 'source_lines' (line 101)
        source_lines_181305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'source_lines', False)
        # Getting the type of 'globs' (line 102)
        globs_181306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'globs', False)
        # Getting the type of 'seen' (line 102)
        seen_181307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 38), 'seen', False)
        # Processing the call keyword arguments (line 101)
        kwargs_181308 = {}
        # Getting the type of 'self' (line 101)
        self_181299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'self', False)
        # Obtaining the member '_find' of a type (line 101)
        _find_181300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 20), self_181299, '_find')
        # Calling _find(args, kwargs) (line 101)
        _find_call_result_181309 = invoke(stypy.reporting.localization.Localization(__file__, 101, 20), _find_181300, *[tests_181301, val_181302, valname_181303, module_181304, source_lines_181305, globs_181306, seen_181307], **kwargs_181308)
        
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_find(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_find' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_181310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181310)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_find'
        return stypy_return_type_181310


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDocTestFinder.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NumpyDocTestFinder' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'NumpyDocTestFinder', NumpyDocTestFinder)
# Declaration of the 'NumpyOutputChecker' class
# Getting the type of 'doctest' (line 108)
doctest_181311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'doctest')
# Obtaining the member 'OutputChecker' of a type (line 108)
OutputChecker_181312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 25), doctest_181311, 'OutputChecker')

class NumpyOutputChecker(OutputChecker_181312, ):

    @norecursion
    def check_output(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_output'
        module_type_store = module_type_store.open_function_context('check_output', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_localization', localization)
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_function_name', 'NumpyOutputChecker.check_output')
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_param_names_list', ['want', 'got', 'optionflags'])
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyOutputChecker.check_output.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyOutputChecker.check_output', ['want', 'got', 'optionflags'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_output', localization, ['want', 'got', 'optionflags'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_output(...)' code ##################

        
        # Assigning a Call to a Name (line 110):
        
        # Call to check_output(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'self' (line 110)
        self_181316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 49), 'self', False)
        # Getting the type of 'want' (line 110)
        want_181317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 55), 'want', False)
        # Getting the type of 'got' (line 110)
        got_181318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 61), 'got', False)
        # Getting the type of 'optionflags' (line 111)
        optionflags_181319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 49), 'optionflags', False)
        # Processing the call keyword arguments (line 110)
        kwargs_181320 = {}
        # Getting the type of 'doctest' (line 110)
        doctest_181313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'doctest', False)
        # Obtaining the member 'OutputChecker' of a type (line 110)
        OutputChecker_181314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 14), doctest_181313, 'OutputChecker')
        # Obtaining the member 'check_output' of a type (line 110)
        check_output_181315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 14), OutputChecker_181314, 'check_output')
        # Calling check_output(args, kwargs) (line 110)
        check_output_call_result_181321 = invoke(stypy.reporting.localization.Localization(__file__, 110, 14), check_output_181315, *[self_181316, want_181317, got_181318, optionflags_181319], **kwargs_181320)
        
        # Assigning a type to the variable 'ret' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'ret', check_output_call_result_181321)
        
        
        # Getting the type of 'ret' (line 112)
        ret_181322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'ret')
        # Applying the 'not' unary operator (line 112)
        result_not__181323 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), 'not', ret_181322)
        
        # Testing the type of an if condition (line 112)
        if_condition_181324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_not__181323)
        # Assigning a type to the variable 'if_condition_181324' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_181324', if_condition_181324)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        str_181325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 15), 'str', '#random')
        # Getting the type of 'want' (line 113)
        want_181326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'want')
        # Applying the binary operator 'in' (line 113)
        result_contains_181327 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), 'in', str_181325, want_181326)
        
        # Testing the type of an if condition (line 113)
        if_condition_181328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 12), result_contains_181327)
        # Assigning a type to the variable 'if_condition_181328' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'if_condition_181328', if_condition_181328)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 114)
        True_181329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'stypy_return_type', True_181329)
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 120):
        
        # Call to replace(...): (line 120)
        # Processing the call arguments (line 120)
        str_181332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 30), 'str', "'>")
        str_181333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 36), 'str', "'<")
        # Processing the call keyword arguments (line 120)
        kwargs_181334 = {}
        # Getting the type of 'got' (line 120)
        got_181330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'got', False)
        # Obtaining the member 'replace' of a type (line 120)
        replace_181331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 18), got_181330, 'replace')
        # Calling replace(args, kwargs) (line 120)
        replace_call_result_181335 = invoke(stypy.reporting.localization.Localization(__file__, 120, 18), replace_181331, *[str_181332, str_181333], **kwargs_181334)
        
        # Assigning a type to the variable 'got' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'got', replace_call_result_181335)
        
        # Assigning a Call to a Name (line 121):
        
        # Call to replace(...): (line 121)
        # Processing the call arguments (line 121)
        str_181338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 32), 'str', "'>")
        str_181339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 38), 'str', "'<")
        # Processing the call keyword arguments (line 121)
        kwargs_181340 = {}
        # Getting the type of 'want' (line 121)
        want_181336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'want', False)
        # Obtaining the member 'replace' of a type (line 121)
        replace_181337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 19), want_181336, 'replace')
        # Calling replace(args, kwargs) (line 121)
        replace_call_result_181341 = invoke(stypy.reporting.localization.Localization(__file__, 121, 19), replace_181337, *[str_181338, str_181339], **kwargs_181340)
        
        # Assigning a type to the variable 'want' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'want', replace_call_result_181341)
        
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_181342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        # Adding element type (line 124)
        int_181343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 22), list_181342, int_181343)
        # Adding element type (line 124)
        int_181344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 22), list_181342, int_181344)
        
        # Testing the type of a for loop iterable (line 124)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 12), list_181342)
        # Getting the type of the for loop variable (line 124)
        for_loop_var_181345 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 12), list_181342)
        # Assigning a type to the variable 'sz' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'sz', for_loop_var_181345)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 125):
        
        # Call to replace(...): (line 125)
        # Processing the call arguments (line 125)
        str_181348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 34), 'str', "'<i%d'")
        # Getting the type of 'sz' (line 125)
        sz_181349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 45), 'sz', False)
        # Applying the binary operator '%' (line 125)
        result_mod_181350 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 34), '%', str_181348, sz_181349)
        
        str_181351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 49), 'str', 'int')
        # Processing the call keyword arguments (line 125)
        kwargs_181352 = {}
        # Getting the type of 'got' (line 125)
        got_181346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'got', False)
        # Obtaining the member 'replace' of a type (line 125)
        replace_181347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 22), got_181346, 'replace')
        # Calling replace(args, kwargs) (line 125)
        replace_call_result_181353 = invoke(stypy.reporting.localization.Localization(__file__, 125, 22), replace_181347, *[result_mod_181350, str_181351], **kwargs_181352)
        
        # Assigning a type to the variable 'got' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'got', replace_call_result_181353)
        
        # Assigning a Call to a Name (line 126):
        
        # Call to replace(...): (line 126)
        # Processing the call arguments (line 126)
        str_181356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 36), 'str', "'<i%d'")
        # Getting the type of 'sz' (line 126)
        sz_181357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 47), 'sz', False)
        # Applying the binary operator '%' (line 126)
        result_mod_181358 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 36), '%', str_181356, sz_181357)
        
        str_181359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 51), 'str', 'int')
        # Processing the call keyword arguments (line 126)
        kwargs_181360 = {}
        # Getting the type of 'want' (line 126)
        want_181354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'want', False)
        # Obtaining the member 'replace' of a type (line 126)
        replace_181355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), want_181354, 'replace')
        # Calling replace(args, kwargs) (line 126)
        replace_call_result_181361 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), replace_181355, *[result_mod_181358, str_181359], **kwargs_181360)
        
        # Assigning a type to the variable 'want' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'want', replace_call_result_181361)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 128):
        
        # Call to check_output(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'self' (line 128)
        self_181365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 53), 'self', False)
        # Getting the type of 'want' (line 128)
        want_181366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'want', False)
        # Getting the type of 'got' (line 129)
        got_181367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'got', False)
        # Getting the type of 'optionflags' (line 129)
        optionflags_181368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'optionflags', False)
        # Processing the call keyword arguments (line 128)
        kwargs_181369 = {}
        # Getting the type of 'doctest' (line 128)
        doctest_181362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 18), 'doctest', False)
        # Obtaining the member 'OutputChecker' of a type (line 128)
        OutputChecker_181363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 18), doctest_181362, 'OutputChecker')
        # Obtaining the member 'check_output' of a type (line 128)
        check_output_181364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 18), OutputChecker_181363, 'check_output')
        # Calling check_output(args, kwargs) (line 128)
        check_output_call_result_181370 = invoke(stypy.reporting.localization.Localization(__file__, 128, 18), check_output_181364, *[self_181365, want_181366, got_181367, optionflags_181368], **kwargs_181369)
        
        # Assigning a type to the variable 'ret' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'ret', check_output_call_result_181370)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'ret' (line 131)
        ret_181371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', ret_181371)
        
        # ################# End of 'check_output(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_output' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_181372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181372)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_output'
        return stypy_return_type_181372


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 108, 0, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyOutputChecker.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NumpyOutputChecker' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'NumpyOutputChecker', NumpyOutputChecker)
# Declaration of the 'NumpyDocTestCase' class
# Getting the type of 'npd' (line 137)
npd_181373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'npd')
# Obtaining the member 'DocTestCase' of a type (line 137)
DocTestCase_181374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 23), npd_181373, 'DocTestCase')

class NumpyDocTestCase(DocTestCase_181374, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_181375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 41), 'int')
        # Getting the type of 'None' (line 138)
        None_181376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 50), 'None')
        # Getting the type of 'None' (line 138)
        None_181377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 65), 'None')
        # Getting the type of 'None' (line 139)
        None_181378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'None')
        # Getting the type of 'None' (line 139)
        None_181379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 'None')
        str_181380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 52), 'str', '_')
        defaults = [int_181375, None_181376, None_181377, None_181378, None_181379, str_181380]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDocTestCase.__init__', ['test', 'optionflags', 'setUp', 'tearDown', 'checker', 'obj', 'result_var'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['test', 'optionflags', 'setUp', 'tearDown', 'checker', 'obj', 'result_var'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 140):
        # Getting the type of 'result_var' (line 140)
        result_var_181381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'result_var')
        # Getting the type of 'self' (line 140)
        self_181382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self')
        # Setting the type of the member '_result_var' of a type (line 140)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_181382, '_result_var', result_var_181381)
        
        # Assigning a Name to a Attribute (line 141):
        # Getting the type of 'obj' (line 141)
        obj_181383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'obj')
        # Getting the type of 'self' (line 141)
        self_181384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'self')
        # Setting the type of the member '_nose_obj' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), self_181384, '_nose_obj', obj_181383)
        
        # Call to __init__(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_181388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 37), 'self', False)
        # Getting the type of 'test' (line 142)
        test_181389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 43), 'test', False)
        # Processing the call keyword arguments (line 142)
        # Getting the type of 'optionflags' (line 143)
        optionflags_181390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 49), 'optionflags', False)
        keyword_181391 = optionflags_181390
        # Getting the type of 'setUp' (line 144)
        setUp_181392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 43), 'setUp', False)
        keyword_181393 = setUp_181392
        # Getting the type of 'tearDown' (line 144)
        tearDown_181394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 59), 'tearDown', False)
        keyword_181395 = tearDown_181394
        # Getting the type of 'checker' (line 145)
        checker_181396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 45), 'checker', False)
        keyword_181397 = checker_181396
        kwargs_181398 = {'tearDown': keyword_181395, 'setUp': keyword_181393, 'optionflags': keyword_181391, 'checker': keyword_181397}
        # Getting the type of 'doctest' (line 142)
        doctest_181385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'doctest', False)
        # Obtaining the member 'DocTestCase' of a type (line 142)
        DocTestCase_181386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), doctest_181385, 'DocTestCase')
        # Obtaining the member '__init__' of a type (line 142)
        init___181387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), DocTestCase_181386, '__init__')
        # Calling __init__(args, kwargs) (line 142)
        init___call_result_181399 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), init___181387, *[self_181388, test_181389], **kwargs_181398)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'NumpyDocTestCase' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'NumpyDocTestCase', NumpyDocTestCase)

# Assigning a Call to a Name (line 148):

# Call to get_printoptions(...): (line 148)
# Processing the call keyword arguments (line 148)
kwargs_181402 = {}
# Getting the type of 'numpy' (line 148)
numpy_181400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 'numpy', False)
# Obtaining the member 'get_printoptions' of a type (line 148)
get_printoptions_181401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 14), numpy_181400, 'get_printoptions')
# Calling get_printoptions(args, kwargs) (line 148)
get_printoptions_call_result_181403 = invoke(stypy.reporting.localization.Localization(__file__, 148, 14), get_printoptions_181401, *[], **kwargs_181402)

# Assigning a type to the variable 'print_state' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'print_state', get_printoptions_call_result_181403)
# Declaration of the 'NumpyDoctest' class
# Getting the type of 'npd' (line 150)
npd_181404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'npd')
# Obtaining the member 'Doctest' of a type (line 150)
Doctest_181405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 19), npd_181404, 'Doctest')

class NumpyDoctest(Doctest_181405, ):

    @norecursion
    def options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'os' (line 167)
        os_181406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 34), 'os')
        # Obtaining the member 'environ' of a type (line 167)
        environ_181407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 34), os_181406, 'environ')
        defaults = [environ_181407]
        # Create a new context for function 'options'
        module_type_store = module_type_store.open_function_context('options', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyDoctest.options.__dict__.__setitem__('stypy_localization', localization)
        NumpyDoctest.options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyDoctest.options.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyDoctest.options.__dict__.__setitem__('stypy_function_name', 'NumpyDoctest.options')
        NumpyDoctest.options.__dict__.__setitem__('stypy_param_names_list', ['parser', 'env'])
        NumpyDoctest.options.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyDoctest.options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyDoctest.options.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyDoctest.options.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyDoctest.options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyDoctest.options.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDoctest.options', ['parser', 'env'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'options', localization, ['parser', 'env'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'options(...)' code ##################

        
        # Call to options(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'self' (line 168)
        self_181410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), 'self', False)
        # Getting the type of 'parser' (line 168)
        parser_181411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 29), 'parser', False)
        # Getting the type of 'env' (line 168)
        env_181412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'env', False)
        # Processing the call keyword arguments (line 168)
        kwargs_181413 = {}
        # Getting the type of 'Plugin' (line 168)
        Plugin_181408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'Plugin', False)
        # Obtaining the member 'options' of a type (line 168)
        options_181409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), Plugin_181408, 'options')
        # Calling options(args, kwargs) (line 168)
        options_call_result_181414 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), options_181409, *[self_181410, parser_181411, env_181412], **kwargs_181413)
        
        
        # Assigning a Name to a Attribute (line 171):
        # Getting the type of 'True' (line 171)
        True_181415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'True')
        # Getting the type of 'self' (line 171)
        self_181416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self')
        # Setting the type of the member 'doctest_tests' of a type (line 171)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_181416, 'doctest_tests', True_181415)
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'None' (line 174)
        None_181417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 34), 'None')
        # Getting the type of 'self' (line 174)
        self_181418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'doctest_result_var' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_181418, 'doctest_result_var', None_181417)
        
        # ################# End of 'options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'options' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_181419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'options'
        return stypy_return_type_181419


    @norecursion
    def configure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'configure'
        module_type_store = module_type_store.open_function_context('configure', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyDoctest.configure.__dict__.__setitem__('stypy_localization', localization)
        NumpyDoctest.configure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyDoctest.configure.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyDoctest.configure.__dict__.__setitem__('stypy_function_name', 'NumpyDoctest.configure')
        NumpyDoctest.configure.__dict__.__setitem__('stypy_param_names_list', ['options', 'config'])
        NumpyDoctest.configure.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyDoctest.configure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyDoctest.configure.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyDoctest.configure.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyDoctest.configure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyDoctest.configure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDoctest.configure', ['options', 'config'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'configure', localization, ['options', 'config'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'configure(...)' code ##################

        
        # Call to configure(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'self' (line 178)
        self_181422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'self', False)
        # Getting the type of 'options' (line 178)
        options_181423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 31), 'options', False)
        # Getting the type of 'config' (line 178)
        config_181424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 40), 'config', False)
        # Processing the call keyword arguments (line 178)
        kwargs_181425 = {}
        # Getting the type of 'Plugin' (line 178)
        Plugin_181420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'Plugin', False)
        # Obtaining the member 'configure' of a type (line 178)
        configure_181421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), Plugin_181420, 'configure')
        # Calling configure(args, kwargs) (line 178)
        configure_call_result_181426 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), configure_181421, *[self_181422, options_181423, config_181424], **kwargs_181425)
        
        
        # Assigning a Call to a Attribute (line 179):
        
        # Call to test_finder_class(...): (line 179)
        # Processing the call keyword arguments (line 179)
        kwargs_181429 = {}
        # Getting the type of 'self' (line 179)
        self_181427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'self', False)
        # Obtaining the member 'test_finder_class' of a type (line 179)
        test_finder_class_181428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 22), self_181427, 'test_finder_class')
        # Calling test_finder_class(args, kwargs) (line 179)
        test_finder_class_call_result_181430 = invoke(stypy.reporting.localization.Localization(__file__, 179, 22), test_finder_class_181428, *[], **kwargs_181429)
        
        # Getting the type of 'self' (line 179)
        self_181431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self')
        # Setting the type of the member 'finder' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_181431, 'finder', test_finder_class_call_result_181430)
        
        # Assigning a Call to a Attribute (line 180):
        
        # Call to DocTestParser(...): (line 180)
        # Processing the call keyword arguments (line 180)
        kwargs_181434 = {}
        # Getting the type of 'doctest' (line 180)
        doctest_181432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'doctest', False)
        # Obtaining the member 'DocTestParser' of a type (line 180)
        DocTestParser_181433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 22), doctest_181432, 'DocTestParser')
        # Calling DocTestParser(args, kwargs) (line 180)
        DocTestParser_call_result_181435 = invoke(stypy.reporting.localization.Localization(__file__, 180, 22), DocTestParser_181433, *[], **kwargs_181434)
        
        # Getting the type of 'self' (line 180)
        self_181436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self')
        # Setting the type of the member 'parser' of a type (line 180)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_181436, 'parser', DocTestParser_call_result_181435)
        
        # Getting the type of 'self' (line 181)
        self_181437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'self')
        # Obtaining the member 'enabled' of a type (line 181)
        enabled_181438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 11), self_181437, 'enabled')
        # Testing the type of an if condition (line 181)
        if_condition_181439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), enabled_181438)
        # Assigning a type to the variable 'if_condition_181439' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_181439', if_condition_181439)
        # SSA begins for if statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Attribute (line 186):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'config' (line 186)
        config_181445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 49), 'config')
        # Obtaining the member 'plugins' of a type (line 186)
        plugins_181446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 49), config_181445, 'plugins')
        # Obtaining the member 'plugins' of a type (line 186)
        plugins_181447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 49), plugins_181446, 'plugins')
        comprehension_181448 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 38), plugins_181447)
        # Assigning a type to the variable 'p' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 38), 'p', comprehension_181448)
        
        # Getting the type of 'p' (line 187)
        p_181441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 41), 'p')
        # Obtaining the member 'name' of a type (line 187)
        name_181442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 41), p_181441, 'name')
        str_181443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 51), 'str', 'doctest')
        # Applying the binary operator '!=' (line 187)
        result_ne_181444 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 41), '!=', name_181442, str_181443)
        
        # Getting the type of 'p' (line 186)
        p_181440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 38), 'p')
        list_181449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 38), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 38), list_181449, p_181440)
        # Getting the type of 'config' (line 186)
        config_181450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'config')
        # Obtaining the member 'plugins' of a type (line 186)
        plugins_181451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), config_181450, 'plugins')
        # Setting the type of the member 'plugins' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), plugins_181451, 'plugins', list_181449)
        # SSA join for if statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'configure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'configure' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_181452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181452)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'configure'
        return stypy_return_type_181452


    @norecursion
    def set_test_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_test_context'
        module_type_store = module_type_store.open_function_context('set_test_context', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_localization', localization)
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_function_name', 'NumpyDoctest.set_test_context')
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_param_names_list', ['test'])
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyDoctest.set_test_context.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDoctest.set_test_context', ['test'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_test_context', localization, ['test'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_test_context(...)' code ##################

        str_181453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, (-1)), 'str', ' Configure `test` object to set test context\n\n        We set the numpy / scipy standard doctest namespace\n\n        Parameters\n        ----------\n        test : test object\n            with ``globs`` dictionary defining namespace\n\n        Returns\n        -------\n        None\n\n        Notes\n        -----\n        `test` object modified in place\n        ')
        
        # Assigning a Call to a Name (line 208):
        
        # Call to get_package_name(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Call to dirname(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'test' (line 208)
        test_181458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 52), 'test', False)
        # Obtaining the member 'filename' of a type (line 208)
        filename_181459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 52), test_181458, 'filename')
        # Processing the call keyword arguments (line 208)
        kwargs_181460 = {}
        # Getting the type of 'os' (line 208)
        os_181455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 36), 'os', False)
        # Obtaining the member 'path' of a type (line 208)
        path_181456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 36), os_181455, 'path')
        # Obtaining the member 'dirname' of a type (line 208)
        dirname_181457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 36), path_181456, 'dirname')
        # Calling dirname(args, kwargs) (line 208)
        dirname_call_result_181461 = invoke(stypy.reporting.localization.Localization(__file__, 208, 36), dirname_181457, *[filename_181459], **kwargs_181460)
        
        # Processing the call keyword arguments (line 208)
        kwargs_181462 = {}
        # Getting the type of 'get_package_name' (line 208)
        get_package_name_181454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 19), 'get_package_name', False)
        # Calling get_package_name(args, kwargs) (line 208)
        get_package_name_call_result_181463 = invoke(stypy.reporting.localization.Localization(__file__, 208, 19), get_package_name_181454, *[dirname_call_result_181461], **kwargs_181462)
        
        # Assigning a type to the variable 'pkg_name' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'pkg_name', get_package_name_call_result_181463)
        
        # Assigning a Dict to a Attribute (line 218):
        
        # Obtaining an instance of the builtin type 'dict' (line 218)
        dict_181464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 218)
        # Adding element type (key, value) (line 218)
        str_181465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'str', '__builtins__')
        # Getting the type of '__builtins__' (line 218)
        builtins___181466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 37), '__builtins__')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), dict_181464, (str_181465, builtins___181466))
        # Adding element type (key, value) (line 218)
        str_181467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 22), 'str', '__file__')
        str_181468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 33), 'str', '__main__')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), dict_181464, (str_181467, str_181468))
        # Adding element type (key, value) (line 218)
        str_181469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 22), 'str', '__name__')
        str_181470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 33), 'str', '__main__')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), dict_181464, (str_181469, str_181470))
        # Adding element type (key, value) (line 218)
        str_181471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 22), 'str', 'np')
        # Getting the type of 'numpy' (line 221)
        numpy_181472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 27), 'numpy')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), dict_181464, (str_181471, numpy_181472))
        
        # Getting the type of 'test' (line 218)
        test_181473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'test')
        # Setting the type of the member 'globs' of a type (line 218)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), test_181473, 'globs', dict_181464)
        
        
        str_181474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 11), 'str', 'scipy')
        # Getting the type of 'pkg_name' (line 223)
        pkg_name_181475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 22), 'pkg_name')
        # Applying the binary operator 'in' (line 223)
        result_contains_181476 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 11), 'in', str_181474, pkg_name_181475)
        
        # Testing the type of an if condition (line 223)
        if_condition_181477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), result_contains_181476)
        # Assigning a type to the variable 'if_condition_181477' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_181477', if_condition_181477)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 224):
        
        # Call to split(...): (line 224)
        # Processing the call arguments (line 224)
        str_181480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 31), 'str', '.')
        # Processing the call keyword arguments (line 224)
        kwargs_181481 = {}
        # Getting the type of 'pkg_name' (line 224)
        pkg_name_181478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'pkg_name', False)
        # Obtaining the member 'split' of a type (line 224)
        split_181479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 16), pkg_name_181478, 'split')
        # Calling split(args, kwargs) (line 224)
        split_call_result_181482 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), split_181479, *[str_181480], **kwargs_181481)
        
        # Assigning a type to the variable 'p' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'p', split_call_result_181482)
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_181483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 19), 'int')
        # Getting the type of 'p' (line 225)
        p_181484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), 'p')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___181485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 17), p_181484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_181486 = invoke(stypy.reporting.localization.Localization(__file__, 225, 17), getitem___181485, int_181483)
        
        # Assigning a type to the variable 'p2' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'p2', subscript_call_result_181486)
        
        # Assigning a Call to a Subscript (line 226):
        
        # Call to __import__(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'pkg_name' (line 226)
        pkg_name_181488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'pkg_name', False)
        # Getting the type of 'test' (line 226)
        test_181489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 50), 'test', False)
        # Obtaining the member 'globs' of a type (line 226)
        globs_181490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 50), test_181489, 'globs')
        
        # Obtaining an instance of the builtin type 'dict' (line 226)
        dict_181491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 62), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 226)
        
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_181492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'p2' (line 226)
        p2_181493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 67), 'p2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 66), list_181492, p2_181493)
        
        # Processing the call keyword arguments (line 226)
        kwargs_181494 = {}
        # Getting the type of '__import__' (line 226)
        import___181487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), '__import__', False)
        # Calling __import__(args, kwargs) (line 226)
        import___call_result_181495 = invoke(stypy.reporting.localization.Localization(__file__, 226, 29), import___181487, *[pkg_name_181488, globs_181490, dict_181491, list_181492], **kwargs_181494)
        
        # Getting the type of 'test' (line 226)
        test_181496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'test')
        # Obtaining the member 'globs' of a type (line 226)
        globs_181497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), test_181496, 'globs')
        # Getting the type of 'p2' (line 226)
        p2_181498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'p2')
        # Storing an element on a container (line 226)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 12), globs_181497, (p2_181498, import___call_result_181495))
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_test_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_test_context' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_181499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181499)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_test_context'
        return stypy_return_type_181499


    @norecursion
    def loadTestsFromModule(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'loadTestsFromModule'
        module_type_store = module_type_store.open_function_context('loadTestsFromModule', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_localization', localization)
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_function_name', 'NumpyDoctest.loadTestsFromModule')
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_param_names_list', ['module'])
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyDoctest.loadTestsFromModule.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDoctest.loadTestsFromModule', ['module'], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Call to matches(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'module' (line 232)
        module_181502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'module', False)
        # Obtaining the member '__name__' of a type (line 232)
        name___181503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 28), module_181502, '__name__')
        # Processing the call keyword arguments (line 232)
        kwargs_181504 = {}
        # Getting the type of 'self' (line 232)
        self_181500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'self', False)
        # Obtaining the member 'matches' of a type (line 232)
        matches_181501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), self_181500, 'matches')
        # Calling matches(args, kwargs) (line 232)
        matches_call_result_181505 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), matches_181501, *[name___181503], **kwargs_181504)
        
        # Applying the 'not' unary operator (line 232)
        result_not__181506 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 11), 'not', matches_call_result_181505)
        
        # Testing the type of an if condition (line 232)
        if_condition_181507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), result_not__181506)
        # Assigning a type to the variable 'if_condition_181507' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_181507', if_condition_181507)
        # SSA begins for if statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug(...): (line 233)
        # Processing the call arguments (line 233)
        str_181511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 26), 'str', "Doctest doesn't want module %s")
        # Getting the type of 'module' (line 233)
        module_181512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 60), 'module', False)
        # Processing the call keyword arguments (line 233)
        kwargs_181513 = {}
        # Getting the type of 'npd' (line 233)
        npd_181508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'npd', False)
        # Obtaining the member 'log' of a type (line 233)
        log_181509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 12), npd_181508, 'log')
        # Obtaining the member 'debug' of a type (line 233)
        debug_181510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 12), log_181509, 'debug')
        # Calling debug(args, kwargs) (line 233)
        debug_call_result_181514 = invoke(stypy.reporting.localization.Localization(__file__, 233, 12), debug_181510, *[str_181511, module_181512], **kwargs_181513)
        
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 232)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 236):
        
        # Call to find(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'module' (line 236)
        module_181518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 37), 'module', False)
        # Processing the call keyword arguments (line 236)
        kwargs_181519 = {}
        # Getting the type of 'self' (line 236)
        self_181515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'self', False)
        # Obtaining the member 'finder' of a type (line 236)
        finder_181516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 20), self_181515, 'finder')
        # Obtaining the member 'find' of a type (line 236)
        find_181517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 20), finder_181516, 'find')
        # Calling find(args, kwargs) (line 236)
        find_call_result_181520 = invoke(stypy.reporting.localization.Localization(__file__, 236, 20), find_181517, *[module_181518], **kwargs_181519)
        
        # Assigning a type to the variable 'tests' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'tests', find_call_result_181520)
        # SSA branch for the except part of a try statement (line 235)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 235)
        module_type_store.open_ssa_branch('except')
        # Assigning a type to the variable 'stypy_return_type' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'tests' (line 241)
        tests_181521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'tests')
        # Applying the 'not' unary operator (line 241)
        result_not__181522 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 11), 'not', tests_181521)
        
        # Testing the type of an if condition (line 241)
        if_condition_181523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), result_not__181522)
        # Assigning a type to the variable 'if_condition_181523' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'if_condition_181523', if_condition_181523)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to sort(...): (line 243)
        # Processing the call keyword arguments (line 243)
        kwargs_181526 = {}
        # Getting the type of 'tests' (line 243)
        tests_181524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tests', False)
        # Obtaining the member 'sort' of a type (line 243)
        sort_181525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), tests_181524, 'sort')
        # Calling sort(args, kwargs) (line 243)
        sort_call_result_181527 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), sort_181525, *[], **kwargs_181526)
        
        
        # Assigning a Call to a Name (line 244):
        
        # Call to src(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'module' (line 244)
        module_181529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 26), 'module', False)
        # Obtaining the member '__file__' of a type (line 244)
        file___181530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 26), module_181529, '__file__')
        # Processing the call keyword arguments (line 244)
        kwargs_181531 = {}
        # Getting the type of 'src' (line 244)
        src_181528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 22), 'src', False)
        # Calling src(args, kwargs) (line 244)
        src_call_result_181532 = invoke(stypy.reporting.localization.Localization(__file__, 244, 22), src_181528, *[file___181530], **kwargs_181531)
        
        # Assigning a type to the variable 'module_file' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'module_file', src_call_result_181532)
        
        # Getting the type of 'tests' (line 245)
        tests_181533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'tests')
        # Testing the type of a for loop iterable (line 245)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 8), tests_181533)
        # Getting the type of the for loop variable (line 245)
        for_loop_var_181534 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 8), tests_181533)
        # Assigning a type to the variable 'test' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'test', for_loop_var_181534)
        # SSA begins for a for statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'test' (line 246)
        test_181535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'test')
        # Obtaining the member 'examples' of a type (line 246)
        examples_181536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), test_181535, 'examples')
        # Applying the 'not' unary operator (line 246)
        result_not__181537 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 15), 'not', examples_181536)
        
        # Testing the type of an if condition (line 246)
        if_condition_181538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 12), result_not__181537)
        # Assigning a type to the variable 'if_condition_181538' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'if_condition_181538', if_condition_181538)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'test' (line 248)
        test_181539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'test')
        # Obtaining the member 'filename' of a type (line 248)
        filename_181540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 19), test_181539, 'filename')
        # Applying the 'not' unary operator (line 248)
        result_not__181541 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 15), 'not', filename_181540)
        
        # Testing the type of an if condition (line 248)
        if_condition_181542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 12), result_not__181541)
        # Assigning a type to the variable 'if_condition_181542' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'if_condition_181542', if_condition_181542)
        # SSA begins for if statement (line 248)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'module_file' (line 249)
        module_file_181543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 32), 'module_file')
        # Getting the type of 'test' (line 249)
        test_181544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'test')
        # Setting the type of the member 'filename' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 16), test_181544, 'filename', module_file_181543)
        # SSA join for if statement (line 248)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_test_context(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'test' (line 251)
        test_181547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 34), 'test', False)
        # Processing the call keyword arguments (line 251)
        kwargs_181548 = {}
        # Getting the type of 'self' (line 251)
        self_181545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'self', False)
        # Obtaining the member 'set_test_context' of a type (line 251)
        set_test_context_181546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), self_181545, 'set_test_context')
        # Calling set_test_context(args, kwargs) (line 251)
        set_test_context_call_result_181549 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), set_test_context_181546, *[test_181547], **kwargs_181548)
        
        # Creating a generator
        
        # Call to doctest_case_class(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'test' (line 252)
        test_181552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 42), 'test', False)
        # Processing the call keyword arguments (line 252)
        # Getting the type of 'self' (line 253)
        self_181553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 54), 'self', False)
        # Obtaining the member 'doctest_optflags' of a type (line 253)
        doctest_optflags_181554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 54), self_181553, 'doctest_optflags')
        keyword_181555 = doctest_optflags_181554
        
        # Call to out_check_class(...): (line 254)
        # Processing the call keyword arguments (line 254)
        kwargs_181558 = {}
        # Getting the type of 'self' (line 254)
        self_181556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 50), 'self', False)
        # Obtaining the member 'out_check_class' of a type (line 254)
        out_check_class_181557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 50), self_181556, 'out_check_class')
        # Calling out_check_class(args, kwargs) (line 254)
        out_check_class_call_result_181559 = invoke(stypy.reporting.localization.Localization(__file__, 254, 50), out_check_class_181557, *[], **kwargs_181558)
        
        keyword_181560 = out_check_class_call_result_181559
        # Getting the type of 'self' (line 255)
        self_181561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 53), 'self', False)
        # Obtaining the member 'doctest_result_var' of a type (line 255)
        doctest_result_var_181562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 53), self_181561, 'doctest_result_var')
        keyword_181563 = doctest_result_var_181562
        kwargs_181564 = {'checker': keyword_181560, 'optionflags': keyword_181555, 'result_var': keyword_181563}
        # Getting the type of 'self' (line 252)
        self_181550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'self', False)
        # Obtaining the member 'doctest_case_class' of a type (line 252)
        doctest_case_class_181551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 18), self_181550, 'doctest_case_class')
        # Calling doctest_case_class(args, kwargs) (line 252)
        doctest_case_class_call_result_181565 = invoke(stypy.reporting.localization.Localization(__file__, 252, 18), doctest_case_class_181551, *[test_181552], **kwargs_181564)
        
        GeneratorType_181566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 12), GeneratorType_181566, doctest_case_class_call_result_181565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'stypy_return_type', GeneratorType_181566)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'loadTestsFromModule(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'loadTestsFromModule' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_181567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181567)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'loadTestsFromModule'
        return stypy_return_type_181567


    @norecursion
    def afterContext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'afterContext'
        module_type_store = module_type_store.open_function_context('afterContext', 259, 4, False)
        # Assigning a type to the variable 'self' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_localization', localization)
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_function_name', 'NumpyDoctest.afterContext')
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_param_names_list', [])
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyDoctest.afterContext.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDoctest.afterContext', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'afterContext', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'afterContext(...)' code ##################

        
        # Call to set_printoptions(...): (line 260)
        # Processing the call keyword arguments (line 260)
        # Getting the type of 'print_state' (line 260)
        print_state_181570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 33), 'print_state', False)
        kwargs_181571 = {'print_state_181570': print_state_181570}
        # Getting the type of 'numpy' (line 260)
        numpy_181568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'numpy', False)
        # Obtaining the member 'set_printoptions' of a type (line 260)
        set_printoptions_181569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), numpy_181568, 'set_printoptions')
        # Calling set_printoptions(args, kwargs) (line 260)
        set_printoptions_call_result_181572 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), set_printoptions_181569, *[], **kwargs_181571)
        
        
        # ################# End of 'afterContext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'afterContext' in the type store
        # Getting the type of 'stypy_return_type' (line 259)
        stypy_return_type_181573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181573)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'afterContext'
        return stypy_return_type_181573


    @norecursion
    def wantFile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wantFile'
        module_type_store = module_type_store.open_function_context('wantFile', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_localization', localization)
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_function_name', 'NumpyDoctest.wantFile')
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_param_names_list', ['file'])
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyDoctest.wantFile.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDoctest.wantFile', ['file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wantFile', localization, ['file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wantFile(...)' code ##################

        
        # Assigning a Call to a Name (line 264):
        
        # Call to basename(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'file' (line 264)
        file_181577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'file', False)
        # Processing the call keyword arguments (line 264)
        kwargs_181578 = {}
        # Getting the type of 'os' (line 264)
        os_181574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 264)
        path_181575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 13), os_181574, 'path')
        # Obtaining the member 'basename' of a type (line 264)
        basename_181576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 13), path_181575, 'basename')
        # Calling basename(args, kwargs) (line 264)
        basename_call_result_181579 = invoke(stypy.reporting.localization.Localization(__file__, 264, 13), basename_181576, *[file_181577], **kwargs_181578)
        
        # Assigning a type to the variable 'bn' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'bn', basename_call_result_181579)
        
        
        # Getting the type of 'bn' (line 265)
        bn_181580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'bn')
        # Getting the type of 'self' (line 265)
        self_181581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'self')
        # Obtaining the member 'doctest_ignore' of a type (line 265)
        doctest_ignore_181582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 17), self_181581, 'doctest_ignore')
        # Applying the binary operator 'in' (line 265)
        result_contains_181583 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'in', bn_181580, doctest_ignore_181582)
        
        # Testing the type of an if condition (line 265)
        if_condition_181584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), result_contains_181583)
        # Assigning a type to the variable 'if_condition_181584' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_181584', if_condition_181584)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 266)
        False_181585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'stypy_return_type', False_181585)
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to wantFile(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'self' (line 267)
        self_181589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 36), 'self', False)
        # Getting the type of 'file' (line 267)
        file_181590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 42), 'file', False)
        # Processing the call keyword arguments (line 267)
        kwargs_181591 = {}
        # Getting the type of 'npd' (line 267)
        npd_181586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 15), 'npd', False)
        # Obtaining the member 'Doctest' of a type (line 267)
        Doctest_181587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 15), npd_181586, 'Doctest')
        # Obtaining the member 'wantFile' of a type (line 267)
        wantFile_181588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 15), Doctest_181587, 'wantFile')
        # Calling wantFile(args, kwargs) (line 267)
        wantFile_call_result_181592 = invoke(stypy.reporting.localization.Localization(__file__, 267, 15), wantFile_181588, *[self_181589, file_181590], **kwargs_181591)
        
        # Assigning a type to the variable 'stypy_return_type' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'stypy_return_type', wantFile_call_result_181592)
        
        # ################# End of 'wantFile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wantFile' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_181593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181593)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wantFile'
        return stypy_return_type_181593


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 150, 0, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDoctest.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NumpyDoctest' (line 150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'NumpyDoctest', NumpyDoctest)

# Assigning a Str to a Name (line 151):
str_181594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 11), 'str', 'numpydoctest')
# Getting the type of 'NumpyDoctest'
NumpyDoctest_181595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NumpyDoctest')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NumpyDoctest_181595, 'name', str_181594)

# Assigning a Num to a Name (line 152):
int_181596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'int')
# Getting the type of 'NumpyDoctest'
NumpyDoctest_181597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NumpyDoctest')
# Setting the type of the member 'score' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NumpyDoctest_181597, 'score', int_181596)

# Assigning a BinOp to a Name (line 155):
# Getting the type of 'doctest' (line 155)
doctest_181598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 23), 'doctest')
# Obtaining the member 'NORMALIZE_WHITESPACE' of a type (line 155)
NORMALIZE_WHITESPACE_181599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 23), doctest_181598, 'NORMALIZE_WHITESPACE')
# Getting the type of 'doctest' (line 155)
doctest_181600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 54), 'doctest')
# Obtaining the member 'ELLIPSIS' of a type (line 155)
ELLIPSIS_181601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 54), doctest_181600, 'ELLIPSIS')
# Applying the binary operator '|' (line 155)
result_or__181602 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 23), '|', NORMALIZE_WHITESPACE_181599, ELLIPSIS_181601)

# Getting the type of 'NumpyDoctest'
NumpyDoctest_181603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NumpyDoctest')
# Setting the type of the member 'doctest_optflags' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NumpyDoctest_181603, 'doctest_optflags', result_or__181602)

# Assigning a List to a Name (line 158):

# Obtaining an instance of the builtin type 'list' (line 158)
list_181604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 158)
# Adding element type (line 158)
str_181605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 22), 'str', 'generate_numpy_api.py')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_181604, str_181605)
# Adding element type (line 158)
str_181606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'str', 'setup.py')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_181604, str_181606)

# Getting the type of 'NumpyDoctest'
NumpyDoctest_181607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NumpyDoctest')
# Setting the type of the member 'doctest_ignore' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NumpyDoctest_181607, 'doctest_ignore', list_181604)

# Assigning a Name to a Name (line 162):
# Getting the type of 'NumpyDocTestCase' (line 162)
NumpyDocTestCase_181608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'NumpyDocTestCase')
# Getting the type of 'NumpyDoctest'
NumpyDoctest_181609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NumpyDoctest')
# Setting the type of the member 'doctest_case_class' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NumpyDoctest_181609, 'doctest_case_class', NumpyDocTestCase_181608)

# Assigning a Name to a Name (line 163):
# Getting the type of 'NumpyOutputChecker' (line 163)
NumpyOutputChecker_181610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'NumpyOutputChecker')
# Getting the type of 'NumpyDoctest'
NumpyDoctest_181611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NumpyDoctest')
# Setting the type of the member 'out_check_class' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NumpyDoctest_181611, 'out_check_class', NumpyOutputChecker_181610)

# Assigning a Name to a Name (line 164):
# Getting the type of 'NumpyDocTestFinder' (line 164)
NumpyDocTestFinder_181612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'NumpyDocTestFinder')
# Getting the type of 'NumpyDoctest'
NumpyDoctest_181613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NumpyDoctest')
# Setting the type of the member 'test_finder_class' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NumpyDoctest_181613, 'test_finder_class', NumpyDocTestFinder_181612)
# Declaration of the 'Unplugger' class

class Unplugger(object, ):
    str_181614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, (-1)), 'str', ' Nose plugin to remove named plugin late in loading\n\n    By default it removes the "doctest" plugin.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_181615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 33), 'str', 'doctest')
        defaults = [str_181615]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Unplugger.__init__', ['to_unplug'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['to_unplug'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 280):
        # Getting the type of 'to_unplug' (line 280)
        to_unplug_181616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'to_unplug')
        # Getting the type of 'self' (line 280)
        self_181617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'self')
        # Setting the type of the member 'to_unplug' of a type (line 280)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), self_181617, 'to_unplug', to_unplug_181616)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'options'
        module_type_store = module_type_store.open_function_context('options', 282, 4, False)
        # Assigning a type to the variable 'self' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Unplugger.options.__dict__.__setitem__('stypy_localization', localization)
        Unplugger.options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Unplugger.options.__dict__.__setitem__('stypy_type_store', module_type_store)
        Unplugger.options.__dict__.__setitem__('stypy_function_name', 'Unplugger.options')
        Unplugger.options.__dict__.__setitem__('stypy_param_names_list', ['parser', 'env'])
        Unplugger.options.__dict__.__setitem__('stypy_varargs_param_name', None)
        Unplugger.options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Unplugger.options.__dict__.__setitem__('stypy_call_defaults', defaults)
        Unplugger.options.__dict__.__setitem__('stypy_call_varargs', varargs)
        Unplugger.options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Unplugger.options.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Unplugger.options', ['parser', 'env'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'options', localization, ['parser', 'env'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'options(...)' code ##################

        pass
        
        # ################# End of 'options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'options' in the type store
        # Getting the type of 'stypy_return_type' (line 282)
        stypy_return_type_181618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'options'
        return stypy_return_type_181618


    @norecursion
    def configure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'configure'
        module_type_store = module_type_store.open_function_context('configure', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Unplugger.configure.__dict__.__setitem__('stypy_localization', localization)
        Unplugger.configure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Unplugger.configure.__dict__.__setitem__('stypy_type_store', module_type_store)
        Unplugger.configure.__dict__.__setitem__('stypy_function_name', 'Unplugger.configure')
        Unplugger.configure.__dict__.__setitem__('stypy_param_names_list', ['options', 'config'])
        Unplugger.configure.__dict__.__setitem__('stypy_varargs_param_name', None)
        Unplugger.configure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Unplugger.configure.__dict__.__setitem__('stypy_call_defaults', defaults)
        Unplugger.configure.__dict__.__setitem__('stypy_call_varargs', varargs)
        Unplugger.configure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Unplugger.configure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Unplugger.configure', ['options', 'config'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'configure', localization, ['options', 'config'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'configure(...)' code ##################

        
        # Assigning a ListComp to a Attribute (line 287):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'config' (line 287)
        config_181625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 45), 'config')
        # Obtaining the member 'plugins' of a type (line 287)
        plugins_181626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 45), config_181625, 'plugins')
        # Obtaining the member 'plugins' of a type (line 287)
        plugins_181627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 45), plugins_181626, 'plugins')
        comprehension_181628 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 34), plugins_181627)
        # Assigning a type to the variable 'p' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 34), 'p', comprehension_181628)
        
        # Getting the type of 'p' (line 288)
        p_181620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 'p')
        # Obtaining the member 'name' of a type (line 288)
        name_181621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 37), p_181620, 'name')
        # Getting the type of 'self' (line 288)
        self_181622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 47), 'self')
        # Obtaining the member 'to_unplug' of a type (line 288)
        to_unplug_181623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 47), self_181622, 'to_unplug')
        # Applying the binary operator '!=' (line 288)
        result_ne_181624 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 37), '!=', name_181621, to_unplug_181623)
        
        # Getting the type of 'p' (line 287)
        p_181619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 34), 'p')
        list_181629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 34), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 34), list_181629, p_181619)
        # Getting the type of 'config' (line 287)
        config_181630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'config')
        # Obtaining the member 'plugins' of a type (line 287)
        plugins_181631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), config_181630, 'plugins')
        # Setting the type of the member 'plugins' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), plugins_181631, 'plugins', list_181629)
        
        # ################# End of 'configure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'configure' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_181632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181632)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'configure'
        return stypy_return_type_181632


# Assigning a type to the variable 'Unplugger' (line 270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'Unplugger', Unplugger)

# Assigning a Str to a Name (line 275):
str_181633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 11), 'str', 'unplugger')
# Getting the type of 'Unplugger'
Unplugger_181634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Unplugger')
# Setting the type of the member 'name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Unplugger_181634, 'name', str_181633)

# Assigning a Name to a Name (line 276):
# Getting the type of 'True' (line 276)
True_181635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 14), 'True')
# Getting the type of 'Unplugger'
Unplugger_181636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Unplugger')
# Setting the type of the member 'enabled' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Unplugger_181636, 'enabled', True_181635)

# Assigning a Num to a Name (line 277):
int_181637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 12), 'int')
# Getting the type of 'Unplugger'
Unplugger_181638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Unplugger')
# Setting the type of the member 'score' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Unplugger_181638, 'score', int_181637)
# Declaration of the 'KnownFailurePlugin' class
# Getting the type of 'ErrorClassPlugin' (line 291)
ErrorClassPlugin_181639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 25), 'ErrorClassPlugin')

class KnownFailurePlugin(ErrorClassPlugin_181639, ):
    str_181640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, (-1)), 'str', "Plugin that installs a KNOWNFAIL error class for the\n    KnownFailureClass exception.  When KnownFailure is raised,\n    the exception will be logged in the knownfail attribute of the\n    result, 'K' or 'KNOWNFAIL' (verbose) will be output, and the\n    exception will not be counted as an error or failure.")

    @norecursion
    def options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'os' (line 302)
        os_181641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 34), 'os')
        # Obtaining the member 'environ' of a type (line 302)
        environ_181642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 34), os_181641, 'environ')
        defaults = [environ_181642]
        # Create a new context for function 'options'
        module_type_store = module_type_store.open_function_context('options', 302, 4, False)
        # Assigning a type to the variable 'self' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_localization', localization)
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_type_store', module_type_store)
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_function_name', 'KnownFailurePlugin.options')
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_param_names_list', ['parser', 'env'])
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_varargs_param_name', None)
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_call_defaults', defaults)
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_call_varargs', varargs)
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KnownFailurePlugin.options.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KnownFailurePlugin.options', ['parser', 'env'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'options', localization, ['parser', 'env'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'options(...)' code ##################

        
        # Assigning a Str to a Name (line 303):
        str_181643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 18), 'str', 'NOSE_WITHOUT_KNOWNFAIL')
        # Assigning a type to the variable 'env_opt' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'env_opt', str_181643)
        
        # Call to add_option(...): (line 304)
        # Processing the call arguments (line 304)
        str_181646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 26), 'str', '--no-knownfail')
        # Processing the call keyword arguments (line 304)
        str_181647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 51), 'str', 'store_true')
        keyword_181648 = str_181647
        str_181649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 31), 'str', 'noKnownFail')
        keyword_181650 = str_181649
        
        # Call to get(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'env_opt' (line 305)
        env_opt_181653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 62), 'env_opt', False)
        # Getting the type of 'False' (line 305)
        False_181654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 71), 'False', False)
        # Processing the call keyword arguments (line 305)
        kwargs_181655 = {}
        # Getting the type of 'env' (line 305)
        env_181651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 54), 'env', False)
        # Obtaining the member 'get' of a type (line 305)
        get_181652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 54), env_181651, 'get')
        # Calling get(args, kwargs) (line 305)
        get_call_result_181656 = invoke(stypy.reporting.localization.Localization(__file__, 305, 54), get_181652, *[env_opt_181653, False_181654], **kwargs_181655)
        
        keyword_181657 = get_call_result_181656
        str_181658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 31), 'str', 'Disable special handling of KnownFailure exceptions')
        keyword_181659 = str_181658
        kwargs_181660 = {'action': keyword_181648, 'dest': keyword_181650, 'default': keyword_181657, 'help': keyword_181659}
        # Getting the type of 'parser' (line 304)
        parser_181644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'parser', False)
        # Obtaining the member 'add_option' of a type (line 304)
        add_option_181645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), parser_181644, 'add_option')
        # Calling add_option(args, kwargs) (line 304)
        add_option_call_result_181661 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), add_option_181645, *[str_181646], **kwargs_181660)
        
        
        # ################# End of 'options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'options' in the type store
        # Getting the type of 'stypy_return_type' (line 302)
        stypy_return_type_181662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181662)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'options'
        return stypy_return_type_181662


    @norecursion
    def configure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'configure'
        module_type_store = module_type_store.open_function_context('configure', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_localization', localization)
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_type_store', module_type_store)
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_function_name', 'KnownFailurePlugin.configure')
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_param_names_list', ['options', 'conf'])
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_varargs_param_name', None)
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_call_defaults', defaults)
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_call_varargs', varargs)
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KnownFailurePlugin.configure.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KnownFailurePlugin.configure', ['options', 'conf'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'configure', localization, ['options', 'conf'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'configure(...)' code ##################

        
        
        # Getting the type of 'self' (line 310)
        self_181663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'self')
        # Obtaining the member 'can_configure' of a type (line 310)
        can_configure_181664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 15), self_181663, 'can_configure')
        # Applying the 'not' unary operator (line 310)
        result_not__181665 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 11), 'not', can_configure_181664)
        
        # Testing the type of an if condition (line 310)
        if_condition_181666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 8), result_not__181665)
        # Assigning a type to the variable 'if_condition_181666' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'if_condition_181666', if_condition_181666)
        # SSA begins for if statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 312):
        # Getting the type of 'conf' (line 312)
        conf_181667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'conf')
        # Getting the type of 'self' (line 312)
        self_181668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self')
        # Setting the type of the member 'conf' of a type (line 312)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), self_181668, 'conf', conf_181667)
        
        # Assigning a Call to a Name (line 313):
        
        # Call to getattr(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'options' (line 313)
        options_181670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 26), 'options', False)
        str_181671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 35), 'str', 'noKnownFail')
        # Getting the type of 'False' (line 313)
        False_181672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 50), 'False', False)
        # Processing the call keyword arguments (line 313)
        kwargs_181673 = {}
        # Getting the type of 'getattr' (line 313)
        getattr_181669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 313)
        getattr_call_result_181674 = invoke(stypy.reporting.localization.Localization(__file__, 313, 18), getattr_181669, *[options_181670, str_181671, False_181672], **kwargs_181673)
        
        # Assigning a type to the variable 'disable' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'disable', getattr_call_result_181674)
        
        # Getting the type of 'disable' (line 314)
        disable_181675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'disable')
        # Testing the type of an if condition (line 314)
        if_condition_181676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 8), disable_181675)
        # Assigning a type to the variable 'if_condition_181676' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'if_condition_181676', if_condition_181676)
        # SSA begins for if statement (line 314)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 315):
        # Getting the type of 'False' (line 315)
        False_181677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 27), 'False')
        # Getting the type of 'self' (line 315)
        self_181678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'self')
        # Setting the type of the member 'enabled' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), self_181678, 'enabled', False_181677)
        # SSA join for if statement (line 314)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'configure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'configure' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_181679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'configure'
        return stypy_return_type_181679


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 291, 0, False)
        # Assigning a type to the variable 'self' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KnownFailurePlugin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'KnownFailurePlugin' (line 291)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), 'KnownFailurePlugin', KnownFailurePlugin)

# Assigning a Name to a Name (line 297):
# Getting the type of 'True' (line 297)
True_181680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 14), 'True')
# Getting the type of 'KnownFailurePlugin'
KnownFailurePlugin_181681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'KnownFailurePlugin')
# Setting the type of the member 'enabled' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), KnownFailurePlugin_181681, 'enabled', True_181680)

# Assigning a Call to a Name (line 298):

# Call to ErrorClass(...): (line 298)
# Processing the call arguments (line 298)
# Getting the type of 'KnownFailureException' (line 298)
KnownFailureException_181683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 27), 'KnownFailureException', False)
# Processing the call keyword arguments (line 298)
str_181684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 33), 'str', 'KNOWNFAIL')
keyword_181685 = str_181684
# Getting the type of 'False' (line 300)
False_181686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 37), 'False', False)
keyword_181687 = False_181686
kwargs_181688 = {'isfailure': keyword_181687, 'label': keyword_181685}
# Getting the type of 'ErrorClass' (line 298)
ErrorClass_181682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'ErrorClass', False)
# Calling ErrorClass(args, kwargs) (line 298)
ErrorClass_call_result_181689 = invoke(stypy.reporting.localization.Localization(__file__, 298, 16), ErrorClass_181682, *[KnownFailureException_181683], **kwargs_181688)

# Getting the type of 'KnownFailurePlugin'
KnownFailurePlugin_181690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'KnownFailurePlugin')
# Setting the type of the member 'knownfail' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), KnownFailurePlugin_181690, 'knownfail', ErrorClass_call_result_181689)

# Assigning a Name to a Name (line 317):
# Getting the type of 'KnownFailurePlugin' (line 317)
KnownFailurePlugin_181691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'KnownFailurePlugin')
# Assigning a type to the variable 'KnownFailure' (line 317)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'KnownFailure', KnownFailurePlugin_181691)
# Declaration of the 'NumpyTestProgram' class
# Getting the type of 'nose' (line 322)
nose_181692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 23), 'nose')
# Obtaining the member 'core' of a type (line 322)
core_181693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 23), nose_181692, 'core')
# Obtaining the member 'TestProgram' of a type (line 322)
TestProgram_181694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 23), core_181693, 'TestProgram')

class NumpyTestProgram(TestProgram_181694, ):

    @norecursion
    def runTests(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runTests'
        module_type_store = module_type_store.open_function_context('runTests', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_localization', localization)
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_function_name', 'NumpyTestProgram.runTests')
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_param_names_list', [])
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyTestProgram.runTests.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyTestProgram.runTests', [], None, None, defaults, varargs, kwargs)

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

        str_181695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, (-1)), 'str', 'Run Tests. Returns true on success, false on failure, and\n        sets self.success to the same value.\n\n        Because nose currently discards the test result object, but we need\n        to return it to the user, override TestProgram.runTests to retain\n        the result\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 331)
        # Getting the type of 'self' (line 331)
        self_181696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'self')
        # Obtaining the member 'testRunner' of a type (line 331)
        testRunner_181697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 11), self_181696, 'testRunner')
        # Getting the type of 'None' (line 331)
        None_181698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 30), 'None')
        
        (may_be_181699, more_types_in_union_181700) = may_be_none(testRunner_181697, None_181698)

        if may_be_181699:

            if more_types_in_union_181700:
                # Runtime conditional SSA (line 331)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 332):
            
            # Call to TextTestRunner(...): (line 332)
            # Processing the call keyword arguments (line 332)
            # Getting the type of 'self' (line 332)
            self_181704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 62), 'self', False)
            # Obtaining the member 'config' of a type (line 332)
            config_181705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 62), self_181704, 'config')
            # Obtaining the member 'stream' of a type (line 332)
            stream_181706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 62), config_181705, 'stream')
            keyword_181707 = stream_181706
            # Getting the type of 'self' (line 333)
            self_181708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 65), 'self', False)
            # Obtaining the member 'config' of a type (line 333)
            config_181709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 65), self_181708, 'config')
            # Obtaining the member 'verbosity' of a type (line 333)
            verbosity_181710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 65), config_181709, 'verbosity')
            keyword_181711 = verbosity_181710
            # Getting the type of 'self' (line 334)
            self_181712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 62), 'self', False)
            # Obtaining the member 'config' of a type (line 334)
            config_181713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 62), self_181712, 'config')
            keyword_181714 = config_181713
            kwargs_181715 = {'verbosity': keyword_181711, 'config': keyword_181714, 'stream': keyword_181707}
            # Getting the type of 'nose' (line 332)
            nose_181701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 30), 'nose', False)
            # Obtaining the member 'core' of a type (line 332)
            core_181702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 30), nose_181701, 'core')
            # Obtaining the member 'TextTestRunner' of a type (line 332)
            TextTestRunner_181703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 30), core_181702, 'TextTestRunner')
            # Calling TextTestRunner(args, kwargs) (line 332)
            TextTestRunner_call_result_181716 = invoke(stypy.reporting.localization.Localization(__file__, 332, 30), TextTestRunner_181703, *[], **kwargs_181715)
            
            # Getting the type of 'self' (line 332)
            self_181717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'self')
            # Setting the type of the member 'testRunner' of a type (line 332)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 12), self_181717, 'testRunner', TextTestRunner_call_result_181716)

            if more_types_in_union_181700:
                # SSA join for if statement (line 331)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 335):
        
        # Call to prepareTestRunner(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'self' (line 335)
        self_181722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 60), 'self', False)
        # Obtaining the member 'testRunner' of a type (line 335)
        testRunner_181723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 60), self_181722, 'testRunner')
        # Processing the call keyword arguments (line 335)
        kwargs_181724 = {}
        # Getting the type of 'self' (line 335)
        self_181718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'self', False)
        # Obtaining the member 'config' of a type (line 335)
        config_181719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 22), self_181718, 'config')
        # Obtaining the member 'plugins' of a type (line 335)
        plugins_181720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 22), config_181719, 'plugins')
        # Obtaining the member 'prepareTestRunner' of a type (line 335)
        prepareTestRunner_181721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 22), plugins_181720, 'prepareTestRunner')
        # Calling prepareTestRunner(args, kwargs) (line 335)
        prepareTestRunner_call_result_181725 = invoke(stypy.reporting.localization.Localization(__file__, 335, 22), prepareTestRunner_181721, *[testRunner_181723], **kwargs_181724)
        
        # Assigning a type to the variable 'plug_runner' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'plug_runner', prepareTestRunner_call_result_181725)
        
        # Type idiom detected: calculating its left and rigth part (line 336)
        # Getting the type of 'plug_runner' (line 336)
        plug_runner_181726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'plug_runner')
        # Getting the type of 'None' (line 336)
        None_181727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 30), 'None')
        
        (may_be_181728, more_types_in_union_181729) = may_not_be_none(plug_runner_181726, None_181727)

        if may_be_181728:

            if more_types_in_union_181729:
                # Runtime conditional SSA (line 336)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 337):
            # Getting the type of 'plug_runner' (line 337)
            plug_runner_181730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 30), 'plug_runner')
            # Getting the type of 'self' (line 337)
            self_181731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'self')
            # Setting the type of the member 'testRunner' of a type (line 337)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), self_181731, 'testRunner', plug_runner_181730)

            if more_types_in_union_181729:
                # SSA join for if statement (line 336)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 338):
        
        # Call to run(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'self' (line 338)
        self_181735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 42), 'self', False)
        # Obtaining the member 'test' of a type (line 338)
        test_181736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 42), self_181735, 'test')
        # Processing the call keyword arguments (line 338)
        kwargs_181737 = {}
        # Getting the type of 'self' (line 338)
        self_181732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 22), 'self', False)
        # Obtaining the member 'testRunner' of a type (line 338)
        testRunner_181733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 22), self_181732, 'testRunner')
        # Obtaining the member 'run' of a type (line 338)
        run_181734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 22), testRunner_181733, 'run')
        # Calling run(args, kwargs) (line 338)
        run_call_result_181738 = invoke(stypy.reporting.localization.Localization(__file__, 338, 22), run_181734, *[test_181736], **kwargs_181737)
        
        # Getting the type of 'self' (line 338)
        self_181739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'self')
        # Setting the type of the member 'result' of a type (line 338)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), self_181739, 'result', run_call_result_181738)
        
        # Assigning a Call to a Attribute (line 339):
        
        # Call to wasSuccessful(...): (line 339)
        # Processing the call keyword arguments (line 339)
        kwargs_181743 = {}
        # Getting the type of 'self' (line 339)
        self_181740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 23), 'self', False)
        # Obtaining the member 'result' of a type (line 339)
        result_181741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 23), self_181740, 'result')
        # Obtaining the member 'wasSuccessful' of a type (line 339)
        wasSuccessful_181742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 23), result_181741, 'wasSuccessful')
        # Calling wasSuccessful(args, kwargs) (line 339)
        wasSuccessful_call_result_181744 = invoke(stypy.reporting.localization.Localization(__file__, 339, 23), wasSuccessful_181742, *[], **kwargs_181743)
        
        # Getting the type of 'self' (line 339)
        self_181745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'self')
        # Setting the type of the member 'success' of a type (line 339)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), self_181745, 'success', wasSuccessful_call_result_181744)
        # Getting the type of 'self' (line 340)
        self_181746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'self')
        # Obtaining the member 'success' of a type (line 340)
        success_181747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 15), self_181746, 'success')
        # Assigning a type to the variable 'stypy_return_type' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'stypy_return_type', success_181747)
        
        # ################# End of 'runTests(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runTests' in the type store
        # Getting the type of 'stypy_return_type' (line 323)
        stypy_return_type_181748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_181748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runTests'
        return stypy_return_type_181748


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 322, 0, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyTestProgram.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NumpyTestProgram' (line 322)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 0), 'NumpyTestProgram', NumpyTestProgram)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
