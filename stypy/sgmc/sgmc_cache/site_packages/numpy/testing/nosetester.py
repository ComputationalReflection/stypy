
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Nose test running.
3: 
4: This module implements ``test()`` and ``bench()`` functions for NumPy modules.
5: 
6: '''
7: from __future__ import division, absolute_import, print_function
8: 
9: import os
10: import sys
11: import warnings
12: from numpy.compat import basestring
13: import numpy as np
14: 
15: 
16: def get_package_name(filepath):
17:     '''
18:     Given a path where a package is installed, determine its name.
19: 
20:     Parameters
21:     ----------
22:     filepath : str
23:         Path to a file. If the determination fails, "numpy" is returned.
24: 
25:     Examples
26:     --------
27:     >>> np.testing.nosetester.get_package_name('nonsense')
28:     'numpy'
29: 
30:     '''
31: 
32:     fullpath = filepath[:]
33:     pkg_name = []
34:     while 'site-packages' in filepath or 'dist-packages' in filepath:
35:         filepath, p2 = os.path.split(filepath)
36:         if p2 in ('site-packages', 'dist-packages'):
37:             break
38:         pkg_name.append(p2)
39: 
40:     # if package name determination failed, just default to numpy/scipy
41:     if not pkg_name:
42:         if 'scipy' in fullpath:
43:             return 'scipy'
44:         else:
45:             return 'numpy'
46: 
47:     # otherwise, reverse to get correct order and return
48:     pkg_name.reverse()
49: 
50:     # don't include the outer egg directory
51:     if pkg_name[0].endswith('.egg'):
52:         pkg_name.pop(0)
53: 
54:     return '.'.join(pkg_name)
55: 
56: def import_nose():
57:     ''' Import nose only when needed.
58:     '''
59:     fine_nose = True
60:     minimum_nose_version = (1, 0, 0)
61:     try:
62:         import nose
63:     except ImportError:
64:         fine_nose = False
65:     else:
66:         if nose.__versioninfo__ < minimum_nose_version:
67:             fine_nose = False
68: 
69:     if not fine_nose:
70:         msg = ('Need nose >= %d.%d.%d for tests - see '
71:                'http://somethingaboutorange.com/mrl/projects/nose' %
72:                minimum_nose_version)
73:         raise ImportError(msg)
74: 
75:     return nose
76: 
77: def run_module_suite(file_to_run=None, argv=None):
78:     '''
79:     Run a test module.
80: 
81:     Equivalent to calling ``$ nosetests <argv> <file_to_run>`` from
82:     the command line
83: 
84:     Parameters
85:     ----------
86:     file_to_run : str, optional
87:         Path to test module, or None.
88:         By default, run the module from which this function is called.
89:     argv : list of strings
90:         Arguments to be passed to the nose test runner. ``argv[0]`` is
91:         ignored. All command line arguments accepted by ``nosetests``
92:         will work. If it is the default value None, sys.argv is used.
93: 
94:         .. versionadded:: 1.9.0
95: 
96:     Examples
97:     --------
98:     Adding the following::
99: 
100:         if __name__ == "__main__" :
101:             run_module_suite(argv=sys.argv)
102: 
103:     at the end of a test module will run the tests when that module is
104:     called in the python interpreter.
105: 
106:     Alternatively, calling::
107: 
108:     >>> run_module_suite(file_to_run="numpy/tests/test_matlib.py")
109: 
110:     from an interpreter will run all the test routine in 'test_matlib.py'.
111:     '''
112:     if file_to_run is None:
113:         f = sys._getframe(1)
114:         file_to_run = f.f_locals.get('__file__', None)
115:         if file_to_run is None:
116:             raise AssertionError
117: 
118:     if argv is None:
119:         argv = sys.argv + [file_to_run]
120:     else:
121:         argv = argv + [file_to_run]
122: 
123:     nose = import_nose()
124:     from .noseclasses import KnownFailurePlugin
125:     nose.run(argv=argv, addplugins=[KnownFailurePlugin()])
126: 
127: 
128: class NoseTester(object):
129:     '''
130:     Nose test runner.
131: 
132:     This class is made available as numpy.testing.Tester, and a test function
133:     is typically added to a package's __init__.py like so::
134: 
135:       from numpy.testing import Tester
136:       test = Tester().test
137: 
138:     Calling this test function finds and runs all tests associated with the
139:     package and all its sub-packages.
140: 
141:     Attributes
142:     ----------
143:     package_path : str
144:         Full path to the package to test.
145:     package_name : str
146:         Name of the package to test.
147: 
148:     Parameters
149:     ----------
150:     package : module, str or None, optional
151:         The package to test. If a string, this should be the full path to
152:         the package. If None (default), `package` is set to the module from
153:         which `NoseTester` is initialized.
154:     raise_warnings : None, str or sequence of warnings, optional
155:         This specifies which warnings to configure as 'raise' instead
156:         of 'warn' during the test execution.  Valid strings are:
157: 
158:           - "develop" : equals ``(DeprecationWarning, RuntimeWarning)``
159:           - "release" : equals ``()``, don't raise on any warnings.
160: 
161:         Default is "release".
162:     depth : int, optional
163:         If `package` is None, then this can be used to initialize from the
164:         module of the caller of (the caller of (...)) the code that
165:         initializes `NoseTester`. Default of 0 means the module of the
166:         immediate caller; higher values are useful for utility routines that
167:         want to initialize `NoseTester` objects on behalf of other code.
168: 
169:     '''
170:     def __init__(self, package=None, raise_warnings="release", depth=0):
171:         # Back-compat: 'None' used to mean either "release" or "develop"
172:         # depending on whether this was a release or develop version of
173:         # numpy. Those semantics were fine for testing numpy, but not so
174:         # helpful for downstream projects like scipy that use
175:         # numpy.testing. (They want to set this based on whether *they* are a
176:         # release or develop version, not whether numpy is.) So we continue to
177:         # accept 'None' for back-compat, but it's now just an alias for the
178:         # default "release".
179:         if raise_warnings is None:
180:             raise_warnings = "release"
181: 
182:         package_name = None
183:         if package is None:
184:             f = sys._getframe(1 + depth)
185:             package_path = f.f_locals.get('__file__', None)
186:             if package_path is None:
187:                 raise AssertionError
188:             package_path = os.path.dirname(package_path)
189:             package_name = f.f_locals.get('__name__', None)
190:         elif isinstance(package, type(os)):
191:             package_path = os.path.dirname(package.__file__)
192:             package_name = getattr(package, '__name__', None)
193:         else:
194:             package_path = str(package)
195: 
196:         self.package_path = package_path
197: 
198:         # Find the package name under test; this name is used to limit coverage
199:         # reporting (if enabled).
200:         if package_name is None:
201:             package_name = get_package_name(package_path)
202:         self.package_name = package_name
203: 
204:         # Set to "release" in constructor in maintenance branches.
205:         self.raise_warnings = raise_warnings
206: 
207:     def _test_argv(self, label, verbose, extra_argv):
208:         ''' Generate argv for nosetest command
209: 
210:         Parameters
211:         ----------
212:         label : {'fast', 'full', '', attribute identifier}, optional
213:             see ``test`` docstring
214:         verbose : int, optional
215:             Verbosity value for test outputs, in the range 1-10. Default is 1.
216:         extra_argv : list, optional
217:             List with any extra arguments to pass to nosetests.
218: 
219:         Returns
220:         -------
221:         argv : list
222:             command line arguments that will be passed to nose
223:         '''
224:         argv = [__file__, self.package_path, '-s']
225:         if label and label != 'full':
226:             if not isinstance(label, basestring):
227:                 raise TypeError('Selection label should be a string')
228:             if label == 'fast':
229:                 label = 'not slow'
230:             argv += ['-A', label]
231:         argv += ['--verbosity', str(verbose)]
232: 
233:         # When installing with setuptools, and also in some other cases, the
234:         # test_*.py files end up marked +x executable. Nose, by default, does
235:         # not run files marked with +x as they might be scripts. However, in
236:         # our case nose only looks for test_*.py files under the package
237:         # directory, which should be safe.
238:         argv += ['--exe']
239: 
240:         if extra_argv:
241:             argv += extra_argv
242:         return argv
243: 
244:     def _show_system_info(self):
245:         nose = import_nose()
246: 
247:         import numpy
248:         print("NumPy version %s" % numpy.__version__)
249:         relaxed_strides = numpy.ones((10, 1), order="C").flags.f_contiguous
250:         print("NumPy relaxed strides checking option:", relaxed_strides)
251:         npdir = os.path.dirname(numpy.__file__)
252:         print("NumPy is installed in %s" % npdir)
253: 
254:         if 'scipy' in self.package_name:
255:             import scipy
256:             print("SciPy version %s" % scipy.__version__)
257:             spdir = os.path.dirname(scipy.__file__)
258:             print("SciPy is installed in %s" % spdir)
259: 
260:         pyversion = sys.version.replace('\n', '')
261:         print("Python version %s" % pyversion)
262:         print("nose version %d.%d.%d" % nose.__versioninfo__)
263: 
264:     def _get_custom_doctester(self):
265:         ''' Return instantiated plugin for doctests
266: 
267:         Allows subclassing of this class to override doctester
268: 
269:         A return value of None means use the nose builtin doctest plugin
270:         '''
271:         from .noseclasses import NumpyDoctest
272:         return NumpyDoctest()
273: 
274:     def prepare_test_args(self, label='fast', verbose=1, extra_argv=None,
275:                           doctests=False, coverage=False):
276:         '''
277:         Run tests for module using nose.
278: 
279:         This method does the heavy lifting for the `test` method. It takes all
280:         the same arguments, for details see `test`.
281: 
282:         See Also
283:         --------
284:         test
285: 
286:         '''
287:         # fail with nice error message if nose is not present
288:         import_nose()
289:         # compile argv
290:         argv = self._test_argv(label, verbose, extra_argv)
291:         # our way of doing coverage
292:         if coverage:
293:             argv += ['--cover-package=%s' % self.package_name, '--with-coverage',
294:                    '--cover-tests', '--cover-erase']
295:         # construct list of plugins
296:         import nose.plugins.builtin
297:         from .noseclasses import KnownFailurePlugin, Unplugger
298:         plugins = [KnownFailurePlugin()]
299:         plugins += [p() for p in nose.plugins.builtin.plugins]
300:         # add doctesting if required
301:         doctest_argv = '--with-doctest' in argv
302:         if doctests == False and doctest_argv:
303:             doctests = True
304:         plug = self._get_custom_doctester()
305:         if plug is None:
306:             # use standard doctesting
307:             if doctests and not doctest_argv:
308:                 argv += ['--with-doctest']
309:         else:  # custom doctesting
310:             if doctest_argv:  # in fact the unplugger would take care of this
311:                 argv.remove('--with-doctest')
312:             plugins += [Unplugger('doctest'), plug]
313:             if doctests:
314:                 argv += ['--with-' + plug.name]
315:         return argv, plugins
316: 
317:     def test(self, label='fast', verbose=1, extra_argv=None,
318:             doctests=False, coverage=False,
319:             raise_warnings=None):
320:         '''
321:         Run tests for module using nose.
322: 
323:         Parameters
324:         ----------
325:         label : {'fast', 'full', '', attribute identifier}, optional
326:             Identifies the tests to run. This can be a string to pass to
327:             the nosetests executable with the '-A' option, or one of several
328:             special values.  Special values are:
329:             * 'fast' - the default - which corresponds to the ``nosetests -A``
330:               option of 'not slow'.
331:             * 'full' - fast (as above) and slow tests as in the
332:               'no -A' option to nosetests - this is the same as ''.
333:             * None or '' - run all tests.
334:             attribute_identifier - string passed directly to nosetests as '-A'.
335:         verbose : int, optional
336:             Verbosity value for test outputs, in the range 1-10. Default is 1.
337:         extra_argv : list, optional
338:             List with any extra arguments to pass to nosetests.
339:         doctests : bool, optional
340:             If True, run doctests in module. Default is False.
341:         coverage : bool, optional
342:             If True, report coverage of NumPy code. Default is False.
343:             (This requires the `coverage module:
344:              <http://nedbatchelder.com/code/modules/coverage.html>`_).
345:         raise_warnings : str or sequence of warnings, optional
346:             This specifies which warnings to configure as 'raise' instead
347:             of 'warn' during the test execution.  Valid strings are:
348: 
349:               - "develop" : equals ``(DeprecationWarning, RuntimeWarning)``
350:               - "release" : equals ``()``, don't raise on any warnings.
351: 
352:         Returns
353:         -------
354:         result : object
355:             Returns the result of running the tests as a
356:             ``nose.result.TextTestResult`` object.
357: 
358:         Notes
359:         -----
360:         Each NumPy module exposes `test` in its namespace to run all tests for it.
361:         For example, to run all tests for numpy.lib:
362: 
363:         >>> np.lib.test() #doctest: +SKIP
364: 
365:         Examples
366:         --------
367:         >>> result = np.lib.test() #doctest: +SKIP
368:         Running unit tests for numpy.lib
369:         ...
370:         Ran 976 tests in 3.933s
371: 
372:         OK
373: 
374:         >>> result.errors #doctest: +SKIP
375:         []
376:         >>> result.knownfail #doctest: +SKIP
377:         []
378:         '''
379: 
380:         # cap verbosity at 3 because nose becomes *very* verbose beyond that
381:         verbose = min(verbose, 3)
382: 
383:         from . import utils
384:         utils.verbose = verbose
385: 
386:         if doctests:
387:             print("Running unit tests and doctests for %s" % self.package_name)
388:         else:
389:             print("Running unit tests for %s" % self.package_name)
390: 
391:         self._show_system_info()
392: 
393:         # reset doctest state on every run
394:         import doctest
395:         doctest.master = None
396: 
397:         if raise_warnings is None:
398:             raise_warnings = self.raise_warnings
399: 
400:         _warn_opts = dict(develop=(DeprecationWarning, RuntimeWarning),
401:                           release=())
402:         if isinstance(raise_warnings, basestring):
403:             raise_warnings = _warn_opts[raise_warnings]
404: 
405:         with warnings.catch_warnings():
406:             # Reset the warning filters to the default state,
407:             # so that running the tests is more repeatable.
408:             warnings.resetwarnings()
409:             # Set all warnings to 'warn', this is because the default 'once'
410:             # has the bad property of possibly shadowing later warnings.
411:             warnings.filterwarnings('always')
412:             # Force the requested warnings to raise
413:             for warningtype in raise_warnings:
414:                 warnings.filterwarnings('error', category=warningtype)
415:             # Filter out annoying import messages.
416:             warnings.filterwarnings('ignore', message='Not importing directory')
417:             warnings.filterwarnings("ignore", message="numpy.dtype size changed")
418:             warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
419:             warnings.filterwarnings("ignore", category=np.ModuleDeprecationWarning)
420:             warnings.filterwarnings("ignore", category=FutureWarning)
421:             # Filter out boolean '-' deprecation messages. This allows
422:             # older versions of scipy to test without a flood of messages.
423:             warnings.filterwarnings("ignore", message=".*boolean negative.*")
424:             warnings.filterwarnings("ignore", message=".*boolean subtract.*")
425:             # Filter out some deprecation warnings inside nose 1.3.7 when run
426:             # on python 3.5b2. See
427:             #     https://github.com/nose-devs/nose/issues/929
428:             warnings.filterwarnings("ignore", message=".*getargspec.*",
429:                                     category=DeprecationWarning,
430:                                     module="nose\.")
431: 
432:             from .noseclasses import NumpyTestProgram
433: 
434:             argv, plugins = self.prepare_test_args(
435:                     label, verbose, extra_argv, doctests, coverage)
436:             t = NumpyTestProgram(argv=argv, exit=False, plugins=plugins)
437: 
438:         return t.result
439: 
440:     def bench(self, label='fast', verbose=1, extra_argv=None):
441:         '''
442:         Run benchmarks for module using nose.
443: 
444:         Parameters
445:         ----------
446:         label : {'fast', 'full', '', attribute identifier}, optional
447:             Identifies the benchmarks to run. This can be a string to pass to
448:             the nosetests executable with the '-A' option, or one of several
449:             special values.  Special values are:
450:             * 'fast' - the default - which corresponds to the ``nosetests -A``
451:               option of 'not slow'.
452:             * 'full' - fast (as above) and slow benchmarks as in the
453:               'no -A' option to nosetests - this is the same as ''.
454:             * None or '' - run all tests.
455:             attribute_identifier - string passed directly to nosetests as '-A'.
456:         verbose : int, optional
457:             Verbosity value for benchmark outputs, in the range 1-10. Default is 1.
458:         extra_argv : list, optional
459:             List with any extra arguments to pass to nosetests.
460: 
461:         Returns
462:         -------
463:         success : bool
464:             Returns True if running the benchmarks works, False if an error
465:             occurred.
466: 
467:         Notes
468:         -----
469:         Benchmarks are like tests, but have names starting with "bench" instead
470:         of "test", and can be found under the "benchmarks" sub-directory of the
471:         module.
472: 
473:         Each NumPy module exposes `bench` in its namespace to run all benchmarks
474:         for it.
475: 
476:         Examples
477:         --------
478:         >>> success = np.lib.bench() #doctest: +SKIP
479:         Running benchmarks for numpy.lib
480:         ...
481:         using 562341 items:
482:         unique:
483:         0.11
484:         unique1d:
485:         0.11
486:         ratio: 1.0
487:         nUnique: 56230 == 56230
488:         ...
489:         OK
490: 
491:         >>> success #doctest: +SKIP
492:         True
493: 
494:         '''
495: 
496:         print("Running benchmarks for %s" % self.package_name)
497:         self._show_system_info()
498: 
499:         argv = self._test_argv(label, verbose, extra_argv)
500:         argv += ['--match', r'(?:^|[\\b_\\.%s-])[Bb]ench' % os.sep]
501: 
502:         # import nose or make informative error
503:         nose = import_nose()
504: 
505:         # get plugin to disable doctests
506:         from .noseclasses import Unplugger
507:         add_plugins = [Unplugger('doctest')]
508: 
509:         return nose.run(argv=argv, addplugins=add_plugins)
510: 
511: def _numpy_tester():
512:     if hasattr(np, "__version__") and ".dev0" in np.__version__:
513:         mode = "develop"
514:     else:
515:         mode = "release"
516:     return NoseTester(raise_warnings=mode, depth=1)
517: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_181755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nNose test running.\n\nThis module implements ``test()`` and ``bench()`` functions for NumPy modules.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import sys' statement (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import warnings' statement (line 11)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.compat import basestring' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181756 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat')

if (type(import_181756) is not StypyTypeError):

    if (import_181756 != 'pyd_module'):
        __import__(import_181756)
        sys_modules_181757 = sys.modules[import_181756]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat', sys_modules_181757.module_type_store, module_type_store, ['basestring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_181757, sys_modules_181757.module_type_store, module_type_store)
    else:
        from numpy.compat import basestring

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat', None, module_type_store, ['basestring'], [basestring])

else:
    # Assigning a type to the variable 'numpy.compat' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat', import_181756)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_181758 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_181758) is not StypyTypeError):

    if (import_181758 != 'pyd_module'):
        __import__(import_181758)
        sys_modules_181759 = sys.modules[import_181758]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_181759.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_181758)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')


@norecursion
def get_package_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_package_name'
    module_type_store = module_type_store.open_function_context('get_package_name', 16, 0, False)
    
    # Passed parameters checking function
    get_package_name.stypy_localization = localization
    get_package_name.stypy_type_of_self = None
    get_package_name.stypy_type_store = module_type_store
    get_package_name.stypy_function_name = 'get_package_name'
    get_package_name.stypy_param_names_list = ['filepath']
    get_package_name.stypy_varargs_param_name = None
    get_package_name.stypy_kwargs_param_name = None
    get_package_name.stypy_call_defaults = defaults
    get_package_name.stypy_call_varargs = varargs
    get_package_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_package_name', ['filepath'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_package_name', localization, ['filepath'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_package_name(...)' code ##################

    str_181760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '\n    Given a path where a package is installed, determine its name.\n\n    Parameters\n    ----------\n    filepath : str\n        Path to a file. If the determination fails, "numpy" is returned.\n\n    Examples\n    --------\n    >>> np.testing.nosetester.get_package_name(\'nonsense\')\n    \'numpy\'\n\n    ')
    
    # Assigning a Subscript to a Name (line 32):
    
    # Assigning a Subscript to a Name (line 32):
    
    # Obtaining the type of the subscript
    slice_181761 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 32, 15), None, None, None)
    # Getting the type of 'filepath' (line 32)
    filepath_181762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'filepath')
    # Obtaining the member '__getitem__' of a type (line 32)
    getitem___181763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), filepath_181762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 32)
    subscript_call_result_181764 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), getitem___181763, slice_181761)
    
    # Assigning a type to the variable 'fullpath' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'fullpath', subscript_call_result_181764)
    
    # Assigning a List to a Name (line 33):
    
    # Assigning a List to a Name (line 33):
    
    # Obtaining an instance of the builtin type 'list' (line 33)
    list_181765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 33)
    
    # Assigning a type to the variable 'pkg_name' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'pkg_name', list_181765)
    
    
    # Evaluating a boolean operation
    
    str_181766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 10), 'str', 'site-packages')
    # Getting the type of 'filepath' (line 34)
    filepath_181767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 29), 'filepath')
    # Applying the binary operator 'in' (line 34)
    result_contains_181768 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 10), 'in', str_181766, filepath_181767)
    
    
    str_181769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 41), 'str', 'dist-packages')
    # Getting the type of 'filepath' (line 34)
    filepath_181770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 60), 'filepath')
    # Applying the binary operator 'in' (line 34)
    result_contains_181771 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 41), 'in', str_181769, filepath_181770)
    
    # Applying the binary operator 'or' (line 34)
    result_or_keyword_181772 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 10), 'or', result_contains_181768, result_contains_181771)
    
    # Testing the type of an if condition (line 34)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 4), result_or_keyword_181772)
    # SSA begins for while statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 35):
    
    # Assigning a Call to a Name:
    
    # Call to split(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'filepath' (line 35)
    filepath_181776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 37), 'filepath', False)
    # Processing the call keyword arguments (line 35)
    kwargs_181777 = {}
    # Getting the type of 'os' (line 35)
    os_181773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 35)
    path_181774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 23), os_181773, 'path')
    # Obtaining the member 'split' of a type (line 35)
    split_181775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 23), path_181774, 'split')
    # Calling split(args, kwargs) (line 35)
    split_call_result_181778 = invoke(stypy.reporting.localization.Localization(__file__, 35, 23), split_181775, *[filepath_181776], **kwargs_181777)
    
    # Assigning a type to the variable 'call_assignment_181749' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'call_assignment_181749', split_call_result_181778)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_181781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'int')
    # Processing the call keyword arguments
    kwargs_181782 = {}
    # Getting the type of 'call_assignment_181749' (line 35)
    call_assignment_181749_181779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'call_assignment_181749', False)
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___181780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), call_assignment_181749_181779, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_181783 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___181780, *[int_181781], **kwargs_181782)
    
    # Assigning a type to the variable 'call_assignment_181750' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'call_assignment_181750', getitem___call_result_181783)
    
    # Assigning a Name to a Name (line 35):
    # Getting the type of 'call_assignment_181750' (line 35)
    call_assignment_181750_181784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'call_assignment_181750')
    # Assigning a type to the variable 'filepath' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'filepath', call_assignment_181750_181784)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_181787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'int')
    # Processing the call keyword arguments
    kwargs_181788 = {}
    # Getting the type of 'call_assignment_181749' (line 35)
    call_assignment_181749_181785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'call_assignment_181749', False)
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___181786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), call_assignment_181749_181785, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_181789 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___181786, *[int_181787], **kwargs_181788)
    
    # Assigning a type to the variable 'call_assignment_181751' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'call_assignment_181751', getitem___call_result_181789)
    
    # Assigning a Name to a Name (line 35):
    # Getting the type of 'call_assignment_181751' (line 35)
    call_assignment_181751_181790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'call_assignment_181751')
    # Assigning a type to the variable 'p2' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'p2', call_assignment_181751_181790)
    
    
    # Getting the type of 'p2' (line 36)
    p2_181791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'p2')
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_181792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    str_181793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'str', 'site-packages')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 18), tuple_181792, str_181793)
    # Adding element type (line 36)
    str_181794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'str', 'dist-packages')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 18), tuple_181792, str_181794)
    
    # Applying the binary operator 'in' (line 36)
    result_contains_181795 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 11), 'in', p2_181791, tuple_181792)
    
    # Testing the type of an if condition (line 36)
    if_condition_181796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 8), result_contains_181795)
    # Assigning a type to the variable 'if_condition_181796' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'if_condition_181796', if_condition_181796)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'p2' (line 38)
    p2_181799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'p2', False)
    # Processing the call keyword arguments (line 38)
    kwargs_181800 = {}
    # Getting the type of 'pkg_name' (line 38)
    pkg_name_181797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'pkg_name', False)
    # Obtaining the member 'append' of a type (line 38)
    append_181798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), pkg_name_181797, 'append')
    # Calling append(args, kwargs) (line 38)
    append_call_result_181801 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), append_181798, *[p2_181799], **kwargs_181800)
    
    # SSA join for while statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'pkg_name' (line 41)
    pkg_name_181802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'pkg_name')
    # Applying the 'not' unary operator (line 41)
    result_not__181803 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), 'not', pkg_name_181802)
    
    # Testing the type of an if condition (line 41)
    if_condition_181804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), result_not__181803)
    # Assigning a type to the variable 'if_condition_181804' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_181804', if_condition_181804)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_181805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'str', 'scipy')
    # Getting the type of 'fullpath' (line 42)
    fullpath_181806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'fullpath')
    # Applying the binary operator 'in' (line 42)
    result_contains_181807 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), 'in', str_181805, fullpath_181806)
    
    # Testing the type of an if condition (line 42)
    if_condition_181808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), result_contains_181807)
    # Assigning a type to the variable 'if_condition_181808' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_181808', if_condition_181808)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_181809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'str', 'scipy')
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'stypy_return_type', str_181809)
    # SSA branch for the else part of an if statement (line 42)
    module_type_store.open_ssa_branch('else')
    str_181810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'str', 'numpy')
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', str_181810)
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reverse(...): (line 48)
    # Processing the call keyword arguments (line 48)
    kwargs_181813 = {}
    # Getting the type of 'pkg_name' (line 48)
    pkg_name_181811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'pkg_name', False)
    # Obtaining the member 'reverse' of a type (line 48)
    reverse_181812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 4), pkg_name_181811, 'reverse')
    # Calling reverse(args, kwargs) (line 48)
    reverse_call_result_181814 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), reverse_181812, *[], **kwargs_181813)
    
    
    
    # Call to endswith(...): (line 51)
    # Processing the call arguments (line 51)
    str_181820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'str', '.egg')
    # Processing the call keyword arguments (line 51)
    kwargs_181821 = {}
    
    # Obtaining the type of the subscript
    int_181815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 16), 'int')
    # Getting the type of 'pkg_name' (line 51)
    pkg_name_181816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), 'pkg_name', False)
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___181817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 7), pkg_name_181816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_181818 = invoke(stypy.reporting.localization.Localization(__file__, 51, 7), getitem___181817, int_181815)
    
    # Obtaining the member 'endswith' of a type (line 51)
    endswith_181819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 7), subscript_call_result_181818, 'endswith')
    # Calling endswith(args, kwargs) (line 51)
    endswith_call_result_181822 = invoke(stypy.reporting.localization.Localization(__file__, 51, 7), endswith_181819, *[str_181820], **kwargs_181821)
    
    # Testing the type of an if condition (line 51)
    if_condition_181823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 4), endswith_call_result_181822)
    # Assigning a type to the variable 'if_condition_181823' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'if_condition_181823', if_condition_181823)
    # SSA begins for if statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to pop(...): (line 52)
    # Processing the call arguments (line 52)
    int_181826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'int')
    # Processing the call keyword arguments (line 52)
    kwargs_181827 = {}
    # Getting the type of 'pkg_name' (line 52)
    pkg_name_181824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'pkg_name', False)
    # Obtaining the member 'pop' of a type (line 52)
    pop_181825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), pkg_name_181824, 'pop')
    # Calling pop(args, kwargs) (line 52)
    pop_call_result_181828 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), pop_181825, *[int_181826], **kwargs_181827)
    
    # SSA join for if statement (line 51)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'pkg_name' (line 54)
    pkg_name_181831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'pkg_name', False)
    # Processing the call keyword arguments (line 54)
    kwargs_181832 = {}
    str_181829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'str', '.')
    # Obtaining the member 'join' of a type (line 54)
    join_181830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), str_181829, 'join')
    # Calling join(args, kwargs) (line 54)
    join_call_result_181833 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), join_181830, *[pkg_name_181831], **kwargs_181832)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', join_call_result_181833)
    
    # ################# End of 'get_package_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_package_name' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_181834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_181834)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_package_name'
    return stypy_return_type_181834

# Assigning a type to the variable 'get_package_name' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'get_package_name', get_package_name)

@norecursion
def import_nose(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'import_nose'
    module_type_store = module_type_store.open_function_context('import_nose', 56, 0, False)
    
    # Passed parameters checking function
    import_nose.stypy_localization = localization
    import_nose.stypy_type_of_self = None
    import_nose.stypy_type_store = module_type_store
    import_nose.stypy_function_name = 'import_nose'
    import_nose.stypy_param_names_list = []
    import_nose.stypy_varargs_param_name = None
    import_nose.stypy_kwargs_param_name = None
    import_nose.stypy_call_defaults = defaults
    import_nose.stypy_call_varargs = varargs
    import_nose.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'import_nose', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'import_nose', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'import_nose(...)' code ##################

    str_181835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', ' Import nose only when needed.\n    ')
    
    # Assigning a Name to a Name (line 59):
    
    # Assigning a Name to a Name (line 59):
    # Getting the type of 'True' (line 59)
    True_181836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'True')
    # Assigning a type to the variable 'fine_nose' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'fine_nose', True_181836)
    
    # Assigning a Tuple to a Name (line 60):
    
    # Assigning a Tuple to a Name (line 60):
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_181837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    int_181838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 28), tuple_181837, int_181838)
    # Adding element type (line 60)
    int_181839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 28), tuple_181837, int_181839)
    # Adding element type (line 60)
    int_181840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 28), tuple_181837, int_181840)
    
    # Assigning a type to the variable 'minimum_nose_version' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'minimum_nose_version', tuple_181837)
    
    
    # SSA begins for try-except statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 62, 8))
    
    # 'import nose' statement (line 62)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
    import_181841 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 62, 8), 'nose')

    if (type(import_181841) is not StypyTypeError):

        if (import_181841 != 'pyd_module'):
            __import__(import_181841)
            sys_modules_181842 = sys.modules[import_181841]
            import_module(stypy.reporting.localization.Localization(__file__, 62, 8), 'nose', sys_modules_181842.module_type_store, module_type_store)
        else:
            import nose

            import_module(stypy.reporting.localization.Localization(__file__, 62, 8), 'nose', nose, module_type_store)

    else:
        # Assigning a type to the variable 'nose' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'nose', import_181841)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
    
    # SSA branch for the except part of a try statement (line 61)
    # SSA branch for the except 'ImportError' branch of a try statement (line 61)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 64):
    
    # Assigning a Name to a Name (line 64):
    # Getting the type of 'False' (line 64)
    False_181843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'False')
    # Assigning a type to the variable 'fine_nose' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'fine_nose', False_181843)
    # SSA branch for the else branch of a try statement (line 61)
    module_type_store.open_ssa_branch('except else')
    
    
    # Getting the type of 'nose' (line 66)
    nose_181844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'nose')
    # Obtaining the member '__versioninfo__' of a type (line 66)
    versioninfo___181845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), nose_181844, '__versioninfo__')
    # Getting the type of 'minimum_nose_version' (line 66)
    minimum_nose_version_181846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'minimum_nose_version')
    # Applying the binary operator '<' (line 66)
    result_lt_181847 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), '<', versioninfo___181845, minimum_nose_version_181846)
    
    # Testing the type of an if condition (line 66)
    if_condition_181848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_lt_181847)
    # Assigning a type to the variable 'if_condition_181848' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_181848', if_condition_181848)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 67):
    
    # Assigning a Name to a Name (line 67):
    # Getting the type of 'False' (line 67)
    False_181849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'False')
    # Assigning a type to the variable 'fine_nose' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'fine_nose', False_181849)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'fine_nose' (line 69)
    fine_nose_181850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'fine_nose')
    # Applying the 'not' unary operator (line 69)
    result_not__181851 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 7), 'not', fine_nose_181850)
    
    # Testing the type of an if condition (line 69)
    if_condition_181852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 4), result_not__181851)
    # Assigning a type to the variable 'if_condition_181852' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'if_condition_181852', if_condition_181852)
    # SSA begins for if statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 70):
    
    # Assigning a BinOp to a Name (line 70):
    str_181853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 15), 'str', 'Need nose >= %d.%d.%d for tests - see http://somethingaboutorange.com/mrl/projects/nose')
    # Getting the type of 'minimum_nose_version' (line 72)
    minimum_nose_version_181854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'minimum_nose_version')
    # Applying the binary operator '%' (line 70)
    result_mod_181855 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 15), '%', str_181853, minimum_nose_version_181854)
    
    # Assigning a type to the variable 'msg' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'msg', result_mod_181855)
    
    # Call to ImportError(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'msg' (line 73)
    msg_181857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'msg', False)
    # Processing the call keyword arguments (line 73)
    kwargs_181858 = {}
    # Getting the type of 'ImportError' (line 73)
    ImportError_181856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'ImportError', False)
    # Calling ImportError(args, kwargs) (line 73)
    ImportError_call_result_181859 = invoke(stypy.reporting.localization.Localization(__file__, 73, 14), ImportError_181856, *[msg_181857], **kwargs_181858)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 73, 8), ImportError_call_result_181859, 'raise parameter', BaseException)
    # SSA join for if statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'nose' (line 75)
    nose_181860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'nose')
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', nose_181860)
    
    # ################# End of 'import_nose(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_nose' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_181861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_181861)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_nose'
    return stypy_return_type_181861

# Assigning a type to the variable 'import_nose' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'import_nose', import_nose)

@norecursion
def run_module_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 77)
    None_181862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'None')
    # Getting the type of 'None' (line 77)
    None_181863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 44), 'None')
    defaults = [None_181862, None_181863]
    # Create a new context for function 'run_module_suite'
    module_type_store = module_type_store.open_function_context('run_module_suite', 77, 0, False)
    
    # Passed parameters checking function
    run_module_suite.stypy_localization = localization
    run_module_suite.stypy_type_of_self = None
    run_module_suite.stypy_type_store = module_type_store
    run_module_suite.stypy_function_name = 'run_module_suite'
    run_module_suite.stypy_param_names_list = ['file_to_run', 'argv']
    run_module_suite.stypy_varargs_param_name = None
    run_module_suite.stypy_kwargs_param_name = None
    run_module_suite.stypy_call_defaults = defaults
    run_module_suite.stypy_call_varargs = varargs
    run_module_suite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run_module_suite', ['file_to_run', 'argv'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run_module_suite', localization, ['file_to_run', 'argv'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run_module_suite(...)' code ##################

    str_181864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, (-1)), 'str', '\n    Run a test module.\n\n    Equivalent to calling ``$ nosetests <argv> <file_to_run>`` from\n    the command line\n\n    Parameters\n    ----------\n    file_to_run : str, optional\n        Path to test module, or None.\n        By default, run the module from which this function is called.\n    argv : list of strings\n        Arguments to be passed to the nose test runner. ``argv[0]`` is\n        ignored. All command line arguments accepted by ``nosetests``\n        will work. If it is the default value None, sys.argv is used.\n\n        .. versionadded:: 1.9.0\n\n    Examples\n    --------\n    Adding the following::\n\n        if __name__ == "__main__" :\n            run_module_suite(argv=sys.argv)\n\n    at the end of a test module will run the tests when that module is\n    called in the python interpreter.\n\n    Alternatively, calling::\n\n    >>> run_module_suite(file_to_run="numpy/tests/test_matlib.py")\n\n    from an interpreter will run all the test routine in \'test_matlib.py\'.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 112)
    # Getting the type of 'file_to_run' (line 112)
    file_to_run_181865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'file_to_run')
    # Getting the type of 'None' (line 112)
    None_181866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'None')
    
    (may_be_181867, more_types_in_union_181868) = may_be_none(file_to_run_181865, None_181866)

    if may_be_181867:

        if more_types_in_union_181868:
            # Runtime conditional SSA (line 112)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 113):
        
        # Assigning a Call to a Name (line 113):
        
        # Call to _getframe(...): (line 113)
        # Processing the call arguments (line 113)
        int_181871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 26), 'int')
        # Processing the call keyword arguments (line 113)
        kwargs_181872 = {}
        # Getting the type of 'sys' (line 113)
        sys_181869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'sys', False)
        # Obtaining the member '_getframe' of a type (line 113)
        _getframe_181870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), sys_181869, '_getframe')
        # Calling _getframe(args, kwargs) (line 113)
        _getframe_call_result_181873 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), _getframe_181870, *[int_181871], **kwargs_181872)
        
        # Assigning a type to the variable 'f' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'f', _getframe_call_result_181873)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to get(...): (line 114)
        # Processing the call arguments (line 114)
        str_181877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 37), 'str', '__file__')
        # Getting the type of 'None' (line 114)
        None_181878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 49), 'None', False)
        # Processing the call keyword arguments (line 114)
        kwargs_181879 = {}
        # Getting the type of 'f' (line 114)
        f_181874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'f', False)
        # Obtaining the member 'f_locals' of a type (line 114)
        f_locals_181875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), f_181874, 'f_locals')
        # Obtaining the member 'get' of a type (line 114)
        get_181876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), f_locals_181875, 'get')
        # Calling get(args, kwargs) (line 114)
        get_call_result_181880 = invoke(stypy.reporting.localization.Localization(__file__, 114, 22), get_181876, *[str_181877, None_181878], **kwargs_181879)
        
        # Assigning a type to the variable 'file_to_run' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'file_to_run', get_call_result_181880)
        
        # Type idiom detected: calculating its left and rigth part (line 115)
        # Getting the type of 'file_to_run' (line 115)
        file_to_run_181881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'file_to_run')
        # Getting the type of 'None' (line 115)
        None_181882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'None')
        
        (may_be_181883, more_types_in_union_181884) = may_be_none(file_to_run_181881, None_181882)

        if may_be_181883:

            if more_types_in_union_181884:
                # Runtime conditional SSA (line 115)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'AssertionError' (line 116)
            AssertionError_181885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'AssertionError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 116, 12), AssertionError_181885, 'raise parameter', BaseException)

            if more_types_in_union_181884:
                # SSA join for if statement (line 115)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_181868:
            # SSA join for if statement (line 112)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 118)
    # Getting the type of 'argv' (line 118)
    argv_181886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'argv')
    # Getting the type of 'None' (line 118)
    None_181887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'None')
    
    (may_be_181888, more_types_in_union_181889) = may_be_none(argv_181886, None_181887)

    if may_be_181888:

        if more_types_in_union_181889:
            # Runtime conditional SSA (line 118)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 119):
        
        # Assigning a BinOp to a Name (line 119):
        # Getting the type of 'sys' (line 119)
        sys_181890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'sys')
        # Obtaining the member 'argv' of a type (line 119)
        argv_181891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), sys_181890, 'argv')
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_181892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        # Getting the type of 'file_to_run' (line 119)
        file_to_run_181893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'file_to_run')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_181892, file_to_run_181893)
        
        # Applying the binary operator '+' (line 119)
        result_add_181894 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 15), '+', argv_181891, list_181892)
        
        # Assigning a type to the variable 'argv' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'argv', result_add_181894)

        if more_types_in_union_181889:
            # Runtime conditional SSA for else branch (line 118)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_181888) or more_types_in_union_181889):
        
        # Assigning a BinOp to a Name (line 121):
        
        # Assigning a BinOp to a Name (line 121):
        # Getting the type of 'argv' (line 121)
        argv_181895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'argv')
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_181896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        # Getting the type of 'file_to_run' (line 121)
        file_to_run_181897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'file_to_run')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), list_181896, file_to_run_181897)
        
        # Applying the binary operator '+' (line 121)
        result_add_181898 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 15), '+', argv_181895, list_181896)
        
        # Assigning a type to the variable 'argv' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'argv', result_add_181898)

        if (may_be_181888 and more_types_in_union_181889):
            # SSA join for if statement (line 118)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 123):
    
    # Assigning a Call to a Name (line 123):
    
    # Call to import_nose(...): (line 123)
    # Processing the call keyword arguments (line 123)
    kwargs_181900 = {}
    # Getting the type of 'import_nose' (line 123)
    import_nose_181899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'import_nose', False)
    # Calling import_nose(args, kwargs) (line 123)
    import_nose_call_result_181901 = invoke(stypy.reporting.localization.Localization(__file__, 123, 11), import_nose_181899, *[], **kwargs_181900)
    
    # Assigning a type to the variable 'nose' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'nose', import_nose_call_result_181901)
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 124, 4))
    
    # 'from numpy.testing.noseclasses import KnownFailurePlugin' statement (line 124)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
    import_181902 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 124, 4), 'numpy.testing.noseclasses')

    if (type(import_181902) is not StypyTypeError):

        if (import_181902 != 'pyd_module'):
            __import__(import_181902)
            sys_modules_181903 = sys.modules[import_181902]
            import_from_module(stypy.reporting.localization.Localization(__file__, 124, 4), 'numpy.testing.noseclasses', sys_modules_181903.module_type_store, module_type_store, ['KnownFailurePlugin'])
            nest_module(stypy.reporting.localization.Localization(__file__, 124, 4), __file__, sys_modules_181903, sys_modules_181903.module_type_store, module_type_store)
        else:
            from numpy.testing.noseclasses import KnownFailurePlugin

            import_from_module(stypy.reporting.localization.Localization(__file__, 124, 4), 'numpy.testing.noseclasses', None, module_type_store, ['KnownFailurePlugin'], [KnownFailurePlugin])

    else:
        # Assigning a type to the variable 'numpy.testing.noseclasses' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'numpy.testing.noseclasses', import_181902)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
    
    
    # Call to run(...): (line 125)
    # Processing the call keyword arguments (line 125)
    # Getting the type of 'argv' (line 125)
    argv_181906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 18), 'argv', False)
    keyword_181907 = argv_181906
    
    # Obtaining an instance of the builtin type 'list' (line 125)
    list_181908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 125)
    # Adding element type (line 125)
    
    # Call to KnownFailurePlugin(...): (line 125)
    # Processing the call keyword arguments (line 125)
    kwargs_181910 = {}
    # Getting the type of 'KnownFailurePlugin' (line 125)
    KnownFailurePlugin_181909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'KnownFailurePlugin', False)
    # Calling KnownFailurePlugin(args, kwargs) (line 125)
    KnownFailurePlugin_call_result_181911 = invoke(stypy.reporting.localization.Localization(__file__, 125, 36), KnownFailurePlugin_181909, *[], **kwargs_181910)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 35), list_181908, KnownFailurePlugin_call_result_181911)
    
    keyword_181912 = list_181908
    kwargs_181913 = {'addplugins': keyword_181912, 'argv': keyword_181907}
    # Getting the type of 'nose' (line 125)
    nose_181904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'nose', False)
    # Obtaining the member 'run' of a type (line 125)
    run_181905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 4), nose_181904, 'run')
    # Calling run(args, kwargs) (line 125)
    run_call_result_181914 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), run_181905, *[], **kwargs_181913)
    
    
    # ################# End of 'run_module_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run_module_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_181915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_181915)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run_module_suite'
    return stypy_return_type_181915

# Assigning a type to the variable 'run_module_suite' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'run_module_suite', run_module_suite)
# Declaration of the 'NoseTester' class

class NoseTester(object, ):
    str_181916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, (-1)), 'str', '\n    Nose test runner.\n\n    This class is made available as numpy.testing.Tester, and a test function\n    is typically added to a package\'s __init__.py like so::\n\n      from numpy.testing import Tester\n      test = Tester().test\n\n    Calling this test function finds and runs all tests associated with the\n    package and all its sub-packages.\n\n    Attributes\n    ----------\n    package_path : str\n        Full path to the package to test.\n    package_name : str\n        Name of the package to test.\n\n    Parameters\n    ----------\n    package : module, str or None, optional\n        The package to test. If a string, this should be the full path to\n        the package. If None (default), `package` is set to the module from\n        which `NoseTester` is initialized.\n    raise_warnings : None, str or sequence of warnings, optional\n        This specifies which warnings to configure as \'raise\' instead\n        of \'warn\' during the test execution.  Valid strings are:\n\n          - "develop" : equals ``(DeprecationWarning, RuntimeWarning)``\n          - "release" : equals ``()``, don\'t raise on any warnings.\n\n        Default is "release".\n    depth : int, optional\n        If `package` is None, then this can be used to initialize from the\n        module of the caller of (the caller of (...)) the code that\n        initializes `NoseTester`. Default of 0 means the module of the\n        immediate caller; higher values are useful for utility routines that\n        want to initialize `NoseTester` objects on behalf of other code.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 170)
        None_181917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'None')
        str_181918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 52), 'str', 'release')
        int_181919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 69), 'int')
        defaults = [None_181917, str_181918, int_181919]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NoseTester.__init__', ['package', 'raise_warnings', 'depth'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['package', 'raise_warnings', 'depth'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 179)
        # Getting the type of 'raise_warnings' (line 179)
        raise_warnings_181920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'raise_warnings')
        # Getting the type of 'None' (line 179)
        None_181921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 29), 'None')
        
        (may_be_181922, more_types_in_union_181923) = may_be_none(raise_warnings_181920, None_181921)

        if may_be_181922:

            if more_types_in_union_181923:
                # Runtime conditional SSA (line 179)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 180):
            
            # Assigning a Str to a Name (line 180):
            str_181924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 29), 'str', 'release')
            # Assigning a type to the variable 'raise_warnings' (line 180)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'raise_warnings', str_181924)

            if more_types_in_union_181923:
                # SSA join for if statement (line 179)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 182):
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'None' (line 182)
        None_181925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 23), 'None')
        # Assigning a type to the variable 'package_name' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'package_name', None_181925)
        
        # Type idiom detected: calculating its left and rigth part (line 183)
        # Getting the type of 'package' (line 183)
        package_181926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'package')
        # Getting the type of 'None' (line 183)
        None_181927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'None')
        
        (may_be_181928, more_types_in_union_181929) = may_be_none(package_181926, None_181927)

        if may_be_181928:

            if more_types_in_union_181929:
                # Runtime conditional SSA (line 183)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 184):
            
            # Assigning a Call to a Name (line 184):
            
            # Call to _getframe(...): (line 184)
            # Processing the call arguments (line 184)
            int_181932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 30), 'int')
            # Getting the type of 'depth' (line 184)
            depth_181933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'depth', False)
            # Applying the binary operator '+' (line 184)
            result_add_181934 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 30), '+', int_181932, depth_181933)
            
            # Processing the call keyword arguments (line 184)
            kwargs_181935 = {}
            # Getting the type of 'sys' (line 184)
            sys_181930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'sys', False)
            # Obtaining the member '_getframe' of a type (line 184)
            _getframe_181931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), sys_181930, '_getframe')
            # Calling _getframe(args, kwargs) (line 184)
            _getframe_call_result_181936 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), _getframe_181931, *[result_add_181934], **kwargs_181935)
            
            # Assigning a type to the variable 'f' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'f', _getframe_call_result_181936)
            
            # Assigning a Call to a Name (line 185):
            
            # Assigning a Call to a Name (line 185):
            
            # Call to get(...): (line 185)
            # Processing the call arguments (line 185)
            str_181940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 42), 'str', '__file__')
            # Getting the type of 'None' (line 185)
            None_181941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 54), 'None', False)
            # Processing the call keyword arguments (line 185)
            kwargs_181942 = {}
            # Getting the type of 'f' (line 185)
            f_181937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'f', False)
            # Obtaining the member 'f_locals' of a type (line 185)
            f_locals_181938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 27), f_181937, 'f_locals')
            # Obtaining the member 'get' of a type (line 185)
            get_181939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 27), f_locals_181938, 'get')
            # Calling get(args, kwargs) (line 185)
            get_call_result_181943 = invoke(stypy.reporting.localization.Localization(__file__, 185, 27), get_181939, *[str_181940, None_181941], **kwargs_181942)
            
            # Assigning a type to the variable 'package_path' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'package_path', get_call_result_181943)
            
            # Type idiom detected: calculating its left and rigth part (line 186)
            # Getting the type of 'package_path' (line 186)
            package_path_181944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'package_path')
            # Getting the type of 'None' (line 186)
            None_181945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 31), 'None')
            
            (may_be_181946, more_types_in_union_181947) = may_be_none(package_path_181944, None_181945)

            if may_be_181946:

                if more_types_in_union_181947:
                    # Runtime conditional SSA (line 186)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'AssertionError' (line 187)
                AssertionError_181948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), 'AssertionError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 187, 16), AssertionError_181948, 'raise parameter', BaseException)

                if more_types_in_union_181947:
                    # SSA join for if statement (line 186)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 188):
            
            # Assigning a Call to a Name (line 188):
            
            # Call to dirname(...): (line 188)
            # Processing the call arguments (line 188)
            # Getting the type of 'package_path' (line 188)
            package_path_181952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 43), 'package_path', False)
            # Processing the call keyword arguments (line 188)
            kwargs_181953 = {}
            # Getting the type of 'os' (line 188)
            os_181949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 27), 'os', False)
            # Obtaining the member 'path' of a type (line 188)
            path_181950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 27), os_181949, 'path')
            # Obtaining the member 'dirname' of a type (line 188)
            dirname_181951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 27), path_181950, 'dirname')
            # Calling dirname(args, kwargs) (line 188)
            dirname_call_result_181954 = invoke(stypy.reporting.localization.Localization(__file__, 188, 27), dirname_181951, *[package_path_181952], **kwargs_181953)
            
            # Assigning a type to the variable 'package_path' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'package_path', dirname_call_result_181954)
            
            # Assigning a Call to a Name (line 189):
            
            # Assigning a Call to a Name (line 189):
            
            # Call to get(...): (line 189)
            # Processing the call arguments (line 189)
            str_181958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 42), 'str', '__name__')
            # Getting the type of 'None' (line 189)
            None_181959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 54), 'None', False)
            # Processing the call keyword arguments (line 189)
            kwargs_181960 = {}
            # Getting the type of 'f' (line 189)
            f_181955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 27), 'f', False)
            # Obtaining the member 'f_locals' of a type (line 189)
            f_locals_181956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 27), f_181955, 'f_locals')
            # Obtaining the member 'get' of a type (line 189)
            get_181957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 27), f_locals_181956, 'get')
            # Calling get(args, kwargs) (line 189)
            get_call_result_181961 = invoke(stypy.reporting.localization.Localization(__file__, 189, 27), get_181957, *[str_181958, None_181959], **kwargs_181960)
            
            # Assigning a type to the variable 'package_name' (line 189)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'package_name', get_call_result_181961)

            if more_types_in_union_181929:
                # Runtime conditional SSA for else branch (line 183)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_181928) or more_types_in_union_181929):
            
            # Type idiom detected: calculating its left and rigth part (line 190)
            
            # Call to type(...): (line 190)
            # Processing the call arguments (line 190)
            # Getting the type of 'os' (line 190)
            os_181963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 38), 'os', False)
            # Processing the call keyword arguments (line 190)
            kwargs_181964 = {}
            # Getting the type of 'type' (line 190)
            type_181962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'type', False)
            # Calling type(args, kwargs) (line 190)
            type_call_result_181965 = invoke(stypy.reporting.localization.Localization(__file__, 190, 33), type_181962, *[os_181963], **kwargs_181964)
            
            # Getting the type of 'package' (line 190)
            package_181966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'package')
            
            (may_be_181967, more_types_in_union_181968) = may_be_subtype(type_call_result_181965, package_181966)

            if may_be_181967:

                if more_types_in_union_181968:
                    # Runtime conditional SSA (line 190)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'package' (line 190)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 13), 'package', remove_not_subtype_from_union(package_181966, type(os)))
                
                # Assigning a Call to a Name (line 191):
                
                # Assigning a Call to a Name (line 191):
                
                # Call to dirname(...): (line 191)
                # Processing the call arguments (line 191)
                # Getting the type of 'package' (line 191)
                package_181972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 43), 'package', False)
                # Obtaining the member '__file__' of a type (line 191)
                file___181973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 43), package_181972, '__file__')
                # Processing the call keyword arguments (line 191)
                kwargs_181974 = {}
                # Getting the type of 'os' (line 191)
                os_181969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 27), 'os', False)
                # Obtaining the member 'path' of a type (line 191)
                path_181970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 27), os_181969, 'path')
                # Obtaining the member 'dirname' of a type (line 191)
                dirname_181971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 27), path_181970, 'dirname')
                # Calling dirname(args, kwargs) (line 191)
                dirname_call_result_181975 = invoke(stypy.reporting.localization.Localization(__file__, 191, 27), dirname_181971, *[file___181973], **kwargs_181974)
                
                # Assigning a type to the variable 'package_path' (line 191)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'package_path', dirname_call_result_181975)
                
                # Assigning a Call to a Name (line 192):
                
                # Assigning a Call to a Name (line 192):
                
                # Call to getattr(...): (line 192)
                # Processing the call arguments (line 192)
                # Getting the type of 'package' (line 192)
                package_181977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 35), 'package', False)
                str_181978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 44), 'str', '__name__')
                # Getting the type of 'None' (line 192)
                None_181979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 56), 'None', False)
                # Processing the call keyword arguments (line 192)
                kwargs_181980 = {}
                # Getting the type of 'getattr' (line 192)
                getattr_181976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'getattr', False)
                # Calling getattr(args, kwargs) (line 192)
                getattr_call_result_181981 = invoke(stypy.reporting.localization.Localization(__file__, 192, 27), getattr_181976, *[package_181977, str_181978, None_181979], **kwargs_181980)
                
                # Assigning a type to the variable 'package_name' (line 192)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'package_name', getattr_call_result_181981)

                if more_types_in_union_181968:
                    # Runtime conditional SSA for else branch (line 190)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_181967) or more_types_in_union_181968):
                # Assigning a type to the variable 'package' (line 190)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 13), 'package', remove_subtype_from_union(package_181966, type(os)))
                
                # Assigning a Call to a Name (line 194):
                
                # Assigning a Call to a Name (line 194):
                
                # Call to str(...): (line 194)
                # Processing the call arguments (line 194)
                # Getting the type of 'package' (line 194)
                package_181983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 31), 'package', False)
                # Processing the call keyword arguments (line 194)
                kwargs_181984 = {}
                # Getting the type of 'str' (line 194)
                str_181982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'str', False)
                # Calling str(args, kwargs) (line 194)
                str_call_result_181985 = invoke(stypy.reporting.localization.Localization(__file__, 194, 27), str_181982, *[package_181983], **kwargs_181984)
                
                # Assigning a type to the variable 'package_path' (line 194)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'package_path', str_call_result_181985)

                if (may_be_181967 and more_types_in_union_181968):
                    # SSA join for if statement (line 190)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_181928 and more_types_in_union_181929):
                # SSA join for if statement (line 183)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 196):
        
        # Assigning a Name to a Attribute (line 196):
        # Getting the type of 'package_path' (line 196)
        package_path_181986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 28), 'package_path')
        # Getting the type of 'self' (line 196)
        self_181987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Setting the type of the member 'package_path' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_181987, 'package_path', package_path_181986)
        
        # Type idiom detected: calculating its left and rigth part (line 200)
        # Getting the type of 'package_name' (line 200)
        package_name_181988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'package_name')
        # Getting the type of 'None' (line 200)
        None_181989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 27), 'None')
        
        (may_be_181990, more_types_in_union_181991) = may_be_none(package_name_181988, None_181989)

        if may_be_181990:

            if more_types_in_union_181991:
                # Runtime conditional SSA (line 200)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 201):
            
            # Assigning a Call to a Name (line 201):
            
            # Call to get_package_name(...): (line 201)
            # Processing the call arguments (line 201)
            # Getting the type of 'package_path' (line 201)
            package_path_181993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 44), 'package_path', False)
            # Processing the call keyword arguments (line 201)
            kwargs_181994 = {}
            # Getting the type of 'get_package_name' (line 201)
            get_package_name_181992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'get_package_name', False)
            # Calling get_package_name(args, kwargs) (line 201)
            get_package_name_call_result_181995 = invoke(stypy.reporting.localization.Localization(__file__, 201, 27), get_package_name_181992, *[package_path_181993], **kwargs_181994)
            
            # Assigning a type to the variable 'package_name' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'package_name', get_package_name_call_result_181995)

            if more_types_in_union_181991:
                # SSA join for if statement (line 200)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 202):
        
        # Assigning a Name to a Attribute (line 202):
        # Getting the type of 'package_name' (line 202)
        package_name_181996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'package_name')
        # Getting the type of 'self' (line 202)
        self_181997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Setting the type of the member 'package_name' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_181997, 'package_name', package_name_181996)
        
        # Assigning a Name to a Attribute (line 205):
        
        # Assigning a Name to a Attribute (line 205):
        # Getting the type of 'raise_warnings' (line 205)
        raise_warnings_181998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'raise_warnings')
        # Getting the type of 'self' (line 205)
        self_181999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member 'raise_warnings' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_181999, 'raise_warnings', raise_warnings_181998)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _test_argv(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_test_argv'
        module_type_store = module_type_store.open_function_context('_test_argv', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NoseTester._test_argv.__dict__.__setitem__('stypy_localization', localization)
        NoseTester._test_argv.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NoseTester._test_argv.__dict__.__setitem__('stypy_type_store', module_type_store)
        NoseTester._test_argv.__dict__.__setitem__('stypy_function_name', 'NoseTester._test_argv')
        NoseTester._test_argv.__dict__.__setitem__('stypy_param_names_list', ['label', 'verbose', 'extra_argv'])
        NoseTester._test_argv.__dict__.__setitem__('stypy_varargs_param_name', None)
        NoseTester._test_argv.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NoseTester._test_argv.__dict__.__setitem__('stypy_call_defaults', defaults)
        NoseTester._test_argv.__dict__.__setitem__('stypy_call_varargs', varargs)
        NoseTester._test_argv.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NoseTester._test_argv.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NoseTester._test_argv', ['label', 'verbose', 'extra_argv'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_test_argv', localization, ['label', 'verbose', 'extra_argv'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_test_argv(...)' code ##################

        str_182000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, (-1)), 'str', " Generate argv for nosetest command\n\n        Parameters\n        ----------\n        label : {'fast', 'full', '', attribute identifier}, optional\n            see ``test`` docstring\n        verbose : int, optional\n            Verbosity value for test outputs, in the range 1-10. Default is 1.\n        extra_argv : list, optional\n            List with any extra arguments to pass to nosetests.\n\n        Returns\n        -------\n        argv : list\n            command line arguments that will be passed to nose\n        ")
        
        # Assigning a List to a Name (line 224):
        
        # Assigning a List to a Name (line 224):
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_182001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        # Getting the type of '__file__' (line 224)
        file___182002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), '__file__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 15), list_182001, file___182002)
        # Adding element type (line 224)
        # Getting the type of 'self' (line 224)
        self_182003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 26), 'self')
        # Obtaining the member 'package_path' of a type (line 224)
        package_path_182004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 26), self_182003, 'package_path')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 15), list_182001, package_path_182004)
        # Adding element type (line 224)
        str_182005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 45), 'str', '-s')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 15), list_182001, str_182005)
        
        # Assigning a type to the variable 'argv' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'argv', list_182001)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'label' (line 225)
        label_182006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'label')
        
        # Getting the type of 'label' (line 225)
        label_182007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 21), 'label')
        str_182008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 30), 'str', 'full')
        # Applying the binary operator '!=' (line 225)
        result_ne_182009 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 21), '!=', label_182007, str_182008)
        
        # Applying the binary operator 'and' (line 225)
        result_and_keyword_182010 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 11), 'and', label_182006, result_ne_182009)
        
        # Testing the type of an if condition (line 225)
        if_condition_182011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 8), result_and_keyword_182010)
        # Assigning a type to the variable 'if_condition_182011' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'if_condition_182011', if_condition_182011)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 226)
        # Getting the type of 'basestring' (line 226)
        basestring_182012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 37), 'basestring')
        # Getting the type of 'label' (line 226)
        label_182013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 30), 'label')
        
        (may_be_182014, more_types_in_union_182015) = may_not_be_subtype(basestring_182012, label_182013)

        if may_be_182014:

            if more_types_in_union_182015:
                # Runtime conditional SSA (line 226)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'label' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'label', remove_subtype_from_union(label_182013, basestring))
            
            # Call to TypeError(...): (line 227)
            # Processing the call arguments (line 227)
            str_182017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 32), 'str', 'Selection label should be a string')
            # Processing the call keyword arguments (line 227)
            kwargs_182018 = {}
            # Getting the type of 'TypeError' (line 227)
            TypeError_182016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 227)
            TypeError_call_result_182019 = invoke(stypy.reporting.localization.Localization(__file__, 227, 22), TypeError_182016, *[str_182017], **kwargs_182018)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 227, 16), TypeError_call_result_182019, 'raise parameter', BaseException)

            if more_types_in_union_182015:
                # SSA join for if statement (line 226)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'label' (line 228)
        label_182020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'label')
        str_182021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 24), 'str', 'fast')
        # Applying the binary operator '==' (line 228)
        result_eq_182022 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 15), '==', label_182020, str_182021)
        
        # Testing the type of an if condition (line 228)
        if_condition_182023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 12), result_eq_182022)
        # Assigning a type to the variable 'if_condition_182023' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'if_condition_182023', if_condition_182023)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 229):
        
        # Assigning a Str to a Name (line 229):
        str_182024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 24), 'str', 'not slow')
        # Assigning a type to the variable 'label' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'label', str_182024)
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'argv' (line 230)
        argv_182025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'argv')
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_182026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        str_182027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 21), 'str', '-A')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 20), list_182026, str_182027)
        # Adding element type (line 230)
        # Getting the type of 'label' (line 230)
        label_182028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 27), 'label')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 20), list_182026, label_182028)
        
        # Applying the binary operator '+=' (line 230)
        result_iadd_182029 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 12), '+=', argv_182025, list_182026)
        # Assigning a type to the variable 'argv' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'argv', result_iadd_182029)
        
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'argv' (line 231)
        argv_182030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'argv')
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_182031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        str_182032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 17), 'str', '--verbosity')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 16), list_182031, str_182032)
        # Adding element type (line 231)
        
        # Call to str(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'verbose' (line 231)
        verbose_182034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 36), 'verbose', False)
        # Processing the call keyword arguments (line 231)
        kwargs_182035 = {}
        # Getting the type of 'str' (line 231)
        str_182033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 32), 'str', False)
        # Calling str(args, kwargs) (line 231)
        str_call_result_182036 = invoke(stypy.reporting.localization.Localization(__file__, 231, 32), str_182033, *[verbose_182034], **kwargs_182035)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 16), list_182031, str_call_result_182036)
        
        # Applying the binary operator '+=' (line 231)
        result_iadd_182037 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 8), '+=', argv_182030, list_182031)
        # Assigning a type to the variable 'argv' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'argv', result_iadd_182037)
        
        
        # Getting the type of 'argv' (line 238)
        argv_182038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'argv')
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_182039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        str_182040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 17), 'str', '--exe')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 16), list_182039, str_182040)
        
        # Applying the binary operator '+=' (line 238)
        result_iadd_182041 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 8), '+=', argv_182038, list_182039)
        # Assigning a type to the variable 'argv' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'argv', result_iadd_182041)
        
        
        # Getting the type of 'extra_argv' (line 240)
        extra_argv_182042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'extra_argv')
        # Testing the type of an if condition (line 240)
        if_condition_182043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), extra_argv_182042)
        # Assigning a type to the variable 'if_condition_182043' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_182043', if_condition_182043)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'argv' (line 241)
        argv_182044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'argv')
        # Getting the type of 'extra_argv' (line 241)
        extra_argv_182045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'extra_argv')
        # Applying the binary operator '+=' (line 241)
        result_iadd_182046 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 12), '+=', argv_182044, extra_argv_182045)
        # Assigning a type to the variable 'argv' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'argv', result_iadd_182046)
        
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'argv' (line 242)
        argv_182047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'argv')
        # Assigning a type to the variable 'stypy_return_type' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'stypy_return_type', argv_182047)
        
        # ################# End of '_test_argv(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_test_argv' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_182048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_182048)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_test_argv'
        return stypy_return_type_182048


    @norecursion
    def _show_system_info(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_show_system_info'
        module_type_store = module_type_store.open_function_context('_show_system_info', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NoseTester._show_system_info.__dict__.__setitem__('stypy_localization', localization)
        NoseTester._show_system_info.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NoseTester._show_system_info.__dict__.__setitem__('stypy_type_store', module_type_store)
        NoseTester._show_system_info.__dict__.__setitem__('stypy_function_name', 'NoseTester._show_system_info')
        NoseTester._show_system_info.__dict__.__setitem__('stypy_param_names_list', [])
        NoseTester._show_system_info.__dict__.__setitem__('stypy_varargs_param_name', None)
        NoseTester._show_system_info.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NoseTester._show_system_info.__dict__.__setitem__('stypy_call_defaults', defaults)
        NoseTester._show_system_info.__dict__.__setitem__('stypy_call_varargs', varargs)
        NoseTester._show_system_info.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NoseTester._show_system_info.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NoseTester._show_system_info', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_show_system_info', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_show_system_info(...)' code ##################

        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to import_nose(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_182050 = {}
        # Getting the type of 'import_nose' (line 245)
        import_nose_182049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'import_nose', False)
        # Calling import_nose(args, kwargs) (line 245)
        import_nose_call_result_182051 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), import_nose_182049, *[], **kwargs_182050)
        
        # Assigning a type to the variable 'nose' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'nose', import_nose_call_result_182051)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 247, 8))
        
        # 'import numpy' statement (line 247)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_182052 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 247, 8), 'numpy')

        if (type(import_182052) is not StypyTypeError):

            if (import_182052 != 'pyd_module'):
                __import__(import_182052)
                sys_modules_182053 = sys.modules[import_182052]
                import_module(stypy.reporting.localization.Localization(__file__, 247, 8), 'numpy', sys_modules_182053.module_type_store, module_type_store)
            else:
                import numpy

                import_module(stypy.reporting.localization.Localization(__file__, 247, 8), 'numpy', numpy, module_type_store)

        else:
            # Assigning a type to the variable 'numpy' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'numpy', import_182052)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        
        
        # Call to print(...): (line 248)
        # Processing the call arguments (line 248)
        str_182055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 14), 'str', 'NumPy version %s')
        # Getting the type of 'numpy' (line 248)
        numpy_182056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 35), 'numpy', False)
        # Obtaining the member '__version__' of a type (line 248)
        version___182057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 35), numpy_182056, '__version__')
        # Applying the binary operator '%' (line 248)
        result_mod_182058 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 14), '%', str_182055, version___182057)
        
        # Processing the call keyword arguments (line 248)
        kwargs_182059 = {}
        # Getting the type of 'print' (line 248)
        print_182054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'print', False)
        # Calling print(args, kwargs) (line 248)
        print_call_result_182060 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), print_182054, *[result_mod_182058], **kwargs_182059)
        
        
        # Assigning a Attribute to a Name (line 249):
        
        # Assigning a Attribute to a Name (line 249):
        
        # Call to ones(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Obtaining an instance of the builtin type 'tuple' (line 249)
        tuple_182063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 249)
        # Adding element type (line 249)
        int_182064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 38), tuple_182063, int_182064)
        # Adding element type (line 249)
        int_182065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 38), tuple_182063, int_182065)
        
        # Processing the call keyword arguments (line 249)
        str_182066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 52), 'str', 'C')
        keyword_182067 = str_182066
        kwargs_182068 = {'order': keyword_182067}
        # Getting the type of 'numpy' (line 249)
        numpy_182061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 26), 'numpy', False)
        # Obtaining the member 'ones' of a type (line 249)
        ones_182062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 26), numpy_182061, 'ones')
        # Calling ones(args, kwargs) (line 249)
        ones_call_result_182069 = invoke(stypy.reporting.localization.Localization(__file__, 249, 26), ones_182062, *[tuple_182063], **kwargs_182068)
        
        # Obtaining the member 'flags' of a type (line 249)
        flags_182070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 26), ones_call_result_182069, 'flags')
        # Obtaining the member 'f_contiguous' of a type (line 249)
        f_contiguous_182071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 26), flags_182070, 'f_contiguous')
        # Assigning a type to the variable 'relaxed_strides' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'relaxed_strides', f_contiguous_182071)
        
        # Call to print(...): (line 250)
        # Processing the call arguments (line 250)
        str_182073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 14), 'str', 'NumPy relaxed strides checking option:')
        # Getting the type of 'relaxed_strides' (line 250)
        relaxed_strides_182074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 56), 'relaxed_strides', False)
        # Processing the call keyword arguments (line 250)
        kwargs_182075 = {}
        # Getting the type of 'print' (line 250)
        print_182072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'print', False)
        # Calling print(args, kwargs) (line 250)
        print_call_result_182076 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), print_182072, *[str_182073, relaxed_strides_182074], **kwargs_182075)
        
        
        # Assigning a Call to a Name (line 251):
        
        # Assigning a Call to a Name (line 251):
        
        # Call to dirname(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'numpy' (line 251)
        numpy_182080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 32), 'numpy', False)
        # Obtaining the member '__file__' of a type (line 251)
        file___182081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 32), numpy_182080, '__file__')
        # Processing the call keyword arguments (line 251)
        kwargs_182082 = {}
        # Getting the type of 'os' (line 251)
        os_182077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 251)
        path_182078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 16), os_182077, 'path')
        # Obtaining the member 'dirname' of a type (line 251)
        dirname_182079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 16), path_182078, 'dirname')
        # Calling dirname(args, kwargs) (line 251)
        dirname_call_result_182083 = invoke(stypy.reporting.localization.Localization(__file__, 251, 16), dirname_182079, *[file___182081], **kwargs_182082)
        
        # Assigning a type to the variable 'npdir' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'npdir', dirname_call_result_182083)
        
        # Call to print(...): (line 252)
        # Processing the call arguments (line 252)
        str_182085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 14), 'str', 'NumPy is installed in %s')
        # Getting the type of 'npdir' (line 252)
        npdir_182086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 43), 'npdir', False)
        # Applying the binary operator '%' (line 252)
        result_mod_182087 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 14), '%', str_182085, npdir_182086)
        
        # Processing the call keyword arguments (line 252)
        kwargs_182088 = {}
        # Getting the type of 'print' (line 252)
        print_182084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'print', False)
        # Calling print(args, kwargs) (line 252)
        print_call_result_182089 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), print_182084, *[result_mod_182087], **kwargs_182088)
        
        
        
        str_182090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 11), 'str', 'scipy')
        # Getting the type of 'self' (line 254)
        self_182091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 22), 'self')
        # Obtaining the member 'package_name' of a type (line 254)
        package_name_182092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 22), self_182091, 'package_name')
        # Applying the binary operator 'in' (line 254)
        result_contains_182093 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 11), 'in', str_182090, package_name_182092)
        
        # Testing the type of an if condition (line 254)
        if_condition_182094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 8), result_contains_182093)
        # Assigning a type to the variable 'if_condition_182094' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'if_condition_182094', if_condition_182094)
        # SSA begins for if statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 255, 12))
        
        # 'import scipy' statement (line 255)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_182095 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 255, 12), 'scipy')

        if (type(import_182095) is not StypyTypeError):

            if (import_182095 != 'pyd_module'):
                __import__(import_182095)
                sys_modules_182096 = sys.modules[import_182095]
                import_module(stypy.reporting.localization.Localization(__file__, 255, 12), 'scipy', sys_modules_182096.module_type_store, module_type_store)
            else:
                import scipy

                import_module(stypy.reporting.localization.Localization(__file__, 255, 12), 'scipy', scipy, module_type_store)

        else:
            # Assigning a type to the variable 'scipy' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'scipy', import_182095)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        
        
        # Call to print(...): (line 256)
        # Processing the call arguments (line 256)
        str_182098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 18), 'str', 'SciPy version %s')
        # Getting the type of 'scipy' (line 256)
        scipy_182099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 39), 'scipy', False)
        # Obtaining the member '__version__' of a type (line 256)
        version___182100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 39), scipy_182099, '__version__')
        # Applying the binary operator '%' (line 256)
        result_mod_182101 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 18), '%', str_182098, version___182100)
        
        # Processing the call keyword arguments (line 256)
        kwargs_182102 = {}
        # Getting the type of 'print' (line 256)
        print_182097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'print', False)
        # Calling print(args, kwargs) (line 256)
        print_call_result_182103 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), print_182097, *[result_mod_182101], **kwargs_182102)
        
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to dirname(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'scipy' (line 257)
        scipy_182107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 36), 'scipy', False)
        # Obtaining the member '__file__' of a type (line 257)
        file___182108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 36), scipy_182107, '__file__')
        # Processing the call keyword arguments (line 257)
        kwargs_182109 = {}
        # Getting the type of 'os' (line 257)
        os_182104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 257)
        path_182105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 20), os_182104, 'path')
        # Obtaining the member 'dirname' of a type (line 257)
        dirname_182106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 20), path_182105, 'dirname')
        # Calling dirname(args, kwargs) (line 257)
        dirname_call_result_182110 = invoke(stypy.reporting.localization.Localization(__file__, 257, 20), dirname_182106, *[file___182108], **kwargs_182109)
        
        # Assigning a type to the variable 'spdir' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'spdir', dirname_call_result_182110)
        
        # Call to print(...): (line 258)
        # Processing the call arguments (line 258)
        str_182112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 18), 'str', 'SciPy is installed in %s')
        # Getting the type of 'spdir' (line 258)
        spdir_182113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'spdir', False)
        # Applying the binary operator '%' (line 258)
        result_mod_182114 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 18), '%', str_182112, spdir_182113)
        
        # Processing the call keyword arguments (line 258)
        kwargs_182115 = {}
        # Getting the type of 'print' (line 258)
        print_182111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'print', False)
        # Calling print(args, kwargs) (line 258)
        print_call_result_182116 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), print_182111, *[result_mod_182114], **kwargs_182115)
        
        # SSA join for if statement (line 254)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to replace(...): (line 260)
        # Processing the call arguments (line 260)
        str_182120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 40), 'str', '\n')
        str_182121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 46), 'str', '')
        # Processing the call keyword arguments (line 260)
        kwargs_182122 = {}
        # Getting the type of 'sys' (line 260)
        sys_182117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'sys', False)
        # Obtaining the member 'version' of a type (line 260)
        version_182118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 20), sys_182117, 'version')
        # Obtaining the member 'replace' of a type (line 260)
        replace_182119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 20), version_182118, 'replace')
        # Calling replace(args, kwargs) (line 260)
        replace_call_result_182123 = invoke(stypy.reporting.localization.Localization(__file__, 260, 20), replace_182119, *[str_182120, str_182121], **kwargs_182122)
        
        # Assigning a type to the variable 'pyversion' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'pyversion', replace_call_result_182123)
        
        # Call to print(...): (line 261)
        # Processing the call arguments (line 261)
        str_182125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 14), 'str', 'Python version %s')
        # Getting the type of 'pyversion' (line 261)
        pyversion_182126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 36), 'pyversion', False)
        # Applying the binary operator '%' (line 261)
        result_mod_182127 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 14), '%', str_182125, pyversion_182126)
        
        # Processing the call keyword arguments (line 261)
        kwargs_182128 = {}
        # Getting the type of 'print' (line 261)
        print_182124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'print', False)
        # Calling print(args, kwargs) (line 261)
        print_call_result_182129 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), print_182124, *[result_mod_182127], **kwargs_182128)
        
        
        # Call to print(...): (line 262)
        # Processing the call arguments (line 262)
        str_182131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 14), 'str', 'nose version %d.%d.%d')
        # Getting the type of 'nose' (line 262)
        nose_182132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 40), 'nose', False)
        # Obtaining the member '__versioninfo__' of a type (line 262)
        versioninfo___182133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 40), nose_182132, '__versioninfo__')
        # Applying the binary operator '%' (line 262)
        result_mod_182134 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 14), '%', str_182131, versioninfo___182133)
        
        # Processing the call keyword arguments (line 262)
        kwargs_182135 = {}
        # Getting the type of 'print' (line 262)
        print_182130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'print', False)
        # Calling print(args, kwargs) (line 262)
        print_call_result_182136 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), print_182130, *[result_mod_182134], **kwargs_182135)
        
        
        # ################# End of '_show_system_info(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_show_system_info' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_182137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_182137)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_show_system_info'
        return stypy_return_type_182137


    @norecursion
    def _get_custom_doctester(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_custom_doctester'
        module_type_store = module_type_store.open_function_context('_get_custom_doctester', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_localization', localization)
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_type_store', module_type_store)
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_function_name', 'NoseTester._get_custom_doctester')
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_param_names_list', [])
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_varargs_param_name', None)
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_call_defaults', defaults)
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_call_varargs', varargs)
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NoseTester._get_custom_doctester.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NoseTester._get_custom_doctester', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_custom_doctester', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_custom_doctester(...)' code ##################

        str_182138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, (-1)), 'str', ' Return instantiated plugin for doctests\n\n        Allows subclassing of this class to override doctester\n\n        A return value of None means use the nose builtin doctest plugin\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 271, 8))
        
        # 'from numpy.testing.noseclasses import NumpyDoctest' statement (line 271)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_182139 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 271, 8), 'numpy.testing.noseclasses')

        if (type(import_182139) is not StypyTypeError):

            if (import_182139 != 'pyd_module'):
                __import__(import_182139)
                sys_modules_182140 = sys.modules[import_182139]
                import_from_module(stypy.reporting.localization.Localization(__file__, 271, 8), 'numpy.testing.noseclasses', sys_modules_182140.module_type_store, module_type_store, ['NumpyDoctest'])
                nest_module(stypy.reporting.localization.Localization(__file__, 271, 8), __file__, sys_modules_182140, sys_modules_182140.module_type_store, module_type_store)
            else:
                from numpy.testing.noseclasses import NumpyDoctest

                import_from_module(stypy.reporting.localization.Localization(__file__, 271, 8), 'numpy.testing.noseclasses', None, module_type_store, ['NumpyDoctest'], [NumpyDoctest])

        else:
            # Assigning a type to the variable 'numpy.testing.noseclasses' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'numpy.testing.noseclasses', import_182139)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        
        
        # Call to NumpyDoctest(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_182142 = {}
        # Getting the type of 'NumpyDoctest' (line 272)
        NumpyDoctest_182141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'NumpyDoctest', False)
        # Calling NumpyDoctest(args, kwargs) (line 272)
        NumpyDoctest_call_result_182143 = invoke(stypy.reporting.localization.Localization(__file__, 272, 15), NumpyDoctest_182141, *[], **kwargs_182142)
        
        # Assigning a type to the variable 'stypy_return_type' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'stypy_return_type', NumpyDoctest_call_result_182143)
        
        # ################# End of '_get_custom_doctester(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_custom_doctester' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_182144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_182144)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_custom_doctester'
        return stypy_return_type_182144


    @norecursion
    def prepare_test_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_182145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 38), 'str', 'fast')
        int_182146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 54), 'int')
        # Getting the type of 'None' (line 274)
        None_182147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 68), 'None')
        # Getting the type of 'False' (line 275)
        False_182148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'False')
        # Getting the type of 'False' (line 275)
        False_182149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 51), 'False')
        defaults = [str_182145, int_182146, None_182147, False_182148, False_182149]
        # Create a new context for function 'prepare_test_args'
        module_type_store = module_type_store.open_function_context('prepare_test_args', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_localization', localization)
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_function_name', 'NoseTester.prepare_test_args')
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_param_names_list', ['label', 'verbose', 'extra_argv', 'doctests', 'coverage'])
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NoseTester.prepare_test_args.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NoseTester.prepare_test_args', ['label', 'verbose', 'extra_argv', 'doctests', 'coverage'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'prepare_test_args', localization, ['label', 'verbose', 'extra_argv', 'doctests', 'coverage'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'prepare_test_args(...)' code ##################

        str_182150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, (-1)), 'str', '\n        Run tests for module using nose.\n\n        This method does the heavy lifting for the `test` method. It takes all\n        the same arguments, for details see `test`.\n\n        See Also\n        --------\n        test\n\n        ')
        
        # Call to import_nose(...): (line 288)
        # Processing the call keyword arguments (line 288)
        kwargs_182152 = {}
        # Getting the type of 'import_nose' (line 288)
        import_nose_182151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'import_nose', False)
        # Calling import_nose(args, kwargs) (line 288)
        import_nose_call_result_182153 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), import_nose_182151, *[], **kwargs_182152)
        
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to _test_argv(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'label' (line 290)
        label_182156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 31), 'label', False)
        # Getting the type of 'verbose' (line 290)
        verbose_182157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 38), 'verbose', False)
        # Getting the type of 'extra_argv' (line 290)
        extra_argv_182158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 47), 'extra_argv', False)
        # Processing the call keyword arguments (line 290)
        kwargs_182159 = {}
        # Getting the type of 'self' (line 290)
        self_182154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'self', False)
        # Obtaining the member '_test_argv' of a type (line 290)
        _test_argv_182155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 15), self_182154, '_test_argv')
        # Calling _test_argv(args, kwargs) (line 290)
        _test_argv_call_result_182160 = invoke(stypy.reporting.localization.Localization(__file__, 290, 15), _test_argv_182155, *[label_182156, verbose_182157, extra_argv_182158], **kwargs_182159)
        
        # Assigning a type to the variable 'argv' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'argv', _test_argv_call_result_182160)
        
        # Getting the type of 'coverage' (line 292)
        coverage_182161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'coverage')
        # Testing the type of an if condition (line 292)
        if_condition_182162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 8), coverage_182161)
        # Assigning a type to the variable 'if_condition_182162' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'if_condition_182162', if_condition_182162)
        # SSA begins for if statement (line 292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'argv' (line 293)
        argv_182163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'argv')
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_182164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        str_182165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 21), 'str', '--cover-package=%s')
        # Getting the type of 'self' (line 293)
        self_182166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 44), 'self')
        # Obtaining the member 'package_name' of a type (line 293)
        package_name_182167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 44), self_182166, 'package_name')
        # Applying the binary operator '%' (line 293)
        result_mod_182168 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 21), '%', str_182165, package_name_182167)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 20), list_182164, result_mod_182168)
        # Adding element type (line 293)
        str_182169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 63), 'str', '--with-coverage')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 20), list_182164, str_182169)
        # Adding element type (line 293)
        str_182170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 19), 'str', '--cover-tests')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 20), list_182164, str_182170)
        # Adding element type (line 293)
        str_182171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 36), 'str', '--cover-erase')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 20), list_182164, str_182171)
        
        # Applying the binary operator '+=' (line 293)
        result_iadd_182172 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 12), '+=', argv_182163, list_182164)
        # Assigning a type to the variable 'argv' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'argv', result_iadd_182172)
        
        # SSA join for if statement (line 292)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 296, 8))
        
        # 'import nose.plugins.builtin' statement (line 296)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_182173 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 296, 8), 'nose.plugins.builtin')

        if (type(import_182173) is not StypyTypeError):

            if (import_182173 != 'pyd_module'):
                __import__(import_182173)
                sys_modules_182174 = sys.modules[import_182173]
                import_module(stypy.reporting.localization.Localization(__file__, 296, 8), 'nose.plugins.builtin', sys_modules_182174.module_type_store, module_type_store)
            else:
                import nose.plugins.builtin

                import_module(stypy.reporting.localization.Localization(__file__, 296, 8), 'nose.plugins.builtin', nose.plugins.builtin, module_type_store)

        else:
            # Assigning a type to the variable 'nose.plugins.builtin' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'nose.plugins.builtin', import_182173)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 297, 8))
        
        # 'from numpy.testing.noseclasses import KnownFailurePlugin, Unplugger' statement (line 297)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_182175 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 297, 8), 'numpy.testing.noseclasses')

        if (type(import_182175) is not StypyTypeError):

            if (import_182175 != 'pyd_module'):
                __import__(import_182175)
                sys_modules_182176 = sys.modules[import_182175]
                import_from_module(stypy.reporting.localization.Localization(__file__, 297, 8), 'numpy.testing.noseclasses', sys_modules_182176.module_type_store, module_type_store, ['KnownFailurePlugin', 'Unplugger'])
                nest_module(stypy.reporting.localization.Localization(__file__, 297, 8), __file__, sys_modules_182176, sys_modules_182176.module_type_store, module_type_store)
            else:
                from numpy.testing.noseclasses import KnownFailurePlugin, Unplugger

                import_from_module(stypy.reporting.localization.Localization(__file__, 297, 8), 'numpy.testing.noseclasses', None, module_type_store, ['KnownFailurePlugin', 'Unplugger'], [KnownFailurePlugin, Unplugger])

        else:
            # Assigning a type to the variable 'numpy.testing.noseclasses' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'numpy.testing.noseclasses', import_182175)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        
        
        # Assigning a List to a Name (line 298):
        
        # Assigning a List to a Name (line 298):
        
        # Obtaining an instance of the builtin type 'list' (line 298)
        list_182177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 298)
        # Adding element type (line 298)
        
        # Call to KnownFailurePlugin(...): (line 298)
        # Processing the call keyword arguments (line 298)
        kwargs_182179 = {}
        # Getting the type of 'KnownFailurePlugin' (line 298)
        KnownFailurePlugin_182178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 19), 'KnownFailurePlugin', False)
        # Calling KnownFailurePlugin(args, kwargs) (line 298)
        KnownFailurePlugin_call_result_182180 = invoke(stypy.reporting.localization.Localization(__file__, 298, 19), KnownFailurePlugin_182178, *[], **kwargs_182179)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 18), list_182177, KnownFailurePlugin_call_result_182180)
        
        # Assigning a type to the variable 'plugins' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'plugins', list_182177)
        
        # Getting the type of 'plugins' (line 299)
        plugins_182181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'plugins')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'nose' (line 299)
        nose_182185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 33), 'nose')
        # Obtaining the member 'plugins' of a type (line 299)
        plugins_182186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 33), nose_182185, 'plugins')
        # Obtaining the member 'builtin' of a type (line 299)
        builtin_182187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 33), plugins_182186, 'builtin')
        # Obtaining the member 'plugins' of a type (line 299)
        plugins_182188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 33), builtin_182187, 'plugins')
        comprehension_182189 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 20), plugins_182188)
        # Assigning a type to the variable 'p' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 20), 'p', comprehension_182189)
        
        # Call to p(...): (line 299)
        # Processing the call keyword arguments (line 299)
        kwargs_182183 = {}
        # Getting the type of 'p' (line 299)
        p_182182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 20), 'p', False)
        # Calling p(args, kwargs) (line 299)
        p_call_result_182184 = invoke(stypy.reporting.localization.Localization(__file__, 299, 20), p_182182, *[], **kwargs_182183)
        
        list_182190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 20), list_182190, p_call_result_182184)
        # Applying the binary operator '+=' (line 299)
        result_iadd_182191 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 8), '+=', plugins_182181, list_182190)
        # Assigning a type to the variable 'plugins' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'plugins', result_iadd_182191)
        
        
        # Assigning a Compare to a Name (line 301):
        
        # Assigning a Compare to a Name (line 301):
        
        str_182192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 23), 'str', '--with-doctest')
        # Getting the type of 'argv' (line 301)
        argv_182193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 43), 'argv')
        # Applying the binary operator 'in' (line 301)
        result_contains_182194 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 23), 'in', str_182192, argv_182193)
        
        # Assigning a type to the variable 'doctest_argv' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'doctest_argv', result_contains_182194)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'doctests' (line 302)
        doctests_182195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'doctests')
        # Getting the type of 'False' (line 302)
        False_182196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 23), 'False')
        # Applying the binary operator '==' (line 302)
        result_eq_182197 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 11), '==', doctests_182195, False_182196)
        
        # Getting the type of 'doctest_argv' (line 302)
        doctest_argv_182198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 33), 'doctest_argv')
        # Applying the binary operator 'and' (line 302)
        result_and_keyword_182199 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 11), 'and', result_eq_182197, doctest_argv_182198)
        
        # Testing the type of an if condition (line 302)
        if_condition_182200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 8), result_and_keyword_182199)
        # Assigning a type to the variable 'if_condition_182200' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'if_condition_182200', if_condition_182200)
        # SSA begins for if statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 303):
        
        # Assigning a Name to a Name (line 303):
        # Getting the type of 'True' (line 303)
        True_182201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'True')
        # Assigning a type to the variable 'doctests' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'doctests', True_182201)
        # SSA join for if statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 304):
        
        # Assigning a Call to a Name (line 304):
        
        # Call to _get_custom_doctester(...): (line 304)
        # Processing the call keyword arguments (line 304)
        kwargs_182204 = {}
        # Getting the type of 'self' (line 304)
        self_182202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'self', False)
        # Obtaining the member '_get_custom_doctester' of a type (line 304)
        _get_custom_doctester_182203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 15), self_182202, '_get_custom_doctester')
        # Calling _get_custom_doctester(args, kwargs) (line 304)
        _get_custom_doctester_call_result_182205 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), _get_custom_doctester_182203, *[], **kwargs_182204)
        
        # Assigning a type to the variable 'plug' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'plug', _get_custom_doctester_call_result_182205)
        
        # Type idiom detected: calculating its left and rigth part (line 305)
        # Getting the type of 'plug' (line 305)
        plug_182206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'plug')
        # Getting the type of 'None' (line 305)
        None_182207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'None')
        
        (may_be_182208, more_types_in_union_182209) = may_be_none(plug_182206, None_182207)

        if may_be_182208:

            if more_types_in_union_182209:
                # Runtime conditional SSA (line 305)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Evaluating a boolean operation
            # Getting the type of 'doctests' (line 307)
            doctests_182210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'doctests')
            
            # Getting the type of 'doctest_argv' (line 307)
            doctest_argv_182211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 32), 'doctest_argv')
            # Applying the 'not' unary operator (line 307)
            result_not__182212 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 28), 'not', doctest_argv_182211)
            
            # Applying the binary operator 'and' (line 307)
            result_and_keyword_182213 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 15), 'and', doctests_182210, result_not__182212)
            
            # Testing the type of an if condition (line 307)
            if_condition_182214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 12), result_and_keyword_182213)
            # Assigning a type to the variable 'if_condition_182214' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'if_condition_182214', if_condition_182214)
            # SSA begins for if statement (line 307)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'argv' (line 308)
            argv_182215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'argv')
            
            # Obtaining an instance of the builtin type 'list' (line 308)
            list_182216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 308)
            # Adding element type (line 308)
            str_182217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 25), 'str', '--with-doctest')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 24), list_182216, str_182217)
            
            # Applying the binary operator '+=' (line 308)
            result_iadd_182218 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 16), '+=', argv_182215, list_182216)
            # Assigning a type to the variable 'argv' (line 308)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'argv', result_iadd_182218)
            
            # SSA join for if statement (line 307)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_182209:
                # Runtime conditional SSA for else branch (line 305)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_182208) or more_types_in_union_182209):
            
            # Getting the type of 'doctest_argv' (line 310)
            doctest_argv_182219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'doctest_argv')
            # Testing the type of an if condition (line 310)
            if_condition_182220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 12), doctest_argv_182219)
            # Assigning a type to the variable 'if_condition_182220' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'if_condition_182220', if_condition_182220)
            # SSA begins for if statement (line 310)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove(...): (line 311)
            # Processing the call arguments (line 311)
            str_182223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 28), 'str', '--with-doctest')
            # Processing the call keyword arguments (line 311)
            kwargs_182224 = {}
            # Getting the type of 'argv' (line 311)
            argv_182221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'argv', False)
            # Obtaining the member 'remove' of a type (line 311)
            remove_182222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 16), argv_182221, 'remove')
            # Calling remove(args, kwargs) (line 311)
            remove_call_result_182225 = invoke(stypy.reporting.localization.Localization(__file__, 311, 16), remove_182222, *[str_182223], **kwargs_182224)
            
            # SSA join for if statement (line 310)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'plugins' (line 312)
            plugins_182226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'plugins')
            
            # Obtaining an instance of the builtin type 'list' (line 312)
            list_182227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 312)
            # Adding element type (line 312)
            
            # Call to Unplugger(...): (line 312)
            # Processing the call arguments (line 312)
            str_182229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 34), 'str', 'doctest')
            # Processing the call keyword arguments (line 312)
            kwargs_182230 = {}
            # Getting the type of 'Unplugger' (line 312)
            Unplugger_182228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'Unplugger', False)
            # Calling Unplugger(args, kwargs) (line 312)
            Unplugger_call_result_182231 = invoke(stypy.reporting.localization.Localization(__file__, 312, 24), Unplugger_182228, *[str_182229], **kwargs_182230)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 23), list_182227, Unplugger_call_result_182231)
            # Adding element type (line 312)
            # Getting the type of 'plug' (line 312)
            plug_182232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 46), 'plug')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 23), list_182227, plug_182232)
            
            # Applying the binary operator '+=' (line 312)
            result_iadd_182233 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 12), '+=', plugins_182226, list_182227)
            # Assigning a type to the variable 'plugins' (line 312)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'plugins', result_iadd_182233)
            
            
            # Getting the type of 'doctests' (line 313)
            doctests_182234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'doctests')
            # Testing the type of an if condition (line 313)
            if_condition_182235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 12), doctests_182234)
            # Assigning a type to the variable 'if_condition_182235' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'if_condition_182235', if_condition_182235)
            # SSA begins for if statement (line 313)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'argv' (line 314)
            argv_182236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'argv')
            
            # Obtaining an instance of the builtin type 'list' (line 314)
            list_182237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 314)
            # Adding element type (line 314)
            str_182238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 25), 'str', '--with-')
            # Getting the type of 'plug' (line 314)
            plug_182239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 37), 'plug')
            # Obtaining the member 'name' of a type (line 314)
            name_182240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 37), plug_182239, 'name')
            # Applying the binary operator '+' (line 314)
            result_add_182241 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 25), '+', str_182238, name_182240)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 24), list_182237, result_add_182241)
            
            # Applying the binary operator '+=' (line 314)
            result_iadd_182242 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 16), '+=', argv_182236, list_182237)
            # Assigning a type to the variable 'argv' (line 314)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'argv', result_iadd_182242)
            
            # SSA join for if statement (line 313)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_182208 and more_types_in_union_182209):
                # SSA join for if statement (line 305)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 315)
        tuple_182243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 315)
        # Adding element type (line 315)
        # Getting the type of 'argv' (line 315)
        argv_182244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'argv')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 15), tuple_182243, argv_182244)
        # Adding element type (line 315)
        # Getting the type of 'plugins' (line 315)
        plugins_182245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 21), 'plugins')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 15), tuple_182243, plugins_182245)
        
        # Assigning a type to the variable 'stypy_return_type' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'stypy_return_type', tuple_182243)
        
        # ################# End of 'prepare_test_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'prepare_test_args' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_182246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_182246)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'prepare_test_args'
        return stypy_return_type_182246


    @norecursion
    def test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_182247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 25), 'str', 'fast')
        int_182248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 41), 'int')
        # Getting the type of 'None' (line 317)
        None_182249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 55), 'None')
        # Getting the type of 'False' (line 318)
        False_182250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 21), 'False')
        # Getting the type of 'False' (line 318)
        False_182251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 37), 'False')
        # Getting the type of 'None' (line 319)
        None_182252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 27), 'None')
        defaults = [str_182247, int_182248, None_182249, False_182250, False_182251, None_182252]
        # Create a new context for function 'test'
        module_type_store = module_type_store.open_function_context('test', 317, 4, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NoseTester.test.__dict__.__setitem__('stypy_localization', localization)
        NoseTester.test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NoseTester.test.__dict__.__setitem__('stypy_type_store', module_type_store)
        NoseTester.test.__dict__.__setitem__('stypy_function_name', 'NoseTester.test')
        NoseTester.test.__dict__.__setitem__('stypy_param_names_list', ['label', 'verbose', 'extra_argv', 'doctests', 'coverage', 'raise_warnings'])
        NoseTester.test.__dict__.__setitem__('stypy_varargs_param_name', None)
        NoseTester.test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NoseTester.test.__dict__.__setitem__('stypy_call_defaults', defaults)
        NoseTester.test.__dict__.__setitem__('stypy_call_varargs', varargs)
        NoseTester.test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NoseTester.test.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NoseTester.test', ['label', 'verbose', 'extra_argv', 'doctests', 'coverage', 'raise_warnings'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test', localization, ['label', 'verbose', 'extra_argv', 'doctests', 'coverage', 'raise_warnings'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test(...)' code ##################

        str_182253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, (-1)), 'str', '\n        Run tests for module using nose.\n\n        Parameters\n        ----------\n        label : {\'fast\', \'full\', \'\', attribute identifier}, optional\n            Identifies the tests to run. This can be a string to pass to\n            the nosetests executable with the \'-A\' option, or one of several\n            special values.  Special values are:\n            * \'fast\' - the default - which corresponds to the ``nosetests -A``\n              option of \'not slow\'.\n            * \'full\' - fast (as above) and slow tests as in the\n              \'no -A\' option to nosetests - this is the same as \'\'.\n            * None or \'\' - run all tests.\n            attribute_identifier - string passed directly to nosetests as \'-A\'.\n        verbose : int, optional\n            Verbosity value for test outputs, in the range 1-10. Default is 1.\n        extra_argv : list, optional\n            List with any extra arguments to pass to nosetests.\n        doctests : bool, optional\n            If True, run doctests in module. Default is False.\n        coverage : bool, optional\n            If True, report coverage of NumPy code. Default is False.\n            (This requires the `coverage module:\n             <http://nedbatchelder.com/code/modules/coverage.html>`_).\n        raise_warnings : str or sequence of warnings, optional\n            This specifies which warnings to configure as \'raise\' instead\n            of \'warn\' during the test execution.  Valid strings are:\n\n              - "develop" : equals ``(DeprecationWarning, RuntimeWarning)``\n              - "release" : equals ``()``, don\'t raise on any warnings.\n\n        Returns\n        -------\n        result : object\n            Returns the result of running the tests as a\n            ``nose.result.TextTestResult`` object.\n\n        Notes\n        -----\n        Each NumPy module exposes `test` in its namespace to run all tests for it.\n        For example, to run all tests for numpy.lib:\n\n        >>> np.lib.test() #doctest: +SKIP\n\n        Examples\n        --------\n        >>> result = np.lib.test() #doctest: +SKIP\n        Running unit tests for numpy.lib\n        ...\n        Ran 976 tests in 3.933s\n\n        OK\n\n        >>> result.errors #doctest: +SKIP\n        []\n        >>> result.knownfail #doctest: +SKIP\n        []\n        ')
        
        # Assigning a Call to a Name (line 381):
        
        # Assigning a Call to a Name (line 381):
        
        # Call to min(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'verbose' (line 381)
        verbose_182255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 22), 'verbose', False)
        int_182256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 31), 'int')
        # Processing the call keyword arguments (line 381)
        kwargs_182257 = {}
        # Getting the type of 'min' (line 381)
        min_182254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 18), 'min', False)
        # Calling min(args, kwargs) (line 381)
        min_call_result_182258 = invoke(stypy.reporting.localization.Localization(__file__, 381, 18), min_182254, *[verbose_182255, int_182256], **kwargs_182257)
        
        # Assigning a type to the variable 'verbose' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'verbose', min_call_result_182258)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 383, 8))
        
        # 'from numpy.testing import utils' statement (line 383)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_182259 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'numpy.testing')

        if (type(import_182259) is not StypyTypeError):

            if (import_182259 != 'pyd_module'):
                __import__(import_182259)
                sys_modules_182260 = sys.modules[import_182259]
                import_from_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'numpy.testing', sys_modules_182260.module_type_store, module_type_store, ['utils'])
                nest_module(stypy.reporting.localization.Localization(__file__, 383, 8), __file__, sys_modules_182260, sys_modules_182260.module_type_store, module_type_store)
            else:
                from numpy.testing import utils

                import_from_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'numpy.testing', None, module_type_store, ['utils'], [utils])

        else:
            # Assigning a type to the variable 'numpy.testing' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'numpy.testing', import_182259)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        
        
        # Assigning a Name to a Attribute (line 384):
        
        # Assigning a Name to a Attribute (line 384):
        # Getting the type of 'verbose' (line 384)
        verbose_182261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'verbose')
        # Getting the type of 'utils' (line 384)
        utils_182262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'utils')
        # Setting the type of the member 'verbose' of a type (line 384)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), utils_182262, 'verbose', verbose_182261)
        
        # Getting the type of 'doctests' (line 386)
        doctests_182263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 11), 'doctests')
        # Testing the type of an if condition (line 386)
        if_condition_182264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 8), doctests_182263)
        # Assigning a type to the variable 'if_condition_182264' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'if_condition_182264', if_condition_182264)
        # SSA begins for if statement (line 386)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to print(...): (line 387)
        # Processing the call arguments (line 387)
        str_182266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 18), 'str', 'Running unit tests and doctests for %s')
        # Getting the type of 'self' (line 387)
        self_182267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 61), 'self', False)
        # Obtaining the member 'package_name' of a type (line 387)
        package_name_182268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 61), self_182267, 'package_name')
        # Applying the binary operator '%' (line 387)
        result_mod_182269 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 18), '%', str_182266, package_name_182268)
        
        # Processing the call keyword arguments (line 387)
        kwargs_182270 = {}
        # Getting the type of 'print' (line 387)
        print_182265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'print', False)
        # Calling print(args, kwargs) (line 387)
        print_call_result_182271 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), print_182265, *[result_mod_182269], **kwargs_182270)
        
        # SSA branch for the else part of an if statement (line 386)
        module_type_store.open_ssa_branch('else')
        
        # Call to print(...): (line 389)
        # Processing the call arguments (line 389)
        str_182273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 18), 'str', 'Running unit tests for %s')
        # Getting the type of 'self' (line 389)
        self_182274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 48), 'self', False)
        # Obtaining the member 'package_name' of a type (line 389)
        package_name_182275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 48), self_182274, 'package_name')
        # Applying the binary operator '%' (line 389)
        result_mod_182276 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 18), '%', str_182273, package_name_182275)
        
        # Processing the call keyword arguments (line 389)
        kwargs_182277 = {}
        # Getting the type of 'print' (line 389)
        print_182272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'print', False)
        # Calling print(args, kwargs) (line 389)
        print_call_result_182278 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), print_182272, *[result_mod_182276], **kwargs_182277)
        
        # SSA join for if statement (line 386)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _show_system_info(...): (line 391)
        # Processing the call keyword arguments (line 391)
        kwargs_182281 = {}
        # Getting the type of 'self' (line 391)
        self_182279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'self', False)
        # Obtaining the member '_show_system_info' of a type (line 391)
        _show_system_info_182280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), self_182279, '_show_system_info')
        # Calling _show_system_info(args, kwargs) (line 391)
        _show_system_info_call_result_182282 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), _show_system_info_182280, *[], **kwargs_182281)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 394, 8))
        
        # 'import doctest' statement (line 394)
        import doctest

        import_module(stypy.reporting.localization.Localization(__file__, 394, 8), 'doctest', doctest, module_type_store)
        
        
        # Assigning a Name to a Attribute (line 395):
        
        # Assigning a Name to a Attribute (line 395):
        # Getting the type of 'None' (line 395)
        None_182283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 25), 'None')
        # Getting the type of 'doctest' (line 395)
        doctest_182284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'doctest')
        # Setting the type of the member 'master' of a type (line 395)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), doctest_182284, 'master', None_182283)
        
        # Type idiom detected: calculating its left and rigth part (line 397)
        # Getting the type of 'raise_warnings' (line 397)
        raise_warnings_182285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 11), 'raise_warnings')
        # Getting the type of 'None' (line 397)
        None_182286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 29), 'None')
        
        (may_be_182287, more_types_in_union_182288) = may_be_none(raise_warnings_182285, None_182286)

        if may_be_182287:

            if more_types_in_union_182288:
                # Runtime conditional SSA (line 397)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 398):
            
            # Assigning a Attribute to a Name (line 398):
            # Getting the type of 'self' (line 398)
            self_182289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 29), 'self')
            # Obtaining the member 'raise_warnings' of a type (line 398)
            raise_warnings_182290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 29), self_182289, 'raise_warnings')
            # Assigning a type to the variable 'raise_warnings' (line 398)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'raise_warnings', raise_warnings_182290)

            if more_types_in_union_182288:
                # SSA join for if statement (line 397)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 400):
        
        # Assigning a Call to a Name (line 400):
        
        # Call to dict(...): (line 400)
        # Processing the call keyword arguments (line 400)
        
        # Obtaining an instance of the builtin type 'tuple' (line 400)
        tuple_182292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 400)
        # Adding element type (line 400)
        # Getting the type of 'DeprecationWarning' (line 400)
        DeprecationWarning_182293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 35), 'DeprecationWarning', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 35), tuple_182292, DeprecationWarning_182293)
        # Adding element type (line 400)
        # Getting the type of 'RuntimeWarning' (line 400)
        RuntimeWarning_182294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 55), 'RuntimeWarning', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 35), tuple_182292, RuntimeWarning_182294)
        
        keyword_182295 = tuple_182292
        
        # Obtaining an instance of the builtin type 'tuple' (line 401)
        tuple_182296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 401)
        
        keyword_182297 = tuple_182296
        kwargs_182298 = {'release': keyword_182297, 'develop': keyword_182295}
        # Getting the type of 'dict' (line 400)
        dict_182291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 21), 'dict', False)
        # Calling dict(args, kwargs) (line 400)
        dict_call_result_182299 = invoke(stypy.reporting.localization.Localization(__file__, 400, 21), dict_182291, *[], **kwargs_182298)
        
        # Assigning a type to the variable '_warn_opts' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), '_warn_opts', dict_call_result_182299)
        
        # Type idiom detected: calculating its left and rigth part (line 402)
        # Getting the type of 'basestring' (line 402)
        basestring_182300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 38), 'basestring')
        # Getting the type of 'raise_warnings' (line 402)
        raise_warnings_182301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 22), 'raise_warnings')
        
        (may_be_182302, more_types_in_union_182303) = may_be_subtype(basestring_182300, raise_warnings_182301)

        if may_be_182302:

            if more_types_in_union_182303:
                # Runtime conditional SSA (line 402)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'raise_warnings' (line 402)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'raise_warnings', remove_not_subtype_from_union(raise_warnings_182301, basestring))
            
            # Assigning a Subscript to a Name (line 403):
            
            # Assigning a Subscript to a Name (line 403):
            
            # Obtaining the type of the subscript
            # Getting the type of 'raise_warnings' (line 403)
            raise_warnings_182304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 40), 'raise_warnings')
            # Getting the type of '_warn_opts' (line 403)
            _warn_opts_182305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 29), '_warn_opts')
            # Obtaining the member '__getitem__' of a type (line 403)
            getitem___182306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 29), _warn_opts_182305, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 403)
            subscript_call_result_182307 = invoke(stypy.reporting.localization.Localization(__file__, 403, 29), getitem___182306, raise_warnings_182304)
            
            # Assigning a type to the variable 'raise_warnings' (line 403)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'raise_warnings', subscript_call_result_182307)

            if more_types_in_union_182303:
                # SSA join for if statement (line 402)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to catch_warnings(...): (line 405)
        # Processing the call keyword arguments (line 405)
        kwargs_182310 = {}
        # Getting the type of 'warnings' (line 405)
        warnings_182308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 13), 'warnings', False)
        # Obtaining the member 'catch_warnings' of a type (line 405)
        catch_warnings_182309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 13), warnings_182308, 'catch_warnings')
        # Calling catch_warnings(args, kwargs) (line 405)
        catch_warnings_call_result_182311 = invoke(stypy.reporting.localization.Localization(__file__, 405, 13), catch_warnings_182309, *[], **kwargs_182310)
        
        with_182312 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 405, 13), catch_warnings_call_result_182311, 'with parameter', '__enter__', '__exit__')

        if with_182312:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 405)
            enter___182313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 13), catch_warnings_call_result_182311, '__enter__')
            with_enter_182314 = invoke(stypy.reporting.localization.Localization(__file__, 405, 13), enter___182313)
            
            # Call to resetwarnings(...): (line 408)
            # Processing the call keyword arguments (line 408)
            kwargs_182317 = {}
            # Getting the type of 'warnings' (line 408)
            warnings_182315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'warnings', False)
            # Obtaining the member 'resetwarnings' of a type (line 408)
            resetwarnings_182316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), warnings_182315, 'resetwarnings')
            # Calling resetwarnings(args, kwargs) (line 408)
            resetwarnings_call_result_182318 = invoke(stypy.reporting.localization.Localization(__file__, 408, 12), resetwarnings_182316, *[], **kwargs_182317)
            
            
            # Call to filterwarnings(...): (line 411)
            # Processing the call arguments (line 411)
            str_182321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 36), 'str', 'always')
            # Processing the call keyword arguments (line 411)
            kwargs_182322 = {}
            # Getting the type of 'warnings' (line 411)
            warnings_182319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 411)
            filterwarnings_182320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 12), warnings_182319, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 411)
            filterwarnings_call_result_182323 = invoke(stypy.reporting.localization.Localization(__file__, 411, 12), filterwarnings_182320, *[str_182321], **kwargs_182322)
            
            
            # Getting the type of 'raise_warnings' (line 413)
            raise_warnings_182324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 31), 'raise_warnings')
            # Testing the type of a for loop iterable (line 413)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 413, 12), raise_warnings_182324)
            # Getting the type of the for loop variable (line 413)
            for_loop_var_182325 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 413, 12), raise_warnings_182324)
            # Assigning a type to the variable 'warningtype' (line 413)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'warningtype', for_loop_var_182325)
            # SSA begins for a for statement (line 413)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to filterwarnings(...): (line 414)
            # Processing the call arguments (line 414)
            str_182328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 40), 'str', 'error')
            # Processing the call keyword arguments (line 414)
            # Getting the type of 'warningtype' (line 414)
            warningtype_182329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 58), 'warningtype', False)
            keyword_182330 = warningtype_182329
            kwargs_182331 = {'category': keyword_182330}
            # Getting the type of 'warnings' (line 414)
            warnings_182326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 16), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 414)
            filterwarnings_182327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 16), warnings_182326, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 414)
            filterwarnings_call_result_182332 = invoke(stypy.reporting.localization.Localization(__file__, 414, 16), filterwarnings_182327, *[str_182328], **kwargs_182331)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to filterwarnings(...): (line 416)
            # Processing the call arguments (line 416)
            str_182335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 36), 'str', 'ignore')
            # Processing the call keyword arguments (line 416)
            str_182336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 54), 'str', 'Not importing directory')
            keyword_182337 = str_182336
            kwargs_182338 = {'message': keyword_182337}
            # Getting the type of 'warnings' (line 416)
            warnings_182333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 416)
            filterwarnings_182334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 12), warnings_182333, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 416)
            filterwarnings_call_result_182339 = invoke(stypy.reporting.localization.Localization(__file__, 416, 12), filterwarnings_182334, *[str_182335], **kwargs_182338)
            
            
            # Call to filterwarnings(...): (line 417)
            # Processing the call arguments (line 417)
            str_182342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 36), 'str', 'ignore')
            # Processing the call keyword arguments (line 417)
            str_182343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 54), 'str', 'numpy.dtype size changed')
            keyword_182344 = str_182343
            kwargs_182345 = {'message': keyword_182344}
            # Getting the type of 'warnings' (line 417)
            warnings_182340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 417)
            filterwarnings_182341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 12), warnings_182340, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 417)
            filterwarnings_call_result_182346 = invoke(stypy.reporting.localization.Localization(__file__, 417, 12), filterwarnings_182341, *[str_182342], **kwargs_182345)
            
            
            # Call to filterwarnings(...): (line 418)
            # Processing the call arguments (line 418)
            str_182349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 36), 'str', 'ignore')
            # Processing the call keyword arguments (line 418)
            str_182350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 54), 'str', 'numpy.ufunc size changed')
            keyword_182351 = str_182350
            kwargs_182352 = {'message': keyword_182351}
            # Getting the type of 'warnings' (line 418)
            warnings_182347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 418)
            filterwarnings_182348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 12), warnings_182347, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 418)
            filterwarnings_call_result_182353 = invoke(stypy.reporting.localization.Localization(__file__, 418, 12), filterwarnings_182348, *[str_182349], **kwargs_182352)
            
            
            # Call to filterwarnings(...): (line 419)
            # Processing the call arguments (line 419)
            str_182356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 36), 'str', 'ignore')
            # Processing the call keyword arguments (line 419)
            # Getting the type of 'np' (line 419)
            np_182357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 55), 'np', False)
            # Obtaining the member 'ModuleDeprecationWarning' of a type (line 419)
            ModuleDeprecationWarning_182358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 55), np_182357, 'ModuleDeprecationWarning')
            keyword_182359 = ModuleDeprecationWarning_182358
            kwargs_182360 = {'category': keyword_182359}
            # Getting the type of 'warnings' (line 419)
            warnings_182354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 419)
            filterwarnings_182355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 12), warnings_182354, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 419)
            filterwarnings_call_result_182361 = invoke(stypy.reporting.localization.Localization(__file__, 419, 12), filterwarnings_182355, *[str_182356], **kwargs_182360)
            
            
            # Call to filterwarnings(...): (line 420)
            # Processing the call arguments (line 420)
            str_182364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 36), 'str', 'ignore')
            # Processing the call keyword arguments (line 420)
            # Getting the type of 'FutureWarning' (line 420)
            FutureWarning_182365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 55), 'FutureWarning', False)
            keyword_182366 = FutureWarning_182365
            kwargs_182367 = {'category': keyword_182366}
            # Getting the type of 'warnings' (line 420)
            warnings_182362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 420)
            filterwarnings_182363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 12), warnings_182362, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 420)
            filterwarnings_call_result_182368 = invoke(stypy.reporting.localization.Localization(__file__, 420, 12), filterwarnings_182363, *[str_182364], **kwargs_182367)
            
            
            # Call to filterwarnings(...): (line 423)
            # Processing the call arguments (line 423)
            str_182371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 36), 'str', 'ignore')
            # Processing the call keyword arguments (line 423)
            str_182372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 54), 'str', '.*boolean negative.*')
            keyword_182373 = str_182372
            kwargs_182374 = {'message': keyword_182373}
            # Getting the type of 'warnings' (line 423)
            warnings_182369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 423)
            filterwarnings_182370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), warnings_182369, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 423)
            filterwarnings_call_result_182375 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), filterwarnings_182370, *[str_182371], **kwargs_182374)
            
            
            # Call to filterwarnings(...): (line 424)
            # Processing the call arguments (line 424)
            str_182378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 36), 'str', 'ignore')
            # Processing the call keyword arguments (line 424)
            str_182379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 54), 'str', '.*boolean subtract.*')
            keyword_182380 = str_182379
            kwargs_182381 = {'message': keyword_182380}
            # Getting the type of 'warnings' (line 424)
            warnings_182376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 424)
            filterwarnings_182377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), warnings_182376, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 424)
            filterwarnings_call_result_182382 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), filterwarnings_182377, *[str_182378], **kwargs_182381)
            
            
            # Call to filterwarnings(...): (line 428)
            # Processing the call arguments (line 428)
            str_182385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 36), 'str', 'ignore')
            # Processing the call keyword arguments (line 428)
            str_182386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 54), 'str', '.*getargspec.*')
            keyword_182387 = str_182386
            # Getting the type of 'DeprecationWarning' (line 429)
            DeprecationWarning_182388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 45), 'DeprecationWarning', False)
            keyword_182389 = DeprecationWarning_182388
            str_182390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 43), 'str', 'nose\\.')
            keyword_182391 = str_182390
            kwargs_182392 = {'category': keyword_182389, 'message': keyword_182387, 'module': keyword_182391}
            # Getting the type of 'warnings' (line 428)
            warnings_182383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 428)
            filterwarnings_182384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 12), warnings_182383, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 428)
            filterwarnings_call_result_182393 = invoke(stypy.reporting.localization.Localization(__file__, 428, 12), filterwarnings_182384, *[str_182385], **kwargs_182392)
            
            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 432, 12))
            
            # 'from numpy.testing.noseclasses import NumpyTestProgram' statement (line 432)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
            import_182394 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 432, 12), 'numpy.testing.noseclasses')

            if (type(import_182394) is not StypyTypeError):

                if (import_182394 != 'pyd_module'):
                    __import__(import_182394)
                    sys_modules_182395 = sys.modules[import_182394]
                    import_from_module(stypy.reporting.localization.Localization(__file__, 432, 12), 'numpy.testing.noseclasses', sys_modules_182395.module_type_store, module_type_store, ['NumpyTestProgram'])
                    nest_module(stypy.reporting.localization.Localization(__file__, 432, 12), __file__, sys_modules_182395, sys_modules_182395.module_type_store, module_type_store)
                else:
                    from numpy.testing.noseclasses import NumpyTestProgram

                    import_from_module(stypy.reporting.localization.Localization(__file__, 432, 12), 'numpy.testing.noseclasses', None, module_type_store, ['NumpyTestProgram'], [NumpyTestProgram])

            else:
                # Assigning a type to the variable 'numpy.testing.noseclasses' (line 432)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'numpy.testing.noseclasses', import_182394)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
            
            
            # Assigning a Call to a Tuple (line 434):
            
            # Assigning a Call to a Name:
            
            # Call to prepare_test_args(...): (line 434)
            # Processing the call arguments (line 434)
            # Getting the type of 'label' (line 435)
            label_182398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'label', False)
            # Getting the type of 'verbose' (line 435)
            verbose_182399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 27), 'verbose', False)
            # Getting the type of 'extra_argv' (line 435)
            extra_argv_182400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 36), 'extra_argv', False)
            # Getting the type of 'doctests' (line 435)
            doctests_182401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 48), 'doctests', False)
            # Getting the type of 'coverage' (line 435)
            coverage_182402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 58), 'coverage', False)
            # Processing the call keyword arguments (line 434)
            kwargs_182403 = {}
            # Getting the type of 'self' (line 434)
            self_182396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 28), 'self', False)
            # Obtaining the member 'prepare_test_args' of a type (line 434)
            prepare_test_args_182397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 28), self_182396, 'prepare_test_args')
            # Calling prepare_test_args(args, kwargs) (line 434)
            prepare_test_args_call_result_182404 = invoke(stypy.reporting.localization.Localization(__file__, 434, 28), prepare_test_args_182397, *[label_182398, verbose_182399, extra_argv_182400, doctests_182401, coverage_182402], **kwargs_182403)
            
            # Assigning a type to the variable 'call_assignment_181752' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'call_assignment_181752', prepare_test_args_call_result_182404)
            
            # Assigning a Call to a Name (line 434):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_182407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 12), 'int')
            # Processing the call keyword arguments
            kwargs_182408 = {}
            # Getting the type of 'call_assignment_181752' (line 434)
            call_assignment_181752_182405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'call_assignment_181752', False)
            # Obtaining the member '__getitem__' of a type (line 434)
            getitem___182406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), call_assignment_181752_182405, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_182409 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___182406, *[int_182407], **kwargs_182408)
            
            # Assigning a type to the variable 'call_assignment_181753' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'call_assignment_181753', getitem___call_result_182409)
            
            # Assigning a Name to a Name (line 434):
            # Getting the type of 'call_assignment_181753' (line 434)
            call_assignment_181753_182410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'call_assignment_181753')
            # Assigning a type to the variable 'argv' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'argv', call_assignment_181753_182410)
            
            # Assigning a Call to a Name (line 434):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_182413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 12), 'int')
            # Processing the call keyword arguments
            kwargs_182414 = {}
            # Getting the type of 'call_assignment_181752' (line 434)
            call_assignment_181752_182411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'call_assignment_181752', False)
            # Obtaining the member '__getitem__' of a type (line 434)
            getitem___182412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), call_assignment_181752_182411, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_182415 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___182412, *[int_182413], **kwargs_182414)
            
            # Assigning a type to the variable 'call_assignment_181754' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'call_assignment_181754', getitem___call_result_182415)
            
            # Assigning a Name to a Name (line 434):
            # Getting the type of 'call_assignment_181754' (line 434)
            call_assignment_181754_182416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'call_assignment_181754')
            # Assigning a type to the variable 'plugins' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 18), 'plugins', call_assignment_181754_182416)
            
            # Assigning a Call to a Name (line 436):
            
            # Assigning a Call to a Name (line 436):
            
            # Call to NumpyTestProgram(...): (line 436)
            # Processing the call keyword arguments (line 436)
            # Getting the type of 'argv' (line 436)
            argv_182418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 38), 'argv', False)
            keyword_182419 = argv_182418
            # Getting the type of 'False' (line 436)
            False_182420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 49), 'False', False)
            keyword_182421 = False_182420
            # Getting the type of 'plugins' (line 436)
            plugins_182422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 64), 'plugins', False)
            keyword_182423 = plugins_182422
            kwargs_182424 = {'exit': keyword_182421, 'argv': keyword_182419, 'plugins': keyword_182423}
            # Getting the type of 'NumpyTestProgram' (line 436)
            NumpyTestProgram_182417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'NumpyTestProgram', False)
            # Calling NumpyTestProgram(args, kwargs) (line 436)
            NumpyTestProgram_call_result_182425 = invoke(stypy.reporting.localization.Localization(__file__, 436, 16), NumpyTestProgram_182417, *[], **kwargs_182424)
            
            # Assigning a type to the variable 't' (line 436)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 't', NumpyTestProgram_call_result_182425)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 405)
            exit___182426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 13), catch_warnings_call_result_182311, '__exit__')
            with_exit_182427 = invoke(stypy.reporting.localization.Localization(__file__, 405, 13), exit___182426, None, None, None)

        # Getting the type of 't' (line 438)
        t_182428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 't')
        # Obtaining the member 'result' of a type (line 438)
        result_182429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 15), t_182428, 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'stypy_return_type', result_182429)
        
        # ################# End of 'test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_182430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_182430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test'
        return stypy_return_type_182430


    @norecursion
    def bench(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_182431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 26), 'str', 'fast')
        int_182432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 42), 'int')
        # Getting the type of 'None' (line 440)
        None_182433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 56), 'None')
        defaults = [str_182431, int_182432, None_182433]
        # Create a new context for function 'bench'
        module_type_store = module_type_store.open_function_context('bench', 440, 4, False)
        # Assigning a type to the variable 'self' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NoseTester.bench.__dict__.__setitem__('stypy_localization', localization)
        NoseTester.bench.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NoseTester.bench.__dict__.__setitem__('stypy_type_store', module_type_store)
        NoseTester.bench.__dict__.__setitem__('stypy_function_name', 'NoseTester.bench')
        NoseTester.bench.__dict__.__setitem__('stypy_param_names_list', ['label', 'verbose', 'extra_argv'])
        NoseTester.bench.__dict__.__setitem__('stypy_varargs_param_name', None)
        NoseTester.bench.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NoseTester.bench.__dict__.__setitem__('stypy_call_defaults', defaults)
        NoseTester.bench.__dict__.__setitem__('stypy_call_varargs', varargs)
        NoseTester.bench.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NoseTester.bench.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NoseTester.bench', ['label', 'verbose', 'extra_argv'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bench', localization, ['label', 'verbose', 'extra_argv'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bench(...)' code ##################

        str_182434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, (-1)), 'str', '\n        Run benchmarks for module using nose.\n\n        Parameters\n        ----------\n        label : {\'fast\', \'full\', \'\', attribute identifier}, optional\n            Identifies the benchmarks to run. This can be a string to pass to\n            the nosetests executable with the \'-A\' option, or one of several\n            special values.  Special values are:\n            * \'fast\' - the default - which corresponds to the ``nosetests -A``\n              option of \'not slow\'.\n            * \'full\' - fast (as above) and slow benchmarks as in the\n              \'no -A\' option to nosetests - this is the same as \'\'.\n            * None or \'\' - run all tests.\n            attribute_identifier - string passed directly to nosetests as \'-A\'.\n        verbose : int, optional\n            Verbosity value for benchmark outputs, in the range 1-10. Default is 1.\n        extra_argv : list, optional\n            List with any extra arguments to pass to nosetests.\n\n        Returns\n        -------\n        success : bool\n            Returns True if running the benchmarks works, False if an error\n            occurred.\n\n        Notes\n        -----\n        Benchmarks are like tests, but have names starting with "bench" instead\n        of "test", and can be found under the "benchmarks" sub-directory of the\n        module.\n\n        Each NumPy module exposes `bench` in its namespace to run all benchmarks\n        for it.\n\n        Examples\n        --------\n        >>> success = np.lib.bench() #doctest: +SKIP\n        Running benchmarks for numpy.lib\n        ...\n        using 562341 items:\n        unique:\n        0.11\n        unique1d:\n        0.11\n        ratio: 1.0\n        nUnique: 56230 == 56230\n        ...\n        OK\n\n        >>> success #doctest: +SKIP\n        True\n\n        ')
        
        # Call to print(...): (line 496)
        # Processing the call arguments (line 496)
        str_182436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 14), 'str', 'Running benchmarks for %s')
        # Getting the type of 'self' (line 496)
        self_182437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 44), 'self', False)
        # Obtaining the member 'package_name' of a type (line 496)
        package_name_182438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 44), self_182437, 'package_name')
        # Applying the binary operator '%' (line 496)
        result_mod_182439 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 14), '%', str_182436, package_name_182438)
        
        # Processing the call keyword arguments (line 496)
        kwargs_182440 = {}
        # Getting the type of 'print' (line 496)
        print_182435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'print', False)
        # Calling print(args, kwargs) (line 496)
        print_call_result_182441 = invoke(stypy.reporting.localization.Localization(__file__, 496, 8), print_182435, *[result_mod_182439], **kwargs_182440)
        
        
        # Call to _show_system_info(...): (line 497)
        # Processing the call keyword arguments (line 497)
        kwargs_182444 = {}
        # Getting the type of 'self' (line 497)
        self_182442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'self', False)
        # Obtaining the member '_show_system_info' of a type (line 497)
        _show_system_info_182443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), self_182442, '_show_system_info')
        # Calling _show_system_info(args, kwargs) (line 497)
        _show_system_info_call_result_182445 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), _show_system_info_182443, *[], **kwargs_182444)
        
        
        # Assigning a Call to a Name (line 499):
        
        # Assigning a Call to a Name (line 499):
        
        # Call to _test_argv(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 'label' (line 499)
        label_182448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 31), 'label', False)
        # Getting the type of 'verbose' (line 499)
        verbose_182449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 38), 'verbose', False)
        # Getting the type of 'extra_argv' (line 499)
        extra_argv_182450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 47), 'extra_argv', False)
        # Processing the call keyword arguments (line 499)
        kwargs_182451 = {}
        # Getting the type of 'self' (line 499)
        self_182446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), 'self', False)
        # Obtaining the member '_test_argv' of a type (line 499)
        _test_argv_182447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 15), self_182446, '_test_argv')
        # Calling _test_argv(args, kwargs) (line 499)
        _test_argv_call_result_182452 = invoke(stypy.reporting.localization.Localization(__file__, 499, 15), _test_argv_182447, *[label_182448, verbose_182449, extra_argv_182450], **kwargs_182451)
        
        # Assigning a type to the variable 'argv' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'argv', _test_argv_call_result_182452)
        
        # Getting the type of 'argv' (line 500)
        argv_182453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'argv')
        
        # Obtaining an instance of the builtin type 'list' (line 500)
        list_182454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 500)
        # Adding element type (line 500)
        str_182455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 17), 'str', '--match')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 16), list_182454, str_182455)
        # Adding element type (line 500)
        str_182456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 28), 'str', '(?:^|[\\\\b_\\\\.%s-])[Bb]ench')
        # Getting the type of 'os' (line 500)
        os_182457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 60), 'os')
        # Obtaining the member 'sep' of a type (line 500)
        sep_182458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 60), os_182457, 'sep')
        # Applying the binary operator '%' (line 500)
        result_mod_182459 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 28), '%', str_182456, sep_182458)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 500, 16), list_182454, result_mod_182459)
        
        # Applying the binary operator '+=' (line 500)
        result_iadd_182460 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 8), '+=', argv_182453, list_182454)
        # Assigning a type to the variable 'argv' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'argv', result_iadd_182460)
        
        
        # Assigning a Call to a Name (line 503):
        
        # Assigning a Call to a Name (line 503):
        
        # Call to import_nose(...): (line 503)
        # Processing the call keyword arguments (line 503)
        kwargs_182462 = {}
        # Getting the type of 'import_nose' (line 503)
        import_nose_182461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 15), 'import_nose', False)
        # Calling import_nose(args, kwargs) (line 503)
        import_nose_call_result_182463 = invoke(stypy.reporting.localization.Localization(__file__, 503, 15), import_nose_182461, *[], **kwargs_182462)
        
        # Assigning a type to the variable 'nose' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'nose', import_nose_call_result_182463)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 506, 8))
        
        # 'from numpy.testing.noseclasses import Unplugger' statement (line 506)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
        import_182464 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 506, 8), 'numpy.testing.noseclasses')

        if (type(import_182464) is not StypyTypeError):

            if (import_182464 != 'pyd_module'):
                __import__(import_182464)
                sys_modules_182465 = sys.modules[import_182464]
                import_from_module(stypy.reporting.localization.Localization(__file__, 506, 8), 'numpy.testing.noseclasses', sys_modules_182465.module_type_store, module_type_store, ['Unplugger'])
                nest_module(stypy.reporting.localization.Localization(__file__, 506, 8), __file__, sys_modules_182465, sys_modules_182465.module_type_store, module_type_store)
            else:
                from numpy.testing.noseclasses import Unplugger

                import_from_module(stypy.reporting.localization.Localization(__file__, 506, 8), 'numpy.testing.noseclasses', None, module_type_store, ['Unplugger'], [Unplugger])

        else:
            # Assigning a type to the variable 'numpy.testing.noseclasses' (line 506)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'numpy.testing.noseclasses', import_182464)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
        
        
        # Assigning a List to a Name (line 507):
        
        # Assigning a List to a Name (line 507):
        
        # Obtaining an instance of the builtin type 'list' (line 507)
        list_182466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 507)
        # Adding element type (line 507)
        
        # Call to Unplugger(...): (line 507)
        # Processing the call arguments (line 507)
        str_182468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 33), 'str', 'doctest')
        # Processing the call keyword arguments (line 507)
        kwargs_182469 = {}
        # Getting the type of 'Unplugger' (line 507)
        Unplugger_182467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 23), 'Unplugger', False)
        # Calling Unplugger(args, kwargs) (line 507)
        Unplugger_call_result_182470 = invoke(stypy.reporting.localization.Localization(__file__, 507, 23), Unplugger_182467, *[str_182468], **kwargs_182469)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 22), list_182466, Unplugger_call_result_182470)
        
        # Assigning a type to the variable 'add_plugins' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'add_plugins', list_182466)
        
        # Call to run(...): (line 509)
        # Processing the call keyword arguments (line 509)
        # Getting the type of 'argv' (line 509)
        argv_182473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 29), 'argv', False)
        keyword_182474 = argv_182473
        # Getting the type of 'add_plugins' (line 509)
        add_plugins_182475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 46), 'add_plugins', False)
        keyword_182476 = add_plugins_182475
        kwargs_182477 = {'addplugins': keyword_182476, 'argv': keyword_182474}
        # Getting the type of 'nose' (line 509)
        nose_182471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 15), 'nose', False)
        # Obtaining the member 'run' of a type (line 509)
        run_182472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 15), nose_182471, 'run')
        # Calling run(args, kwargs) (line 509)
        run_call_result_182478 = invoke(stypy.reporting.localization.Localization(__file__, 509, 15), run_182472, *[], **kwargs_182477)
        
        # Assigning a type to the variable 'stypy_return_type' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'stypy_return_type', run_call_result_182478)
        
        # ################# End of 'bench(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bench' in the type store
        # Getting the type of 'stypy_return_type' (line 440)
        stypy_return_type_182479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_182479)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bench'
        return stypy_return_type_182479


# Assigning a type to the variable 'NoseTester' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'NoseTester', NoseTester)

@norecursion
def _numpy_tester(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_numpy_tester'
    module_type_store = module_type_store.open_function_context('_numpy_tester', 511, 0, False)
    
    # Passed parameters checking function
    _numpy_tester.stypy_localization = localization
    _numpy_tester.stypy_type_of_self = None
    _numpy_tester.stypy_type_store = module_type_store
    _numpy_tester.stypy_function_name = '_numpy_tester'
    _numpy_tester.stypy_param_names_list = []
    _numpy_tester.stypy_varargs_param_name = None
    _numpy_tester.stypy_kwargs_param_name = None
    _numpy_tester.stypy_call_defaults = defaults
    _numpy_tester.stypy_call_varargs = varargs
    _numpy_tester.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_numpy_tester', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_numpy_tester', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_numpy_tester(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'np' (line 512)
    np_182481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 15), 'np', False)
    str_182482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 19), 'str', '__version__')
    # Processing the call keyword arguments (line 512)
    kwargs_182483 = {}
    # Getting the type of 'hasattr' (line 512)
    hasattr_182480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 512)
    hasattr_call_result_182484 = invoke(stypy.reporting.localization.Localization(__file__, 512, 7), hasattr_182480, *[np_182481, str_182482], **kwargs_182483)
    
    
    str_182485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 38), 'str', '.dev0')
    # Getting the type of 'np' (line 512)
    np_182486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 49), 'np')
    # Obtaining the member '__version__' of a type (line 512)
    version___182487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 49), np_182486, '__version__')
    # Applying the binary operator 'in' (line 512)
    result_contains_182488 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 38), 'in', str_182485, version___182487)
    
    # Applying the binary operator 'and' (line 512)
    result_and_keyword_182489 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 7), 'and', hasattr_call_result_182484, result_contains_182488)
    
    # Testing the type of an if condition (line 512)
    if_condition_182490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 4), result_and_keyword_182489)
    # Assigning a type to the variable 'if_condition_182490' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'if_condition_182490', if_condition_182490)
    # SSA begins for if statement (line 512)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 513):
    
    # Assigning a Str to a Name (line 513):
    str_182491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 15), 'str', 'develop')
    # Assigning a type to the variable 'mode' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'mode', str_182491)
    # SSA branch for the else part of an if statement (line 512)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 515):
    
    # Assigning a Str to a Name (line 515):
    str_182492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 15), 'str', 'release')
    # Assigning a type to the variable 'mode' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'mode', str_182492)
    # SSA join for if statement (line 512)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to NoseTester(...): (line 516)
    # Processing the call keyword arguments (line 516)
    # Getting the type of 'mode' (line 516)
    mode_182494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 37), 'mode', False)
    keyword_182495 = mode_182494
    int_182496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 49), 'int')
    keyword_182497 = int_182496
    kwargs_182498 = {'raise_warnings': keyword_182495, 'depth': keyword_182497}
    # Getting the type of 'NoseTester' (line 516)
    NoseTester_182493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'NoseTester', False)
    # Calling NoseTester(args, kwargs) (line 516)
    NoseTester_call_result_182499 = invoke(stypy.reporting.localization.Localization(__file__, 516, 11), NoseTester_182493, *[], **kwargs_182498)
    
    # Assigning a type to the variable 'stypy_return_type' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'stypy_return_type', NoseTester_call_result_182499)
    
    # ################# End of '_numpy_tester(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_numpy_tester' in the type store
    # Getting the type of 'stypy_return_type' (line 511)
    stypy_return_type_182500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_182500)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_numpy_tester'
    return stypy_return_type_182500

# Assigning a type to the variable '_numpy_tester' (line 511)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 0), '_numpy_tester', _numpy_tester)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
