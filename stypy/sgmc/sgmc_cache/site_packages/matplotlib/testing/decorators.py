
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import functools
7: import inspect
8: import os
9: import sys
10: import shutil
11: import warnings
12: import unittest
13: 
14: # Note - don't import nose up here - import it only as needed in functions.
15: # This allows other functions here to be used by pytest-based testing suites
16: # without requiring nose to be installed.
17: 
18: 
19: import matplotlib as mpl
20: import matplotlib.style
21: import matplotlib.units
22: import matplotlib.testing
23: from matplotlib import cbook
24: from matplotlib import ticker
25: from matplotlib import pyplot as plt
26: from matplotlib import ft2font
27: from matplotlib.testing.compare import comparable_formats, compare_images, \
28:      make_test_filename
29: from . import _copy_metadata, is_called_from_pytest
30: from .exceptions import ImageComparisonFailure
31: 
32: 
33: def _knownfailureif(fail_condition, msg=None, known_exception_class=None):
34:     '''
35: 
36:     Assume a will fail if *fail_condition* is True. *fail_condition*
37:     may also be False or the string 'indeterminate'.
38: 
39:     *msg* is the error message displayed for the test.
40: 
41:     If *known_exception_class* is not None, the failure is only known
42:     if the exception is an instance of this class. (Default = None)
43: 
44:     '''
45:     if is_called_from_pytest():
46:         import pytest
47:         if fail_condition == 'indeterminate':
48:             fail_condition, strict = True, False
49:         else:
50:             fail_condition, strict = bool(fail_condition), True
51:         return pytest.mark.xfail(condition=fail_condition, reason=msg,
52:                                  raises=known_exception_class, strict=strict)
53:     else:
54:         from ._nose.decorators import knownfailureif
55:         return knownfailureif(fail_condition, msg, known_exception_class)
56: 
57: 
58: @cbook.deprecated('2.1',
59:                   alternative='pytest.xfail or import the plugin')
60: def knownfailureif(fail_condition, msg=None, known_exception_class=None):
61:     _knownfailureif(fail_condition, msg, known_exception_class)
62: 
63: 
64: def _do_cleanup(original_units_registry, original_settings):
65:     plt.close('all')
66: 
67:     mpl.rcParams.clear()
68:     mpl.rcParams.update(original_settings)
69:     matplotlib.units.registry.clear()
70:     matplotlib.units.registry.update(original_units_registry)
71:     warnings.resetwarnings()  # reset any warning filters set in tests
72: 
73: 
74: class CleanupTest(object):
75:     @classmethod
76:     def setup_class(cls):
77:         cls.original_units_registry = matplotlib.units.registry.copy()
78:         cls.original_settings = mpl.rcParams.copy()
79:         matplotlib.testing.setup()
80: 
81:     @classmethod
82:     def teardown_class(cls):
83:         _do_cleanup(cls.original_units_registry,
84:                     cls.original_settings)
85: 
86:     def test(self):
87:         self._func()
88: 
89: 
90: class CleanupTestCase(unittest.TestCase):
91:     '''A wrapper for unittest.TestCase that includes cleanup operations'''
92:     @classmethod
93:     def setUpClass(cls):
94:         import matplotlib.units
95:         cls.original_units_registry = matplotlib.units.registry.copy()
96:         cls.original_settings = mpl.rcParams.copy()
97: 
98:     @classmethod
99:     def tearDownClass(cls):
100:         _do_cleanup(cls.original_units_registry,
101:                     cls.original_settings)
102: 
103: 
104: def cleanup(style=None):
105:     '''
106:     A decorator to ensure that any global state is reset before
107:     running a test.
108: 
109:     Parameters
110:     ----------
111:     style : str, optional
112:         The name of the style to apply.
113:     '''
114: 
115:     # If cleanup is used without arguments, `style` will be a
116:     # callable, and we pass it directly to the wrapper generator.  If
117:     # cleanup if called with an argument, it is a string naming a
118:     # style, and the function will be passed as an argument to what we
119:     # return.  This is a confusing, but somewhat standard, pattern for
120:     # writing a decorator with optional arguments.
121: 
122:     def make_cleanup(func):
123:         if inspect.isgeneratorfunction(func):
124:             @functools.wraps(func)
125:             def wrapped_callable(*args, **kwargs):
126:                 original_units_registry = matplotlib.units.registry.copy()
127:                 original_settings = mpl.rcParams.copy()
128:                 matplotlib.style.use(style)
129:                 try:
130:                     for yielded in func(*args, **kwargs):
131:                         yield yielded
132:                 finally:
133:                     _do_cleanup(original_units_registry,
134:                                 original_settings)
135:         else:
136:             @functools.wraps(func)
137:             def wrapped_callable(*args, **kwargs):
138:                 original_units_registry = matplotlib.units.registry.copy()
139:                 original_settings = mpl.rcParams.copy()
140:                 matplotlib.style.use(style)
141:                 try:
142:                     func(*args, **kwargs)
143:                 finally:
144:                     _do_cleanup(original_units_registry,
145:                                 original_settings)
146: 
147:         return wrapped_callable
148: 
149:     if isinstance(style, six.string_types):
150:         return make_cleanup
151:     else:
152:         result = make_cleanup(style)
153:         # Default of mpl_test_settings fixture and image_comparison too.
154:         style = '_classic_test'
155:         return result
156: 
157: 
158: def check_freetype_version(ver):
159:     if ver is None:
160:         return True
161: 
162:     from distutils import version
163:     if isinstance(ver, six.string_types):
164:         ver = (ver, ver)
165:     ver = [version.StrictVersion(x) for x in ver]
166:     found = version.StrictVersion(ft2font.__freetype_version__)
167: 
168:     return found >= ver[0] and found <= ver[1]
169: 
170: 
171: def _checked_on_freetype_version(required_freetype_version):
172:     if check_freetype_version(required_freetype_version):
173:         return lambda f: f
174: 
175:     reason = ("Mismatched version of freetype. "
176:               "Test requires '%s', you have '%s'" %
177:               (required_freetype_version, ft2font.__freetype_version__))
178:     return _knownfailureif('indeterminate', msg=reason,
179:                            known_exception_class=ImageComparisonFailure)
180: 
181: 
182: def remove_ticks_and_titles(figure):
183:     figure.suptitle("")
184:     null_formatter = ticker.NullFormatter()
185:     for ax in figure.get_axes():
186:         ax.set_title("")
187:         ax.xaxis.set_major_formatter(null_formatter)
188:         ax.xaxis.set_minor_formatter(null_formatter)
189:         ax.yaxis.set_major_formatter(null_formatter)
190:         ax.yaxis.set_minor_formatter(null_formatter)
191:         try:
192:             ax.zaxis.set_major_formatter(null_formatter)
193:             ax.zaxis.set_minor_formatter(null_formatter)
194:         except AttributeError:
195:             pass
196: 
197: 
198: def _raise_on_image_difference(expected, actual, tol):
199:     __tracebackhide__ = True
200: 
201:     err = compare_images(expected, actual, tol, in_decorator=True)
202: 
203:     if not os.path.exists(expected):
204:         raise ImageComparisonFailure('image does not exist: %s' % expected)
205: 
206:     if err:
207:         for key in ["actual", "expected"]:
208:             err[key] = os.path.relpath(err[key])
209:         raise ImageComparisonFailure(
210:             'images not close (RMS %(rms).3f):\n\t%(actual)s\n\t%(expected)s '
211:              % err)
212: 
213: 
214: def _xfail_if_format_is_uncomparable(extension):
215:     will_fail = extension not in comparable_formats()
216:     if will_fail:
217:         fail_msg = 'Cannot compare %s files on this system' % extension
218:     else:
219:         fail_msg = 'No failure expected'
220: 
221:     return _knownfailureif(will_fail, fail_msg,
222:                            known_exception_class=ImageComparisonFailure)
223: 
224: 
225: def _mark_xfail_if_format_is_uncomparable(extension):
226:     if isinstance(extension, six.string_types):
227:         will_fail = extension not in comparable_formats()
228:     else:
229:         # Extension might be a pytest marker instead of a plain string.
230:         will_fail = extension.args[0] not in comparable_formats()
231:     if will_fail:
232:         fail_msg = 'Cannot compare %s files on this system' % extension
233:         import pytest
234:         return pytest.mark.xfail(extension, reason=fail_msg, strict=False,
235:                                  raises=ImageComparisonFailure)
236:     else:
237:         return extension
238: 
239: 
240: class _ImageComparisonBase(object):
241:     '''
242:     Image comparison base class
243: 
244:     This class provides *just* the comparison-related functionality and avoids
245:     any code that would be specific to any testing framework.
246:     '''
247:     def __init__(self, tol, remove_text, savefig_kwargs):
248:         self.func = self.baseline_dir = self.result_dir = None
249:         self.tol = tol
250:         self.remove_text = remove_text
251:         self.savefig_kwargs = savefig_kwargs
252: 
253:     def delayed_init(self, func):
254:         assert self.func is None, "it looks like same decorator used twice"
255:         self.func = func
256:         self.baseline_dir, self.result_dir = _image_directories(func)
257: 
258:     def copy_baseline(self, baseline, extension):
259:         baseline_path = os.path.join(self.baseline_dir, baseline)
260:         orig_expected_fname = baseline_path + '.' + extension
261:         if extension == 'eps' and not os.path.exists(orig_expected_fname):
262:             orig_expected_fname = baseline_path + '.pdf'
263:         expected_fname = make_test_filename(os.path.join(
264:             self.result_dir, os.path.basename(orig_expected_fname)), 'expected')
265:         if os.path.exists(orig_expected_fname):
266:             shutil.copyfile(orig_expected_fname, expected_fname)
267:         else:
268:             reason = ("Do not have baseline image {0} because this "
269:                       "file does not exist: {1}".format(expected_fname,
270:                                                         orig_expected_fname))
271:             raise ImageComparisonFailure(reason)
272:         return expected_fname
273: 
274:     def compare(self, idx, baseline, extension):
275:         __tracebackhide__ = True
276:         fignum = plt.get_fignums()[idx]
277:         fig = plt.figure(fignum)
278: 
279:         if self.remove_text:
280:             remove_ticks_and_titles(fig)
281: 
282:         actual_fname = os.path.join(self.result_dir, baseline) + '.' + extension
283:         kwargs = self.savefig_kwargs.copy()
284:         if extension == 'pdf':
285:             kwargs.setdefault('metadata',
286:                               {'Creator': None, 'Producer': None,
287:                                'CreationDate': None})
288:         fig.savefig(actual_fname, **kwargs)
289: 
290:         expected_fname = self.copy_baseline(baseline, extension)
291:         _raise_on_image_difference(expected_fname, actual_fname, self.tol)
292: 
293: 
294: class ImageComparisonTest(CleanupTest, _ImageComparisonBase):
295:     '''
296:     Nose-based image comparison class
297: 
298:     This class generates tests for a nose-based testing framework. Ideally,
299:     this class would not be public, and the only publically visible API would
300:     be the :func:`image_comparison` decorator. Unfortunately, there are
301:     existing downstream users of this class (e.g., pytest-mpl) so it cannot yet
302:     be removed.
303:     '''
304:     def __init__(self, baseline_images, extensions, tol,
305:                  freetype_version, remove_text, savefig_kwargs, style):
306:         _ImageComparisonBase.__init__(self, tol, remove_text, savefig_kwargs)
307:         self.baseline_images = baseline_images
308:         self.extensions = extensions
309:         self.freetype_version = freetype_version
310:         self.style = style
311: 
312:     def setup(self):
313:         func = self.func
314:         plt.close('all')
315:         self.setup_class()
316:         try:
317:             matplotlib.style.use(self.style)
318:             matplotlib.testing.set_font_settings_for_testing()
319:             func()
320:             assert len(plt.get_fignums()) == len(self.baseline_images), (
321:                 "Test generated {} images but there are {} baseline images"
322:                 .format(len(plt.get_fignums()), len(self.baseline_images)))
323:         except:
324:             # Restore original settings before raising errors.
325:             self.teardown_class()
326:             raise
327: 
328:     def teardown(self):
329:         self.teardown_class()
330: 
331:     @staticmethod
332:     @cbook.deprecated('2.1',
333:                       alternative='remove_ticks_and_titles')
334:     def remove_text(figure):
335:         remove_ticks_and_titles(figure)
336: 
337:     def nose_runner(self):
338:         func = self.compare
339:         func = _checked_on_freetype_version(self.freetype_version)(func)
340:         funcs = {extension: _xfail_if_format_is_uncomparable(extension)(func)
341:                  for extension in self.extensions}
342:         for idx, baseline in enumerate(self.baseline_images):
343:             for extension in self.extensions:
344:                 yield funcs[extension], idx, baseline, extension
345: 
346:     def __call__(self, func):
347:         self.delayed_init(func)
348:         import nose.tools
349: 
350:         @nose.tools.with_setup(self.setup, self.teardown)
351:         def runner_wrapper():
352:             for case in self.nose_runner():
353:                 yield case
354: 
355:         return _copy_metadata(func, runner_wrapper)
356: 
357: 
358: def _pytest_image_comparison(baseline_images, extensions, tol,
359:                              freetype_version, remove_text, savefig_kwargs,
360:                              style):
361:     '''
362:     Decorate function with image comparison for pytest.
363: 
364:     This function creates a decorator that wraps a figure-generating function
365:     with image comparison code. Pytest can become confused if we change the
366:     signature of the function, so we indirectly pass anything we need via the
367:     `mpl_image_comparison_parameters` fixture and extra markers.
368:     '''
369:     import pytest
370: 
371:     extensions = map(_mark_xfail_if_format_is_uncomparable, extensions)
372: 
373:     def decorator(func):
374:         # Parameter indirection; see docstring above and comment below.
375:         @pytest.mark.usefixtures('mpl_image_comparison_parameters')
376:         @pytest.mark.parametrize('extension', extensions)
377:         @pytest.mark.baseline_images(baseline_images)
378:         # END Parameter indirection.
379:         @pytest.mark.style(style)
380:         @_checked_on_freetype_version(freetype_version)
381:         @functools.wraps(func)
382:         def wrapper(*args, **kwargs):
383:             __tracebackhide__ = True
384:             img = _ImageComparisonBase(tol=tol, remove_text=remove_text,
385:                                        savefig_kwargs=savefig_kwargs)
386:             img.delayed_init(func)
387:             matplotlib.testing.set_font_settings_for_testing()
388:             func(*args, **kwargs)
389: 
390:             # Parameter indirection:
391:             # This is hacked on via the mpl_image_comparison_parameters fixture
392:             # so that we don't need to modify the function's real signature for
393:             # any parametrization. Modifying the signature is very very tricky
394:             # and likely to confuse pytest.
395:             baseline_images, extension = func.parameters
396: 
397:             assert len(plt.get_fignums()) == len(baseline_images), (
398:                 "Test generated {} images but there are {} baseline images"
399:                 .format(len(plt.get_fignums()), len(baseline_images)))
400:             for idx, baseline in enumerate(baseline_images):
401:                 img.compare(idx, baseline, extension)
402: 
403:         wrapper.__wrapped__ = func  # For Python 2.7.
404:         return _copy_metadata(func, wrapper)
405: 
406:     return decorator
407: 
408: 
409: def image_comparison(baseline_images, extensions=None, tol=0,
410:                      freetype_version=None, remove_text=False,
411:                      savefig_kwarg=None,
412:                      # Default of mpl_test_settings fixture and cleanup too.
413:                      style='_classic_test'):
414:     '''
415:     Compare images generated by the test with those specified in
416:     *baseline_images*, which must correspond else an
417:     ImageComparisonFailure exception will be raised.
418: 
419:     Arguments
420:     ---------
421:     baseline_images : list or None
422:         A list of strings specifying the names of the images generated by
423:         calls to :meth:`matplotlib.figure.savefig`.
424: 
425:         If *None*, the test function must use the ``baseline_images`` fixture,
426:         either as a parameter or with pytest.mark.usefixtures. This value is
427:         only allowed when using pytest.
428: 
429:     extensions : [ None | list ]
430: 
431:         If None, defaults to all supported extensions.
432:         Otherwise, a list of extensions to test. For example ['png','pdf'].
433: 
434:     tol : float, optional, default: 0
435:         The RMS threshold above which the test is considered failed.
436: 
437:     freetype_version : str or tuple
438:         The expected freetype version or range of versions for this test to
439:         pass.
440: 
441:     remove_text : bool
442:         Remove the title and tick text from the figure before comparison.
443:         This does not remove other, more deliberate, text, such as legends and
444:         annotations.
445: 
446:     savefig_kwarg : dict
447:         Optional arguments that are passed to the savefig method.
448: 
449:     style : string
450:         Optional name for the base style to apply to the image test. The test
451:         itself can also apply additional styles if desired. Defaults to the
452:         '_classic_test' style.
453: 
454:     '''
455:     if extensions is None:
456:         # default extensions to test
457:         extensions = ['png', 'pdf', 'svg']
458: 
459:     if savefig_kwarg is None:
460:         #default no kwargs to savefig
461:         savefig_kwarg = dict()
462: 
463:     if is_called_from_pytest():
464:         return _pytest_image_comparison(
465:             baseline_images=baseline_images, extensions=extensions, tol=tol,
466:             freetype_version=freetype_version, remove_text=remove_text,
467:             savefig_kwargs=savefig_kwarg, style=style)
468:     else:
469:         if baseline_images is None:
470:             raise ValueError('baseline_images must be specified')
471: 
472:         return ImageComparisonTest(
473:             baseline_images=baseline_images, extensions=extensions, tol=tol,
474:             freetype_version=freetype_version, remove_text=remove_text,
475:             savefig_kwargs=savefig_kwarg, style=style)
476: 
477: 
478: def _image_directories(func):
479:     '''
480:     Compute the baseline and result image directories for testing *func*.
481:     Create the result directory if it doesn't exist.
482:     '''
483:     module_name = func.__module__
484:     if module_name == '__main__':
485:         # FIXME: this won't work for nested packages in matplotlib.tests
486:         warnings.warn('test module run as script. guessing baseline image locations')
487:         script_name = sys.argv[0]
488:         basedir = os.path.abspath(os.path.dirname(script_name))
489:         subdir = os.path.splitext(os.path.split(script_name)[1])[0]
490:     else:
491:         mods = module_name.split('.')
492:         if len(mods) >= 3:
493:             mods.pop(0)
494:             # mods[0] will be the name of the package being tested (in
495:             # most cases "matplotlib") However if this is a
496:             # namespace package pip installed and run via the nose
497:             # multiprocess plugin or as a specific test this may be
498:             # missing. See https://github.com/matplotlib/matplotlib/issues/3314
499:         if mods.pop(0) != 'tests':
500:             warnings.warn(("Module '%s' does not live in a parent module "
501:                 "named 'tests'. This is probably ok, but we may not be able "
502:                 "to guess the correct subdirectory containing the baseline "
503:                 "images. If things go wrong please make sure that there is "
504:                 "a parent directory named 'tests' and that it contains a "
505:                 "__init__.py file (can be empty).") % module_name)
506:         subdir = os.path.join(*mods)
507: 
508:         import imp
509:         def find_dotted_module(module_name, path=None):
510:             '''A version of imp which can handle dots in the module name.
511:                As for imp.find_module(), the return value is a 3-element
512:                tuple (file, pathname, description).'''
513:             res = None
514:             for sub_mod in module_name.split('.'):
515:                 try:
516:                     res = file, path, _ = imp.find_module(sub_mod, path)
517:                     path = [path]
518:                     if file is not None:
519:                         file.close()
520:                 except ImportError:
521:                     # assume namespace package
522:                     path = list(sys.modules[sub_mod].__path__)
523:                     res = None, path, None
524:             return res
525: 
526:         mod_file = find_dotted_module(func.__module__)[1]
527:         basedir = os.path.dirname(mod_file)
528: 
529:     baseline_dir = os.path.join(basedir, 'baseline_images', subdir)
530:     result_dir = os.path.abspath(os.path.join('result_images', subdir))
531: 
532:     if not os.path.exists(result_dir):
533:         cbook.mkdirs(result_dir)
534: 
535:     return baseline_dir, result_dir
536: 
537: 
538: def switch_backend(backend):
539:     # Local import to avoid a hard nose dependency and only incur the
540:     # import time overhead at actual test-time.
541:     def switch_backend_decorator(func):
542:         @functools.wraps(func)
543:         def backend_switcher(*args, **kwargs):
544:             try:
545:                 prev_backend = mpl.get_backend()
546:                 matplotlib.testing.setup()
547:                 plt.switch_backend(backend)
548:                 result = func(*args, **kwargs)
549:             finally:
550:                 plt.switch_backend(prev_backend)
551:             return result
552: 
553:         return _copy_metadata(func, backend_switcher)
554:     return switch_backend_decorator
555: 
556: 
557: def skip_if_command_unavailable(cmd):
558:     '''
559:     skips a test if a command is unavailable.
560: 
561:     Parameters
562:     ----------
563:     cmd : list of str
564:         must be a complete command which should not
565:         return a non zero exit code, something like
566:         ["latex", "-version"]
567:     '''
568:     from matplotlib.compat.subprocess import check_output
569:     try:
570:         check_output(cmd)
571:     except:
572:         import pytest
573:         return pytest.mark.skip(reason='missing command: %s' % cmd[0])
574: 
575:     return lambda f: f
576: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290057 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_290057) is not StypyTypeError):

    if (import_290057 != 'pyd_module'):
        __import__(import_290057)
        sys_modules_290058 = sys.modules[import_290057]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_290058.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_290057)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import functools' statement (line 6)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import inspect' statement (line 7)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import shutil' statement (line 10)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import warnings' statement (line 11)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import unittest' statement (line 12)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import matplotlib' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290059 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib')

if (type(import_290059) is not StypyTypeError):

    if (import_290059 != 'pyd_module'):
        __import__(import_290059)
        sys_modules_290060 = sys.modules[import_290059]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'mpl', sys_modules_290060.module_type_store, module_type_store)
    else:
        import matplotlib as mpl

        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'mpl', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib', import_290059)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import matplotlib.style' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290061 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.style')

if (type(import_290061) is not StypyTypeError):

    if (import_290061 != 'pyd_module'):
        __import__(import_290061)
        sys_modules_290062 = sys.modules[import_290061]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.style', sys_modules_290062.module_type_store, module_type_store)
    else:
        import matplotlib.style

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.style', matplotlib.style, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.style' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.style', import_290061)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import matplotlib.units' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290063 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.units')

if (type(import_290063) is not StypyTypeError):

    if (import_290063 != 'pyd_module'):
        __import__(import_290063)
        sys_modules_290064 = sys.modules[import_290063]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.units', sys_modules_290064.module_type_store, module_type_store)
    else:
        import matplotlib.units

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.units', matplotlib.units, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.units' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'matplotlib.units', import_290063)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import matplotlib.testing' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290065 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.testing')

if (type(import_290065) is not StypyTypeError):

    if (import_290065 != 'pyd_module'):
        __import__(import_290065)
        sys_modules_290066 = sys.modules[import_290065]
        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.testing', sys_modules_290066.module_type_store, module_type_store)
    else:
        import matplotlib.testing

        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.testing', matplotlib.testing, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.testing' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.testing', import_290065)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from matplotlib import cbook' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290067 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib')

if (type(import_290067) is not StypyTypeError):

    if (import_290067 != 'pyd_module'):
        __import__(import_290067)
        sys_modules_290068 = sys.modules[import_290067]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', sys_modules_290068.module_type_store, module_type_store, ['cbook'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_290068, sys_modules_290068.module_type_store, module_type_store)
    else:
        from matplotlib import cbook

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', None, module_type_store, ['cbook'], [cbook])

else:
    # Assigning a type to the variable 'matplotlib' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'matplotlib', import_290067)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from matplotlib import ticker' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290069 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib')

if (type(import_290069) is not StypyTypeError):

    if (import_290069 != 'pyd_module'):
        __import__(import_290069)
        sys_modules_290070 = sys.modules[import_290069]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', sys_modules_290070.module_type_store, module_type_store, ['ticker'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_290070, sys_modules_290070.module_type_store, module_type_store)
    else:
        from matplotlib import ticker

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', None, module_type_store, ['ticker'], [ticker])

else:
    # Assigning a type to the variable 'matplotlib' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', import_290069)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from matplotlib import plt' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290071 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib')

if (type(import_290071) is not StypyTypeError):

    if (import_290071 != 'pyd_module'):
        __import__(import_290071)
        sys_modules_290072 = sys.modules[import_290071]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', sys_modules_290072.module_type_store, module_type_store, ['pyplot'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_290072, sys_modules_290072.module_type_store, module_type_store)
    else:
        from matplotlib import pyplot as plt

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', None, module_type_store, ['pyplot'], [plt])

else:
    # Assigning a type to the variable 'matplotlib' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', import_290071)

# Adding an alias
module_type_store.add_alias('plt', 'pyplot')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from matplotlib import ft2font' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290073 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib')

if (type(import_290073) is not StypyTypeError):

    if (import_290073 != 'pyd_module'):
        __import__(import_290073)
        sys_modules_290074 = sys.modules[import_290073]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib', sys_modules_290074.module_type_store, module_type_store, ['ft2font'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_290074, sys_modules_290074.module_type_store, module_type_store)
    else:
        from matplotlib import ft2font

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib', None, module_type_store, ['ft2font'], [ft2font])

else:
    # Assigning a type to the variable 'matplotlib' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib', import_290073)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from matplotlib.testing.compare import comparable_formats, compare_images, make_test_filename' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290075 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.testing.compare')

if (type(import_290075) is not StypyTypeError):

    if (import_290075 != 'pyd_module'):
        __import__(import_290075)
        sys_modules_290076 = sys.modules[import_290075]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.testing.compare', sys_modules_290076.module_type_store, module_type_store, ['comparable_formats', 'compare_images', 'make_test_filename'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_290076, sys_modules_290076.module_type_store, module_type_store)
    else:
        from matplotlib.testing.compare import comparable_formats, compare_images, make_test_filename

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.testing.compare', None, module_type_store, ['comparable_formats', 'compare_images', 'make_test_filename'], [comparable_formats, compare_images, make_test_filename])

else:
    # Assigning a type to the variable 'matplotlib.testing.compare' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.testing.compare', import_290075)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from matplotlib.testing import _copy_metadata, is_called_from_pytest' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290077 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.testing')

if (type(import_290077) is not StypyTypeError):

    if (import_290077 != 'pyd_module'):
        __import__(import_290077)
        sys_modules_290078 = sys.modules[import_290077]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.testing', sys_modules_290078.module_type_store, module_type_store, ['_copy_metadata', 'is_called_from_pytest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_290078, sys_modules_290078.module_type_store, module_type_store)
    else:
        from matplotlib.testing import _copy_metadata, is_called_from_pytest

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.testing', None, module_type_store, ['_copy_metadata', 'is_called_from_pytest'], [_copy_metadata, is_called_from_pytest])

else:
    # Assigning a type to the variable 'matplotlib.testing' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.testing', import_290077)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from matplotlib.testing.exceptions import ImageComparisonFailure' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_290079 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.testing.exceptions')

if (type(import_290079) is not StypyTypeError):

    if (import_290079 != 'pyd_module'):
        __import__(import_290079)
        sys_modules_290080 = sys.modules[import_290079]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.testing.exceptions', sys_modules_290080.module_type_store, module_type_store, ['ImageComparisonFailure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_290080, sys_modules_290080.module_type_store, module_type_store)
    else:
        from matplotlib.testing.exceptions import ImageComparisonFailure

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.testing.exceptions', None, module_type_store, ['ImageComparisonFailure'], [ImageComparisonFailure])

else:
    # Assigning a type to the variable 'matplotlib.testing.exceptions' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.testing.exceptions', import_290079)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')


@norecursion
def _knownfailureif(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 33)
    None_290081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'None')
    # Getting the type of 'None' (line 33)
    None_290082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 68), 'None')
    defaults = [None_290081, None_290082]
    # Create a new context for function '_knownfailureif'
    module_type_store = module_type_store.open_function_context('_knownfailureif', 33, 0, False)
    
    # Passed parameters checking function
    _knownfailureif.stypy_localization = localization
    _knownfailureif.stypy_type_of_self = None
    _knownfailureif.stypy_type_store = module_type_store
    _knownfailureif.stypy_function_name = '_knownfailureif'
    _knownfailureif.stypy_param_names_list = ['fail_condition', 'msg', 'known_exception_class']
    _knownfailureif.stypy_varargs_param_name = None
    _knownfailureif.stypy_kwargs_param_name = None
    _knownfailureif.stypy_call_defaults = defaults
    _knownfailureif.stypy_call_varargs = varargs
    _knownfailureif.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_knownfailureif', ['fail_condition', 'msg', 'known_exception_class'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_knownfailureif', localization, ['fail_condition', 'msg', 'known_exception_class'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_knownfailureif(...)' code ##################

    unicode_290083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'unicode', u"\n\n    Assume a will fail if *fail_condition* is True. *fail_condition*\n    may also be False or the string 'indeterminate'.\n\n    *msg* is the error message displayed for the test.\n\n    If *known_exception_class* is not None, the failure is only known\n    if the exception is an instance of this class. (Default = None)\n\n    ")
    
    
    # Call to is_called_from_pytest(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_290085 = {}
    # Getting the type of 'is_called_from_pytest' (line 45)
    is_called_from_pytest_290084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 7), 'is_called_from_pytest', False)
    # Calling is_called_from_pytest(args, kwargs) (line 45)
    is_called_from_pytest_call_result_290086 = invoke(stypy.reporting.localization.Localization(__file__, 45, 7), is_called_from_pytest_290084, *[], **kwargs_290085)
    
    # Testing the type of an if condition (line 45)
    if_condition_290087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 4), is_called_from_pytest_call_result_290086)
    # Assigning a type to the variable 'if_condition_290087' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'if_condition_290087', if_condition_290087)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 8))
    
    # 'import pytest' statement (line 46)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_290088 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 46, 8), 'pytest')

    if (type(import_290088) is not StypyTypeError):

        if (import_290088 != 'pyd_module'):
            __import__(import_290088)
            sys_modules_290089 = sys.modules[import_290088]
            import_module(stypy.reporting.localization.Localization(__file__, 46, 8), 'pytest', sys_modules_290089.module_type_store, module_type_store)
        else:
            import pytest

            import_module(stypy.reporting.localization.Localization(__file__, 46, 8), 'pytest', pytest, module_type_store)

    else:
        # Assigning a type to the variable 'pytest' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'pytest', import_290088)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    
    # Getting the type of 'fail_condition' (line 47)
    fail_condition_290090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'fail_condition')
    unicode_290091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'unicode', u'indeterminate')
    # Applying the binary operator '==' (line 47)
    result_eq_290092 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '==', fail_condition_290090, unicode_290091)
    
    # Testing the type of an if condition (line 47)
    if_condition_290093 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_eq_290092)
    # Assigning a type to the variable 'if_condition_290093' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_290093', if_condition_290093)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 48):
    
    # Assigning a Name to a Name (line 48):
    # Getting the type of 'True' (line 48)
    True_290094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'True')
    # Assigning a type to the variable 'tuple_assignment_290044' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'tuple_assignment_290044', True_290094)
    
    # Assigning a Name to a Name (line 48):
    # Getting the type of 'False' (line 48)
    False_290095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 43), 'False')
    # Assigning a type to the variable 'tuple_assignment_290045' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'tuple_assignment_290045', False_290095)
    
    # Assigning a Name to a Name (line 48):
    # Getting the type of 'tuple_assignment_290044' (line 48)
    tuple_assignment_290044_290096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'tuple_assignment_290044')
    # Assigning a type to the variable 'fail_condition' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'fail_condition', tuple_assignment_290044_290096)
    
    # Assigning a Name to a Name (line 48):
    # Getting the type of 'tuple_assignment_290045' (line 48)
    tuple_assignment_290045_290097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'tuple_assignment_290045')
    # Assigning a type to the variable 'strict' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'strict', tuple_assignment_290045_290097)
    # SSA branch for the else part of an if statement (line 47)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 50):
    
    # Assigning a Call to a Name (line 50):
    
    # Call to bool(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'fail_condition' (line 50)
    fail_condition_290099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 42), 'fail_condition', False)
    # Processing the call keyword arguments (line 50)
    kwargs_290100 = {}
    # Getting the type of 'bool' (line 50)
    bool_290098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 37), 'bool', False)
    # Calling bool(args, kwargs) (line 50)
    bool_call_result_290101 = invoke(stypy.reporting.localization.Localization(__file__, 50, 37), bool_290098, *[fail_condition_290099], **kwargs_290100)
    
    # Assigning a type to the variable 'tuple_assignment_290046' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'tuple_assignment_290046', bool_call_result_290101)
    
    # Assigning a Name to a Name (line 50):
    # Getting the type of 'True' (line 50)
    True_290102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 59), 'True')
    # Assigning a type to the variable 'tuple_assignment_290047' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'tuple_assignment_290047', True_290102)
    
    # Assigning a Name to a Name (line 50):
    # Getting the type of 'tuple_assignment_290046' (line 50)
    tuple_assignment_290046_290103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'tuple_assignment_290046')
    # Assigning a type to the variable 'fail_condition' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'fail_condition', tuple_assignment_290046_290103)
    
    # Assigning a Name to a Name (line 50):
    # Getting the type of 'tuple_assignment_290047' (line 50)
    tuple_assignment_290047_290104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'tuple_assignment_290047')
    # Assigning a type to the variable 'strict' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'strict', tuple_assignment_290047_290104)
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to xfail(...): (line 51)
    # Processing the call keyword arguments (line 51)
    # Getting the type of 'fail_condition' (line 51)
    fail_condition_290108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 43), 'fail_condition', False)
    keyword_290109 = fail_condition_290108
    # Getting the type of 'msg' (line 51)
    msg_290110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 66), 'msg', False)
    keyword_290111 = msg_290110
    # Getting the type of 'known_exception_class' (line 52)
    known_exception_class_290112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 40), 'known_exception_class', False)
    keyword_290113 = known_exception_class_290112
    # Getting the type of 'strict' (line 52)
    strict_290114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 70), 'strict', False)
    keyword_290115 = strict_290114
    kwargs_290116 = {'strict': keyword_290115, 'reason': keyword_290111, 'raises': keyword_290113, 'condition': keyword_290109}
    # Getting the type of 'pytest' (line 51)
    pytest_290105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'pytest', False)
    # Obtaining the member 'mark' of a type (line 51)
    mark_290106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), pytest_290105, 'mark')
    # Obtaining the member 'xfail' of a type (line 51)
    xfail_290107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), mark_290106, 'xfail')
    # Calling xfail(args, kwargs) (line 51)
    xfail_call_result_290117 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), xfail_290107, *[], **kwargs_290116)
    
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', xfail_call_result_290117)
    # SSA branch for the else part of an if statement (line 45)
    module_type_store.open_ssa_branch('else')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 54, 8))
    
    # 'from matplotlib.testing._nose.decorators import knownfailureif' statement (line 54)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_290118 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 54, 8), 'matplotlib.testing._nose.decorators')

    if (type(import_290118) is not StypyTypeError):

        if (import_290118 != 'pyd_module'):
            __import__(import_290118)
            sys_modules_290119 = sys.modules[import_290118]
            import_from_module(stypy.reporting.localization.Localization(__file__, 54, 8), 'matplotlib.testing._nose.decorators', sys_modules_290119.module_type_store, module_type_store, ['knownfailureif'])
            nest_module(stypy.reporting.localization.Localization(__file__, 54, 8), __file__, sys_modules_290119, sys_modules_290119.module_type_store, module_type_store)
        else:
            from matplotlib.testing._nose.decorators import knownfailureif

            import_from_module(stypy.reporting.localization.Localization(__file__, 54, 8), 'matplotlib.testing._nose.decorators', None, module_type_store, ['knownfailureif'], [knownfailureif])

    else:
        # Assigning a type to the variable 'matplotlib.testing._nose.decorators' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'matplotlib.testing._nose.decorators', import_290118)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    # Call to knownfailureif(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'fail_condition' (line 55)
    fail_condition_290121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 30), 'fail_condition', False)
    # Getting the type of 'msg' (line 55)
    msg_290122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'msg', False)
    # Getting the type of 'known_exception_class' (line 55)
    known_exception_class_290123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 51), 'known_exception_class', False)
    # Processing the call keyword arguments (line 55)
    kwargs_290124 = {}
    # Getting the type of 'knownfailureif' (line 55)
    knownfailureif_290120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 'knownfailureif', False)
    # Calling knownfailureif(args, kwargs) (line 55)
    knownfailureif_call_result_290125 = invoke(stypy.reporting.localization.Localization(__file__, 55, 15), knownfailureif_290120, *[fail_condition_290121, msg_290122, known_exception_class_290123], **kwargs_290124)
    
    # Assigning a type to the variable 'stypy_return_type' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type', knownfailureif_call_result_290125)
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_knownfailureif(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_knownfailureif' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_290126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290126)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_knownfailureif'
    return stypy_return_type_290126

# Assigning a type to the variable '_knownfailureif' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '_knownfailureif', _knownfailureif)

@norecursion
def knownfailureif(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 60)
    None_290127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'None')
    # Getting the type of 'None' (line 60)
    None_290128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 67), 'None')
    defaults = [None_290127, None_290128]
    # Create a new context for function 'knownfailureif'
    module_type_store = module_type_store.open_function_context('knownfailureif', 58, 0, False)
    
    # Passed parameters checking function
    knownfailureif.stypy_localization = localization
    knownfailureif.stypy_type_of_self = None
    knownfailureif.stypy_type_store = module_type_store
    knownfailureif.stypy_function_name = 'knownfailureif'
    knownfailureif.stypy_param_names_list = ['fail_condition', 'msg', 'known_exception_class']
    knownfailureif.stypy_varargs_param_name = None
    knownfailureif.stypy_kwargs_param_name = None
    knownfailureif.stypy_call_defaults = defaults
    knownfailureif.stypy_call_varargs = varargs
    knownfailureif.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'knownfailureif', ['fail_condition', 'msg', 'known_exception_class'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'knownfailureif', localization, ['fail_condition', 'msg', 'known_exception_class'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'knownfailureif(...)' code ##################

    
    # Call to _knownfailureif(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'fail_condition' (line 61)
    fail_condition_290130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'fail_condition', False)
    # Getting the type of 'msg' (line 61)
    msg_290131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 36), 'msg', False)
    # Getting the type of 'known_exception_class' (line 61)
    known_exception_class_290132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 'known_exception_class', False)
    # Processing the call keyword arguments (line 61)
    kwargs_290133 = {}
    # Getting the type of '_knownfailureif' (line 61)
    _knownfailureif_290129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), '_knownfailureif', False)
    # Calling _knownfailureif(args, kwargs) (line 61)
    _knownfailureif_call_result_290134 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), _knownfailureif_290129, *[fail_condition_290130, msg_290131, known_exception_class_290132], **kwargs_290133)
    
    
    # ################# End of 'knownfailureif(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'knownfailureif' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_290135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290135)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'knownfailureif'
    return stypy_return_type_290135

# Assigning a type to the variable 'knownfailureif' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'knownfailureif', knownfailureif)

@norecursion
def _do_cleanup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_do_cleanup'
    module_type_store = module_type_store.open_function_context('_do_cleanup', 64, 0, False)
    
    # Passed parameters checking function
    _do_cleanup.stypy_localization = localization
    _do_cleanup.stypy_type_of_self = None
    _do_cleanup.stypy_type_store = module_type_store
    _do_cleanup.stypy_function_name = '_do_cleanup'
    _do_cleanup.stypy_param_names_list = ['original_units_registry', 'original_settings']
    _do_cleanup.stypy_varargs_param_name = None
    _do_cleanup.stypy_kwargs_param_name = None
    _do_cleanup.stypy_call_defaults = defaults
    _do_cleanup.stypy_call_varargs = varargs
    _do_cleanup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_do_cleanup', ['original_units_registry', 'original_settings'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_do_cleanup', localization, ['original_units_registry', 'original_settings'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_do_cleanup(...)' code ##################

    
    # Call to close(...): (line 65)
    # Processing the call arguments (line 65)
    unicode_290138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 14), 'unicode', u'all')
    # Processing the call keyword arguments (line 65)
    kwargs_290139 = {}
    # Getting the type of 'plt' (line 65)
    plt_290136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'plt', False)
    # Obtaining the member 'close' of a type (line 65)
    close_290137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), plt_290136, 'close')
    # Calling close(args, kwargs) (line 65)
    close_call_result_290140 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), close_290137, *[unicode_290138], **kwargs_290139)
    
    
    # Call to clear(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_290144 = {}
    # Getting the type of 'mpl' (line 67)
    mpl_290141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'mpl', False)
    # Obtaining the member 'rcParams' of a type (line 67)
    rcParams_290142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), mpl_290141, 'rcParams')
    # Obtaining the member 'clear' of a type (line 67)
    clear_290143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 4), rcParams_290142, 'clear')
    # Calling clear(args, kwargs) (line 67)
    clear_call_result_290145 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), clear_290143, *[], **kwargs_290144)
    
    
    # Call to update(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'original_settings' (line 68)
    original_settings_290149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'original_settings', False)
    # Processing the call keyword arguments (line 68)
    kwargs_290150 = {}
    # Getting the type of 'mpl' (line 68)
    mpl_290146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'mpl', False)
    # Obtaining the member 'rcParams' of a type (line 68)
    rcParams_290147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), mpl_290146, 'rcParams')
    # Obtaining the member 'update' of a type (line 68)
    update_290148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), rcParams_290147, 'update')
    # Calling update(args, kwargs) (line 68)
    update_call_result_290151 = invoke(stypy.reporting.localization.Localization(__file__, 68, 4), update_290148, *[original_settings_290149], **kwargs_290150)
    
    
    # Call to clear(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_290156 = {}
    # Getting the type of 'matplotlib' (line 69)
    matplotlib_290152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'matplotlib', False)
    # Obtaining the member 'units' of a type (line 69)
    units_290153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), matplotlib_290152, 'units')
    # Obtaining the member 'registry' of a type (line 69)
    registry_290154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), units_290153, 'registry')
    # Obtaining the member 'clear' of a type (line 69)
    clear_290155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), registry_290154, 'clear')
    # Calling clear(args, kwargs) (line 69)
    clear_call_result_290157 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), clear_290155, *[], **kwargs_290156)
    
    
    # Call to update(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'original_units_registry' (line 70)
    original_units_registry_290162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 37), 'original_units_registry', False)
    # Processing the call keyword arguments (line 70)
    kwargs_290163 = {}
    # Getting the type of 'matplotlib' (line 70)
    matplotlib_290158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'matplotlib', False)
    # Obtaining the member 'units' of a type (line 70)
    units_290159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), matplotlib_290158, 'units')
    # Obtaining the member 'registry' of a type (line 70)
    registry_290160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), units_290159, 'registry')
    # Obtaining the member 'update' of a type (line 70)
    update_290161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), registry_290160, 'update')
    # Calling update(args, kwargs) (line 70)
    update_call_result_290164 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), update_290161, *[original_units_registry_290162], **kwargs_290163)
    
    
    # Call to resetwarnings(...): (line 71)
    # Processing the call keyword arguments (line 71)
    kwargs_290167 = {}
    # Getting the type of 'warnings' (line 71)
    warnings_290165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'warnings', False)
    # Obtaining the member 'resetwarnings' of a type (line 71)
    resetwarnings_290166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), warnings_290165, 'resetwarnings')
    # Calling resetwarnings(args, kwargs) (line 71)
    resetwarnings_call_result_290168 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), resetwarnings_290166, *[], **kwargs_290167)
    
    
    # ################# End of '_do_cleanup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_do_cleanup' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_290169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290169)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_do_cleanup'
    return stypy_return_type_290169

# Assigning a type to the variable '_do_cleanup' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), '_do_cleanup', _do_cleanup)
# Declaration of the 'CleanupTest' class

class CleanupTest(object, ):

    @norecursion
    def setup_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_class'
        module_type_store = module_type_store.open_function_context('setup_class', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CleanupTest.setup_class.__dict__.__setitem__('stypy_localization', localization)
        CleanupTest.setup_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CleanupTest.setup_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        CleanupTest.setup_class.__dict__.__setitem__('stypy_function_name', 'CleanupTest.setup_class')
        CleanupTest.setup_class.__dict__.__setitem__('stypy_param_names_list', [])
        CleanupTest.setup_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        CleanupTest.setup_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CleanupTest.setup_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        CleanupTest.setup_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        CleanupTest.setup_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CleanupTest.setup_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CleanupTest.setup_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_class(...)' code ##################

        
        # Assigning a Call to a Attribute (line 77):
        
        # Assigning a Call to a Attribute (line 77):
        
        # Call to copy(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_290174 = {}
        # Getting the type of 'matplotlib' (line 77)
        matplotlib_290170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'matplotlib', False)
        # Obtaining the member 'units' of a type (line 77)
        units_290171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 38), matplotlib_290170, 'units')
        # Obtaining the member 'registry' of a type (line 77)
        registry_290172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 38), units_290171, 'registry')
        # Obtaining the member 'copy' of a type (line 77)
        copy_290173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 38), registry_290172, 'copy')
        # Calling copy(args, kwargs) (line 77)
        copy_call_result_290175 = invoke(stypy.reporting.localization.Localization(__file__, 77, 38), copy_290173, *[], **kwargs_290174)
        
        # Getting the type of 'cls' (line 77)
        cls_290176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'cls')
        # Setting the type of the member 'original_units_registry' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), cls_290176, 'original_units_registry', copy_call_result_290175)
        
        # Assigning a Call to a Attribute (line 78):
        
        # Assigning a Call to a Attribute (line 78):
        
        # Call to copy(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_290180 = {}
        # Getting the type of 'mpl' (line 78)
        mpl_290177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 32), 'mpl', False)
        # Obtaining the member 'rcParams' of a type (line 78)
        rcParams_290178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 32), mpl_290177, 'rcParams')
        # Obtaining the member 'copy' of a type (line 78)
        copy_290179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 32), rcParams_290178, 'copy')
        # Calling copy(args, kwargs) (line 78)
        copy_call_result_290181 = invoke(stypy.reporting.localization.Localization(__file__, 78, 32), copy_290179, *[], **kwargs_290180)
        
        # Getting the type of 'cls' (line 78)
        cls_290182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'cls')
        # Setting the type of the member 'original_settings' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), cls_290182, 'original_settings', copy_call_result_290181)
        
        # Call to setup(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_290186 = {}
        # Getting the type of 'matplotlib' (line 79)
        matplotlib_290183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'matplotlib', False)
        # Obtaining the member 'testing' of a type (line 79)
        testing_290184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), matplotlib_290183, 'testing')
        # Obtaining the member 'setup' of a type (line 79)
        setup_290185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), testing_290184, 'setup')
        # Calling setup(args, kwargs) (line 79)
        setup_call_result_290187 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), setup_290185, *[], **kwargs_290186)
        
        
        # ################# End of 'setup_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_class' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_290188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290188)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_class'
        return stypy_return_type_290188


    @norecursion
    def teardown_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'teardown_class'
        module_type_store = module_type_store.open_function_context('teardown_class', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_localization', localization)
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_function_name', 'CleanupTest.teardown_class')
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_param_names_list', [])
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CleanupTest.teardown_class.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CleanupTest.teardown_class', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'teardown_class', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'teardown_class(...)' code ##################

        
        # Call to _do_cleanup(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'cls' (line 83)
        cls_290190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'cls', False)
        # Obtaining the member 'original_units_registry' of a type (line 83)
        original_units_registry_290191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), cls_290190, 'original_units_registry')
        # Getting the type of 'cls' (line 84)
        cls_290192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'cls', False)
        # Obtaining the member 'original_settings' of a type (line 84)
        original_settings_290193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), cls_290192, 'original_settings')
        # Processing the call keyword arguments (line 83)
        kwargs_290194 = {}
        # Getting the type of '_do_cleanup' (line 83)
        _do_cleanup_290189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), '_do_cleanup', False)
        # Calling _do_cleanup(args, kwargs) (line 83)
        _do_cleanup_call_result_290195 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), _do_cleanup_290189, *[original_units_registry_290191, original_settings_290193], **kwargs_290194)
        
        
        # ################# End of 'teardown_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'teardown_class' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_290196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'teardown_class'
        return stypy_return_type_290196


    @norecursion
    def test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test'
        module_type_store = module_type_store.open_function_context('test', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CleanupTest.test.__dict__.__setitem__('stypy_localization', localization)
        CleanupTest.test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CleanupTest.test.__dict__.__setitem__('stypy_type_store', module_type_store)
        CleanupTest.test.__dict__.__setitem__('stypy_function_name', 'CleanupTest.test')
        CleanupTest.test.__dict__.__setitem__('stypy_param_names_list', [])
        CleanupTest.test.__dict__.__setitem__('stypy_varargs_param_name', None)
        CleanupTest.test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CleanupTest.test.__dict__.__setitem__('stypy_call_defaults', defaults)
        CleanupTest.test.__dict__.__setitem__('stypy_call_varargs', varargs)
        CleanupTest.test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CleanupTest.test.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CleanupTest.test', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to _func(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_290199 = {}
        # Getting the type of 'self' (line 87)
        self_290197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member '_func' of a type (line 87)
        _func_290198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_290197, '_func')
        # Calling _func(args, kwargs) (line 87)
        _func_call_result_290200 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), _func_290198, *[], **kwargs_290199)
        
        
        # ################# End of 'test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_290201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290201)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test'
        return stypy_return_type_290201


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 74, 0, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CleanupTest.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CleanupTest' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'CleanupTest', CleanupTest)
# Declaration of the 'CleanupTestCase' class
# Getting the type of 'unittest' (line 90)
unittest_290202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'unittest')
# Obtaining the member 'TestCase' of a type (line 90)
TestCase_290203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 22), unittest_290202, 'TestCase')

class CleanupTestCase(TestCase_290203, ):
    unicode_290204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'unicode', u'A wrapper for unittest.TestCase that includes cleanup operations')

    @norecursion
    def setUpClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUpClass'
        module_type_store = module_type_store.open_function_context('setUpClass', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_localization', localization)
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_function_name', 'CleanupTestCase.setUpClass')
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_param_names_list', [])
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CleanupTestCase.setUpClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CleanupTestCase.setUpClass', [], None, None, defaults, varargs, kwargs)

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

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 94, 8))
        
        # 'import matplotlib.units' statement (line 94)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
        import_290205 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 94, 8), 'matplotlib.units')

        if (type(import_290205) is not StypyTypeError):

            if (import_290205 != 'pyd_module'):
                __import__(import_290205)
                sys_modules_290206 = sys.modules[import_290205]
                import_module(stypy.reporting.localization.Localization(__file__, 94, 8), 'matplotlib.units', sys_modules_290206.module_type_store, module_type_store)
            else:
                import matplotlib.units

                import_module(stypy.reporting.localization.Localization(__file__, 94, 8), 'matplotlib.units', matplotlib.units, module_type_store)

        else:
            # Assigning a type to the variable 'matplotlib.units' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'matplotlib.units', import_290205)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
        
        
        # Assigning a Call to a Attribute (line 95):
        
        # Assigning a Call to a Attribute (line 95):
        
        # Call to copy(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_290211 = {}
        # Getting the type of 'matplotlib' (line 95)
        matplotlib_290207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'matplotlib', False)
        # Obtaining the member 'units' of a type (line 95)
        units_290208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 38), matplotlib_290207, 'units')
        # Obtaining the member 'registry' of a type (line 95)
        registry_290209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 38), units_290208, 'registry')
        # Obtaining the member 'copy' of a type (line 95)
        copy_290210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 38), registry_290209, 'copy')
        # Calling copy(args, kwargs) (line 95)
        copy_call_result_290212 = invoke(stypy.reporting.localization.Localization(__file__, 95, 38), copy_290210, *[], **kwargs_290211)
        
        # Getting the type of 'cls' (line 95)
        cls_290213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'cls')
        # Setting the type of the member 'original_units_registry' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), cls_290213, 'original_units_registry', copy_call_result_290212)
        
        # Assigning a Call to a Attribute (line 96):
        
        # Assigning a Call to a Attribute (line 96):
        
        # Call to copy(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_290217 = {}
        # Getting the type of 'mpl' (line 96)
        mpl_290214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'mpl', False)
        # Obtaining the member 'rcParams' of a type (line 96)
        rcParams_290215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 32), mpl_290214, 'rcParams')
        # Obtaining the member 'copy' of a type (line 96)
        copy_290216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 32), rcParams_290215, 'copy')
        # Calling copy(args, kwargs) (line 96)
        copy_call_result_290218 = invoke(stypy.reporting.localization.Localization(__file__, 96, 32), copy_290216, *[], **kwargs_290217)
        
        # Getting the type of 'cls' (line 96)
        cls_290219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'cls')
        # Setting the type of the member 'original_settings' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), cls_290219, 'original_settings', copy_call_result_290218)
        
        # ################# End of 'setUpClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUpClass' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_290220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290220)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUpClass'
        return stypy_return_type_290220


    @norecursion
    def tearDownClass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDownClass'
        module_type_store = module_type_store.open_function_context('tearDownClass', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_localization', localization)
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_type_store', module_type_store)
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_function_name', 'CleanupTestCase.tearDownClass')
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_param_names_list', [])
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_varargs_param_name', None)
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_call_defaults', defaults)
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_call_varargs', varargs)
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CleanupTestCase.tearDownClass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CleanupTestCase.tearDownClass', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to _do_cleanup(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'cls' (line 100)
        cls_290222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'cls', False)
        # Obtaining the member 'original_units_registry' of a type (line 100)
        original_units_registry_290223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), cls_290222, 'original_units_registry')
        # Getting the type of 'cls' (line 101)
        cls_290224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'cls', False)
        # Obtaining the member 'original_settings' of a type (line 101)
        original_settings_290225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 20), cls_290224, 'original_settings')
        # Processing the call keyword arguments (line 100)
        kwargs_290226 = {}
        # Getting the type of '_do_cleanup' (line 100)
        _do_cleanup_290221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), '_do_cleanup', False)
        # Calling _do_cleanup(args, kwargs) (line 100)
        _do_cleanup_call_result_290227 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), _do_cleanup_290221, *[original_units_registry_290223, original_settings_290225], **kwargs_290226)
        
        
        # ################# End of 'tearDownClass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDownClass' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_290228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDownClass'
        return stypy_return_type_290228


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 90, 0, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CleanupTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CleanupTestCase' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'CleanupTestCase', CleanupTestCase)

@norecursion
def cleanup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 104)
    None_290229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'None')
    defaults = [None_290229]
    # Create a new context for function 'cleanup'
    module_type_store = module_type_store.open_function_context('cleanup', 104, 0, False)
    
    # Passed parameters checking function
    cleanup.stypy_localization = localization
    cleanup.stypy_type_of_self = None
    cleanup.stypy_type_store = module_type_store
    cleanup.stypy_function_name = 'cleanup'
    cleanup.stypy_param_names_list = ['style']
    cleanup.stypy_varargs_param_name = None
    cleanup.stypy_kwargs_param_name = None
    cleanup.stypy_call_defaults = defaults
    cleanup.stypy_call_varargs = varargs
    cleanup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cleanup', ['style'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cleanup', localization, ['style'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cleanup(...)' code ##################

    unicode_290230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, (-1)), 'unicode', u'\n    A decorator to ensure that any global state is reset before\n    running a test.\n\n    Parameters\n    ----------\n    style : str, optional\n        The name of the style to apply.\n    ')

    @norecursion
    def make_cleanup(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_cleanup'
        module_type_store = module_type_store.open_function_context('make_cleanup', 122, 4, False)
        
        # Passed parameters checking function
        make_cleanup.stypy_localization = localization
        make_cleanup.stypy_type_of_self = None
        make_cleanup.stypy_type_store = module_type_store
        make_cleanup.stypy_function_name = 'make_cleanup'
        make_cleanup.stypy_param_names_list = ['func']
        make_cleanup.stypy_varargs_param_name = None
        make_cleanup.stypy_kwargs_param_name = None
        make_cleanup.stypy_call_defaults = defaults
        make_cleanup.stypy_call_varargs = varargs
        make_cleanup.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'make_cleanup', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_cleanup', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_cleanup(...)' code ##################

        
        
        # Call to isgeneratorfunction(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'func' (line 123)
        func_290233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'func', False)
        # Processing the call keyword arguments (line 123)
        kwargs_290234 = {}
        # Getting the type of 'inspect' (line 123)
        inspect_290231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'inspect', False)
        # Obtaining the member 'isgeneratorfunction' of a type (line 123)
        isgeneratorfunction_290232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 11), inspect_290231, 'isgeneratorfunction')
        # Calling isgeneratorfunction(args, kwargs) (line 123)
        isgeneratorfunction_call_result_290235 = invoke(stypy.reporting.localization.Localization(__file__, 123, 11), isgeneratorfunction_290232, *[func_290233], **kwargs_290234)
        
        # Testing the type of an if condition (line 123)
        if_condition_290236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), isgeneratorfunction_call_result_290235)
        # Assigning a type to the variable 'if_condition_290236' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_290236', if_condition_290236)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def wrapped_callable(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'wrapped_callable'
            module_type_store = module_type_store.open_function_context('wrapped_callable', 124, 12, False)
            
            # Passed parameters checking function
            wrapped_callable.stypy_localization = localization
            wrapped_callable.stypy_type_of_self = None
            wrapped_callable.stypy_type_store = module_type_store
            wrapped_callable.stypy_function_name = 'wrapped_callable'
            wrapped_callable.stypy_param_names_list = []
            wrapped_callable.stypy_varargs_param_name = 'args'
            wrapped_callable.stypy_kwargs_param_name = 'kwargs'
            wrapped_callable.stypy_call_defaults = defaults
            wrapped_callable.stypy_call_varargs = varargs
            wrapped_callable.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'wrapped_callable', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'wrapped_callable', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'wrapped_callable(...)' code ##################

            
            # Assigning a Call to a Name (line 126):
            
            # Assigning a Call to a Name (line 126):
            
            # Call to copy(...): (line 126)
            # Processing the call keyword arguments (line 126)
            kwargs_290241 = {}
            # Getting the type of 'matplotlib' (line 126)
            matplotlib_290237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 42), 'matplotlib', False)
            # Obtaining the member 'units' of a type (line 126)
            units_290238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 42), matplotlib_290237, 'units')
            # Obtaining the member 'registry' of a type (line 126)
            registry_290239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 42), units_290238, 'registry')
            # Obtaining the member 'copy' of a type (line 126)
            copy_290240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 42), registry_290239, 'copy')
            # Calling copy(args, kwargs) (line 126)
            copy_call_result_290242 = invoke(stypy.reporting.localization.Localization(__file__, 126, 42), copy_290240, *[], **kwargs_290241)
            
            # Assigning a type to the variable 'original_units_registry' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'original_units_registry', copy_call_result_290242)
            
            # Assigning a Call to a Name (line 127):
            
            # Assigning a Call to a Name (line 127):
            
            # Call to copy(...): (line 127)
            # Processing the call keyword arguments (line 127)
            kwargs_290246 = {}
            # Getting the type of 'mpl' (line 127)
            mpl_290243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 36), 'mpl', False)
            # Obtaining the member 'rcParams' of a type (line 127)
            rcParams_290244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 36), mpl_290243, 'rcParams')
            # Obtaining the member 'copy' of a type (line 127)
            copy_290245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 36), rcParams_290244, 'copy')
            # Calling copy(args, kwargs) (line 127)
            copy_call_result_290247 = invoke(stypy.reporting.localization.Localization(__file__, 127, 36), copy_290245, *[], **kwargs_290246)
            
            # Assigning a type to the variable 'original_settings' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'original_settings', copy_call_result_290247)
            
            # Call to use(...): (line 128)
            # Processing the call arguments (line 128)
            # Getting the type of 'style' (line 128)
            style_290251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'style', False)
            # Processing the call keyword arguments (line 128)
            kwargs_290252 = {}
            # Getting the type of 'matplotlib' (line 128)
            matplotlib_290248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'matplotlib', False)
            # Obtaining the member 'style' of a type (line 128)
            style_290249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 16), matplotlib_290248, 'style')
            # Obtaining the member 'use' of a type (line 128)
            use_290250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 16), style_290249, 'use')
            # Calling use(args, kwargs) (line 128)
            use_call_result_290253 = invoke(stypy.reporting.localization.Localization(__file__, 128, 16), use_290250, *[style_290251], **kwargs_290252)
            
            
            # Try-finally block (line 129)
            
            
            # Call to func(...): (line 130)
            # Getting the type of 'args' (line 130)
            args_290255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 41), 'args', False)
            # Processing the call keyword arguments (line 130)
            # Getting the type of 'kwargs' (line 130)
            kwargs_290256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 49), 'kwargs', False)
            kwargs_290257 = {'kwargs_290256': kwargs_290256}
            # Getting the type of 'func' (line 130)
            func_290254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'func', False)
            # Calling func(args, kwargs) (line 130)
            func_call_result_290258 = invoke(stypy.reporting.localization.Localization(__file__, 130, 35), func_290254, *[args_290255], **kwargs_290257)
            
            # Testing the type of a for loop iterable (line 130)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 130, 20), func_call_result_290258)
            # Getting the type of the for loop variable (line 130)
            for_loop_var_290259 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 130, 20), func_call_result_290258)
            # Assigning a type to the variable 'yielded' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'yielded', for_loop_var_290259)
            # SSA begins for a for statement (line 130)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'yielded' (line 131)
            yielded_290260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'yielded')
            GeneratorType_290261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 24), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 24), GeneratorType_290261, yielded_290260)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'stypy_return_type', GeneratorType_290261)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # finally branch of the try-finally block (line 129)
            
            # Call to _do_cleanup(...): (line 133)
            # Processing the call arguments (line 133)
            # Getting the type of 'original_units_registry' (line 133)
            original_units_registry_290263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 32), 'original_units_registry', False)
            # Getting the type of 'original_settings' (line 134)
            original_settings_290264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 32), 'original_settings', False)
            # Processing the call keyword arguments (line 133)
            kwargs_290265 = {}
            # Getting the type of '_do_cleanup' (line 133)
            _do_cleanup_290262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), '_do_cleanup', False)
            # Calling _do_cleanup(args, kwargs) (line 133)
            _do_cleanup_call_result_290266 = invoke(stypy.reporting.localization.Localization(__file__, 133, 20), _do_cleanup_290262, *[original_units_registry_290263, original_settings_290264], **kwargs_290265)
            
            
            
            # ################# End of 'wrapped_callable(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'wrapped_callable' in the type store
            # Getting the type of 'stypy_return_type' (line 124)
            stypy_return_type_290267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_290267)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'wrapped_callable'
            return stypy_return_type_290267

        # Assigning a type to the variable 'wrapped_callable' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'wrapped_callable', wrapped_callable)
        # SSA branch for the else part of an if statement (line 123)
        module_type_store.open_ssa_branch('else')

        @norecursion
        def wrapped_callable(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'wrapped_callable'
            module_type_store = module_type_store.open_function_context('wrapped_callable', 136, 12, False)
            
            # Passed parameters checking function
            wrapped_callable.stypy_localization = localization
            wrapped_callable.stypy_type_of_self = None
            wrapped_callable.stypy_type_store = module_type_store
            wrapped_callable.stypy_function_name = 'wrapped_callable'
            wrapped_callable.stypy_param_names_list = []
            wrapped_callable.stypy_varargs_param_name = 'args'
            wrapped_callable.stypy_kwargs_param_name = 'kwargs'
            wrapped_callable.stypy_call_defaults = defaults
            wrapped_callable.stypy_call_varargs = varargs
            wrapped_callable.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'wrapped_callable', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'wrapped_callable', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'wrapped_callable(...)' code ##################

            
            # Assigning a Call to a Name (line 138):
            
            # Assigning a Call to a Name (line 138):
            
            # Call to copy(...): (line 138)
            # Processing the call keyword arguments (line 138)
            kwargs_290272 = {}
            # Getting the type of 'matplotlib' (line 138)
            matplotlib_290268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 42), 'matplotlib', False)
            # Obtaining the member 'units' of a type (line 138)
            units_290269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), matplotlib_290268, 'units')
            # Obtaining the member 'registry' of a type (line 138)
            registry_290270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), units_290269, 'registry')
            # Obtaining the member 'copy' of a type (line 138)
            copy_290271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 42), registry_290270, 'copy')
            # Calling copy(args, kwargs) (line 138)
            copy_call_result_290273 = invoke(stypy.reporting.localization.Localization(__file__, 138, 42), copy_290271, *[], **kwargs_290272)
            
            # Assigning a type to the variable 'original_units_registry' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'original_units_registry', copy_call_result_290273)
            
            # Assigning a Call to a Name (line 139):
            
            # Assigning a Call to a Name (line 139):
            
            # Call to copy(...): (line 139)
            # Processing the call keyword arguments (line 139)
            kwargs_290277 = {}
            # Getting the type of 'mpl' (line 139)
            mpl_290274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 36), 'mpl', False)
            # Obtaining the member 'rcParams' of a type (line 139)
            rcParams_290275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 36), mpl_290274, 'rcParams')
            # Obtaining the member 'copy' of a type (line 139)
            copy_290276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 36), rcParams_290275, 'copy')
            # Calling copy(args, kwargs) (line 139)
            copy_call_result_290278 = invoke(stypy.reporting.localization.Localization(__file__, 139, 36), copy_290276, *[], **kwargs_290277)
            
            # Assigning a type to the variable 'original_settings' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'original_settings', copy_call_result_290278)
            
            # Call to use(...): (line 140)
            # Processing the call arguments (line 140)
            # Getting the type of 'style' (line 140)
            style_290282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 37), 'style', False)
            # Processing the call keyword arguments (line 140)
            kwargs_290283 = {}
            # Getting the type of 'matplotlib' (line 140)
            matplotlib_290279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'matplotlib', False)
            # Obtaining the member 'style' of a type (line 140)
            style_290280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), matplotlib_290279, 'style')
            # Obtaining the member 'use' of a type (line 140)
            use_290281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), style_290280, 'use')
            # Calling use(args, kwargs) (line 140)
            use_call_result_290284 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), use_290281, *[style_290282], **kwargs_290283)
            
            
            # Try-finally block (line 141)
            
            # Call to func(...): (line 142)
            # Getting the type of 'args' (line 142)
            args_290286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'args', False)
            # Processing the call keyword arguments (line 142)
            # Getting the type of 'kwargs' (line 142)
            kwargs_290287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'kwargs', False)
            kwargs_290288 = {'kwargs_290287': kwargs_290287}
            # Getting the type of 'func' (line 142)
            func_290285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'func', False)
            # Calling func(args, kwargs) (line 142)
            func_call_result_290289 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), func_290285, *[args_290286], **kwargs_290288)
            
            
            # finally branch of the try-finally block (line 141)
            
            # Call to _do_cleanup(...): (line 144)
            # Processing the call arguments (line 144)
            # Getting the type of 'original_units_registry' (line 144)
            original_units_registry_290291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'original_units_registry', False)
            # Getting the type of 'original_settings' (line 145)
            original_settings_290292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'original_settings', False)
            # Processing the call keyword arguments (line 144)
            kwargs_290293 = {}
            # Getting the type of '_do_cleanup' (line 144)
            _do_cleanup_290290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), '_do_cleanup', False)
            # Calling _do_cleanup(args, kwargs) (line 144)
            _do_cleanup_call_result_290294 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), _do_cleanup_290290, *[original_units_registry_290291, original_settings_290292], **kwargs_290293)
            
            
            
            # ################# End of 'wrapped_callable(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'wrapped_callable' in the type store
            # Getting the type of 'stypy_return_type' (line 136)
            stypy_return_type_290295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_290295)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'wrapped_callable'
            return stypy_return_type_290295

        # Assigning a type to the variable 'wrapped_callable' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'wrapped_callable', wrapped_callable)
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'wrapped_callable' (line 147)
        wrapped_callable_290296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'wrapped_callable')
        # Assigning a type to the variable 'stypy_return_type' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'stypy_return_type', wrapped_callable_290296)
        
        # ################# End of 'make_cleanup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_cleanup' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_290297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_cleanup'
        return stypy_return_type_290297

    # Assigning a type to the variable 'make_cleanup' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'make_cleanup', make_cleanup)
    
    
    # Call to isinstance(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'style' (line 149)
    style_290299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'style', False)
    # Getting the type of 'six' (line 149)
    six_290300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'six', False)
    # Obtaining the member 'string_types' of a type (line 149)
    string_types_290301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 25), six_290300, 'string_types')
    # Processing the call keyword arguments (line 149)
    kwargs_290302 = {}
    # Getting the type of 'isinstance' (line 149)
    isinstance_290298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 149)
    isinstance_call_result_290303 = invoke(stypy.reporting.localization.Localization(__file__, 149, 7), isinstance_290298, *[style_290299, string_types_290301], **kwargs_290302)
    
    # Testing the type of an if condition (line 149)
    if_condition_290304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 4), isinstance_call_result_290303)
    # Assigning a type to the variable 'if_condition_290304' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'if_condition_290304', if_condition_290304)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'make_cleanup' (line 150)
    make_cleanup_290305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'make_cleanup')
    # Assigning a type to the variable 'stypy_return_type' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'stypy_return_type', make_cleanup_290305)
    # SSA branch for the else part of an if statement (line 149)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to make_cleanup(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'style' (line 152)
    style_290307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'style', False)
    # Processing the call keyword arguments (line 152)
    kwargs_290308 = {}
    # Getting the type of 'make_cleanup' (line 152)
    make_cleanup_290306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'make_cleanup', False)
    # Calling make_cleanup(args, kwargs) (line 152)
    make_cleanup_call_result_290309 = invoke(stypy.reporting.localization.Localization(__file__, 152, 17), make_cleanup_290306, *[style_290307], **kwargs_290308)
    
    # Assigning a type to the variable 'result' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'result', make_cleanup_call_result_290309)
    
    # Assigning a Str to a Name (line 154):
    
    # Assigning a Str to a Name (line 154):
    unicode_290310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 16), 'unicode', u'_classic_test')
    # Assigning a type to the variable 'style' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'style', unicode_290310)
    # Getting the type of 'result' (line 155)
    result_290311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'stypy_return_type', result_290311)
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'cleanup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cleanup' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_290312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290312)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cleanup'
    return stypy_return_type_290312

# Assigning a type to the variable 'cleanup' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'cleanup', cleanup)

@norecursion
def check_freetype_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_freetype_version'
    module_type_store = module_type_store.open_function_context('check_freetype_version', 158, 0, False)
    
    # Passed parameters checking function
    check_freetype_version.stypy_localization = localization
    check_freetype_version.stypy_type_of_self = None
    check_freetype_version.stypy_type_store = module_type_store
    check_freetype_version.stypy_function_name = 'check_freetype_version'
    check_freetype_version.stypy_param_names_list = ['ver']
    check_freetype_version.stypy_varargs_param_name = None
    check_freetype_version.stypy_kwargs_param_name = None
    check_freetype_version.stypy_call_defaults = defaults
    check_freetype_version.stypy_call_varargs = varargs
    check_freetype_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_freetype_version', ['ver'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_freetype_version', localization, ['ver'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_freetype_version(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 159)
    # Getting the type of 'ver' (line 159)
    ver_290313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 7), 'ver')
    # Getting the type of 'None' (line 159)
    None_290314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'None')
    
    (may_be_290315, more_types_in_union_290316) = may_be_none(ver_290313, None_290314)

    if may_be_290315:

        if more_types_in_union_290316:
            # Runtime conditional SSA (line 159)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'True' (line 160)
        True_290317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type', True_290317)

        if more_types_in_union_290316:
            # SSA join for if statement (line 159)
            module_type_store = module_type_store.join_ssa_context()


    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 162, 4))
    
    # 'from distutils import version' statement (line 162)
    try:
        from distutils import version

    except:
        version = UndefinedType
    import_from_module(stypy.reporting.localization.Localization(__file__, 162, 4), 'distutils', None, module_type_store, ['version'], [version])
    
    
    
    # Call to isinstance(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'ver' (line 163)
    ver_290319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'ver', False)
    # Getting the type of 'six' (line 163)
    six_290320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 23), 'six', False)
    # Obtaining the member 'string_types' of a type (line 163)
    string_types_290321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 23), six_290320, 'string_types')
    # Processing the call keyword arguments (line 163)
    kwargs_290322 = {}
    # Getting the type of 'isinstance' (line 163)
    isinstance_290318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 163)
    isinstance_call_result_290323 = invoke(stypy.reporting.localization.Localization(__file__, 163, 7), isinstance_290318, *[ver_290319, string_types_290321], **kwargs_290322)
    
    # Testing the type of an if condition (line 163)
    if_condition_290324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 4), isinstance_call_result_290323)
    # Assigning a type to the variable 'if_condition_290324' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'if_condition_290324', if_condition_290324)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 164):
    
    # Assigning a Tuple to a Name (line 164):
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_290325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    # Getting the type of 'ver' (line 164)
    ver_290326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'ver')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 15), tuple_290325, ver_290326)
    # Adding element type (line 164)
    # Getting the type of 'ver' (line 164)
    ver_290327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'ver')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 15), tuple_290325, ver_290327)
    
    # Assigning a type to the variable 'ver' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'ver', tuple_290325)
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 165):
    
    # Assigning a ListComp to a Name (line 165):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ver' (line 165)
    ver_290333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 45), 'ver')
    comprehension_290334 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 11), ver_290333)
    # Assigning a type to the variable 'x' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'x', comprehension_290334)
    
    # Call to StrictVersion(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'x' (line 165)
    x_290330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'x', False)
    # Processing the call keyword arguments (line 165)
    kwargs_290331 = {}
    # Getting the type of 'version' (line 165)
    version_290328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'version', False)
    # Obtaining the member 'StrictVersion' of a type (line 165)
    StrictVersion_290329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 11), version_290328, 'StrictVersion')
    # Calling StrictVersion(args, kwargs) (line 165)
    StrictVersion_call_result_290332 = invoke(stypy.reporting.localization.Localization(__file__, 165, 11), StrictVersion_290329, *[x_290330], **kwargs_290331)
    
    list_290335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 11), list_290335, StrictVersion_call_result_290332)
    # Assigning a type to the variable 'ver' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'ver', list_290335)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to StrictVersion(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'ft2font' (line 166)
    ft2font_290338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 34), 'ft2font', False)
    # Obtaining the member '__freetype_version__' of a type (line 166)
    freetype_version___290339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 34), ft2font_290338, '__freetype_version__')
    # Processing the call keyword arguments (line 166)
    kwargs_290340 = {}
    # Getting the type of 'version' (line 166)
    version_290336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'version', False)
    # Obtaining the member 'StrictVersion' of a type (line 166)
    StrictVersion_290337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), version_290336, 'StrictVersion')
    # Calling StrictVersion(args, kwargs) (line 166)
    StrictVersion_call_result_290341 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), StrictVersion_290337, *[freetype_version___290339], **kwargs_290340)
    
    # Assigning a type to the variable 'found' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'found', StrictVersion_call_result_290341)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'found' (line 168)
    found_290342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'found')
    
    # Obtaining the type of the subscript
    int_290343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 24), 'int')
    # Getting the type of 'ver' (line 168)
    ver_290344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'ver')
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___290345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 20), ver_290344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_290346 = invoke(stypy.reporting.localization.Localization(__file__, 168, 20), getitem___290345, int_290343)
    
    # Applying the binary operator '>=' (line 168)
    result_ge_290347 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), '>=', found_290342, subscript_call_result_290346)
    
    
    # Getting the type of 'found' (line 168)
    found_290348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'found')
    
    # Obtaining the type of the subscript
    int_290349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 44), 'int')
    # Getting the type of 'ver' (line 168)
    ver_290350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 40), 'ver')
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___290351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 40), ver_290350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_290352 = invoke(stypy.reporting.localization.Localization(__file__, 168, 40), getitem___290351, int_290349)
    
    # Applying the binary operator '<=' (line 168)
    result_le_290353 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 31), '<=', found_290348, subscript_call_result_290352)
    
    # Applying the binary operator 'and' (line 168)
    result_and_keyword_290354 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), 'and', result_ge_290347, result_le_290353)
    
    # Assigning a type to the variable 'stypy_return_type' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type', result_and_keyword_290354)
    
    # ################# End of 'check_freetype_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_freetype_version' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_290355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290355)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_freetype_version'
    return stypy_return_type_290355

# Assigning a type to the variable 'check_freetype_version' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'check_freetype_version', check_freetype_version)

@norecursion
def _checked_on_freetype_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_checked_on_freetype_version'
    module_type_store = module_type_store.open_function_context('_checked_on_freetype_version', 171, 0, False)
    
    # Passed parameters checking function
    _checked_on_freetype_version.stypy_localization = localization
    _checked_on_freetype_version.stypy_type_of_self = None
    _checked_on_freetype_version.stypy_type_store = module_type_store
    _checked_on_freetype_version.stypy_function_name = '_checked_on_freetype_version'
    _checked_on_freetype_version.stypy_param_names_list = ['required_freetype_version']
    _checked_on_freetype_version.stypy_varargs_param_name = None
    _checked_on_freetype_version.stypy_kwargs_param_name = None
    _checked_on_freetype_version.stypy_call_defaults = defaults
    _checked_on_freetype_version.stypy_call_varargs = varargs
    _checked_on_freetype_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_checked_on_freetype_version', ['required_freetype_version'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_checked_on_freetype_version', localization, ['required_freetype_version'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_checked_on_freetype_version(...)' code ##################

    
    
    # Call to check_freetype_version(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'required_freetype_version' (line 172)
    required_freetype_version_290357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 30), 'required_freetype_version', False)
    # Processing the call keyword arguments (line 172)
    kwargs_290358 = {}
    # Getting the type of 'check_freetype_version' (line 172)
    check_freetype_version_290356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 7), 'check_freetype_version', False)
    # Calling check_freetype_version(args, kwargs) (line 172)
    check_freetype_version_call_result_290359 = invoke(stypy.reporting.localization.Localization(__file__, 172, 7), check_freetype_version_290356, *[required_freetype_version_290357], **kwargs_290358)
    
    # Testing the type of an if condition (line 172)
    if_condition_290360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 4), check_freetype_version_call_result_290359)
    # Assigning a type to the variable 'if_condition_290360' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'if_condition_290360', if_condition_290360)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def _stypy_temp_lambda_148(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_148'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_148', 173, 15, True)
        # Passed parameters checking function
        _stypy_temp_lambda_148.stypy_localization = localization
        _stypy_temp_lambda_148.stypy_type_of_self = None
        _stypy_temp_lambda_148.stypy_type_store = module_type_store
        _stypy_temp_lambda_148.stypy_function_name = '_stypy_temp_lambda_148'
        _stypy_temp_lambda_148.stypy_param_names_list = ['f']
        _stypy_temp_lambda_148.stypy_varargs_param_name = None
        _stypy_temp_lambda_148.stypy_kwargs_param_name = None
        _stypy_temp_lambda_148.stypy_call_defaults = defaults
        _stypy_temp_lambda_148.stypy_call_varargs = varargs
        _stypy_temp_lambda_148.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_148', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_148', ['f'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'f' (line 173)
        f_290361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 25), 'f')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'stypy_return_type', f_290361)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_148' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_290362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290362)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_148'
        return stypy_return_type_290362

    # Assigning a type to the variable '_stypy_temp_lambda_148' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), '_stypy_temp_lambda_148', _stypy_temp_lambda_148)
    # Getting the type of '_stypy_temp_lambda_148' (line 173)
    _stypy_temp_lambda_148_290363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), '_stypy_temp_lambda_148')
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', _stypy_temp_lambda_148_290363)
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 175):
    
    # Assigning a BinOp to a Name (line 175):
    unicode_290364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 14), 'unicode', u"Mismatched version of freetype. Test requires '%s', you have '%s'")
    
    # Obtaining an instance of the builtin type 'tuple' (line 177)
    tuple_290365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 177)
    # Adding element type (line 177)
    # Getting the type of 'required_freetype_version' (line 177)
    required_freetype_version_290366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'required_freetype_version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 15), tuple_290365, required_freetype_version_290366)
    # Adding element type (line 177)
    # Getting the type of 'ft2font' (line 177)
    ft2font_290367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 42), 'ft2font')
    # Obtaining the member '__freetype_version__' of a type (line 177)
    freetype_version___290368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 42), ft2font_290367, '__freetype_version__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 15), tuple_290365, freetype_version___290368)
    
    # Applying the binary operator '%' (line 175)
    result_mod_290369 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 14), '%', unicode_290364, tuple_290365)
    
    # Assigning a type to the variable 'reason' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'reason', result_mod_290369)
    
    # Call to _knownfailureif(...): (line 178)
    # Processing the call arguments (line 178)
    unicode_290371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 27), 'unicode', u'indeterminate')
    # Processing the call keyword arguments (line 178)
    # Getting the type of 'reason' (line 178)
    reason_290372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 48), 'reason', False)
    keyword_290373 = reason_290372
    # Getting the type of 'ImageComparisonFailure' (line 179)
    ImageComparisonFailure_290374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 49), 'ImageComparisonFailure', False)
    keyword_290375 = ImageComparisonFailure_290374
    kwargs_290376 = {'msg': keyword_290373, 'known_exception_class': keyword_290375}
    # Getting the type of '_knownfailureif' (line 178)
    _knownfailureif_290370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), '_knownfailureif', False)
    # Calling _knownfailureif(args, kwargs) (line 178)
    _knownfailureif_call_result_290377 = invoke(stypy.reporting.localization.Localization(__file__, 178, 11), _knownfailureif_290370, *[unicode_290371], **kwargs_290376)
    
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type', _knownfailureif_call_result_290377)
    
    # ################# End of '_checked_on_freetype_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_checked_on_freetype_version' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_290378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290378)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_checked_on_freetype_version'
    return stypy_return_type_290378

# Assigning a type to the variable '_checked_on_freetype_version' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), '_checked_on_freetype_version', _checked_on_freetype_version)

@norecursion
def remove_ticks_and_titles(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'remove_ticks_and_titles'
    module_type_store = module_type_store.open_function_context('remove_ticks_and_titles', 182, 0, False)
    
    # Passed parameters checking function
    remove_ticks_and_titles.stypy_localization = localization
    remove_ticks_and_titles.stypy_type_of_self = None
    remove_ticks_and_titles.stypy_type_store = module_type_store
    remove_ticks_and_titles.stypy_function_name = 'remove_ticks_and_titles'
    remove_ticks_and_titles.stypy_param_names_list = ['figure']
    remove_ticks_and_titles.stypy_varargs_param_name = None
    remove_ticks_and_titles.stypy_kwargs_param_name = None
    remove_ticks_and_titles.stypy_call_defaults = defaults
    remove_ticks_and_titles.stypy_call_varargs = varargs
    remove_ticks_and_titles.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'remove_ticks_and_titles', ['figure'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'remove_ticks_and_titles', localization, ['figure'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'remove_ticks_and_titles(...)' code ##################

    
    # Call to suptitle(...): (line 183)
    # Processing the call arguments (line 183)
    unicode_290381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 20), 'unicode', u'')
    # Processing the call keyword arguments (line 183)
    kwargs_290382 = {}
    # Getting the type of 'figure' (line 183)
    figure_290379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'figure', False)
    # Obtaining the member 'suptitle' of a type (line 183)
    suptitle_290380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), figure_290379, 'suptitle')
    # Calling suptitle(args, kwargs) (line 183)
    suptitle_call_result_290383 = invoke(stypy.reporting.localization.Localization(__file__, 183, 4), suptitle_290380, *[unicode_290381], **kwargs_290382)
    
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to NullFormatter(...): (line 184)
    # Processing the call keyword arguments (line 184)
    kwargs_290386 = {}
    # Getting the type of 'ticker' (line 184)
    ticker_290384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'ticker', False)
    # Obtaining the member 'NullFormatter' of a type (line 184)
    NullFormatter_290385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 21), ticker_290384, 'NullFormatter')
    # Calling NullFormatter(args, kwargs) (line 184)
    NullFormatter_call_result_290387 = invoke(stypy.reporting.localization.Localization(__file__, 184, 21), NullFormatter_290385, *[], **kwargs_290386)
    
    # Assigning a type to the variable 'null_formatter' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'null_formatter', NullFormatter_call_result_290387)
    
    
    # Call to get_axes(...): (line 185)
    # Processing the call keyword arguments (line 185)
    kwargs_290390 = {}
    # Getting the type of 'figure' (line 185)
    figure_290388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 14), 'figure', False)
    # Obtaining the member 'get_axes' of a type (line 185)
    get_axes_290389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 14), figure_290388, 'get_axes')
    # Calling get_axes(args, kwargs) (line 185)
    get_axes_call_result_290391 = invoke(stypy.reporting.localization.Localization(__file__, 185, 14), get_axes_290389, *[], **kwargs_290390)
    
    # Testing the type of a for loop iterable (line 185)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 185, 4), get_axes_call_result_290391)
    # Getting the type of the for loop variable (line 185)
    for_loop_var_290392 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 185, 4), get_axes_call_result_290391)
    # Assigning a type to the variable 'ax' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'ax', for_loop_var_290392)
    # SSA begins for a for statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to set_title(...): (line 186)
    # Processing the call arguments (line 186)
    unicode_290395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 21), 'unicode', u'')
    # Processing the call keyword arguments (line 186)
    kwargs_290396 = {}
    # Getting the type of 'ax' (line 186)
    ax_290393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'ax', False)
    # Obtaining the member 'set_title' of a type (line 186)
    set_title_290394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), ax_290393, 'set_title')
    # Calling set_title(args, kwargs) (line 186)
    set_title_call_result_290397 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), set_title_290394, *[unicode_290395], **kwargs_290396)
    
    
    # Call to set_major_formatter(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'null_formatter' (line 187)
    null_formatter_290401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 37), 'null_formatter', False)
    # Processing the call keyword arguments (line 187)
    kwargs_290402 = {}
    # Getting the type of 'ax' (line 187)
    ax_290398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'ax', False)
    # Obtaining the member 'xaxis' of a type (line 187)
    xaxis_290399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), ax_290398, 'xaxis')
    # Obtaining the member 'set_major_formatter' of a type (line 187)
    set_major_formatter_290400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), xaxis_290399, 'set_major_formatter')
    # Calling set_major_formatter(args, kwargs) (line 187)
    set_major_formatter_call_result_290403 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), set_major_formatter_290400, *[null_formatter_290401], **kwargs_290402)
    
    
    # Call to set_minor_formatter(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'null_formatter' (line 188)
    null_formatter_290407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 37), 'null_formatter', False)
    # Processing the call keyword arguments (line 188)
    kwargs_290408 = {}
    # Getting the type of 'ax' (line 188)
    ax_290404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'ax', False)
    # Obtaining the member 'xaxis' of a type (line 188)
    xaxis_290405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), ax_290404, 'xaxis')
    # Obtaining the member 'set_minor_formatter' of a type (line 188)
    set_minor_formatter_290406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), xaxis_290405, 'set_minor_formatter')
    # Calling set_minor_formatter(args, kwargs) (line 188)
    set_minor_formatter_call_result_290409 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), set_minor_formatter_290406, *[null_formatter_290407], **kwargs_290408)
    
    
    # Call to set_major_formatter(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'null_formatter' (line 189)
    null_formatter_290413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'null_formatter', False)
    # Processing the call keyword arguments (line 189)
    kwargs_290414 = {}
    # Getting the type of 'ax' (line 189)
    ax_290410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'ax', False)
    # Obtaining the member 'yaxis' of a type (line 189)
    yaxis_290411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), ax_290410, 'yaxis')
    # Obtaining the member 'set_major_formatter' of a type (line 189)
    set_major_formatter_290412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), yaxis_290411, 'set_major_formatter')
    # Calling set_major_formatter(args, kwargs) (line 189)
    set_major_formatter_call_result_290415 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), set_major_formatter_290412, *[null_formatter_290413], **kwargs_290414)
    
    
    # Call to set_minor_formatter(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'null_formatter' (line 190)
    null_formatter_290419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 37), 'null_formatter', False)
    # Processing the call keyword arguments (line 190)
    kwargs_290420 = {}
    # Getting the type of 'ax' (line 190)
    ax_290416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'ax', False)
    # Obtaining the member 'yaxis' of a type (line 190)
    yaxis_290417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), ax_290416, 'yaxis')
    # Obtaining the member 'set_minor_formatter' of a type (line 190)
    set_minor_formatter_290418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), yaxis_290417, 'set_minor_formatter')
    # Calling set_minor_formatter(args, kwargs) (line 190)
    set_minor_formatter_call_result_290421 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), set_minor_formatter_290418, *[null_formatter_290419], **kwargs_290420)
    
    
    
    # SSA begins for try-except statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to set_major_formatter(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'null_formatter' (line 192)
    null_formatter_290425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 41), 'null_formatter', False)
    # Processing the call keyword arguments (line 192)
    kwargs_290426 = {}
    # Getting the type of 'ax' (line 192)
    ax_290422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'ax', False)
    # Obtaining the member 'zaxis' of a type (line 192)
    zaxis_290423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), ax_290422, 'zaxis')
    # Obtaining the member 'set_major_formatter' of a type (line 192)
    set_major_formatter_290424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), zaxis_290423, 'set_major_formatter')
    # Calling set_major_formatter(args, kwargs) (line 192)
    set_major_formatter_call_result_290427 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), set_major_formatter_290424, *[null_formatter_290425], **kwargs_290426)
    
    
    # Call to set_minor_formatter(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'null_formatter' (line 193)
    null_formatter_290431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 41), 'null_formatter', False)
    # Processing the call keyword arguments (line 193)
    kwargs_290432 = {}
    # Getting the type of 'ax' (line 193)
    ax_290428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'ax', False)
    # Obtaining the member 'zaxis' of a type (line 193)
    zaxis_290429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), ax_290428, 'zaxis')
    # Obtaining the member 'set_minor_formatter' of a type (line 193)
    set_minor_formatter_290430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), zaxis_290429, 'set_minor_formatter')
    # Calling set_minor_formatter(args, kwargs) (line 193)
    set_minor_formatter_call_result_290433 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), set_minor_formatter_290430, *[null_formatter_290431], **kwargs_290432)
    
    # SSA branch for the except part of a try statement (line 191)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 191)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'remove_ticks_and_titles(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'remove_ticks_and_titles' in the type store
    # Getting the type of 'stypy_return_type' (line 182)
    stypy_return_type_290434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290434)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'remove_ticks_and_titles'
    return stypy_return_type_290434

# Assigning a type to the variable 'remove_ticks_and_titles' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'remove_ticks_and_titles', remove_ticks_and_titles)

@norecursion
def _raise_on_image_difference(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_raise_on_image_difference'
    module_type_store = module_type_store.open_function_context('_raise_on_image_difference', 198, 0, False)
    
    # Passed parameters checking function
    _raise_on_image_difference.stypy_localization = localization
    _raise_on_image_difference.stypy_type_of_self = None
    _raise_on_image_difference.stypy_type_store = module_type_store
    _raise_on_image_difference.stypy_function_name = '_raise_on_image_difference'
    _raise_on_image_difference.stypy_param_names_list = ['expected', 'actual', 'tol']
    _raise_on_image_difference.stypy_varargs_param_name = None
    _raise_on_image_difference.stypy_kwargs_param_name = None
    _raise_on_image_difference.stypy_call_defaults = defaults
    _raise_on_image_difference.stypy_call_varargs = varargs
    _raise_on_image_difference.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_raise_on_image_difference', ['expected', 'actual', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_raise_on_image_difference', localization, ['expected', 'actual', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_raise_on_image_difference(...)' code ##################

    
    # Assigning a Name to a Name (line 199):
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'True' (line 199)
    True_290435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 24), 'True')
    # Assigning a type to the variable '__tracebackhide__' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), '__tracebackhide__', True_290435)
    
    # Assigning a Call to a Name (line 201):
    
    # Assigning a Call to a Name (line 201):
    
    # Call to compare_images(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'expected' (line 201)
    expected_290437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'expected', False)
    # Getting the type of 'actual' (line 201)
    actual_290438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 35), 'actual', False)
    # Getting the type of 'tol' (line 201)
    tol_290439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 43), 'tol', False)
    # Processing the call keyword arguments (line 201)
    # Getting the type of 'True' (line 201)
    True_290440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 61), 'True', False)
    keyword_290441 = True_290440
    kwargs_290442 = {'in_decorator': keyword_290441}
    # Getting the type of 'compare_images' (line 201)
    compare_images_290436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 10), 'compare_images', False)
    # Calling compare_images(args, kwargs) (line 201)
    compare_images_call_result_290443 = invoke(stypy.reporting.localization.Localization(__file__, 201, 10), compare_images_290436, *[expected_290437, actual_290438, tol_290439], **kwargs_290442)
    
    # Assigning a type to the variable 'err' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'err', compare_images_call_result_290443)
    
    
    
    # Call to exists(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'expected' (line 203)
    expected_290447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 26), 'expected', False)
    # Processing the call keyword arguments (line 203)
    kwargs_290448 = {}
    # Getting the type of 'os' (line 203)
    os_290444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 203)
    path_290445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), os_290444, 'path')
    # Obtaining the member 'exists' of a type (line 203)
    exists_290446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), path_290445, 'exists')
    # Calling exists(args, kwargs) (line 203)
    exists_call_result_290449 = invoke(stypy.reporting.localization.Localization(__file__, 203, 11), exists_290446, *[expected_290447], **kwargs_290448)
    
    # Applying the 'not' unary operator (line 203)
    result_not__290450 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 7), 'not', exists_call_result_290449)
    
    # Testing the type of an if condition (line 203)
    if_condition_290451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 4), result_not__290450)
    # Assigning a type to the variable 'if_condition_290451' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'if_condition_290451', if_condition_290451)
    # SSA begins for if statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ImageComparisonFailure(...): (line 204)
    # Processing the call arguments (line 204)
    unicode_290453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 37), 'unicode', u'image does not exist: %s')
    # Getting the type of 'expected' (line 204)
    expected_290454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 66), 'expected', False)
    # Applying the binary operator '%' (line 204)
    result_mod_290455 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 37), '%', unicode_290453, expected_290454)
    
    # Processing the call keyword arguments (line 204)
    kwargs_290456 = {}
    # Getting the type of 'ImageComparisonFailure' (line 204)
    ImageComparisonFailure_290452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 14), 'ImageComparisonFailure', False)
    # Calling ImageComparisonFailure(args, kwargs) (line 204)
    ImageComparisonFailure_call_result_290457 = invoke(stypy.reporting.localization.Localization(__file__, 204, 14), ImageComparisonFailure_290452, *[result_mod_290455], **kwargs_290456)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 204, 8), ImageComparisonFailure_call_result_290457, 'raise parameter', BaseException)
    # SSA join for if statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'err' (line 206)
    err_290458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'err')
    # Testing the type of an if condition (line 206)
    if_condition_290459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 4), err_290458)
    # Assigning a type to the variable 'if_condition_290459' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'if_condition_290459', if_condition_290459)
    # SSA begins for if statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining an instance of the builtin type 'list' (line 207)
    list_290460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 207)
    # Adding element type (line 207)
    unicode_290461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 20), 'unicode', u'actual')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 19), list_290460, unicode_290461)
    # Adding element type (line 207)
    unicode_290462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 30), 'unicode', u'expected')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 19), list_290460, unicode_290462)
    
    # Testing the type of a for loop iterable (line 207)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 207, 8), list_290460)
    # Getting the type of the for loop variable (line 207)
    for_loop_var_290463 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 207, 8), list_290460)
    # Assigning a type to the variable 'key' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'key', for_loop_var_290463)
    # SSA begins for a for statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 208):
    
    # Assigning a Call to a Subscript (line 208):
    
    # Call to relpath(...): (line 208)
    # Processing the call arguments (line 208)
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 208)
    key_290467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 43), 'key', False)
    # Getting the type of 'err' (line 208)
    err_290468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 39), 'err', False)
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___290469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 39), err_290468, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_290470 = invoke(stypy.reporting.localization.Localization(__file__, 208, 39), getitem___290469, key_290467)
    
    # Processing the call keyword arguments (line 208)
    kwargs_290471 = {}
    # Getting the type of 'os' (line 208)
    os_290464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 208)
    path_290465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 23), os_290464, 'path')
    # Obtaining the member 'relpath' of a type (line 208)
    relpath_290466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 23), path_290465, 'relpath')
    # Calling relpath(args, kwargs) (line 208)
    relpath_call_result_290472 = invoke(stypy.reporting.localization.Localization(__file__, 208, 23), relpath_290466, *[subscript_call_result_290470], **kwargs_290471)
    
    # Getting the type of 'err' (line 208)
    err_290473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'err')
    # Getting the type of 'key' (line 208)
    key_290474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'key')
    # Storing an element on a container (line 208)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 12), err_290473, (key_290474, relpath_call_result_290472))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ImageComparisonFailure(...): (line 209)
    # Processing the call arguments (line 209)
    unicode_290476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 12), 'unicode', u'images not close (RMS %(rms).3f):\n\t%(actual)s\n\t%(expected)s ')
    # Getting the type of 'err' (line 211)
    err_290477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'err', False)
    # Applying the binary operator '%' (line 210)
    result_mod_290478 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 12), '%', unicode_290476, err_290477)
    
    # Processing the call keyword arguments (line 209)
    kwargs_290479 = {}
    # Getting the type of 'ImageComparisonFailure' (line 209)
    ImageComparisonFailure_290475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'ImageComparisonFailure', False)
    # Calling ImageComparisonFailure(args, kwargs) (line 209)
    ImageComparisonFailure_call_result_290480 = invoke(stypy.reporting.localization.Localization(__file__, 209, 14), ImageComparisonFailure_290475, *[result_mod_290478], **kwargs_290479)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 209, 8), ImageComparisonFailure_call_result_290480, 'raise parameter', BaseException)
    # SSA join for if statement (line 206)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_raise_on_image_difference(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_raise_on_image_difference' in the type store
    # Getting the type of 'stypy_return_type' (line 198)
    stypy_return_type_290481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290481)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_raise_on_image_difference'
    return stypy_return_type_290481

# Assigning a type to the variable '_raise_on_image_difference' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), '_raise_on_image_difference', _raise_on_image_difference)

@norecursion
def _xfail_if_format_is_uncomparable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_xfail_if_format_is_uncomparable'
    module_type_store = module_type_store.open_function_context('_xfail_if_format_is_uncomparable', 214, 0, False)
    
    # Passed parameters checking function
    _xfail_if_format_is_uncomparable.stypy_localization = localization
    _xfail_if_format_is_uncomparable.stypy_type_of_self = None
    _xfail_if_format_is_uncomparable.stypy_type_store = module_type_store
    _xfail_if_format_is_uncomparable.stypy_function_name = '_xfail_if_format_is_uncomparable'
    _xfail_if_format_is_uncomparable.stypy_param_names_list = ['extension']
    _xfail_if_format_is_uncomparable.stypy_varargs_param_name = None
    _xfail_if_format_is_uncomparable.stypy_kwargs_param_name = None
    _xfail_if_format_is_uncomparable.stypy_call_defaults = defaults
    _xfail_if_format_is_uncomparable.stypy_call_varargs = varargs
    _xfail_if_format_is_uncomparable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_xfail_if_format_is_uncomparable', ['extension'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_xfail_if_format_is_uncomparable', localization, ['extension'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_xfail_if_format_is_uncomparable(...)' code ##################

    
    # Assigning a Compare to a Name (line 215):
    
    # Assigning a Compare to a Name (line 215):
    
    # Getting the type of 'extension' (line 215)
    extension_290482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'extension')
    
    # Call to comparable_formats(...): (line 215)
    # Processing the call keyword arguments (line 215)
    kwargs_290484 = {}
    # Getting the type of 'comparable_formats' (line 215)
    comparable_formats_290483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 33), 'comparable_formats', False)
    # Calling comparable_formats(args, kwargs) (line 215)
    comparable_formats_call_result_290485 = invoke(stypy.reporting.localization.Localization(__file__, 215, 33), comparable_formats_290483, *[], **kwargs_290484)
    
    # Applying the binary operator 'notin' (line 215)
    result_contains_290486 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 16), 'notin', extension_290482, comparable_formats_call_result_290485)
    
    # Assigning a type to the variable 'will_fail' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'will_fail', result_contains_290486)
    
    # Getting the type of 'will_fail' (line 216)
    will_fail_290487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 7), 'will_fail')
    # Testing the type of an if condition (line 216)
    if_condition_290488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 4), will_fail_290487)
    # Assigning a type to the variable 'if_condition_290488' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'if_condition_290488', if_condition_290488)
    # SSA begins for if statement (line 216)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 217):
    
    # Assigning a BinOp to a Name (line 217):
    unicode_290489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 19), 'unicode', u'Cannot compare %s files on this system')
    # Getting the type of 'extension' (line 217)
    extension_290490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 62), 'extension')
    # Applying the binary operator '%' (line 217)
    result_mod_290491 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 19), '%', unicode_290489, extension_290490)
    
    # Assigning a type to the variable 'fail_msg' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'fail_msg', result_mod_290491)
    # SSA branch for the else part of an if statement (line 216)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 219):
    
    # Assigning a Str to a Name (line 219):
    unicode_290492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 19), 'unicode', u'No failure expected')
    # Assigning a type to the variable 'fail_msg' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'fail_msg', unicode_290492)
    # SSA join for if statement (line 216)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _knownfailureif(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'will_fail' (line 221)
    will_fail_290494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 27), 'will_fail', False)
    # Getting the type of 'fail_msg' (line 221)
    fail_msg_290495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 38), 'fail_msg', False)
    # Processing the call keyword arguments (line 221)
    # Getting the type of 'ImageComparisonFailure' (line 222)
    ImageComparisonFailure_290496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 49), 'ImageComparisonFailure', False)
    keyword_290497 = ImageComparisonFailure_290496
    kwargs_290498 = {'known_exception_class': keyword_290497}
    # Getting the type of '_knownfailureif' (line 221)
    _knownfailureif_290493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), '_knownfailureif', False)
    # Calling _knownfailureif(args, kwargs) (line 221)
    _knownfailureif_call_result_290499 = invoke(stypy.reporting.localization.Localization(__file__, 221, 11), _knownfailureif_290493, *[will_fail_290494, fail_msg_290495], **kwargs_290498)
    
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type', _knownfailureif_call_result_290499)
    
    # ################# End of '_xfail_if_format_is_uncomparable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_xfail_if_format_is_uncomparable' in the type store
    # Getting the type of 'stypy_return_type' (line 214)
    stypy_return_type_290500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290500)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_xfail_if_format_is_uncomparable'
    return stypy_return_type_290500

# Assigning a type to the variable '_xfail_if_format_is_uncomparable' (line 214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), '_xfail_if_format_is_uncomparable', _xfail_if_format_is_uncomparable)

@norecursion
def _mark_xfail_if_format_is_uncomparable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_mark_xfail_if_format_is_uncomparable'
    module_type_store = module_type_store.open_function_context('_mark_xfail_if_format_is_uncomparable', 225, 0, False)
    
    # Passed parameters checking function
    _mark_xfail_if_format_is_uncomparable.stypy_localization = localization
    _mark_xfail_if_format_is_uncomparable.stypy_type_of_self = None
    _mark_xfail_if_format_is_uncomparable.stypy_type_store = module_type_store
    _mark_xfail_if_format_is_uncomparable.stypy_function_name = '_mark_xfail_if_format_is_uncomparable'
    _mark_xfail_if_format_is_uncomparable.stypy_param_names_list = ['extension']
    _mark_xfail_if_format_is_uncomparable.stypy_varargs_param_name = None
    _mark_xfail_if_format_is_uncomparable.stypy_kwargs_param_name = None
    _mark_xfail_if_format_is_uncomparable.stypy_call_defaults = defaults
    _mark_xfail_if_format_is_uncomparable.stypy_call_varargs = varargs
    _mark_xfail_if_format_is_uncomparable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_mark_xfail_if_format_is_uncomparable', ['extension'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_mark_xfail_if_format_is_uncomparable', localization, ['extension'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_mark_xfail_if_format_is_uncomparable(...)' code ##################

    
    
    # Call to isinstance(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'extension' (line 226)
    extension_290502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'extension', False)
    # Getting the type of 'six' (line 226)
    six_290503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'six', False)
    # Obtaining the member 'string_types' of a type (line 226)
    string_types_290504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 29), six_290503, 'string_types')
    # Processing the call keyword arguments (line 226)
    kwargs_290505 = {}
    # Getting the type of 'isinstance' (line 226)
    isinstance_290501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 226)
    isinstance_call_result_290506 = invoke(stypy.reporting.localization.Localization(__file__, 226, 7), isinstance_290501, *[extension_290502, string_types_290504], **kwargs_290505)
    
    # Testing the type of an if condition (line 226)
    if_condition_290507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 4), isinstance_call_result_290506)
    # Assigning a type to the variable 'if_condition_290507' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'if_condition_290507', if_condition_290507)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Compare to a Name (line 227):
    
    # Assigning a Compare to a Name (line 227):
    
    # Getting the type of 'extension' (line 227)
    extension_290508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'extension')
    
    # Call to comparable_formats(...): (line 227)
    # Processing the call keyword arguments (line 227)
    kwargs_290510 = {}
    # Getting the type of 'comparable_formats' (line 227)
    comparable_formats_290509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 37), 'comparable_formats', False)
    # Calling comparable_formats(args, kwargs) (line 227)
    comparable_formats_call_result_290511 = invoke(stypy.reporting.localization.Localization(__file__, 227, 37), comparable_formats_290509, *[], **kwargs_290510)
    
    # Applying the binary operator 'notin' (line 227)
    result_contains_290512 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 20), 'notin', extension_290508, comparable_formats_call_result_290511)
    
    # Assigning a type to the variable 'will_fail' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'will_fail', result_contains_290512)
    # SSA branch for the else part of an if statement (line 226)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Compare to a Name (line 230):
    
    # Assigning a Compare to a Name (line 230):
    
    
    # Obtaining the type of the subscript
    int_290513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 35), 'int')
    # Getting the type of 'extension' (line 230)
    extension_290514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'extension')
    # Obtaining the member 'args' of a type (line 230)
    args_290515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 20), extension_290514, 'args')
    # Obtaining the member '__getitem__' of a type (line 230)
    getitem___290516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 20), args_290515, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 230)
    subscript_call_result_290517 = invoke(stypy.reporting.localization.Localization(__file__, 230, 20), getitem___290516, int_290513)
    
    
    # Call to comparable_formats(...): (line 230)
    # Processing the call keyword arguments (line 230)
    kwargs_290519 = {}
    # Getting the type of 'comparable_formats' (line 230)
    comparable_formats_290518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'comparable_formats', False)
    # Calling comparable_formats(args, kwargs) (line 230)
    comparable_formats_call_result_290520 = invoke(stypy.reporting.localization.Localization(__file__, 230, 45), comparable_formats_290518, *[], **kwargs_290519)
    
    # Applying the binary operator 'notin' (line 230)
    result_contains_290521 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 20), 'notin', subscript_call_result_290517, comparable_formats_call_result_290520)
    
    # Assigning a type to the variable 'will_fail' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'will_fail', result_contains_290521)
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'will_fail' (line 231)
    will_fail_290522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 7), 'will_fail')
    # Testing the type of an if condition (line 231)
    if_condition_290523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 4), will_fail_290522)
    # Assigning a type to the variable 'if_condition_290523' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'if_condition_290523', if_condition_290523)
    # SSA begins for if statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 232):
    
    # Assigning a BinOp to a Name (line 232):
    unicode_290524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 19), 'unicode', u'Cannot compare %s files on this system')
    # Getting the type of 'extension' (line 232)
    extension_290525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 62), 'extension')
    # Applying the binary operator '%' (line 232)
    result_mod_290526 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 19), '%', unicode_290524, extension_290525)
    
    # Assigning a type to the variable 'fail_msg' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'fail_msg', result_mod_290526)
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 233, 8))
    
    # 'import pytest' statement (line 233)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_290527 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 233, 8), 'pytest')

    if (type(import_290527) is not StypyTypeError):

        if (import_290527 != 'pyd_module'):
            __import__(import_290527)
            sys_modules_290528 = sys.modules[import_290527]
            import_module(stypy.reporting.localization.Localization(__file__, 233, 8), 'pytest', sys_modules_290528.module_type_store, module_type_store)
        else:
            import pytest

            import_module(stypy.reporting.localization.Localization(__file__, 233, 8), 'pytest', pytest, module_type_store)

    else:
        # Assigning a type to the variable 'pytest' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'pytest', import_290527)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    # Call to xfail(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'extension' (line 234)
    extension_290532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 33), 'extension', False)
    # Processing the call keyword arguments (line 234)
    # Getting the type of 'fail_msg' (line 234)
    fail_msg_290533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 51), 'fail_msg', False)
    keyword_290534 = fail_msg_290533
    # Getting the type of 'False' (line 234)
    False_290535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 68), 'False', False)
    keyword_290536 = False_290535
    # Getting the type of 'ImageComparisonFailure' (line 235)
    ImageComparisonFailure_290537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 40), 'ImageComparisonFailure', False)
    keyword_290538 = ImageComparisonFailure_290537
    kwargs_290539 = {'strict': keyword_290536, 'reason': keyword_290534, 'raises': keyword_290538}
    # Getting the type of 'pytest' (line 234)
    pytest_290529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'pytest', False)
    # Obtaining the member 'mark' of a type (line 234)
    mark_290530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 15), pytest_290529, 'mark')
    # Obtaining the member 'xfail' of a type (line 234)
    xfail_290531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 15), mark_290530, 'xfail')
    # Calling xfail(args, kwargs) (line 234)
    xfail_call_result_290540 = invoke(stypy.reporting.localization.Localization(__file__, 234, 15), xfail_290531, *[extension_290532], **kwargs_290539)
    
    # Assigning a type to the variable 'stypy_return_type' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'stypy_return_type', xfail_call_result_290540)
    # SSA branch for the else part of an if statement (line 231)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'extension' (line 237)
    extension_290541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'extension')
    # Assigning a type to the variable 'stypy_return_type' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type', extension_290541)
    # SSA join for if statement (line 231)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_mark_xfail_if_format_is_uncomparable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_mark_xfail_if_format_is_uncomparable' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_290542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290542)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_mark_xfail_if_format_is_uncomparable'
    return stypy_return_type_290542

# Assigning a type to the variable '_mark_xfail_if_format_is_uncomparable' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), '_mark_xfail_if_format_is_uncomparable', _mark_xfail_if_format_is_uncomparable)
# Declaration of the '_ImageComparisonBase' class

class _ImageComparisonBase(object, ):
    unicode_290543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, (-1)), 'unicode', u'\n    Image comparison base class\n\n    This class provides *just* the comparison-related functionality and avoids\n    any code that would be specific to any testing framework.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ImageComparisonBase.__init__', ['tol', 'remove_text', 'savefig_kwargs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['tol', 'remove_text', 'savefig_kwargs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Multiple assignment of 3 elements.
        
        # Assigning a Name to a Attribute (line 248):
        # Getting the type of 'None' (line 248)
        None_290544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 58), 'None')
        # Getting the type of 'self' (line 248)
        self_290545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 40), 'self')
        # Setting the type of the member 'result_dir' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 40), self_290545, 'result_dir', None_290544)
        
        # Assigning a Attribute to a Attribute (line 248):
        # Getting the type of 'self' (line 248)
        self_290546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 40), 'self')
        # Obtaining the member 'result_dir' of a type (line 248)
        result_dir_290547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 40), self_290546, 'result_dir')
        # Getting the type of 'self' (line 248)
        self_290548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'self')
        # Setting the type of the member 'baseline_dir' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 20), self_290548, 'baseline_dir', result_dir_290547)
        
        # Assigning a Attribute to a Attribute (line 248):
        # Getting the type of 'self' (line 248)
        self_290549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'self')
        # Obtaining the member 'baseline_dir' of a type (line 248)
        baseline_dir_290550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 20), self_290549, 'baseline_dir')
        # Getting the type of 'self' (line 248)
        self_290551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member 'func' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_290551, 'func', baseline_dir_290550)
        
        # Assigning a Name to a Attribute (line 249):
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'tol' (line 249)
        tol_290552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'tol')
        # Getting the type of 'self' (line 249)
        self_290553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member 'tol' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_290553, 'tol', tol_290552)
        
        # Assigning a Name to a Attribute (line 250):
        
        # Assigning a Name to a Attribute (line 250):
        # Getting the type of 'remove_text' (line 250)
        remove_text_290554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'remove_text')
        # Getting the type of 'self' (line 250)
        self_290555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'remove_text' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_290555, 'remove_text', remove_text_290554)
        
        # Assigning a Name to a Attribute (line 251):
        
        # Assigning a Name to a Attribute (line 251):
        # Getting the type of 'savefig_kwargs' (line 251)
        savefig_kwargs_290556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 30), 'savefig_kwargs')
        # Getting the type of 'self' (line 251)
        self_290557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self')
        # Setting the type of the member 'savefig_kwargs' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_290557, 'savefig_kwargs', savefig_kwargs_290556)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def delayed_init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'delayed_init'
        module_type_store = module_type_store.open_function_context('delayed_init', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_localization', localization)
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_function_name', '_ImageComparisonBase.delayed_init')
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_param_names_list', ['func'])
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ImageComparisonBase.delayed_init.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ImageComparisonBase.delayed_init', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'delayed_init', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'delayed_init(...)' code ##################

        # Evaluating assert statement condition
        
        # Getting the type of 'self' (line 254)
        self_290558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'self')
        # Obtaining the member 'func' of a type (line 254)
        func_290559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 15), self_290558, 'func')
        # Getting the type of 'None' (line 254)
        None_290560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 28), 'None')
        # Applying the binary operator 'is' (line 254)
        result_is__290561 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 15), 'is', func_290559, None_290560)
        
        
        # Assigning a Name to a Attribute (line 255):
        
        # Assigning a Name to a Attribute (line 255):
        # Getting the type of 'func' (line 255)
        func_290562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'func')
        # Getting the type of 'self' (line 255)
        self_290563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self')
        # Setting the type of the member 'func' of a type (line 255)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_290563, 'func', func_290562)
        
        # Assigning a Call to a Tuple (line 256):
        
        # Assigning a Call to a Name:
        
        # Call to _image_directories(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'func' (line 256)
        func_290565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 64), 'func', False)
        # Processing the call keyword arguments (line 256)
        kwargs_290566 = {}
        # Getting the type of '_image_directories' (line 256)
        _image_directories_290564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 45), '_image_directories', False)
        # Calling _image_directories(args, kwargs) (line 256)
        _image_directories_call_result_290567 = invoke(stypy.reporting.localization.Localization(__file__, 256, 45), _image_directories_290564, *[func_290565], **kwargs_290566)
        
        # Assigning a type to the variable 'call_assignment_290048' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'call_assignment_290048', _image_directories_call_result_290567)
        
        # Assigning a Call to a Name (line 256):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_290570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 8), 'int')
        # Processing the call keyword arguments
        kwargs_290571 = {}
        # Getting the type of 'call_assignment_290048' (line 256)
        call_assignment_290048_290568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'call_assignment_290048', False)
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___290569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), call_assignment_290048_290568, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_290572 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___290569, *[int_290570], **kwargs_290571)
        
        # Assigning a type to the variable 'call_assignment_290049' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'call_assignment_290049', getitem___call_result_290572)
        
        # Assigning a Name to a Attribute (line 256):
        # Getting the type of 'call_assignment_290049' (line 256)
        call_assignment_290049_290573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'call_assignment_290049')
        # Getting the type of 'self' (line 256)
        self_290574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'self')
        # Setting the type of the member 'baseline_dir' of a type (line 256)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), self_290574, 'baseline_dir', call_assignment_290049_290573)
        
        # Assigning a Call to a Name (line 256):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_290577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 8), 'int')
        # Processing the call keyword arguments
        kwargs_290578 = {}
        # Getting the type of 'call_assignment_290048' (line 256)
        call_assignment_290048_290575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'call_assignment_290048', False)
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___290576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), call_assignment_290048_290575, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_290579 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___290576, *[int_290577], **kwargs_290578)
        
        # Assigning a type to the variable 'call_assignment_290050' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'call_assignment_290050', getitem___call_result_290579)
        
        # Assigning a Name to a Attribute (line 256):
        # Getting the type of 'call_assignment_290050' (line 256)
        call_assignment_290050_290580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'call_assignment_290050')
        # Getting the type of 'self' (line 256)
        self_290581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 27), 'self')
        # Setting the type of the member 'result_dir' of a type (line 256)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 27), self_290581, 'result_dir', call_assignment_290050_290580)
        
        # ################# End of 'delayed_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'delayed_init' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_290582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290582)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'delayed_init'
        return stypy_return_type_290582


    @norecursion
    def copy_baseline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy_baseline'
        module_type_store = module_type_store.open_function_context('copy_baseline', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_localization', localization)
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_function_name', '_ImageComparisonBase.copy_baseline')
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_param_names_list', ['baseline', 'extension'])
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ImageComparisonBase.copy_baseline.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ImageComparisonBase.copy_baseline', ['baseline', 'extension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy_baseline', localization, ['baseline', 'extension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy_baseline(...)' code ##################

        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to join(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'self' (line 259)
        self_290586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 37), 'self', False)
        # Obtaining the member 'baseline_dir' of a type (line 259)
        baseline_dir_290587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 37), self_290586, 'baseline_dir')
        # Getting the type of 'baseline' (line 259)
        baseline_290588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 56), 'baseline', False)
        # Processing the call keyword arguments (line 259)
        kwargs_290589 = {}
        # Getting the type of 'os' (line 259)
        os_290583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 259)
        path_290584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), os_290583, 'path')
        # Obtaining the member 'join' of a type (line 259)
        join_290585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), path_290584, 'join')
        # Calling join(args, kwargs) (line 259)
        join_call_result_290590 = invoke(stypy.reporting.localization.Localization(__file__, 259, 24), join_290585, *[baseline_dir_290587, baseline_290588], **kwargs_290589)
        
        # Assigning a type to the variable 'baseline_path' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'baseline_path', join_call_result_290590)
        
        # Assigning a BinOp to a Name (line 260):
        
        # Assigning a BinOp to a Name (line 260):
        # Getting the type of 'baseline_path' (line 260)
        baseline_path_290591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 30), 'baseline_path')
        unicode_290592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 46), 'unicode', u'.')
        # Applying the binary operator '+' (line 260)
        result_add_290593 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 30), '+', baseline_path_290591, unicode_290592)
        
        # Getting the type of 'extension' (line 260)
        extension_290594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 52), 'extension')
        # Applying the binary operator '+' (line 260)
        result_add_290595 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 50), '+', result_add_290593, extension_290594)
        
        # Assigning a type to the variable 'orig_expected_fname' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'orig_expected_fname', result_add_290595)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'extension' (line 261)
        extension_290596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'extension')
        unicode_290597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 24), 'unicode', u'eps')
        # Applying the binary operator '==' (line 261)
        result_eq_290598 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), '==', extension_290596, unicode_290597)
        
        
        
        # Call to exists(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'orig_expected_fname' (line 261)
        orig_expected_fname_290602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 53), 'orig_expected_fname', False)
        # Processing the call keyword arguments (line 261)
        kwargs_290603 = {}
        # Getting the type of 'os' (line 261)
        os_290599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 38), 'os', False)
        # Obtaining the member 'path' of a type (line 261)
        path_290600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 38), os_290599, 'path')
        # Obtaining the member 'exists' of a type (line 261)
        exists_290601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 38), path_290600, 'exists')
        # Calling exists(args, kwargs) (line 261)
        exists_call_result_290604 = invoke(stypy.reporting.localization.Localization(__file__, 261, 38), exists_290601, *[orig_expected_fname_290602], **kwargs_290603)
        
        # Applying the 'not' unary operator (line 261)
        result_not__290605 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 34), 'not', exists_call_result_290604)
        
        # Applying the binary operator 'and' (line 261)
        result_and_keyword_290606 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), 'and', result_eq_290598, result_not__290605)
        
        # Testing the type of an if condition (line 261)
        if_condition_290607 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), result_and_keyword_290606)
        # Assigning a type to the variable 'if_condition_290607' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_290607', if_condition_290607)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 262):
        
        # Assigning a BinOp to a Name (line 262):
        # Getting the type of 'baseline_path' (line 262)
        baseline_path_290608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 34), 'baseline_path')
        unicode_290609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 50), 'unicode', u'.pdf')
        # Applying the binary operator '+' (line 262)
        result_add_290610 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 34), '+', baseline_path_290608, unicode_290609)
        
        # Assigning a type to the variable 'orig_expected_fname' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'orig_expected_fname', result_add_290610)
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 263):
        
        # Assigning a Call to a Name (line 263):
        
        # Call to make_test_filename(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Call to join(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'self' (line 264)
        self_290615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'self', False)
        # Obtaining the member 'result_dir' of a type (line 264)
        result_dir_290616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), self_290615, 'result_dir')
        
        # Call to basename(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'orig_expected_fname' (line 264)
        orig_expected_fname_290620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 46), 'orig_expected_fname', False)
        # Processing the call keyword arguments (line 264)
        kwargs_290621 = {}
        # Getting the type of 'os' (line 264)
        os_290617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 264)
        path_290618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 29), os_290617, 'path')
        # Obtaining the member 'basename' of a type (line 264)
        basename_290619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 29), path_290618, 'basename')
        # Calling basename(args, kwargs) (line 264)
        basename_call_result_290622 = invoke(stypy.reporting.localization.Localization(__file__, 264, 29), basename_290619, *[orig_expected_fname_290620], **kwargs_290621)
        
        # Processing the call keyword arguments (line 263)
        kwargs_290623 = {}
        # Getting the type of 'os' (line 263)
        os_290612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 263)
        path_290613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 44), os_290612, 'path')
        # Obtaining the member 'join' of a type (line 263)
        join_290614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 44), path_290613, 'join')
        # Calling join(args, kwargs) (line 263)
        join_call_result_290624 = invoke(stypy.reporting.localization.Localization(__file__, 263, 44), join_290614, *[result_dir_290616, basename_call_result_290622], **kwargs_290623)
        
        unicode_290625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 69), 'unicode', u'expected')
        # Processing the call keyword arguments (line 263)
        kwargs_290626 = {}
        # Getting the type of 'make_test_filename' (line 263)
        make_test_filename_290611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), 'make_test_filename', False)
        # Calling make_test_filename(args, kwargs) (line 263)
        make_test_filename_call_result_290627 = invoke(stypy.reporting.localization.Localization(__file__, 263, 25), make_test_filename_290611, *[join_call_result_290624, unicode_290625], **kwargs_290626)
        
        # Assigning a type to the variable 'expected_fname' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'expected_fname', make_test_filename_call_result_290627)
        
        
        # Call to exists(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'orig_expected_fname' (line 265)
        orig_expected_fname_290631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 26), 'orig_expected_fname', False)
        # Processing the call keyword arguments (line 265)
        kwargs_290632 = {}
        # Getting the type of 'os' (line 265)
        os_290628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 265)
        path_290629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), os_290628, 'path')
        # Obtaining the member 'exists' of a type (line 265)
        exists_290630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), path_290629, 'exists')
        # Calling exists(args, kwargs) (line 265)
        exists_call_result_290633 = invoke(stypy.reporting.localization.Localization(__file__, 265, 11), exists_290630, *[orig_expected_fname_290631], **kwargs_290632)
        
        # Testing the type of an if condition (line 265)
        if_condition_290634 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), exists_call_result_290633)
        # Assigning a type to the variable 'if_condition_290634' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_290634', if_condition_290634)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copyfile(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'orig_expected_fname' (line 266)
        orig_expected_fname_290637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'orig_expected_fname', False)
        # Getting the type of 'expected_fname' (line 266)
        expected_fname_290638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 49), 'expected_fname', False)
        # Processing the call keyword arguments (line 266)
        kwargs_290639 = {}
        # Getting the type of 'shutil' (line 266)
        shutil_290635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'shutil', False)
        # Obtaining the member 'copyfile' of a type (line 266)
        copyfile_290636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), shutil_290635, 'copyfile')
        # Calling copyfile(args, kwargs) (line 266)
        copyfile_call_result_290640 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), copyfile_290636, *[orig_expected_fname_290637, expected_fname_290638], **kwargs_290639)
        
        # SSA branch for the else part of an if statement (line 265)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 268):
        
        # Assigning a Call to a Name (line 268):
        
        # Call to format(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'expected_fname' (line 269)
        expected_fname_290643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 56), 'expected_fname', False)
        # Getting the type of 'orig_expected_fname' (line 270)
        orig_expected_fname_290644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 56), 'orig_expected_fname', False)
        # Processing the call keyword arguments (line 268)
        kwargs_290645 = {}
        unicode_290641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 22), 'unicode', u'Do not have baseline image {0} because this file does not exist: {1}')
        # Obtaining the member 'format' of a type (line 268)
        format_290642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 22), unicode_290641, 'format')
        # Calling format(args, kwargs) (line 268)
        format_call_result_290646 = invoke(stypy.reporting.localization.Localization(__file__, 268, 22), format_290642, *[expected_fname_290643, orig_expected_fname_290644], **kwargs_290645)
        
        # Assigning a type to the variable 'reason' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'reason', format_call_result_290646)
        
        # Call to ImageComparisonFailure(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'reason' (line 271)
        reason_290648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 41), 'reason', False)
        # Processing the call keyword arguments (line 271)
        kwargs_290649 = {}
        # Getting the type of 'ImageComparisonFailure' (line 271)
        ImageComparisonFailure_290647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'ImageComparisonFailure', False)
        # Calling ImageComparisonFailure(args, kwargs) (line 271)
        ImageComparisonFailure_call_result_290650 = invoke(stypy.reporting.localization.Localization(__file__, 271, 18), ImageComparisonFailure_290647, *[reason_290648], **kwargs_290649)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 271, 12), ImageComparisonFailure_call_result_290650, 'raise parameter', BaseException)
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'expected_fname' (line 272)
        expected_fname_290651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'expected_fname')
        # Assigning a type to the variable 'stypy_return_type' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'stypy_return_type', expected_fname_290651)
        
        # ################# End of 'copy_baseline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy_baseline' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_290652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy_baseline'
        return stypy_return_type_290652


    @norecursion
    def compare(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'compare'
        module_type_store = module_type_store.open_function_context('compare', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_localization', localization)
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_function_name', '_ImageComparisonBase.compare')
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_param_names_list', ['idx', 'baseline', 'extension'])
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ImageComparisonBase.compare.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ImageComparisonBase.compare', ['idx', 'baseline', 'extension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'compare', localization, ['idx', 'baseline', 'extension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'compare(...)' code ##################

        
        # Assigning a Name to a Name (line 275):
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'True' (line 275)
        True_290653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 28), 'True')
        # Assigning a type to the variable '__tracebackhide__' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), '__tracebackhide__', True_290653)
        
        # Assigning a Subscript to a Name (line 276):
        
        # Assigning a Subscript to a Name (line 276):
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 276)
        idx_290654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'idx')
        
        # Call to get_fignums(...): (line 276)
        # Processing the call keyword arguments (line 276)
        kwargs_290657 = {}
        # Getting the type of 'plt' (line 276)
        plt_290655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 17), 'plt', False)
        # Obtaining the member 'get_fignums' of a type (line 276)
        get_fignums_290656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 17), plt_290655, 'get_fignums')
        # Calling get_fignums(args, kwargs) (line 276)
        get_fignums_call_result_290658 = invoke(stypy.reporting.localization.Localization(__file__, 276, 17), get_fignums_290656, *[], **kwargs_290657)
        
        # Obtaining the member '__getitem__' of a type (line 276)
        getitem___290659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 17), get_fignums_call_result_290658, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 276)
        subscript_call_result_290660 = invoke(stypy.reporting.localization.Localization(__file__, 276, 17), getitem___290659, idx_290654)
        
        # Assigning a type to the variable 'fignum' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'fignum', subscript_call_result_290660)
        
        # Assigning a Call to a Name (line 277):
        
        # Assigning a Call to a Name (line 277):
        
        # Call to figure(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'fignum' (line 277)
        fignum_290663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 'fignum', False)
        # Processing the call keyword arguments (line 277)
        kwargs_290664 = {}
        # Getting the type of 'plt' (line 277)
        plt_290661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 14), 'plt', False)
        # Obtaining the member 'figure' of a type (line 277)
        figure_290662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 14), plt_290661, 'figure')
        # Calling figure(args, kwargs) (line 277)
        figure_call_result_290665 = invoke(stypy.reporting.localization.Localization(__file__, 277, 14), figure_290662, *[fignum_290663], **kwargs_290664)
        
        # Assigning a type to the variable 'fig' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'fig', figure_call_result_290665)
        
        # Getting the type of 'self' (line 279)
        self_290666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'self')
        # Obtaining the member 'remove_text' of a type (line 279)
        remove_text_290667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 11), self_290666, 'remove_text')
        # Testing the type of an if condition (line 279)
        if_condition_290668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), remove_text_290667)
        # Assigning a type to the variable 'if_condition_290668' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_290668', if_condition_290668)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove_ticks_and_titles(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'fig' (line 280)
        fig_290670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 36), 'fig', False)
        # Processing the call keyword arguments (line 280)
        kwargs_290671 = {}
        # Getting the type of 'remove_ticks_and_titles' (line 280)
        remove_ticks_and_titles_290669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'remove_ticks_and_titles', False)
        # Calling remove_ticks_and_titles(args, kwargs) (line 280)
        remove_ticks_and_titles_call_result_290672 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), remove_ticks_and_titles_290669, *[fig_290670], **kwargs_290671)
        
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 282):
        
        # Assigning a BinOp to a Name (line 282):
        
        # Call to join(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'self' (line 282)
        self_290676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 36), 'self', False)
        # Obtaining the member 'result_dir' of a type (line 282)
        result_dir_290677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 36), self_290676, 'result_dir')
        # Getting the type of 'baseline' (line 282)
        baseline_290678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 53), 'baseline', False)
        # Processing the call keyword arguments (line 282)
        kwargs_290679 = {}
        # Getting the type of 'os' (line 282)
        os_290673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 282)
        path_290674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 23), os_290673, 'path')
        # Obtaining the member 'join' of a type (line 282)
        join_290675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 23), path_290674, 'join')
        # Calling join(args, kwargs) (line 282)
        join_call_result_290680 = invoke(stypy.reporting.localization.Localization(__file__, 282, 23), join_290675, *[result_dir_290677, baseline_290678], **kwargs_290679)
        
        unicode_290681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 65), 'unicode', u'.')
        # Applying the binary operator '+' (line 282)
        result_add_290682 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 23), '+', join_call_result_290680, unicode_290681)
        
        # Getting the type of 'extension' (line 282)
        extension_290683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 71), 'extension')
        # Applying the binary operator '+' (line 282)
        result_add_290684 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 69), '+', result_add_290682, extension_290683)
        
        # Assigning a type to the variable 'actual_fname' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'actual_fname', result_add_290684)
        
        # Assigning a Call to a Name (line 283):
        
        # Assigning a Call to a Name (line 283):
        
        # Call to copy(...): (line 283)
        # Processing the call keyword arguments (line 283)
        kwargs_290688 = {}
        # Getting the type of 'self' (line 283)
        self_290685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 17), 'self', False)
        # Obtaining the member 'savefig_kwargs' of a type (line 283)
        savefig_kwargs_290686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 17), self_290685, 'savefig_kwargs')
        # Obtaining the member 'copy' of a type (line 283)
        copy_290687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 17), savefig_kwargs_290686, 'copy')
        # Calling copy(args, kwargs) (line 283)
        copy_call_result_290689 = invoke(stypy.reporting.localization.Localization(__file__, 283, 17), copy_290687, *[], **kwargs_290688)
        
        # Assigning a type to the variable 'kwargs' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'kwargs', copy_call_result_290689)
        
        
        # Getting the type of 'extension' (line 284)
        extension_290690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'extension')
        unicode_290691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 24), 'unicode', u'pdf')
        # Applying the binary operator '==' (line 284)
        result_eq_290692 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), '==', extension_290690, unicode_290691)
        
        # Testing the type of an if condition (line 284)
        if_condition_290693 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_eq_290692)
        # Assigning a type to the variable 'if_condition_290693' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_290693', if_condition_290693)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setdefault(...): (line 285)
        # Processing the call arguments (line 285)
        unicode_290696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 30), 'unicode', u'metadata')
        
        # Obtaining an instance of the builtin type 'dict' (line 286)
        dict_290697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 30), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 286)
        # Adding element type (key, value) (line 286)
        unicode_290698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 31), 'unicode', u'Creator')
        # Getting the type of 'None' (line 286)
        None_290699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 42), 'None', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 30), dict_290697, (unicode_290698, None_290699))
        # Adding element type (key, value) (line 286)
        unicode_290700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 48), 'unicode', u'Producer')
        # Getting the type of 'None' (line 286)
        None_290701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 60), 'None', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 30), dict_290697, (unicode_290700, None_290701))
        # Adding element type (key, value) (line 286)
        unicode_290702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 31), 'unicode', u'CreationDate')
        # Getting the type of 'None' (line 287)
        None_290703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 47), 'None', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 30), dict_290697, (unicode_290702, None_290703))
        
        # Processing the call keyword arguments (line 285)
        kwargs_290704 = {}
        # Getting the type of 'kwargs' (line 285)
        kwargs_290694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'kwargs', False)
        # Obtaining the member 'setdefault' of a type (line 285)
        setdefault_290695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), kwargs_290694, 'setdefault')
        # Calling setdefault(args, kwargs) (line 285)
        setdefault_call_result_290705 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), setdefault_290695, *[unicode_290696, dict_290697], **kwargs_290704)
        
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to savefig(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'actual_fname' (line 288)
        actual_fname_290708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 20), 'actual_fname', False)
        # Processing the call keyword arguments (line 288)
        # Getting the type of 'kwargs' (line 288)
        kwargs_290709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 36), 'kwargs', False)
        kwargs_290710 = {'kwargs_290709': kwargs_290709}
        # Getting the type of 'fig' (line 288)
        fig_290706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'fig', False)
        # Obtaining the member 'savefig' of a type (line 288)
        savefig_290707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), fig_290706, 'savefig')
        # Calling savefig(args, kwargs) (line 288)
        savefig_call_result_290711 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), savefig_290707, *[actual_fname_290708], **kwargs_290710)
        
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to copy_baseline(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'baseline' (line 290)
        baseline_290714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 44), 'baseline', False)
        # Getting the type of 'extension' (line 290)
        extension_290715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 54), 'extension', False)
        # Processing the call keyword arguments (line 290)
        kwargs_290716 = {}
        # Getting the type of 'self' (line 290)
        self_290712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 25), 'self', False)
        # Obtaining the member 'copy_baseline' of a type (line 290)
        copy_baseline_290713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 25), self_290712, 'copy_baseline')
        # Calling copy_baseline(args, kwargs) (line 290)
        copy_baseline_call_result_290717 = invoke(stypy.reporting.localization.Localization(__file__, 290, 25), copy_baseline_290713, *[baseline_290714, extension_290715], **kwargs_290716)
        
        # Assigning a type to the variable 'expected_fname' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'expected_fname', copy_baseline_call_result_290717)
        
        # Call to _raise_on_image_difference(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'expected_fname' (line 291)
        expected_fname_290719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 35), 'expected_fname', False)
        # Getting the type of 'actual_fname' (line 291)
        actual_fname_290720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 51), 'actual_fname', False)
        # Getting the type of 'self' (line 291)
        self_290721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 65), 'self', False)
        # Obtaining the member 'tol' of a type (line 291)
        tol_290722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 65), self_290721, 'tol')
        # Processing the call keyword arguments (line 291)
        kwargs_290723 = {}
        # Getting the type of '_raise_on_image_difference' (line 291)
        _raise_on_image_difference_290718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), '_raise_on_image_difference', False)
        # Calling _raise_on_image_difference(args, kwargs) (line 291)
        _raise_on_image_difference_call_result_290724 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), _raise_on_image_difference_290718, *[expected_fname_290719, actual_fname_290720, tol_290722], **kwargs_290723)
        
        
        # ################# End of 'compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compare' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_290725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290725)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compare'
        return stypy_return_type_290725


# Assigning a type to the variable '_ImageComparisonBase' (line 240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), '_ImageComparisonBase', _ImageComparisonBase)
# Declaration of the 'ImageComparisonTest' class
# Getting the type of 'CleanupTest' (line 294)
CleanupTest_290726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'CleanupTest')
# Getting the type of '_ImageComparisonBase' (line 294)
_ImageComparisonBase_290727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 39), '_ImageComparisonBase')

class ImageComparisonTest(CleanupTest_290726, _ImageComparisonBase_290727, ):
    unicode_290728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, (-1)), 'unicode', u'\n    Nose-based image comparison class\n\n    This class generates tests for a nose-based testing framework. Ideally,\n    this class would not be public, and the only publically visible API would\n    be the :func:`image_comparison` decorator. Unfortunately, there are\n    existing downstream users of this class (e.g., pytest-mpl) so it cannot yet\n    be removed.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ImageComparisonTest.__init__', ['baseline_images', 'extensions', 'tol', 'freetype_version', 'remove_text', 'savefig_kwargs', 'style'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['baseline_images', 'extensions', 'tol', 'freetype_version', 'remove_text', 'savefig_kwargs', 'style'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'self' (line 306)
        self_290731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 38), 'self', False)
        # Getting the type of 'tol' (line 306)
        tol_290732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 44), 'tol', False)
        # Getting the type of 'remove_text' (line 306)
        remove_text_290733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 49), 'remove_text', False)
        # Getting the type of 'savefig_kwargs' (line 306)
        savefig_kwargs_290734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 62), 'savefig_kwargs', False)
        # Processing the call keyword arguments (line 306)
        kwargs_290735 = {}
        # Getting the type of '_ImageComparisonBase' (line 306)
        _ImageComparisonBase_290729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), '_ImageComparisonBase', False)
        # Obtaining the member '__init__' of a type (line 306)
        init___290730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), _ImageComparisonBase_290729, '__init__')
        # Calling __init__(args, kwargs) (line 306)
        init___call_result_290736 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), init___290730, *[self_290731, tol_290732, remove_text_290733, savefig_kwargs_290734], **kwargs_290735)
        
        
        # Assigning a Name to a Attribute (line 307):
        
        # Assigning a Name to a Attribute (line 307):
        # Getting the type of 'baseline_images' (line 307)
        baseline_images_290737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 31), 'baseline_images')
        # Getting the type of 'self' (line 307)
        self_290738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self')
        # Setting the type of the member 'baseline_images' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_290738, 'baseline_images', baseline_images_290737)
        
        # Assigning a Name to a Attribute (line 308):
        
        # Assigning a Name to a Attribute (line 308):
        # Getting the type of 'extensions' (line 308)
        extensions_290739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 26), 'extensions')
        # Getting the type of 'self' (line 308)
        self_290740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self')
        # Setting the type of the member 'extensions' of a type (line 308)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_290740, 'extensions', extensions_290739)
        
        # Assigning a Name to a Attribute (line 309):
        
        # Assigning a Name to a Attribute (line 309):
        # Getting the type of 'freetype_version' (line 309)
        freetype_version_290741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 32), 'freetype_version')
        # Getting the type of 'self' (line 309)
        self_290742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self')
        # Setting the type of the member 'freetype_version' of a type (line 309)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_290742, 'freetype_version', freetype_version_290741)
        
        # Assigning a Name to a Attribute (line 310):
        
        # Assigning a Name to a Attribute (line 310):
        # Getting the type of 'style' (line 310)
        style_290743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 21), 'style')
        # Getting the type of 'self' (line 310)
        self_290744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self')
        # Setting the type of the member 'style' of a type (line 310)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_290744, 'style', style_290743)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def setup(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup'
        module_type_store = module_type_store.open_function_context('setup', 312, 4, False)
        # Assigning a type to the variable 'self' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_localization', localization)
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_type_store', module_type_store)
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_function_name', 'ImageComparisonTest.setup')
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_param_names_list', [])
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_varargs_param_name', None)
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_call_defaults', defaults)
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_call_varargs', varargs)
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ImageComparisonTest.setup.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ImageComparisonTest.setup', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup(...)' code ##################

        
        # Assigning a Attribute to a Name (line 313):
        
        # Assigning a Attribute to a Name (line 313):
        # Getting the type of 'self' (line 313)
        self_290745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'self')
        # Obtaining the member 'func' of a type (line 313)
        func_290746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 15), self_290745, 'func')
        # Assigning a type to the variable 'func' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'func', func_290746)
        
        # Call to close(...): (line 314)
        # Processing the call arguments (line 314)
        unicode_290749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 18), 'unicode', u'all')
        # Processing the call keyword arguments (line 314)
        kwargs_290750 = {}
        # Getting the type of 'plt' (line 314)
        plt_290747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'plt', False)
        # Obtaining the member 'close' of a type (line 314)
        close_290748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), plt_290747, 'close')
        # Calling close(args, kwargs) (line 314)
        close_call_result_290751 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), close_290748, *[unicode_290749], **kwargs_290750)
        
        
        # Call to setup_class(...): (line 315)
        # Processing the call keyword arguments (line 315)
        kwargs_290754 = {}
        # Getting the type of 'self' (line 315)
        self_290752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self', False)
        # Obtaining the member 'setup_class' of a type (line 315)
        setup_class_290753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_290752, 'setup_class')
        # Calling setup_class(args, kwargs) (line 315)
        setup_class_call_result_290755 = invoke(stypy.reporting.localization.Localization(__file__, 315, 8), setup_class_290753, *[], **kwargs_290754)
        
        
        
        # SSA begins for try-except statement (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to use(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'self' (line 317)
        self_290759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 33), 'self', False)
        # Obtaining the member 'style' of a type (line 317)
        style_290760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 33), self_290759, 'style')
        # Processing the call keyword arguments (line 317)
        kwargs_290761 = {}
        # Getting the type of 'matplotlib' (line 317)
        matplotlib_290756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'matplotlib', False)
        # Obtaining the member 'style' of a type (line 317)
        style_290757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), matplotlib_290756, 'style')
        # Obtaining the member 'use' of a type (line 317)
        use_290758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), style_290757, 'use')
        # Calling use(args, kwargs) (line 317)
        use_call_result_290762 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), use_290758, *[style_290760], **kwargs_290761)
        
        
        # Call to set_font_settings_for_testing(...): (line 318)
        # Processing the call keyword arguments (line 318)
        kwargs_290766 = {}
        # Getting the type of 'matplotlib' (line 318)
        matplotlib_290763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'matplotlib', False)
        # Obtaining the member 'testing' of a type (line 318)
        testing_290764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 12), matplotlib_290763, 'testing')
        # Obtaining the member 'set_font_settings_for_testing' of a type (line 318)
        set_font_settings_for_testing_290765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 12), testing_290764, 'set_font_settings_for_testing')
        # Calling set_font_settings_for_testing(args, kwargs) (line 318)
        set_font_settings_for_testing_call_result_290767 = invoke(stypy.reporting.localization.Localization(__file__, 318, 12), set_font_settings_for_testing_290765, *[], **kwargs_290766)
        
        
        # Call to func(...): (line 319)
        # Processing the call keyword arguments (line 319)
        kwargs_290769 = {}
        # Getting the type of 'func' (line 319)
        func_290768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'func', False)
        # Calling func(args, kwargs) (line 319)
        func_call_result_290770 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), func_290768, *[], **kwargs_290769)
        
        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Call to get_fignums(...): (line 320)
        # Processing the call keyword arguments (line 320)
        kwargs_290774 = {}
        # Getting the type of 'plt' (line 320)
        plt_290772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'plt', False)
        # Obtaining the member 'get_fignums' of a type (line 320)
        get_fignums_290773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 23), plt_290772, 'get_fignums')
        # Calling get_fignums(args, kwargs) (line 320)
        get_fignums_call_result_290775 = invoke(stypy.reporting.localization.Localization(__file__, 320, 23), get_fignums_290773, *[], **kwargs_290774)
        
        # Processing the call keyword arguments (line 320)
        kwargs_290776 = {}
        # Getting the type of 'len' (line 320)
        len_290771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 19), 'len', False)
        # Calling len(args, kwargs) (line 320)
        len_call_result_290777 = invoke(stypy.reporting.localization.Localization(__file__, 320, 19), len_290771, *[get_fignums_call_result_290775], **kwargs_290776)
        
        
        # Call to len(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'self' (line 320)
        self_290779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 49), 'self', False)
        # Obtaining the member 'baseline_images' of a type (line 320)
        baseline_images_290780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 49), self_290779, 'baseline_images')
        # Processing the call keyword arguments (line 320)
        kwargs_290781 = {}
        # Getting the type of 'len' (line 320)
        len_290778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 45), 'len', False)
        # Calling len(args, kwargs) (line 320)
        len_call_result_290782 = invoke(stypy.reporting.localization.Localization(__file__, 320, 45), len_290778, *[baseline_images_290780], **kwargs_290781)
        
        # Applying the binary operator '==' (line 320)
        result_eq_290783 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 19), '==', len_call_result_290777, len_call_result_290782)
        
        # SSA branch for the except part of a try statement (line 316)
        # SSA branch for the except '<any exception>' branch of a try statement (line 316)
        module_type_store.open_ssa_branch('except')
        
        # Call to teardown_class(...): (line 325)
        # Processing the call keyword arguments (line 325)
        kwargs_290786 = {}
        # Getting the type of 'self' (line 325)
        self_290784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'self', False)
        # Obtaining the member 'teardown_class' of a type (line 325)
        teardown_class_290785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), self_290784, 'teardown_class')
        # Calling teardown_class(args, kwargs) (line 325)
        teardown_class_call_result_290787 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), teardown_class_290785, *[], **kwargs_290786)
        
        # SSA join for try-except statement (line 316)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setup(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup' in the type store
        # Getting the type of 'stypy_return_type' (line 312)
        stypy_return_type_290788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290788)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup'
        return stypy_return_type_290788


    @norecursion
    def teardown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'teardown'
        module_type_store = module_type_store.open_function_context('teardown', 328, 4, False)
        # Assigning a type to the variable 'self' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_localization', localization)
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_type_store', module_type_store)
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_function_name', 'ImageComparisonTest.teardown')
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_param_names_list', [])
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_varargs_param_name', None)
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_call_defaults', defaults)
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_call_varargs', varargs)
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ImageComparisonTest.teardown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ImageComparisonTest.teardown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'teardown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'teardown(...)' code ##################

        
        # Call to teardown_class(...): (line 329)
        # Processing the call keyword arguments (line 329)
        kwargs_290791 = {}
        # Getting the type of 'self' (line 329)
        self_290789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'self', False)
        # Obtaining the member 'teardown_class' of a type (line 329)
        teardown_class_290790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), self_290789, 'teardown_class')
        # Calling teardown_class(args, kwargs) (line 329)
        teardown_class_call_result_290792 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), teardown_class_290790, *[], **kwargs_290791)
        
        
        # ################# End of 'teardown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'teardown' in the type store
        # Getting the type of 'stypy_return_type' (line 328)
        stypy_return_type_290793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290793)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'teardown'
        return stypy_return_type_290793


    @staticmethod
    @norecursion
    def remove_text(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_text'
        module_type_store = module_type_store.open_function_context('remove_text', 331, 4, False)
        
        # Passed parameters checking function
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_localization', localization)
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_type_of_self', None)
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_function_name', 'remove_text')
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ImageComparisonTest.remove_text.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'remove_text', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_text', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_text(...)' code ##################

        
        # Call to remove_ticks_and_titles(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'figure' (line 335)
        figure_290795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 32), 'figure', False)
        # Processing the call keyword arguments (line 335)
        kwargs_290796 = {}
        # Getting the type of 'remove_ticks_and_titles' (line 335)
        remove_ticks_and_titles_290794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'remove_ticks_and_titles', False)
        # Calling remove_ticks_and_titles(args, kwargs) (line 335)
        remove_ticks_and_titles_call_result_290797 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), remove_ticks_and_titles_290794, *[figure_290795], **kwargs_290796)
        
        
        # ################# End of 'remove_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_text' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_290798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290798)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_text'
        return stypy_return_type_290798


    @norecursion
    def nose_runner(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'nose_runner'
        module_type_store = module_type_store.open_function_context('nose_runner', 337, 4, False)
        # Assigning a type to the variable 'self' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_localization', localization)
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_type_store', module_type_store)
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_function_name', 'ImageComparisonTest.nose_runner')
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_param_names_list', [])
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_varargs_param_name', None)
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_call_defaults', defaults)
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_call_varargs', varargs)
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ImageComparisonTest.nose_runner.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ImageComparisonTest.nose_runner', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'nose_runner', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'nose_runner(...)' code ##################

        
        # Assigning a Attribute to a Name (line 338):
        
        # Assigning a Attribute to a Name (line 338):
        # Getting the type of 'self' (line 338)
        self_290799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'self')
        # Obtaining the member 'compare' of a type (line 338)
        compare_290800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 15), self_290799, 'compare')
        # Assigning a type to the variable 'func' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'func', compare_290800)
        
        # Assigning a Call to a Name (line 339):
        
        # Assigning a Call to a Name (line 339):
        
        # Call to (...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'func' (line 339)
        func_290806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 67), 'func', False)
        # Processing the call keyword arguments (line 339)
        kwargs_290807 = {}
        
        # Call to _checked_on_freetype_version(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'self' (line 339)
        self_290802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 44), 'self', False)
        # Obtaining the member 'freetype_version' of a type (line 339)
        freetype_version_290803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 44), self_290802, 'freetype_version')
        # Processing the call keyword arguments (line 339)
        kwargs_290804 = {}
        # Getting the type of '_checked_on_freetype_version' (line 339)
        _checked_on_freetype_version_290801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), '_checked_on_freetype_version', False)
        # Calling _checked_on_freetype_version(args, kwargs) (line 339)
        _checked_on_freetype_version_call_result_290805 = invoke(stypy.reporting.localization.Localization(__file__, 339, 15), _checked_on_freetype_version_290801, *[freetype_version_290803], **kwargs_290804)
        
        # Calling (args, kwargs) (line 339)
        _call_result_290808 = invoke(stypy.reporting.localization.Localization(__file__, 339, 15), _checked_on_freetype_version_call_result_290805, *[func_290806], **kwargs_290807)
        
        # Assigning a type to the variable 'func' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'func', _call_result_290808)
        
        # Assigning a DictComp to a Name (line 340):
        
        # Assigning a DictComp to a Name (line 340):
        # Calculating dict comprehension
        module_type_store = module_type_store.open_function_context('dict comprehension expression', 340, 17, True)
        # Calculating comprehension expression
        # Getting the type of 'self' (line 341)
        self_290817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 34), 'self')
        # Obtaining the member 'extensions' of a type (line 341)
        extensions_290818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 34), self_290817, 'extensions')
        comprehension_290819 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 17), extensions_290818)
        # Assigning a type to the variable 'extension' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 17), 'extension', comprehension_290819)
        # Getting the type of 'extension' (line 340)
        extension_290809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 17), 'extension')
        
        # Call to (...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'func' (line 340)
        func_290814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 72), 'func', False)
        # Processing the call keyword arguments (line 340)
        kwargs_290815 = {}
        
        # Call to _xfail_if_format_is_uncomparable(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'extension' (line 340)
        extension_290811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 61), 'extension', False)
        # Processing the call keyword arguments (line 340)
        kwargs_290812 = {}
        # Getting the type of '_xfail_if_format_is_uncomparable' (line 340)
        _xfail_if_format_is_uncomparable_290810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 28), '_xfail_if_format_is_uncomparable', False)
        # Calling _xfail_if_format_is_uncomparable(args, kwargs) (line 340)
        _xfail_if_format_is_uncomparable_call_result_290813 = invoke(stypy.reporting.localization.Localization(__file__, 340, 28), _xfail_if_format_is_uncomparable_290810, *[extension_290811], **kwargs_290812)
        
        # Calling (args, kwargs) (line 340)
        _call_result_290816 = invoke(stypy.reporting.localization.Localization(__file__, 340, 28), _xfail_if_format_is_uncomparable_call_result_290813, *[func_290814], **kwargs_290815)
        
        dict_290820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 17), 'dict')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 17), dict_290820, (extension_290809, _call_result_290816))
        # Assigning a type to the variable 'funcs' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'funcs', dict_290820)
        
        
        # Call to enumerate(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'self' (line 342)
        self_290822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 39), 'self', False)
        # Obtaining the member 'baseline_images' of a type (line 342)
        baseline_images_290823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 39), self_290822, 'baseline_images')
        # Processing the call keyword arguments (line 342)
        kwargs_290824 = {}
        # Getting the type of 'enumerate' (line 342)
        enumerate_290821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 342)
        enumerate_call_result_290825 = invoke(stypy.reporting.localization.Localization(__file__, 342, 29), enumerate_290821, *[baseline_images_290823], **kwargs_290824)
        
        # Testing the type of a for loop iterable (line 342)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 342, 8), enumerate_call_result_290825)
        # Getting the type of the for loop variable (line 342)
        for_loop_var_290826 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 342, 8), enumerate_call_result_290825)
        # Assigning a type to the variable 'idx' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'idx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 8), for_loop_var_290826))
        # Assigning a type to the variable 'baseline' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'baseline', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 8), for_loop_var_290826))
        # SSA begins for a for statement (line 342)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 343)
        self_290827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 29), 'self')
        # Obtaining the member 'extensions' of a type (line 343)
        extensions_290828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 29), self_290827, 'extensions')
        # Testing the type of a for loop iterable (line 343)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 343, 12), extensions_290828)
        # Getting the type of the for loop variable (line 343)
        for_loop_var_290829 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 343, 12), extensions_290828)
        # Assigning a type to the variable 'extension' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'extension', for_loop_var_290829)
        # SSA begins for a for statement (line 343)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 344)
        tuple_290830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 344)
        # Adding element type (line 344)
        
        # Obtaining the type of the subscript
        # Getting the type of 'extension' (line 344)
        extension_290831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 28), 'extension')
        # Getting the type of 'funcs' (line 344)
        funcs_290832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 22), 'funcs')
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___290833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 22), funcs_290832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_290834 = invoke(stypy.reporting.localization.Localization(__file__, 344, 22), getitem___290833, extension_290831)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 22), tuple_290830, subscript_call_result_290834)
        # Adding element type (line 344)
        # Getting the type of 'idx' (line 344)
        idx_290835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 40), 'idx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 22), tuple_290830, idx_290835)
        # Adding element type (line 344)
        # Getting the type of 'baseline' (line 344)
        baseline_290836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 45), 'baseline')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 22), tuple_290830, baseline_290836)
        # Adding element type (line 344)
        # Getting the type of 'extension' (line 344)
        extension_290837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 55), 'extension')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 22), tuple_290830, extension_290837)
        
        GeneratorType_290838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 16), GeneratorType_290838, tuple_290830)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'stypy_return_type', GeneratorType_290838)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'nose_runner(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'nose_runner' in the type store
        # Getting the type of 'stypy_return_type' (line 337)
        stypy_return_type_290839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290839)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'nose_runner'
        return stypy_return_type_290839


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 346, 4, False)
        # Assigning a type to the variable 'self' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_localization', localization)
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_function_name', 'ImageComparisonTest.__call__')
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_param_names_list', ['func'])
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ImageComparisonTest.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ImageComparisonTest.__call__', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to delayed_init(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'func' (line 347)
        func_290842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 26), 'func', False)
        # Processing the call keyword arguments (line 347)
        kwargs_290843 = {}
        # Getting the type of 'self' (line 347)
        self_290840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self', False)
        # Obtaining the member 'delayed_init' of a type (line 347)
        delayed_init_290841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_290840, 'delayed_init')
        # Calling delayed_init(args, kwargs) (line 347)
        delayed_init_call_result_290844 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), delayed_init_290841, *[func_290842], **kwargs_290843)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 348, 8))
        
        # 'import nose.tools' statement (line 348)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
        import_290845 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 348, 8), 'nose.tools')

        if (type(import_290845) is not StypyTypeError):

            if (import_290845 != 'pyd_module'):
                __import__(import_290845)
                sys_modules_290846 = sys.modules[import_290845]
                import_module(stypy.reporting.localization.Localization(__file__, 348, 8), 'nose.tools', sys_modules_290846.module_type_store, module_type_store)
            else:
                import nose.tools

                import_module(stypy.reporting.localization.Localization(__file__, 348, 8), 'nose.tools', nose.tools, module_type_store)

        else:
            # Assigning a type to the variable 'nose.tools' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'nose.tools', import_290845)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
        

        @norecursion
        def runner_wrapper(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'runner_wrapper'
            module_type_store = module_type_store.open_function_context('runner_wrapper', 350, 8, False)
            
            # Passed parameters checking function
            runner_wrapper.stypy_localization = localization
            runner_wrapper.stypy_type_of_self = None
            runner_wrapper.stypy_type_store = module_type_store
            runner_wrapper.stypy_function_name = 'runner_wrapper'
            runner_wrapper.stypy_param_names_list = []
            runner_wrapper.stypy_varargs_param_name = None
            runner_wrapper.stypy_kwargs_param_name = None
            runner_wrapper.stypy_call_defaults = defaults
            runner_wrapper.stypy_call_varargs = varargs
            runner_wrapper.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'runner_wrapper', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'runner_wrapper', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'runner_wrapper(...)' code ##################

            
            
            # Call to nose_runner(...): (line 352)
            # Processing the call keyword arguments (line 352)
            kwargs_290849 = {}
            # Getting the type of 'self' (line 352)
            self_290847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 24), 'self', False)
            # Obtaining the member 'nose_runner' of a type (line 352)
            nose_runner_290848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 24), self_290847, 'nose_runner')
            # Calling nose_runner(args, kwargs) (line 352)
            nose_runner_call_result_290850 = invoke(stypy.reporting.localization.Localization(__file__, 352, 24), nose_runner_290848, *[], **kwargs_290849)
            
            # Testing the type of a for loop iterable (line 352)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 352, 12), nose_runner_call_result_290850)
            # Getting the type of the for loop variable (line 352)
            for_loop_var_290851 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 352, 12), nose_runner_call_result_290850)
            # Assigning a type to the variable 'case' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'case', for_loop_var_290851)
            # SSA begins for a for statement (line 352)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'case' (line 353)
            case_290852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 22), 'case')
            GeneratorType_290853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 16), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 16), GeneratorType_290853, case_290852)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'stypy_return_type', GeneratorType_290853)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'runner_wrapper(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'runner_wrapper' in the type store
            # Getting the type of 'stypy_return_type' (line 350)
            stypy_return_type_290854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_290854)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'runner_wrapper'
            return stypy_return_type_290854

        # Assigning a type to the variable 'runner_wrapper' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'runner_wrapper', runner_wrapper)
        
        # Call to _copy_metadata(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'func' (line 355)
        func_290856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 30), 'func', False)
        # Getting the type of 'runner_wrapper' (line 355)
        runner_wrapper_290857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 36), 'runner_wrapper', False)
        # Processing the call keyword arguments (line 355)
        kwargs_290858 = {}
        # Getting the type of '_copy_metadata' (line 355)
        _copy_metadata_290855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 15), '_copy_metadata', False)
        # Calling _copy_metadata(args, kwargs) (line 355)
        _copy_metadata_call_result_290859 = invoke(stypy.reporting.localization.Localization(__file__, 355, 15), _copy_metadata_290855, *[func_290856, runner_wrapper_290857], **kwargs_290858)
        
        # Assigning a type to the variable 'stypy_return_type' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'stypy_return_type', _copy_metadata_call_result_290859)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 346)
        stypy_return_type_290860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290860)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_290860


# Assigning a type to the variable 'ImageComparisonTest' (line 294)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 0), 'ImageComparisonTest', ImageComparisonTest)

@norecursion
def _pytest_image_comparison(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_pytest_image_comparison'
    module_type_store = module_type_store.open_function_context('_pytest_image_comparison', 358, 0, False)
    
    # Passed parameters checking function
    _pytest_image_comparison.stypy_localization = localization
    _pytest_image_comparison.stypy_type_of_self = None
    _pytest_image_comparison.stypy_type_store = module_type_store
    _pytest_image_comparison.stypy_function_name = '_pytest_image_comparison'
    _pytest_image_comparison.stypy_param_names_list = ['baseline_images', 'extensions', 'tol', 'freetype_version', 'remove_text', 'savefig_kwargs', 'style']
    _pytest_image_comparison.stypy_varargs_param_name = None
    _pytest_image_comparison.stypy_kwargs_param_name = None
    _pytest_image_comparison.stypy_call_defaults = defaults
    _pytest_image_comparison.stypy_call_varargs = varargs
    _pytest_image_comparison.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_pytest_image_comparison', ['baseline_images', 'extensions', 'tol', 'freetype_version', 'remove_text', 'savefig_kwargs', 'style'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_pytest_image_comparison', localization, ['baseline_images', 'extensions', 'tol', 'freetype_version', 'remove_text', 'savefig_kwargs', 'style'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_pytest_image_comparison(...)' code ##################

    unicode_290861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, (-1)), 'unicode', u'\n    Decorate function with image comparison for pytest.\n\n    This function creates a decorator that wraps a figure-generating function\n    with image comparison code. Pytest can become confused if we change the\n    signature of the function, so we indirectly pass anything we need via the\n    `mpl_image_comparison_parameters` fixture and extra markers.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 369, 4))
    
    # 'import pytest' statement (line 369)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_290862 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 369, 4), 'pytest')

    if (type(import_290862) is not StypyTypeError):

        if (import_290862 != 'pyd_module'):
            __import__(import_290862)
            sys_modules_290863 = sys.modules[import_290862]
            import_module(stypy.reporting.localization.Localization(__file__, 369, 4), 'pytest', sys_modules_290863.module_type_store, module_type_store)
        else:
            import pytest

            import_module(stypy.reporting.localization.Localization(__file__, 369, 4), 'pytest', pytest, module_type_store)

    else:
        # Assigning a type to the variable 'pytest' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'pytest', import_290862)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    # Assigning a Call to a Name (line 371):
    
    # Assigning a Call to a Name (line 371):
    
    # Call to map(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of '_mark_xfail_if_format_is_uncomparable' (line 371)
    _mark_xfail_if_format_is_uncomparable_290865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), '_mark_xfail_if_format_is_uncomparable', False)
    # Getting the type of 'extensions' (line 371)
    extensions_290866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 60), 'extensions', False)
    # Processing the call keyword arguments (line 371)
    kwargs_290867 = {}
    # Getting the type of 'map' (line 371)
    map_290864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'map', False)
    # Calling map(args, kwargs) (line 371)
    map_call_result_290868 = invoke(stypy.reporting.localization.Localization(__file__, 371, 17), map_290864, *[_mark_xfail_if_format_is_uncomparable_290865, extensions_290866], **kwargs_290867)
    
    # Assigning a type to the variable 'extensions' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'extensions', map_call_result_290868)

    @norecursion
    def decorator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'decorator'
        module_type_store = module_type_store.open_function_context('decorator', 373, 4, False)
        
        # Passed parameters checking function
        decorator.stypy_localization = localization
        decorator.stypy_type_of_self = None
        decorator.stypy_type_store = module_type_store
        decorator.stypy_function_name = 'decorator'
        decorator.stypy_param_names_list = ['func']
        decorator.stypy_varargs_param_name = None
        decorator.stypy_kwargs_param_name = None
        decorator.stypy_call_defaults = defaults
        decorator.stypy_call_varargs = varargs
        decorator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'decorator', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'decorator', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'decorator(...)' code ##################


        @norecursion
        def wrapper(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'wrapper'
            module_type_store = module_type_store.open_function_context('wrapper', 375, 8, False)
            
            # Passed parameters checking function
            wrapper.stypy_localization = localization
            wrapper.stypy_type_of_self = None
            wrapper.stypy_type_store = module_type_store
            wrapper.stypy_function_name = 'wrapper'
            wrapper.stypy_param_names_list = []
            wrapper.stypy_varargs_param_name = 'args'
            wrapper.stypy_kwargs_param_name = 'kwargs'
            wrapper.stypy_call_defaults = defaults
            wrapper.stypy_call_varargs = varargs
            wrapper.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'wrapper', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'wrapper', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'wrapper(...)' code ##################

            
            # Assigning a Name to a Name (line 383):
            
            # Assigning a Name to a Name (line 383):
            # Getting the type of 'True' (line 383)
            True_290869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 32), 'True')
            # Assigning a type to the variable '__tracebackhide__' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), '__tracebackhide__', True_290869)
            
            # Assigning a Call to a Name (line 384):
            
            # Assigning a Call to a Name (line 384):
            
            # Call to _ImageComparisonBase(...): (line 384)
            # Processing the call keyword arguments (line 384)
            # Getting the type of 'tol' (line 384)
            tol_290871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 43), 'tol', False)
            keyword_290872 = tol_290871
            # Getting the type of 'remove_text' (line 384)
            remove_text_290873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 60), 'remove_text', False)
            keyword_290874 = remove_text_290873
            # Getting the type of 'savefig_kwargs' (line 385)
            savefig_kwargs_290875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 54), 'savefig_kwargs', False)
            keyword_290876 = savefig_kwargs_290875
            kwargs_290877 = {'remove_text': keyword_290874, 'tol': keyword_290872, 'savefig_kwargs': keyword_290876}
            # Getting the type of '_ImageComparisonBase' (line 384)
            _ImageComparisonBase_290870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 18), '_ImageComparisonBase', False)
            # Calling _ImageComparisonBase(args, kwargs) (line 384)
            _ImageComparisonBase_call_result_290878 = invoke(stypy.reporting.localization.Localization(__file__, 384, 18), _ImageComparisonBase_290870, *[], **kwargs_290877)
            
            # Assigning a type to the variable 'img' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'img', _ImageComparisonBase_call_result_290878)
            
            # Call to delayed_init(...): (line 386)
            # Processing the call arguments (line 386)
            # Getting the type of 'func' (line 386)
            func_290881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 29), 'func', False)
            # Processing the call keyword arguments (line 386)
            kwargs_290882 = {}
            # Getting the type of 'img' (line 386)
            img_290879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'img', False)
            # Obtaining the member 'delayed_init' of a type (line 386)
            delayed_init_290880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), img_290879, 'delayed_init')
            # Calling delayed_init(args, kwargs) (line 386)
            delayed_init_call_result_290883 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), delayed_init_290880, *[func_290881], **kwargs_290882)
            
            
            # Call to set_font_settings_for_testing(...): (line 387)
            # Processing the call keyword arguments (line 387)
            kwargs_290887 = {}
            # Getting the type of 'matplotlib' (line 387)
            matplotlib_290884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'matplotlib', False)
            # Obtaining the member 'testing' of a type (line 387)
            testing_290885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), matplotlib_290884, 'testing')
            # Obtaining the member 'set_font_settings_for_testing' of a type (line 387)
            set_font_settings_for_testing_290886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), testing_290885, 'set_font_settings_for_testing')
            # Calling set_font_settings_for_testing(args, kwargs) (line 387)
            set_font_settings_for_testing_call_result_290888 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), set_font_settings_for_testing_290886, *[], **kwargs_290887)
            
            
            # Call to func(...): (line 388)
            # Getting the type of 'args' (line 388)
            args_290890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 18), 'args', False)
            # Processing the call keyword arguments (line 388)
            # Getting the type of 'kwargs' (line 388)
            kwargs_290891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 26), 'kwargs', False)
            kwargs_290892 = {'kwargs_290891': kwargs_290891}
            # Getting the type of 'func' (line 388)
            func_290889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'func', False)
            # Calling func(args, kwargs) (line 388)
            func_call_result_290893 = invoke(stypy.reporting.localization.Localization(__file__, 388, 12), func_290889, *[args_290890], **kwargs_290892)
            
            
            # Assigning a Attribute to a Tuple (line 395):
            
            # Assigning a Subscript to a Name (line 395):
            
            # Obtaining the type of the subscript
            int_290894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 12), 'int')
            # Getting the type of 'func' (line 395)
            func_290895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 41), 'func')
            # Obtaining the member 'parameters' of a type (line 395)
            parameters_290896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 41), func_290895, 'parameters')
            # Obtaining the member '__getitem__' of a type (line 395)
            getitem___290897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), parameters_290896, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 395)
            subscript_call_result_290898 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), getitem___290897, int_290894)
            
            # Assigning a type to the variable 'tuple_var_assignment_290051' (line 395)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'tuple_var_assignment_290051', subscript_call_result_290898)
            
            # Assigning a Subscript to a Name (line 395):
            
            # Obtaining the type of the subscript
            int_290899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 12), 'int')
            # Getting the type of 'func' (line 395)
            func_290900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 41), 'func')
            # Obtaining the member 'parameters' of a type (line 395)
            parameters_290901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 41), func_290900, 'parameters')
            # Obtaining the member '__getitem__' of a type (line 395)
            getitem___290902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), parameters_290901, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 395)
            subscript_call_result_290903 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), getitem___290902, int_290899)
            
            # Assigning a type to the variable 'tuple_var_assignment_290052' (line 395)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'tuple_var_assignment_290052', subscript_call_result_290903)
            
            # Assigning a Name to a Name (line 395):
            # Getting the type of 'tuple_var_assignment_290051' (line 395)
            tuple_var_assignment_290051_290904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'tuple_var_assignment_290051')
            # Assigning a type to the variable 'baseline_images' (line 395)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'baseline_images', tuple_var_assignment_290051_290904)
            
            # Assigning a Name to a Name (line 395):
            # Getting the type of 'tuple_var_assignment_290052' (line 395)
            tuple_var_assignment_290052_290905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'tuple_var_assignment_290052')
            # Assigning a type to the variable 'extension' (line 395)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 29), 'extension', tuple_var_assignment_290052_290905)
            # Evaluating assert statement condition
            
            
            # Call to len(...): (line 397)
            # Processing the call arguments (line 397)
            
            # Call to get_fignums(...): (line 397)
            # Processing the call keyword arguments (line 397)
            kwargs_290909 = {}
            # Getting the type of 'plt' (line 397)
            plt_290907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 23), 'plt', False)
            # Obtaining the member 'get_fignums' of a type (line 397)
            get_fignums_290908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 23), plt_290907, 'get_fignums')
            # Calling get_fignums(args, kwargs) (line 397)
            get_fignums_call_result_290910 = invoke(stypy.reporting.localization.Localization(__file__, 397, 23), get_fignums_290908, *[], **kwargs_290909)
            
            # Processing the call keyword arguments (line 397)
            kwargs_290911 = {}
            # Getting the type of 'len' (line 397)
            len_290906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 'len', False)
            # Calling len(args, kwargs) (line 397)
            len_call_result_290912 = invoke(stypy.reporting.localization.Localization(__file__, 397, 19), len_290906, *[get_fignums_call_result_290910], **kwargs_290911)
            
            
            # Call to len(...): (line 397)
            # Processing the call arguments (line 397)
            # Getting the type of 'baseline_images' (line 397)
            baseline_images_290914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 49), 'baseline_images', False)
            # Processing the call keyword arguments (line 397)
            kwargs_290915 = {}
            # Getting the type of 'len' (line 397)
            len_290913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 45), 'len', False)
            # Calling len(args, kwargs) (line 397)
            len_call_result_290916 = invoke(stypy.reporting.localization.Localization(__file__, 397, 45), len_290913, *[baseline_images_290914], **kwargs_290915)
            
            # Applying the binary operator '==' (line 397)
            result_eq_290917 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 19), '==', len_call_result_290912, len_call_result_290916)
            
            
            
            # Call to enumerate(...): (line 400)
            # Processing the call arguments (line 400)
            # Getting the type of 'baseline_images' (line 400)
            baseline_images_290919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 43), 'baseline_images', False)
            # Processing the call keyword arguments (line 400)
            kwargs_290920 = {}
            # Getting the type of 'enumerate' (line 400)
            enumerate_290918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 33), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 400)
            enumerate_call_result_290921 = invoke(stypy.reporting.localization.Localization(__file__, 400, 33), enumerate_290918, *[baseline_images_290919], **kwargs_290920)
            
            # Testing the type of a for loop iterable (line 400)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 400, 12), enumerate_call_result_290921)
            # Getting the type of the for loop variable (line 400)
            for_loop_var_290922 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 400, 12), enumerate_call_result_290921)
            # Assigning a type to the variable 'idx' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'idx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 12), for_loop_var_290922))
            # Assigning a type to the variable 'baseline' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'baseline', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 12), for_loop_var_290922))
            # SSA begins for a for statement (line 400)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to compare(...): (line 401)
            # Processing the call arguments (line 401)
            # Getting the type of 'idx' (line 401)
            idx_290925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 28), 'idx', False)
            # Getting the type of 'baseline' (line 401)
            baseline_290926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 33), 'baseline', False)
            # Getting the type of 'extension' (line 401)
            extension_290927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 43), 'extension', False)
            # Processing the call keyword arguments (line 401)
            kwargs_290928 = {}
            # Getting the type of 'img' (line 401)
            img_290923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 16), 'img', False)
            # Obtaining the member 'compare' of a type (line 401)
            compare_290924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 16), img_290923, 'compare')
            # Calling compare(args, kwargs) (line 401)
            compare_call_result_290929 = invoke(stypy.reporting.localization.Localization(__file__, 401, 16), compare_290924, *[idx_290925, baseline_290926, extension_290927], **kwargs_290928)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'wrapper(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'wrapper' in the type store
            # Getting the type of 'stypy_return_type' (line 375)
            stypy_return_type_290930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_290930)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'wrapper'
            return stypy_return_type_290930

        # Assigning a type to the variable 'wrapper' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'wrapper', wrapper)
        
        # Assigning a Name to a Attribute (line 403):
        
        # Assigning a Name to a Attribute (line 403):
        # Getting the type of 'func' (line 403)
        func_290931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 30), 'func')
        # Getting the type of 'wrapper' (line 403)
        wrapper_290932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'wrapper')
        # Setting the type of the member '__wrapped__' of a type (line 403)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), wrapper_290932, '__wrapped__', func_290931)
        
        # Call to _copy_metadata(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'func' (line 404)
        func_290934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 30), 'func', False)
        # Getting the type of 'wrapper' (line 404)
        wrapper_290935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 36), 'wrapper', False)
        # Processing the call keyword arguments (line 404)
        kwargs_290936 = {}
        # Getting the type of '_copy_metadata' (line 404)
        _copy_metadata_290933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), '_copy_metadata', False)
        # Calling _copy_metadata(args, kwargs) (line 404)
        _copy_metadata_call_result_290937 = invoke(stypy.reporting.localization.Localization(__file__, 404, 15), _copy_metadata_290933, *[func_290934, wrapper_290935], **kwargs_290936)
        
        # Assigning a type to the variable 'stypy_return_type' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'stypy_return_type', _copy_metadata_call_result_290937)
        
        # ################# End of 'decorator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'decorator' in the type store
        # Getting the type of 'stypy_return_type' (line 373)
        stypy_return_type_290938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'decorator'
        return stypy_return_type_290938

    # Assigning a type to the variable 'decorator' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'decorator', decorator)
    # Getting the type of 'decorator' (line 406)
    decorator_290939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'decorator')
    # Assigning a type to the variable 'stypy_return_type' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type', decorator_290939)
    
    # ################# End of '_pytest_image_comparison(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_pytest_image_comparison' in the type store
    # Getting the type of 'stypy_return_type' (line 358)
    stypy_return_type_290940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290940)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_pytest_image_comparison'
    return stypy_return_type_290940

# Assigning a type to the variable '_pytest_image_comparison' (line 358)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 0), '_pytest_image_comparison', _pytest_image_comparison)

@norecursion
def image_comparison(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 409)
    None_290941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 49), 'None')
    int_290942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 59), 'int')
    # Getting the type of 'None' (line 410)
    None_290943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 38), 'None')
    # Getting the type of 'False' (line 410)
    False_290944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 56), 'False')
    # Getting the type of 'None' (line 411)
    None_290945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 35), 'None')
    unicode_290946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 27), 'unicode', u'_classic_test')
    defaults = [None_290941, int_290942, None_290943, False_290944, None_290945, unicode_290946]
    # Create a new context for function 'image_comparison'
    module_type_store = module_type_store.open_function_context('image_comparison', 409, 0, False)
    
    # Passed parameters checking function
    image_comparison.stypy_localization = localization
    image_comparison.stypy_type_of_self = None
    image_comparison.stypy_type_store = module_type_store
    image_comparison.stypy_function_name = 'image_comparison'
    image_comparison.stypy_param_names_list = ['baseline_images', 'extensions', 'tol', 'freetype_version', 'remove_text', 'savefig_kwarg', 'style']
    image_comparison.stypy_varargs_param_name = None
    image_comparison.stypy_kwargs_param_name = None
    image_comparison.stypy_call_defaults = defaults
    image_comparison.stypy_call_varargs = varargs
    image_comparison.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'image_comparison', ['baseline_images', 'extensions', 'tol', 'freetype_version', 'remove_text', 'savefig_kwarg', 'style'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'image_comparison', localization, ['baseline_images', 'extensions', 'tol', 'freetype_version', 'remove_text', 'savefig_kwarg', 'style'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'image_comparison(...)' code ##################

    unicode_290947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, (-1)), 'unicode', u"\n    Compare images generated by the test with those specified in\n    *baseline_images*, which must correspond else an\n    ImageComparisonFailure exception will be raised.\n\n    Arguments\n    ---------\n    baseline_images : list or None\n        A list of strings specifying the names of the images generated by\n        calls to :meth:`matplotlib.figure.savefig`.\n\n        If *None*, the test function must use the ``baseline_images`` fixture,\n        either as a parameter or with pytest.mark.usefixtures. This value is\n        only allowed when using pytest.\n\n    extensions : [ None | list ]\n\n        If None, defaults to all supported extensions.\n        Otherwise, a list of extensions to test. For example ['png','pdf'].\n\n    tol : float, optional, default: 0\n        The RMS threshold above which the test is considered failed.\n\n    freetype_version : str or tuple\n        The expected freetype version or range of versions for this test to\n        pass.\n\n    remove_text : bool\n        Remove the title and tick text from the figure before comparison.\n        This does not remove other, more deliberate, text, such as legends and\n        annotations.\n\n    savefig_kwarg : dict\n        Optional arguments that are passed to the savefig method.\n\n    style : string\n        Optional name for the base style to apply to the image test. The test\n        itself can also apply additional styles if desired. Defaults to the\n        '_classic_test' style.\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 455)
    # Getting the type of 'extensions' (line 455)
    extensions_290948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 7), 'extensions')
    # Getting the type of 'None' (line 455)
    None_290949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), 'None')
    
    (may_be_290950, more_types_in_union_290951) = may_be_none(extensions_290948, None_290949)

    if may_be_290950:

        if more_types_in_union_290951:
            # Runtime conditional SSA (line 455)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 457):
        
        # Assigning a List to a Name (line 457):
        
        # Obtaining an instance of the builtin type 'list' (line 457)
        list_290952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 457)
        # Adding element type (line 457)
        unicode_290953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 22), 'unicode', u'png')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 21), list_290952, unicode_290953)
        # Adding element type (line 457)
        unicode_290954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 29), 'unicode', u'pdf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 21), list_290952, unicode_290954)
        # Adding element type (line 457)
        unicode_290955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 36), 'unicode', u'svg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 21), list_290952, unicode_290955)
        
        # Assigning a type to the variable 'extensions' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'extensions', list_290952)

        if more_types_in_union_290951:
            # SSA join for if statement (line 455)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 459)
    # Getting the type of 'savefig_kwarg' (line 459)
    savefig_kwarg_290956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 7), 'savefig_kwarg')
    # Getting the type of 'None' (line 459)
    None_290957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 24), 'None')
    
    (may_be_290958, more_types_in_union_290959) = may_be_none(savefig_kwarg_290956, None_290957)

    if may_be_290958:

        if more_types_in_union_290959:
            # Runtime conditional SSA (line 459)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 461):
        
        # Assigning a Call to a Name (line 461):
        
        # Call to dict(...): (line 461)
        # Processing the call keyword arguments (line 461)
        kwargs_290961 = {}
        # Getting the type of 'dict' (line 461)
        dict_290960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 24), 'dict', False)
        # Calling dict(args, kwargs) (line 461)
        dict_call_result_290962 = invoke(stypy.reporting.localization.Localization(__file__, 461, 24), dict_290960, *[], **kwargs_290961)
        
        # Assigning a type to the variable 'savefig_kwarg' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'savefig_kwarg', dict_call_result_290962)

        if more_types_in_union_290959:
            # SSA join for if statement (line 459)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to is_called_from_pytest(...): (line 463)
    # Processing the call keyword arguments (line 463)
    kwargs_290964 = {}
    # Getting the type of 'is_called_from_pytest' (line 463)
    is_called_from_pytest_290963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 7), 'is_called_from_pytest', False)
    # Calling is_called_from_pytest(args, kwargs) (line 463)
    is_called_from_pytest_call_result_290965 = invoke(stypy.reporting.localization.Localization(__file__, 463, 7), is_called_from_pytest_290963, *[], **kwargs_290964)
    
    # Testing the type of an if condition (line 463)
    if_condition_290966 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 4), is_called_from_pytest_call_result_290965)
    # Assigning a type to the variable 'if_condition_290966' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'if_condition_290966', if_condition_290966)
    # SSA begins for if statement (line 463)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _pytest_image_comparison(...): (line 464)
    # Processing the call keyword arguments (line 464)
    # Getting the type of 'baseline_images' (line 465)
    baseline_images_290968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 28), 'baseline_images', False)
    keyword_290969 = baseline_images_290968
    # Getting the type of 'extensions' (line 465)
    extensions_290970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 56), 'extensions', False)
    keyword_290971 = extensions_290970
    # Getting the type of 'tol' (line 465)
    tol_290972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 72), 'tol', False)
    keyword_290973 = tol_290972
    # Getting the type of 'freetype_version' (line 466)
    freetype_version_290974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 29), 'freetype_version', False)
    keyword_290975 = freetype_version_290974
    # Getting the type of 'remove_text' (line 466)
    remove_text_290976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 59), 'remove_text', False)
    keyword_290977 = remove_text_290976
    # Getting the type of 'savefig_kwarg' (line 467)
    savefig_kwarg_290978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 27), 'savefig_kwarg', False)
    keyword_290979 = savefig_kwarg_290978
    # Getting the type of 'style' (line 467)
    style_290980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 48), 'style', False)
    keyword_290981 = style_290980
    kwargs_290982 = {'savefig_kwargs': keyword_290979, 'style': keyword_290981, 'remove_text': keyword_290977, 'freetype_version': keyword_290975, 'extensions': keyword_290971, 'tol': keyword_290973, 'baseline_images': keyword_290969}
    # Getting the type of '_pytest_image_comparison' (line 464)
    _pytest_image_comparison_290967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), '_pytest_image_comparison', False)
    # Calling _pytest_image_comparison(args, kwargs) (line 464)
    _pytest_image_comparison_call_result_290983 = invoke(stypy.reporting.localization.Localization(__file__, 464, 15), _pytest_image_comparison_290967, *[], **kwargs_290982)
    
    # Assigning a type to the variable 'stypy_return_type' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'stypy_return_type', _pytest_image_comparison_call_result_290983)
    # SSA branch for the else part of an if statement (line 463)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 469)
    # Getting the type of 'baseline_images' (line 469)
    baseline_images_290984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 11), 'baseline_images')
    # Getting the type of 'None' (line 469)
    None_290985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 30), 'None')
    
    (may_be_290986, more_types_in_union_290987) = may_be_none(baseline_images_290984, None_290985)

    if may_be_290986:

        if more_types_in_union_290987:
            # Runtime conditional SSA (line 469)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 470)
        # Processing the call arguments (line 470)
        unicode_290989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 29), 'unicode', u'baseline_images must be specified')
        # Processing the call keyword arguments (line 470)
        kwargs_290990 = {}
        # Getting the type of 'ValueError' (line 470)
        ValueError_290988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 470)
        ValueError_call_result_290991 = invoke(stypy.reporting.localization.Localization(__file__, 470, 18), ValueError_290988, *[unicode_290989], **kwargs_290990)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 470, 12), ValueError_call_result_290991, 'raise parameter', BaseException)

        if more_types_in_union_290987:
            # SSA join for if statement (line 469)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to ImageComparisonTest(...): (line 472)
    # Processing the call keyword arguments (line 472)
    # Getting the type of 'baseline_images' (line 473)
    baseline_images_290993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 28), 'baseline_images', False)
    keyword_290994 = baseline_images_290993
    # Getting the type of 'extensions' (line 473)
    extensions_290995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 56), 'extensions', False)
    keyword_290996 = extensions_290995
    # Getting the type of 'tol' (line 473)
    tol_290997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 72), 'tol', False)
    keyword_290998 = tol_290997
    # Getting the type of 'freetype_version' (line 474)
    freetype_version_290999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 29), 'freetype_version', False)
    keyword_291000 = freetype_version_290999
    # Getting the type of 'remove_text' (line 474)
    remove_text_291001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 59), 'remove_text', False)
    keyword_291002 = remove_text_291001
    # Getting the type of 'savefig_kwarg' (line 475)
    savefig_kwarg_291003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 27), 'savefig_kwarg', False)
    keyword_291004 = savefig_kwarg_291003
    # Getting the type of 'style' (line 475)
    style_291005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 48), 'style', False)
    keyword_291006 = style_291005
    kwargs_291007 = {'savefig_kwargs': keyword_291004, 'style': keyword_291006, 'remove_text': keyword_291002, 'freetype_version': keyword_291000, 'extensions': keyword_290996, 'tol': keyword_290998, 'baseline_images': keyword_290994}
    # Getting the type of 'ImageComparisonTest' (line 472)
    ImageComparisonTest_290992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'ImageComparisonTest', False)
    # Calling ImageComparisonTest(args, kwargs) (line 472)
    ImageComparisonTest_call_result_291008 = invoke(stypy.reporting.localization.Localization(__file__, 472, 15), ImageComparisonTest_290992, *[], **kwargs_291007)
    
    # Assigning a type to the variable 'stypy_return_type' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'stypy_return_type', ImageComparisonTest_call_result_291008)
    # SSA join for if statement (line 463)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'image_comparison(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'image_comparison' in the type store
    # Getting the type of 'stypy_return_type' (line 409)
    stypy_return_type_291009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291009)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'image_comparison'
    return stypy_return_type_291009

# Assigning a type to the variable 'image_comparison' (line 409)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'image_comparison', image_comparison)

@norecursion
def _image_directories(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_image_directories'
    module_type_store = module_type_store.open_function_context('_image_directories', 478, 0, False)
    
    # Passed parameters checking function
    _image_directories.stypy_localization = localization
    _image_directories.stypy_type_of_self = None
    _image_directories.stypy_type_store = module_type_store
    _image_directories.stypy_function_name = '_image_directories'
    _image_directories.stypy_param_names_list = ['func']
    _image_directories.stypy_varargs_param_name = None
    _image_directories.stypy_kwargs_param_name = None
    _image_directories.stypy_call_defaults = defaults
    _image_directories.stypy_call_varargs = varargs
    _image_directories.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_image_directories', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_image_directories', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_image_directories(...)' code ##################

    unicode_291010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, (-1)), 'unicode', u"\n    Compute the baseline and result image directories for testing *func*.\n    Create the result directory if it doesn't exist.\n    ")
    
    # Assigning a Attribute to a Name (line 483):
    
    # Assigning a Attribute to a Name (line 483):
    # Getting the type of 'func' (line 483)
    func_291011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 18), 'func')
    # Obtaining the member '__module__' of a type (line 483)
    module___291012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 18), func_291011, '__module__')
    # Assigning a type to the variable 'module_name' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'module_name', module___291012)
    
    
    # Getting the type of 'module_name' (line 484)
    module_name_291013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 7), 'module_name')
    unicode_291014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 22), 'unicode', u'__main__')
    # Applying the binary operator '==' (line 484)
    result_eq_291015 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 7), '==', module_name_291013, unicode_291014)
    
    # Testing the type of an if condition (line 484)
    if_condition_291016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 484, 4), result_eq_291015)
    # Assigning a type to the variable 'if_condition_291016' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'if_condition_291016', if_condition_291016)
    # SSA begins for if statement (line 484)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 486)
    # Processing the call arguments (line 486)
    unicode_291019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 22), 'unicode', u'test module run as script. guessing baseline image locations')
    # Processing the call keyword arguments (line 486)
    kwargs_291020 = {}
    # Getting the type of 'warnings' (line 486)
    warnings_291017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 486)
    warn_291018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 8), warnings_291017, 'warn')
    # Calling warn(args, kwargs) (line 486)
    warn_call_result_291021 = invoke(stypy.reporting.localization.Localization(__file__, 486, 8), warn_291018, *[unicode_291019], **kwargs_291020)
    
    
    # Assigning a Subscript to a Name (line 487):
    
    # Assigning a Subscript to a Name (line 487):
    
    # Obtaining the type of the subscript
    int_291022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 31), 'int')
    # Getting the type of 'sys' (line 487)
    sys_291023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 22), 'sys')
    # Obtaining the member 'argv' of a type (line 487)
    argv_291024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 22), sys_291023, 'argv')
    # Obtaining the member '__getitem__' of a type (line 487)
    getitem___291025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 22), argv_291024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 487)
    subscript_call_result_291026 = invoke(stypy.reporting.localization.Localization(__file__, 487, 22), getitem___291025, int_291022)
    
    # Assigning a type to the variable 'script_name' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'script_name', subscript_call_result_291026)
    
    # Assigning a Call to a Name (line 488):
    
    # Assigning a Call to a Name (line 488):
    
    # Call to abspath(...): (line 488)
    # Processing the call arguments (line 488)
    
    # Call to dirname(...): (line 488)
    # Processing the call arguments (line 488)
    # Getting the type of 'script_name' (line 488)
    script_name_291033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 50), 'script_name', False)
    # Processing the call keyword arguments (line 488)
    kwargs_291034 = {}
    # Getting the type of 'os' (line 488)
    os_291030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 488)
    path_291031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 34), os_291030, 'path')
    # Obtaining the member 'dirname' of a type (line 488)
    dirname_291032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 34), path_291031, 'dirname')
    # Calling dirname(args, kwargs) (line 488)
    dirname_call_result_291035 = invoke(stypy.reporting.localization.Localization(__file__, 488, 34), dirname_291032, *[script_name_291033], **kwargs_291034)
    
    # Processing the call keyword arguments (line 488)
    kwargs_291036 = {}
    # Getting the type of 'os' (line 488)
    os_291027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 488)
    path_291028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 18), os_291027, 'path')
    # Obtaining the member 'abspath' of a type (line 488)
    abspath_291029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 18), path_291028, 'abspath')
    # Calling abspath(args, kwargs) (line 488)
    abspath_call_result_291037 = invoke(stypy.reporting.localization.Localization(__file__, 488, 18), abspath_291029, *[dirname_call_result_291035], **kwargs_291036)
    
    # Assigning a type to the variable 'basedir' (line 488)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'basedir', abspath_call_result_291037)
    
    # Assigning a Subscript to a Name (line 489):
    
    # Assigning a Subscript to a Name (line 489):
    
    # Obtaining the type of the subscript
    int_291038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 65), 'int')
    
    # Call to splitext(...): (line 489)
    # Processing the call arguments (line 489)
    
    # Obtaining the type of the subscript
    int_291042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 61), 'int')
    
    # Call to split(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'script_name' (line 489)
    script_name_291046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 48), 'script_name', False)
    # Processing the call keyword arguments (line 489)
    kwargs_291047 = {}
    # Getting the type of 'os' (line 489)
    os_291043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 489)
    path_291044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 34), os_291043, 'path')
    # Obtaining the member 'split' of a type (line 489)
    split_291045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 34), path_291044, 'split')
    # Calling split(args, kwargs) (line 489)
    split_call_result_291048 = invoke(stypy.reporting.localization.Localization(__file__, 489, 34), split_291045, *[script_name_291046], **kwargs_291047)
    
    # Obtaining the member '__getitem__' of a type (line 489)
    getitem___291049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 34), split_call_result_291048, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 489)
    subscript_call_result_291050 = invoke(stypy.reporting.localization.Localization(__file__, 489, 34), getitem___291049, int_291042)
    
    # Processing the call keyword arguments (line 489)
    kwargs_291051 = {}
    # Getting the type of 'os' (line 489)
    os_291039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 489)
    path_291040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 17), os_291039, 'path')
    # Obtaining the member 'splitext' of a type (line 489)
    splitext_291041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 17), path_291040, 'splitext')
    # Calling splitext(args, kwargs) (line 489)
    splitext_call_result_291052 = invoke(stypy.reporting.localization.Localization(__file__, 489, 17), splitext_291041, *[subscript_call_result_291050], **kwargs_291051)
    
    # Obtaining the member '__getitem__' of a type (line 489)
    getitem___291053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 17), splitext_call_result_291052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 489)
    subscript_call_result_291054 = invoke(stypy.reporting.localization.Localization(__file__, 489, 17), getitem___291053, int_291038)
    
    # Assigning a type to the variable 'subdir' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'subdir', subscript_call_result_291054)
    # SSA branch for the else part of an if statement (line 484)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 491):
    
    # Assigning a Call to a Name (line 491):
    
    # Call to split(...): (line 491)
    # Processing the call arguments (line 491)
    unicode_291057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 33), 'unicode', u'.')
    # Processing the call keyword arguments (line 491)
    kwargs_291058 = {}
    # Getting the type of 'module_name' (line 491)
    module_name_291055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 15), 'module_name', False)
    # Obtaining the member 'split' of a type (line 491)
    split_291056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 15), module_name_291055, 'split')
    # Calling split(args, kwargs) (line 491)
    split_call_result_291059 = invoke(stypy.reporting.localization.Localization(__file__, 491, 15), split_291056, *[unicode_291057], **kwargs_291058)
    
    # Assigning a type to the variable 'mods' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'mods', split_call_result_291059)
    
    
    
    # Call to len(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'mods' (line 492)
    mods_291061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 15), 'mods', False)
    # Processing the call keyword arguments (line 492)
    kwargs_291062 = {}
    # Getting the type of 'len' (line 492)
    len_291060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 11), 'len', False)
    # Calling len(args, kwargs) (line 492)
    len_call_result_291063 = invoke(stypy.reporting.localization.Localization(__file__, 492, 11), len_291060, *[mods_291061], **kwargs_291062)
    
    int_291064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 24), 'int')
    # Applying the binary operator '>=' (line 492)
    result_ge_291065 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 11), '>=', len_call_result_291063, int_291064)
    
    # Testing the type of an if condition (line 492)
    if_condition_291066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 492, 8), result_ge_291065)
    # Assigning a type to the variable 'if_condition_291066' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'if_condition_291066', if_condition_291066)
    # SSA begins for if statement (line 492)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to pop(...): (line 493)
    # Processing the call arguments (line 493)
    int_291069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 21), 'int')
    # Processing the call keyword arguments (line 493)
    kwargs_291070 = {}
    # Getting the type of 'mods' (line 493)
    mods_291067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'mods', False)
    # Obtaining the member 'pop' of a type (line 493)
    pop_291068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 12), mods_291067, 'pop')
    # Calling pop(args, kwargs) (line 493)
    pop_call_result_291071 = invoke(stypy.reporting.localization.Localization(__file__, 493, 12), pop_291068, *[int_291069], **kwargs_291070)
    
    # SSA join for if statement (line 492)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to pop(...): (line 499)
    # Processing the call arguments (line 499)
    int_291074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 20), 'int')
    # Processing the call keyword arguments (line 499)
    kwargs_291075 = {}
    # Getting the type of 'mods' (line 499)
    mods_291072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 11), 'mods', False)
    # Obtaining the member 'pop' of a type (line 499)
    pop_291073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 11), mods_291072, 'pop')
    # Calling pop(args, kwargs) (line 499)
    pop_call_result_291076 = invoke(stypy.reporting.localization.Localization(__file__, 499, 11), pop_291073, *[int_291074], **kwargs_291075)
    
    unicode_291077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 26), 'unicode', u'tests')
    # Applying the binary operator '!=' (line 499)
    result_ne_291078 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 11), '!=', pop_call_result_291076, unicode_291077)
    
    # Testing the type of an if condition (line 499)
    if_condition_291079 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 8), result_ne_291078)
    # Assigning a type to the variable 'if_condition_291079' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'if_condition_291079', if_condition_291079)
    # SSA begins for if statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 500)
    # Processing the call arguments (line 500)
    unicode_291082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 27), 'unicode', u"Module '%s' does not live in a parent module named 'tests'. This is probably ok, but we may not be able to guess the correct subdirectory containing the baseline images. If things go wrong please make sure that there is a parent directory named 'tests' and that it contains a __init__.py file (can be empty).")
    # Getting the type of 'module_name' (line 505)
    module_name_291083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 54), 'module_name', False)
    # Applying the binary operator '%' (line 500)
    result_mod_291084 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 26), '%', unicode_291082, module_name_291083)
    
    # Processing the call keyword arguments (line 500)
    kwargs_291085 = {}
    # Getting the type of 'warnings' (line 500)
    warnings_291080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 500)
    warn_291081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), warnings_291080, 'warn')
    # Calling warn(args, kwargs) (line 500)
    warn_call_result_291086 = invoke(stypy.reporting.localization.Localization(__file__, 500, 12), warn_291081, *[result_mod_291084], **kwargs_291085)
    
    # SSA join for if statement (line 499)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 506):
    
    # Assigning a Call to a Name (line 506):
    
    # Call to join(...): (line 506)
    # Getting the type of 'mods' (line 506)
    mods_291090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 31), 'mods', False)
    # Processing the call keyword arguments (line 506)
    kwargs_291091 = {}
    # Getting the type of 'os' (line 506)
    os_291087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 506)
    path_291088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 17), os_291087, 'path')
    # Obtaining the member 'join' of a type (line 506)
    join_291089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 17), path_291088, 'join')
    # Calling join(args, kwargs) (line 506)
    join_call_result_291092 = invoke(stypy.reporting.localization.Localization(__file__, 506, 17), join_291089, *[mods_291090], **kwargs_291091)
    
    # Assigning a type to the variable 'subdir' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'subdir', join_call_result_291092)
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 508, 8))
    
    # 'import imp' statement (line 508)
    import imp

    import_module(stypy.reporting.localization.Localization(__file__, 508, 8), 'imp', imp, module_type_store)
    

    @norecursion
    def find_dotted_module(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 509)
        None_291093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 49), 'None')
        defaults = [None_291093]
        # Create a new context for function 'find_dotted_module'
        module_type_store = module_type_store.open_function_context('find_dotted_module', 509, 8, False)
        
        # Passed parameters checking function
        find_dotted_module.stypy_localization = localization
        find_dotted_module.stypy_type_of_self = None
        find_dotted_module.stypy_type_store = module_type_store
        find_dotted_module.stypy_function_name = 'find_dotted_module'
        find_dotted_module.stypy_param_names_list = ['module_name', 'path']
        find_dotted_module.stypy_varargs_param_name = None
        find_dotted_module.stypy_kwargs_param_name = None
        find_dotted_module.stypy_call_defaults = defaults
        find_dotted_module.stypy_call_varargs = varargs
        find_dotted_module.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'find_dotted_module', ['module_name', 'path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_dotted_module', localization, ['module_name', 'path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_dotted_module(...)' code ##################

        unicode_291094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, (-1)), 'unicode', u'A version of imp which can handle dots in the module name.\n               As for imp.find_module(), the return value is a 3-element\n               tuple (file, pathname, description).')
        
        # Assigning a Name to a Name (line 513):
        
        # Assigning a Name to a Name (line 513):
        # Getting the type of 'None' (line 513)
        None_291095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 18), 'None')
        # Assigning a type to the variable 'res' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'res', None_291095)
        
        
        # Call to split(...): (line 514)
        # Processing the call arguments (line 514)
        unicode_291098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 45), 'unicode', u'.')
        # Processing the call keyword arguments (line 514)
        kwargs_291099 = {}
        # Getting the type of 'module_name' (line 514)
        module_name_291096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 27), 'module_name', False)
        # Obtaining the member 'split' of a type (line 514)
        split_291097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 27), module_name_291096, 'split')
        # Calling split(args, kwargs) (line 514)
        split_call_result_291100 = invoke(stypy.reporting.localization.Localization(__file__, 514, 27), split_291097, *[unicode_291098], **kwargs_291099)
        
        # Testing the type of a for loop iterable (line 514)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 514, 12), split_call_result_291100)
        # Getting the type of the for loop variable (line 514)
        for_loop_var_291101 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 514, 12), split_call_result_291100)
        # Assigning a type to the variable 'sub_mod' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'sub_mod', for_loop_var_291101)
        # SSA begins for a for statement (line 514)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 515)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Name:
        
        # Call to find_module(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'sub_mod' (line 516)
        sub_mod_291104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 58), 'sub_mod', False)
        # Getting the type of 'path' (line 516)
        path_291105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 67), 'path', False)
        # Processing the call keyword arguments (line 516)
        kwargs_291106 = {}
        # Getting the type of 'imp' (line 516)
        imp_291102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 42), 'imp', False)
        # Obtaining the member 'find_module' of a type (line 516)
        find_module_291103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 42), imp_291102, 'find_module')
        # Calling find_module(args, kwargs) (line 516)
        find_module_call_result_291107 = invoke(stypy.reporting.localization.Localization(__file__, 516, 42), find_module_291103, *[sub_mod_291104, path_291105], **kwargs_291106)
        
        # Assigning a type to the variable 'call_assignment_290053' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290053', find_module_call_result_291107)
        
        # Assigning a Call to a Name (line 516):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_291110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 20), 'int')
        # Processing the call keyword arguments
        kwargs_291111 = {}
        # Getting the type of 'call_assignment_290053' (line 516)
        call_assignment_290053_291108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290053', False)
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___291109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 20), call_assignment_290053_291108, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_291112 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___291109, *[int_291110], **kwargs_291111)
        
        # Assigning a type to the variable 'call_assignment_290054' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290054', getitem___call_result_291112)
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'call_assignment_290054' (line 516)
        call_assignment_290054_291113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290054')
        # Assigning a type to the variable 'file' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 26), 'file', call_assignment_290054_291113)
        
        # Assigning a Call to a Name (line 516):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_291116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 20), 'int')
        # Processing the call keyword arguments
        kwargs_291117 = {}
        # Getting the type of 'call_assignment_290053' (line 516)
        call_assignment_290053_291114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290053', False)
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___291115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 20), call_assignment_290053_291114, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_291118 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___291115, *[int_291116], **kwargs_291117)
        
        # Assigning a type to the variable 'call_assignment_290055' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290055', getitem___call_result_291118)
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'call_assignment_290055' (line 516)
        call_assignment_290055_291119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290055')
        # Assigning a type to the variable 'path' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 32), 'path', call_assignment_290055_291119)
        
        # Assigning a Call to a Name (line 516):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_291122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 20), 'int')
        # Processing the call keyword arguments
        kwargs_291123 = {}
        # Getting the type of 'call_assignment_290053' (line 516)
        call_assignment_290053_291120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290053', False)
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___291121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 20), call_assignment_290053_291120, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_291124 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___291121, *[int_291122], **kwargs_291123)
        
        # Assigning a type to the variable 'call_assignment_290056' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290056', getitem___call_result_291124)
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'call_assignment_290056' (line 516)
        call_assignment_290056_291125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'call_assignment_290056')
        # Assigning a type to the variable '_' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 38), '_', call_assignment_290056_291125)
        
        # Assigning a Tuple to a Name (line 516):
        
        # Obtaining an instance of the builtin type 'tuple' (line 516)
        tuple_291126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 516)
        # Adding element type (line 516)file
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), tuple_291126, )
        # Adding element type (line 516)path
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), tuple_291126, )
        # Adding element type (line 516)_
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 26), tuple_291126, )
        
        # Assigning a type to the variable 'res' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'res', tuple_291126)
        
        # Assigning a List to a Name (line 517):
        
        # Assigning a List to a Name (line 517):
        
        # Obtaining an instance of the builtin type 'list' (line 517)
        list_291127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 517)
        # Adding element type (line 517)
        # Getting the type of 'path' (line 517)
        path_291128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 28), 'path')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 27), list_291127, path_291128)
        
        # Assigning a type to the variable 'path' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 20), 'path', list_291127)
        
        # Type idiom detected: calculating its left and rigth part (line 518)
        # Getting the type of 'file' (line 518)
        file_291129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 20), 'file')
        # Getting the type of 'None' (line 518)
        None_291130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 35), 'None')
        
        (may_be_291131, more_types_in_union_291132) = may_not_be_none(file_291129, None_291130)

        if may_be_291131:

            if more_types_in_union_291132:
                # Runtime conditional SSA (line 518)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to close(...): (line 519)
            # Processing the call keyword arguments (line 519)
            kwargs_291135 = {}
            # Getting the type of 'file' (line 519)
            file_291133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 24), 'file', False)
            # Obtaining the member 'close' of a type (line 519)
            close_291134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 24), file_291133, 'close')
            # Calling close(args, kwargs) (line 519)
            close_call_result_291136 = invoke(stypy.reporting.localization.Localization(__file__, 519, 24), close_291134, *[], **kwargs_291135)
            

            if more_types_in_union_291132:
                # SSA join for if statement (line 518)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the except part of a try statement (line 515)
        # SSA branch for the except 'ImportError' branch of a try statement (line 515)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 522):
        
        # Assigning a Call to a Name (line 522):
        
        # Call to list(...): (line 522)
        # Processing the call arguments (line 522)
        
        # Obtaining the type of the subscript
        # Getting the type of 'sub_mod' (line 522)
        sub_mod_291138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 44), 'sub_mod', False)
        # Getting the type of 'sys' (line 522)
        sys_291139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 32), 'sys', False)
        # Obtaining the member 'modules' of a type (line 522)
        modules_291140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 32), sys_291139, 'modules')
        # Obtaining the member '__getitem__' of a type (line 522)
        getitem___291141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 32), modules_291140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 522)
        subscript_call_result_291142 = invoke(stypy.reporting.localization.Localization(__file__, 522, 32), getitem___291141, sub_mod_291138)
        
        # Obtaining the member '__path__' of a type (line 522)
        path___291143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 32), subscript_call_result_291142, '__path__')
        # Processing the call keyword arguments (line 522)
        kwargs_291144 = {}
        # Getting the type of 'list' (line 522)
        list_291137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 27), 'list', False)
        # Calling list(args, kwargs) (line 522)
        list_call_result_291145 = invoke(stypy.reporting.localization.Localization(__file__, 522, 27), list_291137, *[path___291143], **kwargs_291144)
        
        # Assigning a type to the variable 'path' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 20), 'path', list_call_result_291145)
        
        # Assigning a Tuple to a Name (line 523):
        
        # Assigning a Tuple to a Name (line 523):
        
        # Obtaining an instance of the builtin type 'tuple' (line 523)
        tuple_291146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 523)
        # Adding element type (line 523)
        # Getting the type of 'None' (line 523)
        None_291147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 26), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 26), tuple_291146, None_291147)
        # Adding element type (line 523)
        # Getting the type of 'path' (line 523)
        path_291148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 32), 'path')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 26), tuple_291146, path_291148)
        # Adding element type (line 523)
        # Getting the type of 'None' (line 523)
        None_291149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 38), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 26), tuple_291146, None_291149)
        
        # Assigning a type to the variable 'res' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 20), 'res', tuple_291146)
        # SSA join for try-except statement (line 515)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'res' (line 524)
        res_291150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 19), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'stypy_return_type', res_291150)
        
        # ################# End of 'find_dotted_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_dotted_module' in the type store
        # Getting the type of 'stypy_return_type' (line 509)
        stypy_return_type_291151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291151)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_dotted_module'
        return stypy_return_type_291151

    # Assigning a type to the variable 'find_dotted_module' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'find_dotted_module', find_dotted_module)
    
    # Assigning a Subscript to a Name (line 526):
    
    # Assigning a Subscript to a Name (line 526):
    
    # Obtaining the type of the subscript
    int_291152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 55), 'int')
    
    # Call to find_dotted_module(...): (line 526)
    # Processing the call arguments (line 526)
    # Getting the type of 'func' (line 526)
    func_291154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 38), 'func', False)
    # Obtaining the member '__module__' of a type (line 526)
    module___291155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 38), func_291154, '__module__')
    # Processing the call keyword arguments (line 526)
    kwargs_291156 = {}
    # Getting the type of 'find_dotted_module' (line 526)
    find_dotted_module_291153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 19), 'find_dotted_module', False)
    # Calling find_dotted_module(args, kwargs) (line 526)
    find_dotted_module_call_result_291157 = invoke(stypy.reporting.localization.Localization(__file__, 526, 19), find_dotted_module_291153, *[module___291155], **kwargs_291156)
    
    # Obtaining the member '__getitem__' of a type (line 526)
    getitem___291158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 19), find_dotted_module_call_result_291157, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 526)
    subscript_call_result_291159 = invoke(stypy.reporting.localization.Localization(__file__, 526, 19), getitem___291158, int_291152)
    
    # Assigning a type to the variable 'mod_file' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'mod_file', subscript_call_result_291159)
    
    # Assigning a Call to a Name (line 527):
    
    # Assigning a Call to a Name (line 527):
    
    # Call to dirname(...): (line 527)
    # Processing the call arguments (line 527)
    # Getting the type of 'mod_file' (line 527)
    mod_file_291163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 34), 'mod_file', False)
    # Processing the call keyword arguments (line 527)
    kwargs_291164 = {}
    # Getting the type of 'os' (line 527)
    os_291160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 527)
    path_291161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 18), os_291160, 'path')
    # Obtaining the member 'dirname' of a type (line 527)
    dirname_291162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 18), path_291161, 'dirname')
    # Calling dirname(args, kwargs) (line 527)
    dirname_call_result_291165 = invoke(stypy.reporting.localization.Localization(__file__, 527, 18), dirname_291162, *[mod_file_291163], **kwargs_291164)
    
    # Assigning a type to the variable 'basedir' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'basedir', dirname_call_result_291165)
    # SSA join for if statement (line 484)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 529):
    
    # Assigning a Call to a Name (line 529):
    
    # Call to join(...): (line 529)
    # Processing the call arguments (line 529)
    # Getting the type of 'basedir' (line 529)
    basedir_291169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 32), 'basedir', False)
    unicode_291170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 41), 'unicode', u'baseline_images')
    # Getting the type of 'subdir' (line 529)
    subdir_291171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 60), 'subdir', False)
    # Processing the call keyword arguments (line 529)
    kwargs_291172 = {}
    # Getting the type of 'os' (line 529)
    os_291166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 529)
    path_291167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 19), os_291166, 'path')
    # Obtaining the member 'join' of a type (line 529)
    join_291168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 19), path_291167, 'join')
    # Calling join(args, kwargs) (line 529)
    join_call_result_291173 = invoke(stypy.reporting.localization.Localization(__file__, 529, 19), join_291168, *[basedir_291169, unicode_291170, subdir_291171], **kwargs_291172)
    
    # Assigning a type to the variable 'baseline_dir' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'baseline_dir', join_call_result_291173)
    
    # Assigning a Call to a Name (line 530):
    
    # Assigning a Call to a Name (line 530):
    
    # Call to abspath(...): (line 530)
    # Processing the call arguments (line 530)
    
    # Call to join(...): (line 530)
    # Processing the call arguments (line 530)
    unicode_291180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 46), 'unicode', u'result_images')
    # Getting the type of 'subdir' (line 530)
    subdir_291181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 63), 'subdir', False)
    # Processing the call keyword arguments (line 530)
    kwargs_291182 = {}
    # Getting the type of 'os' (line 530)
    os_291177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 33), 'os', False)
    # Obtaining the member 'path' of a type (line 530)
    path_291178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 33), os_291177, 'path')
    # Obtaining the member 'join' of a type (line 530)
    join_291179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 33), path_291178, 'join')
    # Calling join(args, kwargs) (line 530)
    join_call_result_291183 = invoke(stypy.reporting.localization.Localization(__file__, 530, 33), join_291179, *[unicode_291180, subdir_291181], **kwargs_291182)
    
    # Processing the call keyword arguments (line 530)
    kwargs_291184 = {}
    # Getting the type of 'os' (line 530)
    os_291174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 530)
    path_291175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 17), os_291174, 'path')
    # Obtaining the member 'abspath' of a type (line 530)
    abspath_291176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 17), path_291175, 'abspath')
    # Calling abspath(args, kwargs) (line 530)
    abspath_call_result_291185 = invoke(stypy.reporting.localization.Localization(__file__, 530, 17), abspath_291176, *[join_call_result_291183], **kwargs_291184)
    
    # Assigning a type to the variable 'result_dir' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'result_dir', abspath_call_result_291185)
    
    
    
    # Call to exists(...): (line 532)
    # Processing the call arguments (line 532)
    # Getting the type of 'result_dir' (line 532)
    result_dir_291189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 26), 'result_dir', False)
    # Processing the call keyword arguments (line 532)
    kwargs_291190 = {}
    # Getting the type of 'os' (line 532)
    os_291186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 532)
    path_291187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 11), os_291186, 'path')
    # Obtaining the member 'exists' of a type (line 532)
    exists_291188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 11), path_291187, 'exists')
    # Calling exists(args, kwargs) (line 532)
    exists_call_result_291191 = invoke(stypy.reporting.localization.Localization(__file__, 532, 11), exists_291188, *[result_dir_291189], **kwargs_291190)
    
    # Applying the 'not' unary operator (line 532)
    result_not__291192 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 7), 'not', exists_call_result_291191)
    
    # Testing the type of an if condition (line 532)
    if_condition_291193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 532, 4), result_not__291192)
    # Assigning a type to the variable 'if_condition_291193' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'if_condition_291193', if_condition_291193)
    # SSA begins for if statement (line 532)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to mkdirs(...): (line 533)
    # Processing the call arguments (line 533)
    # Getting the type of 'result_dir' (line 533)
    result_dir_291196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 21), 'result_dir', False)
    # Processing the call keyword arguments (line 533)
    kwargs_291197 = {}
    # Getting the type of 'cbook' (line 533)
    cbook_291194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'cbook', False)
    # Obtaining the member 'mkdirs' of a type (line 533)
    mkdirs_291195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), cbook_291194, 'mkdirs')
    # Calling mkdirs(args, kwargs) (line 533)
    mkdirs_call_result_291198 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), mkdirs_291195, *[result_dir_291196], **kwargs_291197)
    
    # SSA join for if statement (line 532)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 535)
    tuple_291199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 535)
    # Adding element type (line 535)
    # Getting the type of 'baseline_dir' (line 535)
    baseline_dir_291200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 11), 'baseline_dir')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 11), tuple_291199, baseline_dir_291200)
    # Adding element type (line 535)
    # Getting the type of 'result_dir' (line 535)
    result_dir_291201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 25), 'result_dir')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 11), tuple_291199, result_dir_291201)
    
    # Assigning a type to the variable 'stypy_return_type' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type', tuple_291199)
    
    # ################# End of '_image_directories(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_image_directories' in the type store
    # Getting the type of 'stypy_return_type' (line 478)
    stypy_return_type_291202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291202)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_image_directories'
    return stypy_return_type_291202

# Assigning a type to the variable '_image_directories' (line 478)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 0), '_image_directories', _image_directories)

@norecursion
def switch_backend(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'switch_backend'
    module_type_store = module_type_store.open_function_context('switch_backend', 538, 0, False)
    
    # Passed parameters checking function
    switch_backend.stypy_localization = localization
    switch_backend.stypy_type_of_self = None
    switch_backend.stypy_type_store = module_type_store
    switch_backend.stypy_function_name = 'switch_backend'
    switch_backend.stypy_param_names_list = ['backend']
    switch_backend.stypy_varargs_param_name = None
    switch_backend.stypy_kwargs_param_name = None
    switch_backend.stypy_call_defaults = defaults
    switch_backend.stypy_call_varargs = varargs
    switch_backend.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'switch_backend', ['backend'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'switch_backend', localization, ['backend'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'switch_backend(...)' code ##################


    @norecursion
    def switch_backend_decorator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'switch_backend_decorator'
        module_type_store = module_type_store.open_function_context('switch_backend_decorator', 541, 4, False)
        
        # Passed parameters checking function
        switch_backend_decorator.stypy_localization = localization
        switch_backend_decorator.stypy_type_of_self = None
        switch_backend_decorator.stypy_type_store = module_type_store
        switch_backend_decorator.stypy_function_name = 'switch_backend_decorator'
        switch_backend_decorator.stypy_param_names_list = ['func']
        switch_backend_decorator.stypy_varargs_param_name = None
        switch_backend_decorator.stypy_kwargs_param_name = None
        switch_backend_decorator.stypy_call_defaults = defaults
        switch_backend_decorator.stypy_call_varargs = varargs
        switch_backend_decorator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'switch_backend_decorator', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'switch_backend_decorator', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'switch_backend_decorator(...)' code ##################


        @norecursion
        def backend_switcher(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'backend_switcher'
            module_type_store = module_type_store.open_function_context('backend_switcher', 542, 8, False)
            
            # Passed parameters checking function
            backend_switcher.stypy_localization = localization
            backend_switcher.stypy_type_of_self = None
            backend_switcher.stypy_type_store = module_type_store
            backend_switcher.stypy_function_name = 'backend_switcher'
            backend_switcher.stypy_param_names_list = []
            backend_switcher.stypy_varargs_param_name = 'args'
            backend_switcher.stypy_kwargs_param_name = 'kwargs'
            backend_switcher.stypy_call_defaults = defaults
            backend_switcher.stypy_call_varargs = varargs
            backend_switcher.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'backend_switcher', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'backend_switcher', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'backend_switcher(...)' code ##################

            
            # Try-finally block (line 544)
            
            # Assigning a Call to a Name (line 545):
            
            # Assigning a Call to a Name (line 545):
            
            # Call to get_backend(...): (line 545)
            # Processing the call keyword arguments (line 545)
            kwargs_291205 = {}
            # Getting the type of 'mpl' (line 545)
            mpl_291203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 31), 'mpl', False)
            # Obtaining the member 'get_backend' of a type (line 545)
            get_backend_291204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 31), mpl_291203, 'get_backend')
            # Calling get_backend(args, kwargs) (line 545)
            get_backend_call_result_291206 = invoke(stypy.reporting.localization.Localization(__file__, 545, 31), get_backend_291204, *[], **kwargs_291205)
            
            # Assigning a type to the variable 'prev_backend' (line 545)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'prev_backend', get_backend_call_result_291206)
            
            # Call to setup(...): (line 546)
            # Processing the call keyword arguments (line 546)
            kwargs_291210 = {}
            # Getting the type of 'matplotlib' (line 546)
            matplotlib_291207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 16), 'matplotlib', False)
            # Obtaining the member 'testing' of a type (line 546)
            testing_291208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 16), matplotlib_291207, 'testing')
            # Obtaining the member 'setup' of a type (line 546)
            setup_291209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 16), testing_291208, 'setup')
            # Calling setup(args, kwargs) (line 546)
            setup_call_result_291211 = invoke(stypy.reporting.localization.Localization(__file__, 546, 16), setup_291209, *[], **kwargs_291210)
            
            
            # Call to switch_backend(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'backend' (line 547)
            backend_291214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 35), 'backend', False)
            # Processing the call keyword arguments (line 547)
            kwargs_291215 = {}
            # Getting the type of 'plt' (line 547)
            plt_291212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'plt', False)
            # Obtaining the member 'switch_backend' of a type (line 547)
            switch_backend_291213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 16), plt_291212, 'switch_backend')
            # Calling switch_backend(args, kwargs) (line 547)
            switch_backend_call_result_291216 = invoke(stypy.reporting.localization.Localization(__file__, 547, 16), switch_backend_291213, *[backend_291214], **kwargs_291215)
            
            
            # Assigning a Call to a Name (line 548):
            
            # Assigning a Call to a Name (line 548):
            
            # Call to func(...): (line 548)
            # Getting the type of 'args' (line 548)
            args_291218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 31), 'args', False)
            # Processing the call keyword arguments (line 548)
            # Getting the type of 'kwargs' (line 548)
            kwargs_291219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 39), 'kwargs', False)
            kwargs_291220 = {'kwargs_291219': kwargs_291219}
            # Getting the type of 'func' (line 548)
            func_291217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 25), 'func', False)
            # Calling func(args, kwargs) (line 548)
            func_call_result_291221 = invoke(stypy.reporting.localization.Localization(__file__, 548, 25), func_291217, *[args_291218], **kwargs_291220)
            
            # Assigning a type to the variable 'result' (line 548)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'result', func_call_result_291221)
            
            # finally branch of the try-finally block (line 544)
            
            # Call to switch_backend(...): (line 550)
            # Processing the call arguments (line 550)
            # Getting the type of 'prev_backend' (line 550)
            prev_backend_291224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'prev_backend', False)
            # Processing the call keyword arguments (line 550)
            kwargs_291225 = {}
            # Getting the type of 'plt' (line 550)
            plt_291222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 16), 'plt', False)
            # Obtaining the member 'switch_backend' of a type (line 550)
            switch_backend_291223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 16), plt_291222, 'switch_backend')
            # Calling switch_backend(args, kwargs) (line 550)
            switch_backend_call_result_291226 = invoke(stypy.reporting.localization.Localization(__file__, 550, 16), switch_backend_291223, *[prev_backend_291224], **kwargs_291225)
            
            
            # Getting the type of 'result' (line 551)
            result_291227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 19), 'result')
            # Assigning a type to the variable 'stypy_return_type' (line 551)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'stypy_return_type', result_291227)
            
            # ################# End of 'backend_switcher(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'backend_switcher' in the type store
            # Getting the type of 'stypy_return_type' (line 542)
            stypy_return_type_291228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_291228)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'backend_switcher'
            return stypy_return_type_291228

        # Assigning a type to the variable 'backend_switcher' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'backend_switcher', backend_switcher)
        
        # Call to _copy_metadata(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'func' (line 553)
        func_291230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 30), 'func', False)
        # Getting the type of 'backend_switcher' (line 553)
        backend_switcher_291231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 36), 'backend_switcher', False)
        # Processing the call keyword arguments (line 553)
        kwargs_291232 = {}
        # Getting the type of '_copy_metadata' (line 553)
        _copy_metadata_291229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 15), '_copy_metadata', False)
        # Calling _copy_metadata(args, kwargs) (line 553)
        _copy_metadata_call_result_291233 = invoke(stypy.reporting.localization.Localization(__file__, 553, 15), _copy_metadata_291229, *[func_291230, backend_switcher_291231], **kwargs_291232)
        
        # Assigning a type to the variable 'stypy_return_type' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'stypy_return_type', _copy_metadata_call_result_291233)
        
        # ################# End of 'switch_backend_decorator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'switch_backend_decorator' in the type store
        # Getting the type of 'stypy_return_type' (line 541)
        stypy_return_type_291234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291234)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'switch_backend_decorator'
        return stypy_return_type_291234

    # Assigning a type to the variable 'switch_backend_decorator' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'switch_backend_decorator', switch_backend_decorator)
    # Getting the type of 'switch_backend_decorator' (line 554)
    switch_backend_decorator_291235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 'switch_backend_decorator')
    # Assigning a type to the variable 'stypy_return_type' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'stypy_return_type', switch_backend_decorator_291235)
    
    # ################# End of 'switch_backend(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'switch_backend' in the type store
    # Getting the type of 'stypy_return_type' (line 538)
    stypy_return_type_291236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291236)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'switch_backend'
    return stypy_return_type_291236

# Assigning a type to the variable 'switch_backend' (line 538)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 0), 'switch_backend', switch_backend)

@norecursion
def skip_if_command_unavailable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'skip_if_command_unavailable'
    module_type_store = module_type_store.open_function_context('skip_if_command_unavailable', 557, 0, False)
    
    # Passed parameters checking function
    skip_if_command_unavailable.stypy_localization = localization
    skip_if_command_unavailable.stypy_type_of_self = None
    skip_if_command_unavailable.stypy_type_store = module_type_store
    skip_if_command_unavailable.stypy_function_name = 'skip_if_command_unavailable'
    skip_if_command_unavailable.stypy_param_names_list = ['cmd']
    skip_if_command_unavailable.stypy_varargs_param_name = None
    skip_if_command_unavailable.stypy_kwargs_param_name = None
    skip_if_command_unavailable.stypy_call_defaults = defaults
    skip_if_command_unavailable.stypy_call_varargs = varargs
    skip_if_command_unavailable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'skip_if_command_unavailable', ['cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'skip_if_command_unavailable', localization, ['cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'skip_if_command_unavailable(...)' code ##################

    unicode_291237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, (-1)), 'unicode', u'\n    skips a test if a command is unavailable.\n\n    Parameters\n    ----------\n    cmd : list of str\n        must be a complete command which should not\n        return a non zero exit code, something like\n        ["latex", "-version"]\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 568, 4))
    
    # 'from matplotlib.compat.subprocess import check_output' statement (line 568)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_291238 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 568, 4), 'matplotlib.compat.subprocess')

    if (type(import_291238) is not StypyTypeError):

        if (import_291238 != 'pyd_module'):
            __import__(import_291238)
            sys_modules_291239 = sys.modules[import_291238]
            import_from_module(stypy.reporting.localization.Localization(__file__, 568, 4), 'matplotlib.compat.subprocess', sys_modules_291239.module_type_store, module_type_store, ['check_output'])
            nest_module(stypy.reporting.localization.Localization(__file__, 568, 4), __file__, sys_modules_291239, sys_modules_291239.module_type_store, module_type_store)
        else:
            from matplotlib.compat.subprocess import check_output

            import_from_module(stypy.reporting.localization.Localization(__file__, 568, 4), 'matplotlib.compat.subprocess', None, module_type_store, ['check_output'], [check_output])

    else:
        # Assigning a type to the variable 'matplotlib.compat.subprocess' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'matplotlib.compat.subprocess', import_291238)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    
    # SSA begins for try-except statement (line 569)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to check_output(...): (line 570)
    # Processing the call arguments (line 570)
    # Getting the type of 'cmd' (line 570)
    cmd_291241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 21), 'cmd', False)
    # Processing the call keyword arguments (line 570)
    kwargs_291242 = {}
    # Getting the type of 'check_output' (line 570)
    check_output_291240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'check_output', False)
    # Calling check_output(args, kwargs) (line 570)
    check_output_call_result_291243 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), check_output_291240, *[cmd_291241], **kwargs_291242)
    
    # SSA branch for the except part of a try statement (line 569)
    # SSA branch for the except '<any exception>' branch of a try statement (line 569)
    module_type_store.open_ssa_branch('except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 572, 8))
    
    # 'import pytest' statement (line 572)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_291244 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 572, 8), 'pytest')

    if (type(import_291244) is not StypyTypeError):

        if (import_291244 != 'pyd_module'):
            __import__(import_291244)
            sys_modules_291245 = sys.modules[import_291244]
            import_module(stypy.reporting.localization.Localization(__file__, 572, 8), 'pytest', sys_modules_291245.module_type_store, module_type_store)
        else:
            import pytest

            import_module(stypy.reporting.localization.Localization(__file__, 572, 8), 'pytest', pytest, module_type_store)

    else:
        # Assigning a type to the variable 'pytest' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'pytest', import_291244)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    # Call to skip(...): (line 573)
    # Processing the call keyword arguments (line 573)
    unicode_291249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 39), 'unicode', u'missing command: %s')
    
    # Obtaining the type of the subscript
    int_291250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 67), 'int')
    # Getting the type of 'cmd' (line 573)
    cmd_291251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 63), 'cmd', False)
    # Obtaining the member '__getitem__' of a type (line 573)
    getitem___291252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 63), cmd_291251, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 573)
    subscript_call_result_291253 = invoke(stypy.reporting.localization.Localization(__file__, 573, 63), getitem___291252, int_291250)
    
    # Applying the binary operator '%' (line 573)
    result_mod_291254 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 39), '%', unicode_291249, subscript_call_result_291253)
    
    keyword_291255 = result_mod_291254
    kwargs_291256 = {'reason': keyword_291255}
    # Getting the type of 'pytest' (line 573)
    pytest_291246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 15), 'pytest', False)
    # Obtaining the member 'mark' of a type (line 573)
    mark_291247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 15), pytest_291246, 'mark')
    # Obtaining the member 'skip' of a type (line 573)
    skip_291248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 15), mark_291247, 'skip')
    # Calling skip(args, kwargs) (line 573)
    skip_call_result_291257 = invoke(stypy.reporting.localization.Localization(__file__, 573, 15), skip_291248, *[], **kwargs_291256)
    
    # Assigning a type to the variable 'stypy_return_type' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'stypy_return_type', skip_call_result_291257)
    # SSA join for try-except statement (line 569)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def _stypy_temp_lambda_149(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_149'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_149', 575, 11, True)
        # Passed parameters checking function
        _stypy_temp_lambda_149.stypy_localization = localization
        _stypy_temp_lambda_149.stypy_type_of_self = None
        _stypy_temp_lambda_149.stypy_type_store = module_type_store
        _stypy_temp_lambda_149.stypy_function_name = '_stypy_temp_lambda_149'
        _stypy_temp_lambda_149.stypy_param_names_list = ['f']
        _stypy_temp_lambda_149.stypy_varargs_param_name = None
        _stypy_temp_lambda_149.stypy_kwargs_param_name = None
        _stypy_temp_lambda_149.stypy_call_defaults = defaults
        _stypy_temp_lambda_149.stypy_call_varargs = varargs
        _stypy_temp_lambda_149.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_149', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_149', ['f'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'f' (line 575)
        f_291258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 21), 'f')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), 'stypy_return_type', f_291258)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_149' in the type store
        # Getting the type of 'stypy_return_type' (line 575)
        stypy_return_type_291259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291259)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_149'
        return stypy_return_type_291259

    # Assigning a type to the variable '_stypy_temp_lambda_149' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), '_stypy_temp_lambda_149', _stypy_temp_lambda_149)
    # Getting the type of '_stypy_temp_lambda_149' (line 575)
    _stypy_temp_lambda_149_291260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), '_stypy_temp_lambda_149')
    # Assigning a type to the variable 'stypy_return_type' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'stypy_return_type', _stypy_temp_lambda_149_291260)
    
    # ################# End of 'skip_if_command_unavailable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'skip_if_command_unavailable' in the type store
    # Getting the type of 'stypy_return_type' (line 557)
    stypy_return_type_291261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291261)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'skip_if_command_unavailable'
    return stypy_return_type_291261

# Assigning a type to the variable 'skip_if_command_unavailable' (line 557)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 0), 'skip_if_command_unavailable', skip_if_command_unavailable)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
