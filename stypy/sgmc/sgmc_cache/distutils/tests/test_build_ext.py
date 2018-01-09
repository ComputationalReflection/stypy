
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import sys
2: import os
3: from StringIO import StringIO
4: import textwrap
5: 
6: from distutils.core import Extension, Distribution
7: from distutils.command.build_ext import build_ext
8: from distutils import sysconfig
9: from distutils.tests import support
10: from distutils.errors import (DistutilsSetupError, CompileError,
11:                               DistutilsPlatformError)
12: 
13: import unittest
14: from test import test_support
15: 
16: # http://bugs.python.org/issue4373
17: # Don't load the xx module more than once.
18: ALREADY_TESTED = False
19: 
20: 
21: class BuildExtTestCase(support.TempdirManager,
22:                        support.LoggingSilencer,
23:                        unittest.TestCase):
24:     def setUp(self):
25:         super(BuildExtTestCase, self).setUp()
26:         self.tmp_dir = self.mkdtemp()
27:         self.xx_created = False
28:         sys.path.append(self.tmp_dir)
29:         self.addCleanup(sys.path.remove, self.tmp_dir)
30:         if sys.version > "2.6":
31:             import site
32:             self.old_user_base = site.USER_BASE
33:             site.USER_BASE = self.mkdtemp()
34:             from distutils.command import build_ext
35:             build_ext.USER_BASE = site.USER_BASE
36: 
37:     def tearDown(self):
38:         if self.xx_created:
39:             test_support.unload('xx')
40:             # XXX on Windows the test leaves a directory
41:             # with xx module in TEMP
42:         super(BuildExtTestCase, self).tearDown()
43: 
44:     def test_build_ext(self):
45:         global ALREADY_TESTED
46:         support.copy_xxmodule_c(self.tmp_dir)
47:         self.xx_created = True
48:         xx_c = os.path.join(self.tmp_dir, 'xxmodule.c')
49:         xx_ext = Extension('xx', [xx_c])
50:         dist = Distribution({'name': 'xx', 'ext_modules': [xx_ext]})
51:         dist.package_dir = self.tmp_dir
52:         cmd = build_ext(dist)
53:         support.fixup_build_ext(cmd)
54:         cmd.build_lib = self.tmp_dir
55:         cmd.build_temp = self.tmp_dir
56: 
57:         old_stdout = sys.stdout
58:         if not test_support.verbose:
59:             # silence compiler output
60:             sys.stdout = StringIO()
61:         try:
62:             cmd.ensure_finalized()
63:             cmd.run()
64:         finally:
65:             sys.stdout = old_stdout
66: 
67:         if ALREADY_TESTED:
68:             self.skipTest('Already tested in %s' % ALREADY_TESTED)
69:         else:
70:             ALREADY_TESTED = type(self).__name__
71: 
72:         import xx
73: 
74:         for attr in ('error', 'foo', 'new', 'roj'):
75:             self.assertTrue(hasattr(xx, attr))
76: 
77:         self.assertEqual(xx.foo(2, 5), 7)
78:         self.assertEqual(xx.foo(13,15), 28)
79:         self.assertEqual(xx.new().demo(), None)
80:         if test_support.HAVE_DOCSTRINGS:
81:             doc = 'This is a template module just for instruction.'
82:             self.assertEqual(xx.__doc__, doc)
83:         self.assertIsInstance(xx.Null(), xx.Null)
84:         self.assertIsInstance(xx.Str(), xx.Str)
85: 
86:     def test_solaris_enable_shared(self):
87:         dist = Distribution({'name': 'xx'})
88:         cmd = build_ext(dist)
89:         old = sys.platform
90: 
91:         sys.platform = 'sunos' # fooling finalize_options
92:         from distutils.sysconfig import  _config_vars
93:         old_var = _config_vars.get('Py_ENABLE_SHARED')
94:         _config_vars['Py_ENABLE_SHARED'] = 1
95:         try:
96:             cmd.ensure_finalized()
97:         finally:
98:             sys.platform = old
99:             if old_var is None:
100:                 del _config_vars['Py_ENABLE_SHARED']
101:             else:
102:                 _config_vars['Py_ENABLE_SHARED'] = old_var
103: 
104:         # make sure we get some library dirs under solaris
105:         self.assertGreater(len(cmd.library_dirs), 0)
106: 
107:     @unittest.skipIf(sys.version < '2.6',
108:                      'site.USER_SITE was introduced in 2.6')
109:     def test_user_site(self):
110:         import site
111:         dist = Distribution({'name': 'xx'})
112:         cmd = build_ext(dist)
113: 
114:         # making sure the user option is there
115:         options = [name for name, short, label in
116:                    cmd.user_options]
117:         self.assertIn('user', options)
118: 
119:         # setting a value
120:         cmd.user = 1
121: 
122:         # setting user based lib and include
123:         lib = os.path.join(site.USER_BASE, 'lib')
124:         incl = os.path.join(site.USER_BASE, 'include')
125:         os.mkdir(lib)
126:         os.mkdir(incl)
127: 
128:         cmd.ensure_finalized()
129: 
130:         # see if include_dirs and library_dirs were set
131:         self.assertIn(lib, cmd.library_dirs)
132:         self.assertIn(lib, cmd.rpath)
133:         self.assertIn(incl, cmd.include_dirs)
134: 
135:     def test_finalize_options(self):
136:         # Make sure Python's include directories (for Python.h, pyconfig.h,
137:         # etc.) are in the include search path.
138:         modules = [Extension('foo', ['xxx'])]
139:         dist = Distribution({'name': 'xx', 'ext_modules': modules})
140:         cmd = build_ext(dist)
141:         cmd.finalize_options()
142: 
143:         py_include = sysconfig.get_python_inc()
144:         self.assertIn(py_include, cmd.include_dirs)
145: 
146:         plat_py_include = sysconfig.get_python_inc(plat_specific=1)
147:         self.assertIn(plat_py_include, cmd.include_dirs)
148: 
149:         # make sure cmd.libraries is turned into a list
150:         # if it's a string
151:         cmd = build_ext(dist)
152:         cmd.libraries = 'my_lib, other_lib lastlib'
153:         cmd.finalize_options()
154:         self.assertEqual(cmd.libraries, ['my_lib', 'other_lib', 'lastlib'])
155: 
156:         # make sure cmd.library_dirs is turned into a list
157:         # if it's a string
158:         cmd = build_ext(dist)
159:         cmd.library_dirs = 'my_lib_dir%sother_lib_dir' % os.pathsep
160:         cmd.finalize_options()
161:         self.assertIn('my_lib_dir', cmd.library_dirs)
162:         self.assertIn('other_lib_dir', cmd.library_dirs)
163: 
164:         # make sure rpath is turned into a list
165:         # if it's a string
166:         cmd = build_ext(dist)
167:         cmd.rpath = 'one%stwo' % os.pathsep
168:         cmd.finalize_options()
169:         self.assertEqual(cmd.rpath, ['one', 'two'])
170: 
171:         # make sure cmd.link_objects is turned into a list
172:         # if it's a string
173:         cmd = build_ext(dist)
174:         cmd.link_objects = 'one two,three'
175:         cmd.finalize_options()
176:         self.assertEqual(cmd.link_objects, ['one', 'two', 'three'])
177: 
178:         # XXX more tests to perform for win32
179: 
180:         # make sure define is turned into 2-tuples
181:         # strings if they are ','-separated strings
182:         cmd = build_ext(dist)
183:         cmd.define = 'one,two'
184:         cmd.finalize_options()
185:         self.assertEqual(cmd.define, [('one', '1'), ('two', '1')])
186: 
187:         # make sure undef is turned into a list of
188:         # strings if they are ','-separated strings
189:         cmd = build_ext(dist)
190:         cmd.undef = 'one,two'
191:         cmd.finalize_options()
192:         self.assertEqual(cmd.undef, ['one', 'two'])
193: 
194:         # make sure swig_opts is turned into a list
195:         cmd = build_ext(dist)
196:         cmd.swig_opts = None
197:         cmd.finalize_options()
198:         self.assertEqual(cmd.swig_opts, [])
199: 
200:         cmd = build_ext(dist)
201:         cmd.swig_opts = '1 2'
202:         cmd.finalize_options()
203:         self.assertEqual(cmd.swig_opts, ['1', '2'])
204: 
205:     def test_check_extensions_list(self):
206:         dist = Distribution()
207:         cmd = build_ext(dist)
208:         cmd.finalize_options()
209: 
210:         #'extensions' option must be a list of Extension instances
211:         self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, 'foo')
212: 
213:         # each element of 'ext_modules' option must be an
214:         # Extension instance or 2-tuple
215:         exts = [('bar', 'foo', 'bar'), 'foo']
216:         self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, exts)
217: 
218:         # first element of each tuple in 'ext_modules'
219:         # must be the extension name (a string) and match
220:         # a python dotted-separated name
221:         exts = [('foo-bar', '')]
222:         self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, exts)
223: 
224:         # second element of each tuple in 'ext_modules'
225:         # must be a dictionary (build info)
226:         exts = [('foo.bar', '')]
227:         self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, exts)
228: 
229:         # ok this one should pass
230:         exts = [('foo.bar', {'sources': [''], 'libraries': 'foo',
231:                              'some': 'bar'})]
232:         cmd.check_extensions_list(exts)
233:         ext = exts[0]
234:         self.assertIsInstance(ext, Extension)
235: 
236:         # check_extensions_list adds in ext the values passed
237:         # when they are in ('include_dirs', 'library_dirs', 'libraries'
238:         # 'extra_objects', 'extra_compile_args', 'extra_link_args')
239:         self.assertEqual(ext.libraries, 'foo')
240:         self.assertFalse(hasattr(ext, 'some'))
241: 
242:         # 'macros' element of build info dict must be 1- or 2-tuple
243:         exts = [('foo.bar', {'sources': [''], 'libraries': 'foo',
244:                 'some': 'bar', 'macros': [('1', '2', '3'), 'foo']})]
245:         self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, exts)
246: 
247:         exts[0][1]['macros'] = [('1', '2'), ('3',)]
248:         cmd.check_extensions_list(exts)
249:         self.assertEqual(exts[0].undef_macros, ['3'])
250:         self.assertEqual(exts[0].define_macros, [('1', '2')])
251: 
252:     def test_get_source_files(self):
253:         modules = [Extension('foo', ['xxx'])]
254:         dist = Distribution({'name': 'xx', 'ext_modules': modules})
255:         cmd = build_ext(dist)
256:         cmd.ensure_finalized()
257:         self.assertEqual(cmd.get_source_files(), ['xxx'])
258: 
259:     def test_compiler_option(self):
260:         # cmd.compiler is an option and
261:         # should not be overridden by a compiler instance
262:         # when the command is run
263:         dist = Distribution()
264:         cmd = build_ext(dist)
265:         cmd.compiler = 'unix'
266:         cmd.ensure_finalized()
267:         cmd.run()
268:         self.assertEqual(cmd.compiler, 'unix')
269: 
270:     def test_get_outputs(self):
271:         tmp_dir = self.mkdtemp()
272:         c_file = os.path.join(tmp_dir, 'foo.c')
273:         self.write_file(c_file, 'void initfoo(void) {};\n')
274:         ext = Extension('foo', [c_file])
275:         dist = Distribution({'name': 'xx',
276:                              'ext_modules': [ext]})
277:         cmd = build_ext(dist)
278:         support.fixup_build_ext(cmd)
279:         cmd.ensure_finalized()
280:         self.assertEqual(len(cmd.get_outputs()), 1)
281: 
282:         cmd.build_lib = os.path.join(self.tmp_dir, 'build')
283:         cmd.build_temp = os.path.join(self.tmp_dir, 'tempt')
284: 
285:         # issue #5977 : distutils build_ext.get_outputs
286:         # returns wrong result with --inplace
287:         other_tmp_dir = os.path.realpath(self.mkdtemp())
288:         old_wd = os.getcwd()
289:         os.chdir(other_tmp_dir)
290:         try:
291:             cmd.inplace = 1
292:             cmd.run()
293:             so_file = cmd.get_outputs()[0]
294:         finally:
295:             os.chdir(old_wd)
296:         self.assertTrue(os.path.exists(so_file))
297:         self.assertEqual(os.path.splitext(so_file)[-1],
298:                          sysconfig.get_config_var('SO'))
299:         so_dir = os.path.dirname(so_file)
300:         self.assertEqual(so_dir, other_tmp_dir)
301:         cmd.compiler = None
302:         cmd.inplace = 0
303:         cmd.run()
304:         so_file = cmd.get_outputs()[0]
305:         self.assertTrue(os.path.exists(so_file))
306:         self.assertEqual(os.path.splitext(so_file)[-1],
307:                          sysconfig.get_config_var('SO'))
308:         so_dir = os.path.dirname(so_file)
309:         self.assertEqual(so_dir, cmd.build_lib)
310: 
311:         # inplace = 0, cmd.package = 'bar'
312:         build_py = cmd.get_finalized_command('build_py')
313:         build_py.package_dir = {'': 'bar'}
314:         path = cmd.get_ext_fullpath('foo')
315:         # checking that the last directory is the build_dir
316:         path = os.path.split(path)[0]
317:         self.assertEqual(path, cmd.build_lib)
318: 
319:         # inplace = 1, cmd.package = 'bar'
320:         cmd.inplace = 1
321:         other_tmp_dir = os.path.realpath(self.mkdtemp())
322:         old_wd = os.getcwd()
323:         os.chdir(other_tmp_dir)
324:         try:
325:             path = cmd.get_ext_fullpath('foo')
326:         finally:
327:             os.chdir(old_wd)
328:         # checking that the last directory is bar
329:         path = os.path.split(path)[0]
330:         lastdir = os.path.split(path)[-1]
331:         self.assertEqual(lastdir, 'bar')
332: 
333:     def test_ext_fullpath(self):
334:         ext = sysconfig.get_config_vars()['SO']
335:         dist = Distribution()
336:         cmd = build_ext(dist)
337:         cmd.inplace = 1
338:         cmd.distribution.package_dir = {'': 'src'}
339:         cmd.distribution.packages = ['lxml', 'lxml.html']
340:         curdir = os.getcwd()
341:         wanted = os.path.join(curdir, 'src', 'lxml', 'etree' + ext)
342:         path = cmd.get_ext_fullpath('lxml.etree')
343:         self.assertEqual(wanted, path)
344: 
345:         # building lxml.etree not inplace
346:         cmd.inplace = 0
347:         cmd.build_lib = os.path.join(curdir, 'tmpdir')
348:         wanted = os.path.join(curdir, 'tmpdir', 'lxml', 'etree' + ext)
349:         path = cmd.get_ext_fullpath('lxml.etree')
350:         self.assertEqual(wanted, path)
351: 
352:         # building twisted.runner.portmap not inplace
353:         build_py = cmd.get_finalized_command('build_py')
354:         build_py.package_dir = {}
355:         cmd.distribution.packages = ['twisted', 'twisted.runner.portmap']
356:         path = cmd.get_ext_fullpath('twisted.runner.portmap')
357:         wanted = os.path.join(curdir, 'tmpdir', 'twisted', 'runner',
358:                               'portmap' + ext)
359:         self.assertEqual(wanted, path)
360: 
361:         # building twisted.runner.portmap inplace
362:         cmd.inplace = 1
363:         path = cmd.get_ext_fullpath('twisted.runner.portmap')
364:         wanted = os.path.join(curdir, 'twisted', 'runner', 'portmap' + ext)
365:         self.assertEqual(wanted, path)
366: 
367:     def test_build_ext_inplace(self):
368:         etree_c = os.path.join(self.tmp_dir, 'lxml.etree.c')
369:         etree_ext = Extension('lxml.etree', [etree_c])
370:         dist = Distribution({'name': 'lxml', 'ext_modules': [etree_ext]})
371:         cmd = build_ext(dist)
372:         cmd.ensure_finalized()
373:         cmd.inplace = 1
374:         cmd.distribution.package_dir = {'': 'src'}
375:         cmd.distribution.packages = ['lxml', 'lxml.html']
376:         curdir = os.getcwd()
377:         ext = sysconfig.get_config_var("SO")
378:         wanted = os.path.join(curdir, 'src', 'lxml', 'etree' + ext)
379:         path = cmd.get_ext_fullpath('lxml.etree')
380:         self.assertEqual(wanted, path)
381: 
382:     def test_setuptools_compat(self):
383:         import distutils.core, distutils.extension, distutils.command.build_ext
384:         saved_ext = distutils.extension.Extension
385:         try:
386:             # on some platforms, it loads the deprecated "dl" module
387:             test_support.import_module('setuptools_build_ext', deprecated=True)
388: 
389:             # theses import patch Distutils' Extension class
390:             from setuptools_build_ext import build_ext as setuptools_build_ext
391:             from setuptools_extension import Extension
392: 
393:             etree_c = os.path.join(self.tmp_dir, 'lxml.etree.c')
394:             etree_ext = Extension('lxml.etree', [etree_c])
395:             dist = Distribution({'name': 'lxml', 'ext_modules': [etree_ext]})
396:             cmd = setuptools_build_ext(dist)
397:             cmd.ensure_finalized()
398:             cmd.inplace = 1
399:             cmd.distribution.package_dir = {'': 'src'}
400:             cmd.distribution.packages = ['lxml', 'lxml.html']
401:             curdir = os.getcwd()
402:             ext = sysconfig.get_config_var("SO")
403:             wanted = os.path.join(curdir, 'src', 'lxml', 'etree' + ext)
404:             path = cmd.get_ext_fullpath('lxml.etree')
405:             self.assertEqual(wanted, path)
406:         finally:
407:             # restoring Distutils' Extension class otherwise its broken
408:             distutils.extension.Extension = saved_ext
409:             distutils.core.Extension = saved_ext
410:             distutils.command.build_ext.Extension = saved_ext
411: 
412:     def test_build_ext_path_with_os_sep(self):
413:         dist = Distribution({'name': 'UpdateManager'})
414:         cmd = build_ext(dist)
415:         cmd.ensure_finalized()
416:         ext = sysconfig.get_config_var("SO")
417:         ext_name = os.path.join('UpdateManager', 'fdsend')
418:         ext_path = cmd.get_ext_fullpath(ext_name)
419:         wanted = os.path.join(cmd.build_lib, 'UpdateManager', 'fdsend' + ext)
420:         self.assertEqual(ext_path, wanted)
421: 
422:     @unittest.skipUnless(sys.platform == 'win32', 'these tests require Windows')
423:     def test_build_ext_path_cross_platform(self):
424:         dist = Distribution({'name': 'UpdateManager'})
425:         cmd = build_ext(dist)
426:         cmd.ensure_finalized()
427:         ext = sysconfig.get_config_var("SO")
428:         # this needs to work even under win32
429:         ext_name = 'UpdateManager/fdsend'
430:         ext_path = cmd.get_ext_fullpath(ext_name)
431:         wanted = os.path.join(cmd.build_lib, 'UpdateManager', 'fdsend' + ext)
432:         self.assertEqual(ext_path, wanted)
433: 
434:     @unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for MacOSX')
435:     def test_deployment_target_default(self):
436:         # Issue 9516: Test that, in the absence of the environment variable,
437:         # an extension module is compiled with the same deployment target as
438:         #  the interpreter.
439:         self._try_compile_deployment_target('==', None)
440: 
441:     @unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for MacOSX')
442:     def test_deployment_target_too_low(self):
443:         # Issue 9516: Test that an extension module is not allowed to be
444:         # compiled with a deployment target less than that of the interpreter.
445:         self.assertRaises(DistutilsPlatformError,
446:             self._try_compile_deployment_target, '>', '10.1')
447: 
448:     @unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for MacOSX')
449:     def test_deployment_target_higher_ok(self):
450:         # Issue 9516: Test that an extension module can be compiled with a
451:         # deployment target higher than that of the interpreter: the ext
452:         # module may depend on some newer OS feature.
453:         deptarget = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
454:         if deptarget:
455:             # increment the minor version number (i.e. 10.6 -> 10.7)
456:             deptarget = [int(x) for x in deptarget.split('.')]
457:             deptarget[-1] += 1
458:             deptarget = '.'.join(str(i) for i in deptarget)
459:             self._try_compile_deployment_target('<', deptarget)
460: 
461:     def _try_compile_deployment_target(self, operator, target):
462:         orig_environ = os.environ
463:         os.environ = orig_environ.copy()
464:         self.addCleanup(setattr, os, 'environ', orig_environ)
465: 
466:         if target is None:
467:             if os.environ.get('MACOSX_DEPLOYMENT_TARGET'):
468:                 del os.environ['MACOSX_DEPLOYMENT_TARGET']
469:         else:
470:             os.environ['MACOSX_DEPLOYMENT_TARGET'] = target
471: 
472:         deptarget_c = os.path.join(self.tmp_dir, 'deptargetmodule.c')
473: 
474:         with open(deptarget_c, 'w') as fp:
475:             fp.write(textwrap.dedent('''\
476:                 #include <AvailabilityMacros.h>
477: 
478:                 int dummy;
479: 
480:                 #if TARGET %s MAC_OS_X_VERSION_MIN_REQUIRED
481:                 #else
482:                 #error "Unexpected target"
483:                 #endif
484: 
485:             ''' % operator))
486: 
487:         # get the deployment target that the interpreter was built with
488:         target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
489:         target = tuple(map(int, target.split('.')[0:2]))
490:         # format the target value as defined in the Apple
491:         # Availability Macros.  We can't use the macro names since
492:         # at least one value we test with will not exist yet.
493:         if target[1] < 10:
494:             # for 10.1 through 10.9.x -> "10n0"
495:             target = '%02d%01d0' % target
496:         else:
497:             # for 10.10 and beyond -> "10nn00"
498:             target = '%02d%02d00' % target
499:         deptarget_ext = Extension(
500:             'deptarget',
501:             [deptarget_c],
502:             extra_compile_args=['-DTARGET=%s'%(target,)],
503:         )
504:         dist = Distribution({
505:             'name': 'deptarget',
506:             'ext_modules': [deptarget_ext]
507:         })
508:         dist.package_dir = self.tmp_dir
509:         cmd = build_ext(dist)
510:         cmd.build_lib = self.tmp_dir
511:         cmd.build_temp = self.tmp_dir
512: 
513:         try:
514:             cmd.ensure_finalized()
515:             cmd.run()
516:         except CompileError:
517:             self.fail("Wrong deployment target during compilation")
518: 
519: def test_suite():
520:     return unittest.makeSuite(BuildExtTestCase)
521: 
522: if __name__ == '__main__':
523:     test_support.run_unittest(test_suite())
524: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import sys' statement (line 1)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from StringIO import StringIO' statement (line 3)
try:
    from StringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'StringIO', None, module_type_store, ['StringIO'], [StringIO])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import textwrap' statement (line 4)
import textwrap

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'textwrap', textwrap, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.core import Extension, Distribution' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_31558 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.core')

if (type(import_31558) is not StypyTypeError):

    if (import_31558 != 'pyd_module'):
        __import__(import_31558)
        sys_modules_31559 = sys.modules[import_31558]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.core', sys_modules_31559.module_type_store, module_type_store, ['Extension', 'Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_31559, sys_modules_31559.module_type_store, module_type_store)
    else:
        from distutils.core import Extension, Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.core', None, module_type_store, ['Extension', 'Distribution'], [Extension, Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.core', import_31558)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.command.build_ext import build_ext' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_31560 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.build_ext')

if (type(import_31560) is not StypyTypeError):

    if (import_31560 != 'pyd_module'):
        __import__(import_31560)
        sys_modules_31561 = sys.modules[import_31560]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.build_ext', sys_modules_31561.module_type_store, module_type_store, ['build_ext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_31561, sys_modules_31561.module_type_store, module_type_store)
    else:
        from distutils.command.build_ext import build_ext

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.build_ext', None, module_type_store, ['build_ext'], [build_ext])

else:
    # Assigning a type to the variable 'distutils.command.build_ext' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.command.build_ext', import_31560)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils import sysconfig' statement (line 8)
try:
    from distutils import sysconfig

except:
    sysconfig = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.tests import support' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_31562 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.tests')

if (type(import_31562) is not StypyTypeError):

    if (import_31562 != 'pyd_module'):
        __import__(import_31562)
        sys_modules_31563 = sys.modules[import_31562]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.tests', sys_modules_31563.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_31563, sys_modules_31563.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.tests', import_31562)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.errors import DistutilsSetupError, CompileError, DistutilsPlatformError' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_31564 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors')

if (type(import_31564) is not StypyTypeError):

    if (import_31564 != 'pyd_module'):
        __import__(import_31564)
        sys_modules_31565 = sys.modules[import_31564]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', sys_modules_31565.module_type_store, module_type_store, ['DistutilsSetupError', 'CompileError', 'DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_31565, sys_modules_31565.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsSetupError, CompileError, DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', None, module_type_store, ['DistutilsSetupError', 'CompileError', 'DistutilsPlatformError'], [DistutilsSetupError, CompileError, DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.errors', import_31564)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import unittest' statement (line 13)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from test import test_support' statement (line 14)
try:
    from test import test_support

except:
    test_support = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'test', None, module_type_store, ['test_support'], [test_support])


# Assigning a Name to a Name (line 18):
# Getting the type of 'False' (line 18)
False_31566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'False')
# Assigning a type to the variable 'ALREADY_TESTED' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'ALREADY_TESTED', False_31566)
# Declaration of the 'BuildExtTestCase' class
# Getting the type of 'support' (line 21)
support_31567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'support')
# Obtaining the member 'TempdirManager' of a type (line 21)
TempdirManager_31568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 23), support_31567, 'TempdirManager')
# Getting the type of 'support' (line 22)
support_31569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 22)
LoggingSilencer_31570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), support_31569, 'LoggingSilencer')
# Getting the type of 'unittest' (line 23)
unittest_31571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'unittest')
# Obtaining the member 'TestCase' of a type (line 23)
TestCase_31572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 23), unittest_31571, 'TestCase')

class BuildExtTestCase(TempdirManager_31568, LoggingSilencer_31570, TestCase_31572, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.setUp')
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 25)
        # Processing the call keyword arguments (line 25)
        kwargs_31579 = {}
        
        # Call to super(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'BuildExtTestCase' (line 25)
        BuildExtTestCase_31574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'BuildExtTestCase', False)
        # Getting the type of 'self' (line 25)
        self_31575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'self', False)
        # Processing the call keyword arguments (line 25)
        kwargs_31576 = {}
        # Getting the type of 'super' (line 25)
        super_31573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'super', False)
        # Calling super(args, kwargs) (line 25)
        super_call_result_31577 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), super_31573, *[BuildExtTestCase_31574, self_31575], **kwargs_31576)
        
        # Obtaining the member 'setUp' of a type (line 25)
        setUp_31578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), super_call_result_31577, 'setUp')
        # Calling setUp(args, kwargs) (line 25)
        setUp_call_result_31580 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), setUp_31578, *[], **kwargs_31579)
        
        
        # Assigning a Call to a Attribute (line 26):
        
        # Call to mkdtemp(...): (line 26)
        # Processing the call keyword arguments (line 26)
        kwargs_31583 = {}
        # Getting the type of 'self' (line 26)
        self_31581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 26)
        mkdtemp_31582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 23), self_31581, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 26)
        mkdtemp_call_result_31584 = invoke(stypy.reporting.localization.Localization(__file__, 26, 23), mkdtemp_31582, *[], **kwargs_31583)
        
        # Getting the type of 'self' (line 26)
        self_31585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'tmp_dir' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_31585, 'tmp_dir', mkdtemp_call_result_31584)
        
        # Assigning a Name to a Attribute (line 27):
        # Getting the type of 'False' (line 27)
        False_31586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'False')
        # Getting the type of 'self' (line 27)
        self_31587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'xx_created' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_31587, 'xx_created', False_31586)
        
        # Call to append(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_31591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 28)
        tmp_dir_31592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), self_31591, 'tmp_dir')
        # Processing the call keyword arguments (line 28)
        kwargs_31593 = {}
        # Getting the type of 'sys' (line 28)
        sys_31588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'sys', False)
        # Obtaining the member 'path' of a type (line 28)
        path_31589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), sys_31588, 'path')
        # Obtaining the member 'append' of a type (line 28)
        append_31590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), path_31589, 'append')
        # Calling append(args, kwargs) (line 28)
        append_call_result_31594 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), append_31590, *[tmp_dir_31592], **kwargs_31593)
        
        
        # Call to addCleanup(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'sys' (line 29)
        sys_31597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'sys', False)
        # Obtaining the member 'path' of a type (line 29)
        path_31598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 24), sys_31597, 'path')
        # Obtaining the member 'remove' of a type (line 29)
        remove_31599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 24), path_31598, 'remove')
        # Getting the type of 'self' (line 29)
        self_31600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 41), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 29)
        tmp_dir_31601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 41), self_31600, 'tmp_dir')
        # Processing the call keyword arguments (line 29)
        kwargs_31602 = {}
        # Getting the type of 'self' (line 29)
        self_31595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 29)
        addCleanup_31596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_31595, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 29)
        addCleanup_call_result_31603 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), addCleanup_31596, *[remove_31599, tmp_dir_31601], **kwargs_31602)
        
        
        
        # Getting the type of 'sys' (line 30)
        sys_31604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'sys')
        # Obtaining the member 'version' of a type (line 30)
        version_31605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), sys_31604, 'version')
        str_31606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'str', '2.6')
        # Applying the binary operator '>' (line 30)
        result_gt_31607 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), '>', version_31605, str_31606)
        
        # Testing the type of an if condition (line 30)
        if_condition_31608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 8), result_gt_31607)
        # Assigning a type to the variable 'if_condition_31608' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'if_condition_31608', if_condition_31608)
        # SSA begins for if statement (line 30)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 12))
        
        # 'import site' statement (line 31)
        import site

        import_module(stypy.reporting.localization.Localization(__file__, 31, 12), 'site', site, module_type_store)
        
        
        # Assigning a Attribute to a Attribute (line 32):
        # Getting the type of 'site' (line 32)
        site_31609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'site')
        # Obtaining the member 'USER_BASE' of a type (line 32)
        USER_BASE_31610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 33), site_31609, 'USER_BASE')
        # Getting the type of 'self' (line 32)
        self_31611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'self')
        # Setting the type of the member 'old_user_base' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), self_31611, 'old_user_base', USER_BASE_31610)
        
        # Assigning a Call to a Attribute (line 33):
        
        # Call to mkdtemp(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_31614 = {}
        # Getting the type of 'self' (line 33)
        self_31612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 33)
        mkdtemp_31613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 29), self_31612, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 33)
        mkdtemp_call_result_31615 = invoke(stypy.reporting.localization.Localization(__file__, 33, 29), mkdtemp_31613, *[], **kwargs_31614)
        
        # Getting the type of 'site' (line 33)
        site_31616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'site')
        # Setting the type of the member 'USER_BASE' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), site_31616, 'USER_BASE', mkdtemp_call_result_31615)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 12))
        
        # 'from distutils.command import build_ext' statement (line 34)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_31617 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 12), 'distutils.command')

        if (type(import_31617) is not StypyTypeError):

            if (import_31617 != 'pyd_module'):
                __import__(import_31617)
                sys_modules_31618 = sys.modules[import_31617]
                import_from_module(stypy.reporting.localization.Localization(__file__, 34, 12), 'distutils.command', sys_modules_31618.module_type_store, module_type_store, ['build_ext'])
                nest_module(stypy.reporting.localization.Localization(__file__, 34, 12), __file__, sys_modules_31618, sys_modules_31618.module_type_store, module_type_store)
            else:
                from distutils.command import build_ext

                import_from_module(stypy.reporting.localization.Localization(__file__, 34, 12), 'distutils.command', None, module_type_store, ['build_ext'], [build_ext])

        else:
            # Assigning a type to the variable 'distutils.command' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'distutils.command', import_31617)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Assigning a Attribute to a Attribute (line 35):
        # Getting the type of 'site' (line 35)
        site_31619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'site')
        # Obtaining the member 'USER_BASE' of a type (line 35)
        USER_BASE_31620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 34), site_31619, 'USER_BASE')
        # Getting the type of 'build_ext' (line 35)
        build_ext_31621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'build_ext')
        # Setting the type of the member 'USER_BASE' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), build_ext_31621, 'USER_BASE', USER_BASE_31620)
        # SSA join for if statement (line 30)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_31622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_31622


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.tearDown')
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 38)
        self_31623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'self')
        # Obtaining the member 'xx_created' of a type (line 38)
        xx_created_31624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), self_31623, 'xx_created')
        # Testing the type of an if condition (line 38)
        if_condition_31625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), xx_created_31624)
        # Assigning a type to the variable 'if_condition_31625' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_31625', if_condition_31625)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to unload(...): (line 39)
        # Processing the call arguments (line 39)
        str_31628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 32), 'str', 'xx')
        # Processing the call keyword arguments (line 39)
        kwargs_31629 = {}
        # Getting the type of 'test_support' (line 39)
        test_support_31626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'test_support', False)
        # Obtaining the member 'unload' of a type (line 39)
        unload_31627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), test_support_31626, 'unload')
        # Calling unload(args, kwargs) (line 39)
        unload_call_result_31630 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), unload_31627, *[str_31628], **kwargs_31629)
        
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to tearDown(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_31637 = {}
        
        # Call to super(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'BuildExtTestCase' (line 42)
        BuildExtTestCase_31632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'BuildExtTestCase', False)
        # Getting the type of 'self' (line 42)
        self_31633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 'self', False)
        # Processing the call keyword arguments (line 42)
        kwargs_31634 = {}
        # Getting the type of 'super' (line 42)
        super_31631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'super', False)
        # Calling super(args, kwargs) (line 42)
        super_call_result_31635 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), super_31631, *[BuildExtTestCase_31632, self_31633], **kwargs_31634)
        
        # Obtaining the member 'tearDown' of a type (line 42)
        tearDown_31636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), super_call_result_31635, 'tearDown')
        # Calling tearDown(args, kwargs) (line 42)
        tearDown_call_result_31638 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), tearDown_31636, *[], **kwargs_31637)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_31639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31639)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_31639


    @norecursion
    def test_build_ext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_build_ext'
        module_type_store = module_type_store.open_function_context('test_build_ext', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_build_ext')
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_build_ext.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_build_ext', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_build_ext', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_build_ext(...)' code ##################

        # Marking variables as global (line 45)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 45, 8), 'ALREADY_TESTED')
        
        # Call to copy_xxmodule_c(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_31642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 46)
        tmp_dir_31643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 32), self_31642, 'tmp_dir')
        # Processing the call keyword arguments (line 46)
        kwargs_31644 = {}
        # Getting the type of 'support' (line 46)
        support_31640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'support', False)
        # Obtaining the member 'copy_xxmodule_c' of a type (line 46)
        copy_xxmodule_c_31641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), support_31640, 'copy_xxmodule_c')
        # Calling copy_xxmodule_c(args, kwargs) (line 46)
        copy_xxmodule_c_call_result_31645 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), copy_xxmodule_c_31641, *[tmp_dir_31643], **kwargs_31644)
        
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'True' (line 47)
        True_31646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'True')
        # Getting the type of 'self' (line 47)
        self_31647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'xx_created' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_31647, 'xx_created', True_31646)
        
        # Assigning a Call to a Name (line 48):
        
        # Call to join(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'self' (line 48)
        self_31651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 48)
        tmp_dir_31652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 28), self_31651, 'tmp_dir')
        str_31653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 42), 'str', 'xxmodule.c')
        # Processing the call keyword arguments (line 48)
        kwargs_31654 = {}
        # Getting the type of 'os' (line 48)
        os_31648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 48)
        path_31649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), os_31648, 'path')
        # Obtaining the member 'join' of a type (line 48)
        join_31650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), path_31649, 'join')
        # Calling join(args, kwargs) (line 48)
        join_call_result_31655 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), join_31650, *[tmp_dir_31652, str_31653], **kwargs_31654)
        
        # Assigning a type to the variable 'xx_c' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'xx_c', join_call_result_31655)
        
        # Assigning a Call to a Name (line 49):
        
        # Call to Extension(...): (line 49)
        # Processing the call arguments (line 49)
        str_31657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 27), 'str', 'xx')
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_31658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        # Getting the type of 'xx_c' (line 49)
        xx_c_31659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'xx_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 33), list_31658, xx_c_31659)
        
        # Processing the call keyword arguments (line 49)
        kwargs_31660 = {}
        # Getting the type of 'Extension' (line 49)
        Extension_31656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 17), 'Extension', False)
        # Calling Extension(args, kwargs) (line 49)
        Extension_call_result_31661 = invoke(stypy.reporting.localization.Localization(__file__, 49, 17), Extension_31656, *[str_31657, list_31658], **kwargs_31660)
        
        # Assigning a type to the variable 'xx_ext' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'xx_ext', Extension_call_result_31661)
        
        # Assigning a Call to a Name (line 50):
        
        # Call to Distribution(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Obtaining an instance of the builtin type 'dict' (line 50)
        dict_31663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 50)
        # Adding element type (key, value) (line 50)
        str_31664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'str', 'name')
        str_31665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 37), 'str', 'xx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 28), dict_31663, (str_31664, str_31665))
        # Adding element type (key, value) (line 50)
        str_31666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 43), 'str', 'ext_modules')
        
        # Obtaining an instance of the builtin type 'list' (line 50)
        list_31667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 50)
        # Adding element type (line 50)
        # Getting the type of 'xx_ext' (line 50)
        xx_ext_31668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 59), 'xx_ext', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 58), list_31667, xx_ext_31668)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 28), dict_31663, (str_31666, list_31667))
        
        # Processing the call keyword arguments (line 50)
        kwargs_31669 = {}
        # Getting the type of 'Distribution' (line 50)
        Distribution_31662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 50)
        Distribution_call_result_31670 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), Distribution_31662, *[dict_31663], **kwargs_31669)
        
        # Assigning a type to the variable 'dist' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'dist', Distribution_call_result_31670)
        
        # Assigning a Attribute to a Attribute (line 51):
        # Getting the type of 'self' (line 51)
        self_31671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 27), 'self')
        # Obtaining the member 'tmp_dir' of a type (line 51)
        tmp_dir_31672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 27), self_31671, 'tmp_dir')
        # Getting the type of 'dist' (line 51)
        dist_31673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'dist')
        # Setting the type of the member 'package_dir' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), dist_31673, 'package_dir', tmp_dir_31672)
        
        # Assigning a Call to a Name (line 52):
        
        # Call to build_ext(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'dist' (line 52)
        dist_31675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 24), 'dist', False)
        # Processing the call keyword arguments (line 52)
        kwargs_31676 = {}
        # Getting the type of 'build_ext' (line 52)
        build_ext_31674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 52)
        build_ext_call_result_31677 = invoke(stypy.reporting.localization.Localization(__file__, 52, 14), build_ext_31674, *[dist_31675], **kwargs_31676)
        
        # Assigning a type to the variable 'cmd' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'cmd', build_ext_call_result_31677)
        
        # Call to fixup_build_ext(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'cmd' (line 53)
        cmd_31680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'cmd', False)
        # Processing the call keyword arguments (line 53)
        kwargs_31681 = {}
        # Getting the type of 'support' (line 53)
        support_31678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'support', False)
        # Obtaining the member 'fixup_build_ext' of a type (line 53)
        fixup_build_ext_31679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), support_31678, 'fixup_build_ext')
        # Calling fixup_build_ext(args, kwargs) (line 53)
        fixup_build_ext_call_result_31682 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), fixup_build_ext_31679, *[cmd_31680], **kwargs_31681)
        
        
        # Assigning a Attribute to a Attribute (line 54):
        # Getting the type of 'self' (line 54)
        self_31683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'self')
        # Obtaining the member 'tmp_dir' of a type (line 54)
        tmp_dir_31684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), self_31683, 'tmp_dir')
        # Getting the type of 'cmd' (line 54)
        cmd_31685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'cmd')
        # Setting the type of the member 'build_lib' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), cmd_31685, 'build_lib', tmp_dir_31684)
        
        # Assigning a Attribute to a Attribute (line 55):
        # Getting the type of 'self' (line 55)
        self_31686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'self')
        # Obtaining the member 'tmp_dir' of a type (line 55)
        tmp_dir_31687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 25), self_31686, 'tmp_dir')
        # Getting the type of 'cmd' (line 55)
        cmd_31688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'cmd')
        # Setting the type of the member 'build_temp' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), cmd_31688, 'build_temp', tmp_dir_31687)
        
        # Assigning a Attribute to a Name (line 57):
        # Getting the type of 'sys' (line 57)
        sys_31689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'sys')
        # Obtaining the member 'stdout' of a type (line 57)
        stdout_31690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 21), sys_31689, 'stdout')
        # Assigning a type to the variable 'old_stdout' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'old_stdout', stdout_31690)
        
        
        # Getting the type of 'test_support' (line 58)
        test_support_31691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'test_support')
        # Obtaining the member 'verbose' of a type (line 58)
        verbose_31692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 15), test_support_31691, 'verbose')
        # Applying the 'not' unary operator (line 58)
        result_not__31693 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 11), 'not', verbose_31692)
        
        # Testing the type of an if condition (line 58)
        if_condition_31694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), result_not__31693)
        # Assigning a type to the variable 'if_condition_31694' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_31694', if_condition_31694)
        # SSA begins for if statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 60):
        
        # Call to StringIO(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_31696 = {}
        # Getting the type of 'StringIO' (line 60)
        StringIO_31695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 60)
        StringIO_call_result_31697 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), StringIO_31695, *[], **kwargs_31696)
        
        # Getting the type of 'sys' (line 60)
        sys_31698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'sys')
        # Setting the type of the member 'stdout' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), sys_31698, 'stdout', StringIO_call_result_31697)
        # SSA join for if statement (line 58)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Try-finally block (line 61)
        
        # Call to ensure_finalized(...): (line 62)
        # Processing the call keyword arguments (line 62)
        kwargs_31701 = {}
        # Getting the type of 'cmd' (line 62)
        cmd_31699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 62)
        ensure_finalized_31700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), cmd_31699, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 62)
        ensure_finalized_call_result_31702 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), ensure_finalized_31700, *[], **kwargs_31701)
        
        
        # Call to run(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_31705 = {}
        # Getting the type of 'cmd' (line 63)
        cmd_31703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 63)
        run_31704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), cmd_31703, 'run')
        # Calling run(args, kwargs) (line 63)
        run_call_result_31706 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), run_31704, *[], **kwargs_31705)
        
        
        # finally branch of the try-finally block (line 61)
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'old_stdout' (line 65)
        old_stdout_31707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'old_stdout')
        # Getting the type of 'sys' (line 65)
        sys_31708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'sys')
        # Setting the type of the member 'stdout' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), sys_31708, 'stdout', old_stdout_31707)
        
        
        # Getting the type of 'ALREADY_TESTED' (line 67)
        ALREADY_TESTED_31709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'ALREADY_TESTED')
        # Testing the type of an if condition (line 67)
        if_condition_31710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), ALREADY_TESTED_31709)
        # Assigning a type to the variable 'if_condition_31710' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_31710', if_condition_31710)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to skipTest(...): (line 68)
        # Processing the call arguments (line 68)
        str_31713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 26), 'str', 'Already tested in %s')
        # Getting the type of 'ALREADY_TESTED' (line 68)
        ALREADY_TESTED_31714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 51), 'ALREADY_TESTED', False)
        # Applying the binary operator '%' (line 68)
        result_mod_31715 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 26), '%', str_31713, ALREADY_TESTED_31714)
        
        # Processing the call keyword arguments (line 68)
        kwargs_31716 = {}
        # Getting the type of 'self' (line 68)
        self_31711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self', False)
        # Obtaining the member 'skipTest' of a type (line 68)
        skipTest_31712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), self_31711, 'skipTest')
        # Calling skipTest(args, kwargs) (line 68)
        skipTest_call_result_31717 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), skipTest_31712, *[result_mod_31715], **kwargs_31716)
        
        # SSA branch for the else part of an if statement (line 67)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 70):
        
        # Call to type(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'self' (line 70)
        self_31719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 34), 'self', False)
        # Processing the call keyword arguments (line 70)
        kwargs_31720 = {}
        # Getting the type of 'type' (line 70)
        type_31718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 29), 'type', False)
        # Calling type(args, kwargs) (line 70)
        type_call_result_31721 = invoke(stypy.reporting.localization.Localization(__file__, 70, 29), type_31718, *[self_31719], **kwargs_31720)
        
        # Obtaining the member '__name__' of a type (line 70)
        name___31722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 29), type_call_result_31721, '__name__')
        # Assigning a type to the variable 'ALREADY_TESTED' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'ALREADY_TESTED', name___31722)
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 72, 8))
        
        # 'import xx' statement (line 72)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_31723 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 72, 8), 'xx')

        if (type(import_31723) is not StypyTypeError):

            if (import_31723 != 'pyd_module'):
                __import__(import_31723)
                sys_modules_31724 = sys.modules[import_31723]
                import_module(stypy.reporting.localization.Localization(__file__, 72, 8), 'xx', sys_modules_31724.module_type_store, module_type_store)
            else:
                import xx

                import_module(stypy.reporting.localization.Localization(__file__, 72, 8), 'xx', xx, module_type_store)

        else:
            # Assigning a type to the variable 'xx' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'xx', import_31723)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 74)
        tuple_31725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 74)
        # Adding element type (line 74)
        str_31726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'str', 'error')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), tuple_31725, str_31726)
        # Adding element type (line 74)
        str_31727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), tuple_31725, str_31727)
        # Adding element type (line 74)
        str_31728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 37), 'str', 'new')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), tuple_31725, str_31728)
        # Adding element type (line 74)
        str_31729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 44), 'str', 'roj')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 21), tuple_31725, str_31729)
        
        # Testing the type of a for loop iterable (line 74)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 8), tuple_31725)
        # Getting the type of the for loop variable (line 74)
        for_loop_var_31730 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 8), tuple_31725)
        # Assigning a type to the variable 'attr' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'attr', for_loop_var_31730)
        # SSA begins for a for statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertTrue(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Call to hasattr(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'xx' (line 75)
        xx_31734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'xx', False)
        # Getting the type of 'attr' (line 75)
        attr_31735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'attr', False)
        # Processing the call keyword arguments (line 75)
        kwargs_31736 = {}
        # Getting the type of 'hasattr' (line 75)
        hasattr_31733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 28), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 75)
        hasattr_call_result_31737 = invoke(stypy.reporting.localization.Localization(__file__, 75, 28), hasattr_31733, *[xx_31734, attr_31735], **kwargs_31736)
        
        # Processing the call keyword arguments (line 75)
        kwargs_31738 = {}
        # Getting the type of 'self' (line 75)
        self_31731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 75)
        assertTrue_31732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), self_31731, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 75)
        assertTrue_call_result_31739 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), assertTrue_31732, *[hasattr_call_result_31737], **kwargs_31738)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertEqual(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to foo(...): (line 77)
        # Processing the call arguments (line 77)
        int_31744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'int')
        int_31745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 35), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_31746 = {}
        # Getting the type of 'xx' (line 77)
        xx_31742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'xx', False)
        # Obtaining the member 'foo' of a type (line 77)
        foo_31743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), xx_31742, 'foo')
        # Calling foo(args, kwargs) (line 77)
        foo_call_result_31747 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), foo_31743, *[int_31744, int_31745], **kwargs_31746)
        
        int_31748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 39), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_31749 = {}
        # Getting the type of 'self' (line 77)
        self_31740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 77)
        assertEqual_31741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_31740, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 77)
        assertEqual_call_result_31750 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), assertEqual_31741, *[foo_call_result_31747, int_31748], **kwargs_31749)
        
        
        # Call to assertEqual(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Call to foo(...): (line 78)
        # Processing the call arguments (line 78)
        int_31755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'int')
        int_31756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 35), 'int')
        # Processing the call keyword arguments (line 78)
        kwargs_31757 = {}
        # Getting the type of 'xx' (line 78)
        xx_31753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'xx', False)
        # Obtaining the member 'foo' of a type (line 78)
        foo_31754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 25), xx_31753, 'foo')
        # Calling foo(args, kwargs) (line 78)
        foo_call_result_31758 = invoke(stypy.reporting.localization.Localization(__file__, 78, 25), foo_31754, *[int_31755, int_31756], **kwargs_31757)
        
        int_31759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'int')
        # Processing the call keyword arguments (line 78)
        kwargs_31760 = {}
        # Getting the type of 'self' (line 78)
        self_31751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 78)
        assertEqual_31752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_31751, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 78)
        assertEqual_call_result_31761 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assertEqual_31752, *[foo_call_result_31758, int_31759], **kwargs_31760)
        
        
        # Call to assertEqual(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to demo(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_31769 = {}
        
        # Call to new(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_31766 = {}
        # Getting the type of 'xx' (line 79)
        xx_31764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'xx', False)
        # Obtaining the member 'new' of a type (line 79)
        new_31765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), xx_31764, 'new')
        # Calling new(args, kwargs) (line 79)
        new_call_result_31767 = invoke(stypy.reporting.localization.Localization(__file__, 79, 25), new_31765, *[], **kwargs_31766)
        
        # Obtaining the member 'demo' of a type (line 79)
        demo_31768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), new_call_result_31767, 'demo')
        # Calling demo(args, kwargs) (line 79)
        demo_call_result_31770 = invoke(stypy.reporting.localization.Localization(__file__, 79, 25), demo_31768, *[], **kwargs_31769)
        
        # Getting the type of 'None' (line 79)
        None_31771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'None', False)
        # Processing the call keyword arguments (line 79)
        kwargs_31772 = {}
        # Getting the type of 'self' (line 79)
        self_31762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 79)
        assertEqual_31763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_31762, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 79)
        assertEqual_call_result_31773 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assertEqual_31763, *[demo_call_result_31770, None_31771], **kwargs_31772)
        
        
        # Getting the type of 'test_support' (line 80)
        test_support_31774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'test_support')
        # Obtaining the member 'HAVE_DOCSTRINGS' of a type (line 80)
        HAVE_DOCSTRINGS_31775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), test_support_31774, 'HAVE_DOCSTRINGS')
        # Testing the type of an if condition (line 80)
        if_condition_31776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), HAVE_DOCSTRINGS_31775)
        # Assigning a type to the variable 'if_condition_31776' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_31776', if_condition_31776)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 81):
        str_31777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 18), 'str', 'This is a template module just for instruction.')
        # Assigning a type to the variable 'doc' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'doc', str_31777)
        
        # Call to assertEqual(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'xx' (line 82)
        xx_31780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'xx', False)
        # Obtaining the member '__doc__' of a type (line 82)
        doc___31781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 29), xx_31780, '__doc__')
        # Getting the type of 'doc' (line 82)
        doc_31782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'doc', False)
        # Processing the call keyword arguments (line 82)
        kwargs_31783 = {}
        # Getting the type of 'self' (line 82)
        self_31778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 82)
        assertEqual_31779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), self_31778, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 82)
        assertEqual_call_result_31784 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), assertEqual_31779, *[doc___31781, doc_31782], **kwargs_31783)
        
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertIsInstance(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to Null(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_31789 = {}
        # Getting the type of 'xx' (line 83)
        xx_31787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'xx', False)
        # Obtaining the member 'Null' of a type (line 83)
        Null_31788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 30), xx_31787, 'Null')
        # Calling Null(args, kwargs) (line 83)
        Null_call_result_31790 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), Null_31788, *[], **kwargs_31789)
        
        # Getting the type of 'xx' (line 83)
        xx_31791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'xx', False)
        # Obtaining the member 'Null' of a type (line 83)
        Null_31792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 41), xx_31791, 'Null')
        # Processing the call keyword arguments (line 83)
        kwargs_31793 = {}
        # Getting the type of 'self' (line 83)
        self_31785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 83)
        assertIsInstance_31786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_31785, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 83)
        assertIsInstance_call_result_31794 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assertIsInstance_31786, *[Null_call_result_31790, Null_31792], **kwargs_31793)
        
        
        # Call to assertIsInstance(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Call to Str(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_31799 = {}
        # Getting the type of 'xx' (line 84)
        xx_31797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'xx', False)
        # Obtaining the member 'Str' of a type (line 84)
        Str_31798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 30), xx_31797, 'Str')
        # Calling Str(args, kwargs) (line 84)
        Str_call_result_31800 = invoke(stypy.reporting.localization.Localization(__file__, 84, 30), Str_31798, *[], **kwargs_31799)
        
        # Getting the type of 'xx' (line 84)
        xx_31801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 40), 'xx', False)
        # Obtaining the member 'Str' of a type (line 84)
        Str_31802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 40), xx_31801, 'Str')
        # Processing the call keyword arguments (line 84)
        kwargs_31803 = {}
        # Getting the type of 'self' (line 84)
        self_31795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 84)
        assertIsInstance_31796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_31795, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 84)
        assertIsInstance_call_result_31804 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assertIsInstance_31796, *[Str_call_result_31800, Str_31802], **kwargs_31803)
        
        
        # ################# End of 'test_build_ext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_build_ext' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_31805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_build_ext'
        return stypy_return_type_31805


    @norecursion
    def test_solaris_enable_shared(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_solaris_enable_shared'
        module_type_store = module_type_store.open_function_context('test_solaris_enable_shared', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_solaris_enable_shared')
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_solaris_enable_shared.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_solaris_enable_shared', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_solaris_enable_shared', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_solaris_enable_shared(...)' code ##################

        
        # Assigning a Call to a Name (line 87):
        
        # Call to Distribution(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Obtaining an instance of the builtin type 'dict' (line 87)
        dict_31807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 87)
        # Adding element type (key, value) (line 87)
        str_31808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'str', 'name')
        str_31809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 37), 'str', 'xx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 28), dict_31807, (str_31808, str_31809))
        
        # Processing the call keyword arguments (line 87)
        kwargs_31810 = {}
        # Getting the type of 'Distribution' (line 87)
        Distribution_31806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 87)
        Distribution_call_result_31811 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), Distribution_31806, *[dict_31807], **kwargs_31810)
        
        # Assigning a type to the variable 'dist' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'dist', Distribution_call_result_31811)
        
        # Assigning a Call to a Name (line 88):
        
        # Call to build_ext(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'dist' (line 88)
        dist_31813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'dist', False)
        # Processing the call keyword arguments (line 88)
        kwargs_31814 = {}
        # Getting the type of 'build_ext' (line 88)
        build_ext_31812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 88)
        build_ext_call_result_31815 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), build_ext_31812, *[dist_31813], **kwargs_31814)
        
        # Assigning a type to the variable 'cmd' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'cmd', build_ext_call_result_31815)
        
        # Assigning a Attribute to a Name (line 89):
        # Getting the type of 'sys' (line 89)
        sys_31816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 14), 'sys')
        # Obtaining the member 'platform' of a type (line 89)
        platform_31817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 14), sys_31816, 'platform')
        # Assigning a type to the variable 'old' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'old', platform_31817)
        
        # Assigning a Str to a Attribute (line 91):
        str_31818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 23), 'str', 'sunos')
        # Getting the type of 'sys' (line 91)
        sys_31819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'sys')
        # Setting the type of the member 'platform' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), sys_31819, 'platform', str_31818)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 92, 8))
        
        # 'from distutils.sysconfig import _config_vars' statement (line 92)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_31820 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 92, 8), 'distutils.sysconfig')

        if (type(import_31820) is not StypyTypeError):

            if (import_31820 != 'pyd_module'):
                __import__(import_31820)
                sys_modules_31821 = sys.modules[import_31820]
                import_from_module(stypy.reporting.localization.Localization(__file__, 92, 8), 'distutils.sysconfig', sys_modules_31821.module_type_store, module_type_store, ['_config_vars'])
                nest_module(stypy.reporting.localization.Localization(__file__, 92, 8), __file__, sys_modules_31821, sys_modules_31821.module_type_store, module_type_store)
            else:
                from distutils.sysconfig import _config_vars

                import_from_module(stypy.reporting.localization.Localization(__file__, 92, 8), 'distutils.sysconfig', None, module_type_store, ['_config_vars'], [_config_vars])

        else:
            # Assigning a type to the variable 'distutils.sysconfig' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'distutils.sysconfig', import_31820)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Assigning a Call to a Name (line 93):
        
        # Call to get(...): (line 93)
        # Processing the call arguments (line 93)
        str_31824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 35), 'str', 'Py_ENABLE_SHARED')
        # Processing the call keyword arguments (line 93)
        kwargs_31825 = {}
        # Getting the type of '_config_vars' (line 93)
        _config_vars_31822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), '_config_vars', False)
        # Obtaining the member 'get' of a type (line 93)
        get_31823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), _config_vars_31822, 'get')
        # Calling get(args, kwargs) (line 93)
        get_call_result_31826 = invoke(stypy.reporting.localization.Localization(__file__, 93, 18), get_31823, *[str_31824], **kwargs_31825)
        
        # Assigning a type to the variable 'old_var' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'old_var', get_call_result_31826)
        
        # Assigning a Num to a Subscript (line 94):
        int_31827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 43), 'int')
        # Getting the type of '_config_vars' (line 94)
        _config_vars_31828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), '_config_vars')
        str_31829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'str', 'Py_ENABLE_SHARED')
        # Storing an element on a container (line 94)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 8), _config_vars_31828, (str_31829, int_31827))
        
        # Try-finally block (line 95)
        
        # Call to ensure_finalized(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_31832 = {}
        # Getting the type of 'cmd' (line 96)
        cmd_31830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 96)
        ensure_finalized_31831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), cmd_31830, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 96)
        ensure_finalized_call_result_31833 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), ensure_finalized_31831, *[], **kwargs_31832)
        
        
        # finally branch of the try-finally block (line 95)
        
        # Assigning a Name to a Attribute (line 98):
        # Getting the type of 'old' (line 98)
        old_31834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'old')
        # Getting the type of 'sys' (line 98)
        sys_31835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'sys')
        # Setting the type of the member 'platform' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), sys_31835, 'platform', old_31834)
        
        # Type idiom detected: calculating its left and rigth part (line 99)
        # Getting the type of 'old_var' (line 99)
        old_var_31836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'old_var')
        # Getting the type of 'None' (line 99)
        None_31837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'None')
        
        (may_be_31838, more_types_in_union_31839) = may_be_none(old_var_31836, None_31837)

        if may_be_31838:

            if more_types_in_union_31839:
                # Runtime conditional SSA (line 99)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Deleting a member
            # Getting the type of '_config_vars' (line 100)
            _config_vars_31840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), '_config_vars')
            
            # Obtaining the type of the subscript
            str_31841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 33), 'str', 'Py_ENABLE_SHARED')
            # Getting the type of '_config_vars' (line 100)
            _config_vars_31842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), '_config_vars')
            # Obtaining the member '__getitem__' of a type (line 100)
            getitem___31843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), _config_vars_31842, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 100)
            subscript_call_result_31844 = invoke(stypy.reporting.localization.Localization(__file__, 100, 20), getitem___31843, str_31841)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 16), _config_vars_31840, subscript_call_result_31844)

            if more_types_in_union_31839:
                # Runtime conditional SSA for else branch (line 99)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_31838) or more_types_in_union_31839):
            
            # Assigning a Name to a Subscript (line 102):
            # Getting the type of 'old_var' (line 102)
            old_var_31845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 51), 'old_var')
            # Getting the type of '_config_vars' (line 102)
            _config_vars_31846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), '_config_vars')
            str_31847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 29), 'str', 'Py_ENABLE_SHARED')
            # Storing an element on a container (line 102)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), _config_vars_31846, (str_31847, old_var_31845))

            if (may_be_31838 and more_types_in_union_31839):
                # SSA join for if statement (line 99)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to assertGreater(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to len(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'cmd' (line 105)
        cmd_31851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'cmd', False)
        # Obtaining the member 'library_dirs' of a type (line 105)
        library_dirs_31852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 31), cmd_31851, 'library_dirs')
        # Processing the call keyword arguments (line 105)
        kwargs_31853 = {}
        # Getting the type of 'len' (line 105)
        len_31850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'len', False)
        # Calling len(args, kwargs) (line 105)
        len_call_result_31854 = invoke(stypy.reporting.localization.Localization(__file__, 105, 27), len_31850, *[library_dirs_31852], **kwargs_31853)
        
        int_31855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 50), 'int')
        # Processing the call keyword arguments (line 105)
        kwargs_31856 = {}
        # Getting the type of 'self' (line 105)
        self_31848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self', False)
        # Obtaining the member 'assertGreater' of a type (line 105)
        assertGreater_31849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_31848, 'assertGreater')
        # Calling assertGreater(args, kwargs) (line 105)
        assertGreater_call_result_31857 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assertGreater_31849, *[len_call_result_31854, int_31855], **kwargs_31856)
        
        
        # ################# End of 'test_solaris_enable_shared(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_solaris_enable_shared' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_31858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31858)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_solaris_enable_shared'
        return stypy_return_type_31858


    @norecursion
    def test_user_site(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_user_site'
        module_type_store = module_type_store.open_function_context('test_user_site', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_user_site')
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_user_site.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_user_site', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_user_site', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_user_site(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 110, 8))
        
        # 'import site' statement (line 110)
        import site

        import_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'site', site, module_type_store)
        
        
        # Assigning a Call to a Name (line 111):
        
        # Call to Distribution(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining an instance of the builtin type 'dict' (line 111)
        dict_31860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 111)
        # Adding element type (key, value) (line 111)
        str_31861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'str', 'name')
        str_31862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 37), 'str', 'xx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_31860, (str_31861, str_31862))
        
        # Processing the call keyword arguments (line 111)
        kwargs_31863 = {}
        # Getting the type of 'Distribution' (line 111)
        Distribution_31859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 111)
        Distribution_call_result_31864 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), Distribution_31859, *[dict_31860], **kwargs_31863)
        
        # Assigning a type to the variable 'dist' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'dist', Distribution_call_result_31864)
        
        # Assigning a Call to a Name (line 112):
        
        # Call to build_ext(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'dist' (line 112)
        dist_31866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'dist', False)
        # Processing the call keyword arguments (line 112)
        kwargs_31867 = {}
        # Getting the type of 'build_ext' (line 112)
        build_ext_31865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 112)
        build_ext_call_result_31868 = invoke(stypy.reporting.localization.Localization(__file__, 112, 14), build_ext_31865, *[dist_31866], **kwargs_31867)
        
        # Assigning a type to the variable 'cmd' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'cmd', build_ext_call_result_31868)
        
        # Assigning a ListComp to a Name (line 115):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'cmd' (line 116)
        cmd_31870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'cmd')
        # Obtaining the member 'user_options' of a type (line 116)
        user_options_31871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 19), cmd_31870, 'user_options')
        comprehension_31872 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), user_options_31871)
        # Assigning a type to the variable 'name' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), comprehension_31872))
        # Assigning a type to the variable 'short' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'short', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), comprehension_31872))
        # Assigning a type to the variable 'label' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'label', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), comprehension_31872))
        # Getting the type of 'name' (line 115)
        name_31869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'name')
        list_31873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_31873, name_31869)
        # Assigning a type to the variable 'options' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'options', list_31873)
        
        # Call to assertIn(...): (line 117)
        # Processing the call arguments (line 117)
        str_31876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 22), 'str', 'user')
        # Getting the type of 'options' (line 117)
        options_31877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'options', False)
        # Processing the call keyword arguments (line 117)
        kwargs_31878 = {}
        # Getting the type of 'self' (line 117)
        self_31874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 117)
        assertIn_31875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_31874, 'assertIn')
        # Calling assertIn(args, kwargs) (line 117)
        assertIn_call_result_31879 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), assertIn_31875, *[str_31876, options_31877], **kwargs_31878)
        
        
        # Assigning a Num to a Attribute (line 120):
        int_31880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 19), 'int')
        # Getting the type of 'cmd' (line 120)
        cmd_31881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'cmd')
        # Setting the type of the member 'user' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), cmd_31881, 'user', int_31880)
        
        # Assigning a Call to a Name (line 123):
        
        # Call to join(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'site' (line 123)
        site_31885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 27), 'site', False)
        # Obtaining the member 'USER_BASE' of a type (line 123)
        USER_BASE_31886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 27), site_31885, 'USER_BASE')
        str_31887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 43), 'str', 'lib')
        # Processing the call keyword arguments (line 123)
        kwargs_31888 = {}
        # Getting the type of 'os' (line 123)
        os_31882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 123)
        path_31883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 14), os_31882, 'path')
        # Obtaining the member 'join' of a type (line 123)
        join_31884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 14), path_31883, 'join')
        # Calling join(args, kwargs) (line 123)
        join_call_result_31889 = invoke(stypy.reporting.localization.Localization(__file__, 123, 14), join_31884, *[USER_BASE_31886, str_31887], **kwargs_31888)
        
        # Assigning a type to the variable 'lib' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'lib', join_call_result_31889)
        
        # Assigning a Call to a Name (line 124):
        
        # Call to join(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'site' (line 124)
        site_31893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'site', False)
        # Obtaining the member 'USER_BASE' of a type (line 124)
        USER_BASE_31894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 28), site_31893, 'USER_BASE')
        str_31895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 44), 'str', 'include')
        # Processing the call keyword arguments (line 124)
        kwargs_31896 = {}
        # Getting the type of 'os' (line 124)
        os_31890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 124)
        path_31891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), os_31890, 'path')
        # Obtaining the member 'join' of a type (line 124)
        join_31892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), path_31891, 'join')
        # Calling join(args, kwargs) (line 124)
        join_call_result_31897 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), join_31892, *[USER_BASE_31894, str_31895], **kwargs_31896)
        
        # Assigning a type to the variable 'incl' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'incl', join_call_result_31897)
        
        # Call to mkdir(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'lib' (line 125)
        lib_31900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'lib', False)
        # Processing the call keyword arguments (line 125)
        kwargs_31901 = {}
        # Getting the type of 'os' (line 125)
        os_31898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 125)
        mkdir_31899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), os_31898, 'mkdir')
        # Calling mkdir(args, kwargs) (line 125)
        mkdir_call_result_31902 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), mkdir_31899, *[lib_31900], **kwargs_31901)
        
        
        # Call to mkdir(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'incl' (line 126)
        incl_31905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'incl', False)
        # Processing the call keyword arguments (line 126)
        kwargs_31906 = {}
        # Getting the type of 'os' (line 126)
        os_31903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 126)
        mkdir_31904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), os_31903, 'mkdir')
        # Calling mkdir(args, kwargs) (line 126)
        mkdir_call_result_31907 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), mkdir_31904, *[incl_31905], **kwargs_31906)
        
        
        # Call to ensure_finalized(...): (line 128)
        # Processing the call keyword arguments (line 128)
        kwargs_31910 = {}
        # Getting the type of 'cmd' (line 128)
        cmd_31908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 128)
        ensure_finalized_31909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), cmd_31908, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 128)
        ensure_finalized_call_result_31911 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), ensure_finalized_31909, *[], **kwargs_31910)
        
        
        # Call to assertIn(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'lib' (line 131)
        lib_31914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'lib', False)
        # Getting the type of 'cmd' (line 131)
        cmd_31915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'cmd', False)
        # Obtaining the member 'library_dirs' of a type (line 131)
        library_dirs_31916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 27), cmd_31915, 'library_dirs')
        # Processing the call keyword arguments (line 131)
        kwargs_31917 = {}
        # Getting the type of 'self' (line 131)
        self_31912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 131)
        assertIn_31913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_31912, 'assertIn')
        # Calling assertIn(args, kwargs) (line 131)
        assertIn_call_result_31918 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assertIn_31913, *[lib_31914, library_dirs_31916], **kwargs_31917)
        
        
        # Call to assertIn(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'lib' (line 132)
        lib_31921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 22), 'lib', False)
        # Getting the type of 'cmd' (line 132)
        cmd_31922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'cmd', False)
        # Obtaining the member 'rpath' of a type (line 132)
        rpath_31923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 27), cmd_31922, 'rpath')
        # Processing the call keyword arguments (line 132)
        kwargs_31924 = {}
        # Getting the type of 'self' (line 132)
        self_31919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 132)
        assertIn_31920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_31919, 'assertIn')
        # Calling assertIn(args, kwargs) (line 132)
        assertIn_call_result_31925 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), assertIn_31920, *[lib_31921, rpath_31923], **kwargs_31924)
        
        
        # Call to assertIn(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'incl' (line 133)
        incl_31928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'incl', False)
        # Getting the type of 'cmd' (line 133)
        cmd_31929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'cmd', False)
        # Obtaining the member 'include_dirs' of a type (line 133)
        include_dirs_31930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 28), cmd_31929, 'include_dirs')
        # Processing the call keyword arguments (line 133)
        kwargs_31931 = {}
        # Getting the type of 'self' (line 133)
        self_31926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 133)
        assertIn_31927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_31926, 'assertIn')
        # Calling assertIn(args, kwargs) (line 133)
        assertIn_call_result_31932 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assertIn_31927, *[incl_31928, include_dirs_31930], **kwargs_31931)
        
        
        # ################# End of 'test_user_site(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_user_site' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_31933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_31933)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_user_site'
        return stypy_return_type_31933


    @norecursion
    def test_finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_finalize_options'
        module_type_store = module_type_store.open_function_context('test_finalize_options', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_finalize_options')
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_finalize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_finalize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_finalize_options(...)' code ##################

        
        # Assigning a List to a Name (line 138):
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_31934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        
        # Call to Extension(...): (line 138)
        # Processing the call arguments (line 138)
        str_31936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'str', 'foo')
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_31937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        str_31938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 37), 'str', 'xxx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 36), list_31937, str_31938)
        
        # Processing the call keyword arguments (line 138)
        kwargs_31939 = {}
        # Getting the type of 'Extension' (line 138)
        Extension_31935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'Extension', False)
        # Calling Extension(args, kwargs) (line 138)
        Extension_call_result_31940 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), Extension_31935, *[str_31936, list_31937], **kwargs_31939)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 18), list_31934, Extension_call_result_31940)
        
        # Assigning a type to the variable 'modules' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'modules', list_31934)
        
        # Assigning a Call to a Name (line 139):
        
        # Call to Distribution(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Obtaining an instance of the builtin type 'dict' (line 139)
        dict_31942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 139)
        # Adding element type (key, value) (line 139)
        str_31943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 29), 'str', 'name')
        str_31944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 37), 'str', 'xx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 28), dict_31942, (str_31943, str_31944))
        # Adding element type (key, value) (line 139)
        str_31945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 43), 'str', 'ext_modules')
        # Getting the type of 'modules' (line 139)
        modules_31946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 58), 'modules', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 28), dict_31942, (str_31945, modules_31946))
        
        # Processing the call keyword arguments (line 139)
        kwargs_31947 = {}
        # Getting the type of 'Distribution' (line 139)
        Distribution_31941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 139)
        Distribution_call_result_31948 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), Distribution_31941, *[dict_31942], **kwargs_31947)
        
        # Assigning a type to the variable 'dist' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'dist', Distribution_call_result_31948)
        
        # Assigning a Call to a Name (line 140):
        
        # Call to build_ext(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'dist' (line 140)
        dist_31950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'dist', False)
        # Processing the call keyword arguments (line 140)
        kwargs_31951 = {}
        # Getting the type of 'build_ext' (line 140)
        build_ext_31949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 140)
        build_ext_call_result_31952 = invoke(stypy.reporting.localization.Localization(__file__, 140, 14), build_ext_31949, *[dist_31950], **kwargs_31951)
        
        # Assigning a type to the variable 'cmd' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'cmd', build_ext_call_result_31952)
        
        # Call to finalize_options(...): (line 141)
        # Processing the call keyword arguments (line 141)
        kwargs_31955 = {}
        # Getting the type of 'cmd' (line 141)
        cmd_31953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 141)
        finalize_options_31954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), cmd_31953, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 141)
        finalize_options_call_result_31956 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), finalize_options_31954, *[], **kwargs_31955)
        
        
        # Assigning a Call to a Name (line 143):
        
        # Call to get_python_inc(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_31959 = {}
        # Getting the type of 'sysconfig' (line 143)
        sysconfig_31957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'sysconfig', False)
        # Obtaining the member 'get_python_inc' of a type (line 143)
        get_python_inc_31958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 21), sysconfig_31957, 'get_python_inc')
        # Calling get_python_inc(args, kwargs) (line 143)
        get_python_inc_call_result_31960 = invoke(stypy.reporting.localization.Localization(__file__, 143, 21), get_python_inc_31958, *[], **kwargs_31959)
        
        # Assigning a type to the variable 'py_include' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'py_include', get_python_inc_call_result_31960)
        
        # Call to assertIn(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'py_include' (line 144)
        py_include_31963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'py_include', False)
        # Getting the type of 'cmd' (line 144)
        cmd_31964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'cmd', False)
        # Obtaining the member 'include_dirs' of a type (line 144)
        include_dirs_31965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 34), cmd_31964, 'include_dirs')
        # Processing the call keyword arguments (line 144)
        kwargs_31966 = {}
        # Getting the type of 'self' (line 144)
        self_31961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 144)
        assertIn_31962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_31961, 'assertIn')
        # Calling assertIn(args, kwargs) (line 144)
        assertIn_call_result_31967 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assertIn_31962, *[py_include_31963, include_dirs_31965], **kwargs_31966)
        
        
        # Assigning a Call to a Name (line 146):
        
        # Call to get_python_inc(...): (line 146)
        # Processing the call keyword arguments (line 146)
        int_31970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 65), 'int')
        keyword_31971 = int_31970
        kwargs_31972 = {'plat_specific': keyword_31971}
        # Getting the type of 'sysconfig' (line 146)
        sysconfig_31968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'sysconfig', False)
        # Obtaining the member 'get_python_inc' of a type (line 146)
        get_python_inc_31969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 26), sysconfig_31968, 'get_python_inc')
        # Calling get_python_inc(args, kwargs) (line 146)
        get_python_inc_call_result_31973 = invoke(stypy.reporting.localization.Localization(__file__, 146, 26), get_python_inc_31969, *[], **kwargs_31972)
        
        # Assigning a type to the variable 'plat_py_include' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'plat_py_include', get_python_inc_call_result_31973)
        
        # Call to assertIn(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'plat_py_include' (line 147)
        plat_py_include_31976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'plat_py_include', False)
        # Getting the type of 'cmd' (line 147)
        cmd_31977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 39), 'cmd', False)
        # Obtaining the member 'include_dirs' of a type (line 147)
        include_dirs_31978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 39), cmd_31977, 'include_dirs')
        # Processing the call keyword arguments (line 147)
        kwargs_31979 = {}
        # Getting the type of 'self' (line 147)
        self_31974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 147)
        assertIn_31975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_31974, 'assertIn')
        # Calling assertIn(args, kwargs) (line 147)
        assertIn_call_result_31980 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assertIn_31975, *[plat_py_include_31976, include_dirs_31978], **kwargs_31979)
        
        
        # Assigning a Call to a Name (line 151):
        
        # Call to build_ext(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'dist' (line 151)
        dist_31982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'dist', False)
        # Processing the call keyword arguments (line 151)
        kwargs_31983 = {}
        # Getting the type of 'build_ext' (line 151)
        build_ext_31981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 151)
        build_ext_call_result_31984 = invoke(stypy.reporting.localization.Localization(__file__, 151, 14), build_ext_31981, *[dist_31982], **kwargs_31983)
        
        # Assigning a type to the variable 'cmd' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'cmd', build_ext_call_result_31984)
        
        # Assigning a Str to a Attribute (line 152):
        str_31985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 24), 'str', 'my_lib, other_lib lastlib')
        # Getting the type of 'cmd' (line 152)
        cmd_31986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'cmd')
        # Setting the type of the member 'libraries' of a type (line 152)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), cmd_31986, 'libraries', str_31985)
        
        # Call to finalize_options(...): (line 153)
        # Processing the call keyword arguments (line 153)
        kwargs_31989 = {}
        # Getting the type of 'cmd' (line 153)
        cmd_31987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 153)
        finalize_options_31988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), cmd_31987, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 153)
        finalize_options_call_result_31990 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), finalize_options_31988, *[], **kwargs_31989)
        
        
        # Call to assertEqual(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'cmd' (line 154)
        cmd_31993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'cmd', False)
        # Obtaining the member 'libraries' of a type (line 154)
        libraries_31994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 25), cmd_31993, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_31995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        str_31996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 41), 'str', 'my_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 40), list_31995, str_31996)
        # Adding element type (line 154)
        str_31997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 51), 'str', 'other_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 40), list_31995, str_31997)
        # Adding element type (line 154)
        str_31998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 64), 'str', 'lastlib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 40), list_31995, str_31998)
        
        # Processing the call keyword arguments (line 154)
        kwargs_31999 = {}
        # Getting the type of 'self' (line 154)
        self_31991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 154)
        assertEqual_31992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_31991, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 154)
        assertEqual_call_result_32000 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), assertEqual_31992, *[libraries_31994, list_31995], **kwargs_31999)
        
        
        # Assigning a Call to a Name (line 158):
        
        # Call to build_ext(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'dist' (line 158)
        dist_32002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'dist', False)
        # Processing the call keyword arguments (line 158)
        kwargs_32003 = {}
        # Getting the type of 'build_ext' (line 158)
        build_ext_32001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 158)
        build_ext_call_result_32004 = invoke(stypy.reporting.localization.Localization(__file__, 158, 14), build_ext_32001, *[dist_32002], **kwargs_32003)
        
        # Assigning a type to the variable 'cmd' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'cmd', build_ext_call_result_32004)
        
        # Assigning a BinOp to a Attribute (line 159):
        str_32005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 27), 'str', 'my_lib_dir%sother_lib_dir')
        # Getting the type of 'os' (line 159)
        os_32006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 57), 'os')
        # Obtaining the member 'pathsep' of a type (line 159)
        pathsep_32007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 57), os_32006, 'pathsep')
        # Applying the binary operator '%' (line 159)
        result_mod_32008 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 27), '%', str_32005, pathsep_32007)
        
        # Getting the type of 'cmd' (line 159)
        cmd_32009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'cmd')
        # Setting the type of the member 'library_dirs' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), cmd_32009, 'library_dirs', result_mod_32008)
        
        # Call to finalize_options(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_32012 = {}
        # Getting the type of 'cmd' (line 160)
        cmd_32010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 160)
        finalize_options_32011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), cmd_32010, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 160)
        finalize_options_call_result_32013 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), finalize_options_32011, *[], **kwargs_32012)
        
        
        # Call to assertIn(...): (line 161)
        # Processing the call arguments (line 161)
        str_32016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 22), 'str', 'my_lib_dir')
        # Getting the type of 'cmd' (line 161)
        cmd_32017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'cmd', False)
        # Obtaining the member 'library_dirs' of a type (line 161)
        library_dirs_32018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 36), cmd_32017, 'library_dirs')
        # Processing the call keyword arguments (line 161)
        kwargs_32019 = {}
        # Getting the type of 'self' (line 161)
        self_32014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 161)
        assertIn_32015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), self_32014, 'assertIn')
        # Calling assertIn(args, kwargs) (line 161)
        assertIn_call_result_32020 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), assertIn_32015, *[str_32016, library_dirs_32018], **kwargs_32019)
        
        
        # Call to assertIn(...): (line 162)
        # Processing the call arguments (line 162)
        str_32023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 22), 'str', 'other_lib_dir')
        # Getting the type of 'cmd' (line 162)
        cmd_32024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 39), 'cmd', False)
        # Obtaining the member 'library_dirs' of a type (line 162)
        library_dirs_32025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 39), cmd_32024, 'library_dirs')
        # Processing the call keyword arguments (line 162)
        kwargs_32026 = {}
        # Getting the type of 'self' (line 162)
        self_32021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 162)
        assertIn_32022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_32021, 'assertIn')
        # Calling assertIn(args, kwargs) (line 162)
        assertIn_call_result_32027 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), assertIn_32022, *[str_32023, library_dirs_32025], **kwargs_32026)
        
        
        # Assigning a Call to a Name (line 166):
        
        # Call to build_ext(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'dist' (line 166)
        dist_32029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'dist', False)
        # Processing the call keyword arguments (line 166)
        kwargs_32030 = {}
        # Getting the type of 'build_ext' (line 166)
        build_ext_32028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 166)
        build_ext_call_result_32031 = invoke(stypy.reporting.localization.Localization(__file__, 166, 14), build_ext_32028, *[dist_32029], **kwargs_32030)
        
        # Assigning a type to the variable 'cmd' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'cmd', build_ext_call_result_32031)
        
        # Assigning a BinOp to a Attribute (line 167):
        str_32032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'str', 'one%stwo')
        # Getting the type of 'os' (line 167)
        os_32033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'os')
        # Obtaining the member 'pathsep' of a type (line 167)
        pathsep_32034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 33), os_32033, 'pathsep')
        # Applying the binary operator '%' (line 167)
        result_mod_32035 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 20), '%', str_32032, pathsep_32034)
        
        # Getting the type of 'cmd' (line 167)
        cmd_32036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'cmd')
        # Setting the type of the member 'rpath' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), cmd_32036, 'rpath', result_mod_32035)
        
        # Call to finalize_options(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_32039 = {}
        # Getting the type of 'cmd' (line 168)
        cmd_32037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 168)
        finalize_options_32038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), cmd_32037, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 168)
        finalize_options_call_result_32040 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), finalize_options_32038, *[], **kwargs_32039)
        
        
        # Call to assertEqual(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'cmd' (line 169)
        cmd_32043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'cmd', False)
        # Obtaining the member 'rpath' of a type (line 169)
        rpath_32044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 25), cmd_32043, 'rpath')
        
        # Obtaining an instance of the builtin type 'list' (line 169)
        list_32045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 169)
        # Adding element type (line 169)
        str_32046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 37), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 36), list_32045, str_32046)
        # Adding element type (line 169)
        str_32047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 44), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 36), list_32045, str_32047)
        
        # Processing the call keyword arguments (line 169)
        kwargs_32048 = {}
        # Getting the type of 'self' (line 169)
        self_32041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 169)
        assertEqual_32042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_32041, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 169)
        assertEqual_call_result_32049 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), assertEqual_32042, *[rpath_32044, list_32045], **kwargs_32048)
        
        
        # Assigning a Call to a Name (line 173):
        
        # Call to build_ext(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'dist' (line 173)
        dist_32051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'dist', False)
        # Processing the call keyword arguments (line 173)
        kwargs_32052 = {}
        # Getting the type of 'build_ext' (line 173)
        build_ext_32050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 173)
        build_ext_call_result_32053 = invoke(stypy.reporting.localization.Localization(__file__, 173, 14), build_ext_32050, *[dist_32051], **kwargs_32052)
        
        # Assigning a type to the variable 'cmd' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'cmd', build_ext_call_result_32053)
        
        # Assigning a Str to a Attribute (line 174):
        str_32054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 27), 'str', 'one two,three')
        # Getting the type of 'cmd' (line 174)
        cmd_32055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'cmd')
        # Setting the type of the member 'link_objects' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), cmd_32055, 'link_objects', str_32054)
        
        # Call to finalize_options(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_32058 = {}
        # Getting the type of 'cmd' (line 175)
        cmd_32056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 175)
        finalize_options_32057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), cmd_32056, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 175)
        finalize_options_call_result_32059 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), finalize_options_32057, *[], **kwargs_32058)
        
        
        # Call to assertEqual(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'cmd' (line 176)
        cmd_32062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'cmd', False)
        # Obtaining the member 'link_objects' of a type (line 176)
        link_objects_32063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), cmd_32062, 'link_objects')
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_32064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        str_32065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 44), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), list_32064, str_32065)
        # Adding element type (line 176)
        str_32066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 51), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), list_32064, str_32066)
        # Adding element type (line 176)
        str_32067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 58), 'str', 'three')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 43), list_32064, str_32067)
        
        # Processing the call keyword arguments (line 176)
        kwargs_32068 = {}
        # Getting the type of 'self' (line 176)
        self_32060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 176)
        assertEqual_32061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_32060, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 176)
        assertEqual_call_result_32069 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assertEqual_32061, *[link_objects_32063, list_32064], **kwargs_32068)
        
        
        # Assigning a Call to a Name (line 182):
        
        # Call to build_ext(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'dist' (line 182)
        dist_32071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'dist', False)
        # Processing the call keyword arguments (line 182)
        kwargs_32072 = {}
        # Getting the type of 'build_ext' (line 182)
        build_ext_32070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 182)
        build_ext_call_result_32073 = invoke(stypy.reporting.localization.Localization(__file__, 182, 14), build_ext_32070, *[dist_32071], **kwargs_32072)
        
        # Assigning a type to the variable 'cmd' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'cmd', build_ext_call_result_32073)
        
        # Assigning a Str to a Attribute (line 183):
        str_32074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 21), 'str', 'one,two')
        # Getting the type of 'cmd' (line 183)
        cmd_32075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'cmd')
        # Setting the type of the member 'define' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), cmd_32075, 'define', str_32074)
        
        # Call to finalize_options(...): (line 184)
        # Processing the call keyword arguments (line 184)
        kwargs_32078 = {}
        # Getting the type of 'cmd' (line 184)
        cmd_32076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 184)
        finalize_options_32077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), cmd_32076, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 184)
        finalize_options_call_result_32079 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), finalize_options_32077, *[], **kwargs_32078)
        
        
        # Call to assertEqual(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'cmd' (line 185)
        cmd_32082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 'cmd', False)
        # Obtaining the member 'define' of a type (line 185)
        define_32083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 25), cmd_32082, 'define')
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_32084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_32085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        str_32086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 39), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 39), tuple_32085, str_32086)
        # Adding element type (line 185)
        str_32087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 46), 'str', '1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 39), tuple_32085, str_32087)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 37), list_32084, tuple_32085)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_32088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        str_32089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 53), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 53), tuple_32088, str_32089)
        # Adding element type (line 185)
        str_32090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 60), 'str', '1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 53), tuple_32088, str_32090)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 37), list_32084, tuple_32088)
        
        # Processing the call keyword arguments (line 185)
        kwargs_32091 = {}
        # Getting the type of 'self' (line 185)
        self_32080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 185)
        assertEqual_32081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), self_32080, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 185)
        assertEqual_call_result_32092 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assertEqual_32081, *[define_32083, list_32084], **kwargs_32091)
        
        
        # Assigning a Call to a Name (line 189):
        
        # Call to build_ext(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'dist' (line 189)
        dist_32094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'dist', False)
        # Processing the call keyword arguments (line 189)
        kwargs_32095 = {}
        # Getting the type of 'build_ext' (line 189)
        build_ext_32093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 189)
        build_ext_call_result_32096 = invoke(stypy.reporting.localization.Localization(__file__, 189, 14), build_ext_32093, *[dist_32094], **kwargs_32095)
        
        # Assigning a type to the variable 'cmd' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'cmd', build_ext_call_result_32096)
        
        # Assigning a Str to a Attribute (line 190):
        str_32097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 20), 'str', 'one,two')
        # Getting the type of 'cmd' (line 190)
        cmd_32098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'cmd')
        # Setting the type of the member 'undef' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), cmd_32098, 'undef', str_32097)
        
        # Call to finalize_options(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_32101 = {}
        # Getting the type of 'cmd' (line 191)
        cmd_32099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 191)
        finalize_options_32100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), cmd_32099, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 191)
        finalize_options_call_result_32102 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), finalize_options_32100, *[], **kwargs_32101)
        
        
        # Call to assertEqual(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'cmd' (line 192)
        cmd_32105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 25), 'cmd', False)
        # Obtaining the member 'undef' of a type (line 192)
        undef_32106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 25), cmd_32105, 'undef')
        
        # Obtaining an instance of the builtin type 'list' (line 192)
        list_32107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 192)
        # Adding element type (line 192)
        str_32108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 37), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 36), list_32107, str_32108)
        # Adding element type (line 192)
        str_32109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 44), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 36), list_32107, str_32109)
        
        # Processing the call keyword arguments (line 192)
        kwargs_32110 = {}
        # Getting the type of 'self' (line 192)
        self_32103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 192)
        assertEqual_32104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_32103, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 192)
        assertEqual_call_result_32111 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), assertEqual_32104, *[undef_32106, list_32107], **kwargs_32110)
        
        
        # Assigning a Call to a Name (line 195):
        
        # Call to build_ext(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'dist' (line 195)
        dist_32113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'dist', False)
        # Processing the call keyword arguments (line 195)
        kwargs_32114 = {}
        # Getting the type of 'build_ext' (line 195)
        build_ext_32112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 195)
        build_ext_call_result_32115 = invoke(stypy.reporting.localization.Localization(__file__, 195, 14), build_ext_32112, *[dist_32113], **kwargs_32114)
        
        # Assigning a type to the variable 'cmd' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'cmd', build_ext_call_result_32115)
        
        # Assigning a Name to a Attribute (line 196):
        # Getting the type of 'None' (line 196)
        None_32116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'None')
        # Getting the type of 'cmd' (line 196)
        cmd_32117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'cmd')
        # Setting the type of the member 'swig_opts' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), cmd_32117, 'swig_opts', None_32116)
        
        # Call to finalize_options(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_32120 = {}
        # Getting the type of 'cmd' (line 197)
        cmd_32118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 197)
        finalize_options_32119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), cmd_32118, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 197)
        finalize_options_call_result_32121 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), finalize_options_32119, *[], **kwargs_32120)
        
        
        # Call to assertEqual(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'cmd' (line 198)
        cmd_32124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 25), 'cmd', False)
        # Obtaining the member 'swig_opts' of a type (line 198)
        swig_opts_32125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 25), cmd_32124, 'swig_opts')
        
        # Obtaining an instance of the builtin type 'list' (line 198)
        list_32126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 198)
        
        # Processing the call keyword arguments (line 198)
        kwargs_32127 = {}
        # Getting the type of 'self' (line 198)
        self_32122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 198)
        assertEqual_32123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_32122, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 198)
        assertEqual_call_result_32128 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), assertEqual_32123, *[swig_opts_32125, list_32126], **kwargs_32127)
        
        
        # Assigning a Call to a Name (line 200):
        
        # Call to build_ext(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'dist' (line 200)
        dist_32130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 24), 'dist', False)
        # Processing the call keyword arguments (line 200)
        kwargs_32131 = {}
        # Getting the type of 'build_ext' (line 200)
        build_ext_32129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 200)
        build_ext_call_result_32132 = invoke(stypy.reporting.localization.Localization(__file__, 200, 14), build_ext_32129, *[dist_32130], **kwargs_32131)
        
        # Assigning a type to the variable 'cmd' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'cmd', build_ext_call_result_32132)
        
        # Assigning a Str to a Attribute (line 201):
        str_32133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 24), 'str', '1 2')
        # Getting the type of 'cmd' (line 201)
        cmd_32134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'cmd')
        # Setting the type of the member 'swig_opts' of a type (line 201)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), cmd_32134, 'swig_opts', str_32133)
        
        # Call to finalize_options(...): (line 202)
        # Processing the call keyword arguments (line 202)
        kwargs_32137 = {}
        # Getting the type of 'cmd' (line 202)
        cmd_32135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 202)
        finalize_options_32136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), cmd_32135, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 202)
        finalize_options_call_result_32138 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), finalize_options_32136, *[], **kwargs_32137)
        
        
        # Call to assertEqual(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'cmd' (line 203)
        cmd_32141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'cmd', False)
        # Obtaining the member 'swig_opts' of a type (line 203)
        swig_opts_32142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 25), cmd_32141, 'swig_opts')
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_32143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        str_32144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 41), 'str', '1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 40), list_32143, str_32144)
        # Adding element type (line 203)
        str_32145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 46), 'str', '2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 40), list_32143, str_32145)
        
        # Processing the call keyword arguments (line 203)
        kwargs_32146 = {}
        # Getting the type of 'self' (line 203)
        self_32139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 203)
        assertEqual_32140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_32139, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 203)
        assertEqual_call_result_32147 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), assertEqual_32140, *[swig_opts_32142, list_32143], **kwargs_32146)
        
        
        # ################# End of 'test_finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_32148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32148)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_finalize_options'
        return stypy_return_type_32148


    @norecursion
    def test_check_extensions_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_check_extensions_list'
        module_type_store = module_type_store.open_function_context('test_check_extensions_list', 205, 4, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_check_extensions_list')
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_check_extensions_list.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_check_extensions_list', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_check_extensions_list', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_check_extensions_list(...)' code ##################

        
        # Assigning a Call to a Name (line 206):
        
        # Call to Distribution(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_32150 = {}
        # Getting the type of 'Distribution' (line 206)
        Distribution_32149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 206)
        Distribution_call_result_32151 = invoke(stypy.reporting.localization.Localization(__file__, 206, 15), Distribution_32149, *[], **kwargs_32150)
        
        # Assigning a type to the variable 'dist' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'dist', Distribution_call_result_32151)
        
        # Assigning a Call to a Name (line 207):
        
        # Call to build_ext(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'dist' (line 207)
        dist_32153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), 'dist', False)
        # Processing the call keyword arguments (line 207)
        kwargs_32154 = {}
        # Getting the type of 'build_ext' (line 207)
        build_ext_32152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 207)
        build_ext_call_result_32155 = invoke(stypy.reporting.localization.Localization(__file__, 207, 14), build_ext_32152, *[dist_32153], **kwargs_32154)
        
        # Assigning a type to the variable 'cmd' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'cmd', build_ext_call_result_32155)
        
        # Call to finalize_options(...): (line 208)
        # Processing the call keyword arguments (line 208)
        kwargs_32158 = {}
        # Getting the type of 'cmd' (line 208)
        cmd_32156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 208)
        finalize_options_32157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), cmd_32156, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 208)
        finalize_options_call_result_32159 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), finalize_options_32157, *[], **kwargs_32158)
        
        
        # Call to assertRaises(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'DistutilsSetupError' (line 211)
        DistutilsSetupError_32162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 211)
        cmd_32163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 47), 'cmd', False)
        # Obtaining the member 'check_extensions_list' of a type (line 211)
        check_extensions_list_32164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 47), cmd_32163, 'check_extensions_list')
        str_32165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 74), 'str', 'foo')
        # Processing the call keyword arguments (line 211)
        kwargs_32166 = {}
        # Getting the type of 'self' (line 211)
        self_32160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 211)
        assertRaises_32161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_32160, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 211)
        assertRaises_call_result_32167 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assertRaises_32161, *[DistutilsSetupError_32162, check_extensions_list_32164, str_32165], **kwargs_32166)
        
        
        # Assigning a List to a Name (line 215):
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_32168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        # Adding element type (line 215)
        
        # Obtaining an instance of the builtin type 'tuple' (line 215)
        tuple_32169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 215)
        # Adding element type (line 215)
        str_32170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 17), 'str', 'bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 17), tuple_32169, str_32170)
        # Adding element type (line 215)
        str_32171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 24), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 17), tuple_32169, str_32171)
        # Adding element type (line 215)
        str_32172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 31), 'str', 'bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 17), tuple_32169, str_32172)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 15), list_32168, tuple_32169)
        # Adding element type (line 215)
        str_32173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 39), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 15), list_32168, str_32173)
        
        # Assigning a type to the variable 'exts' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'exts', list_32168)
        
        # Call to assertRaises(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'DistutilsSetupError' (line 216)
        DistutilsSetupError_32176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 216)
        cmd_32177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 47), 'cmd', False)
        # Obtaining the member 'check_extensions_list' of a type (line 216)
        check_extensions_list_32178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 47), cmd_32177, 'check_extensions_list')
        # Getting the type of 'exts' (line 216)
        exts_32179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 74), 'exts', False)
        # Processing the call keyword arguments (line 216)
        kwargs_32180 = {}
        # Getting the type of 'self' (line 216)
        self_32174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 216)
        assertRaises_32175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), self_32174, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 216)
        assertRaises_call_result_32181 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), assertRaises_32175, *[DistutilsSetupError_32176, check_extensions_list_32178, exts_32179], **kwargs_32180)
        
        
        # Assigning a List to a Name (line 221):
        
        # Obtaining an instance of the builtin type 'list' (line 221)
        list_32182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 221)
        # Adding element type (line 221)
        
        # Obtaining an instance of the builtin type 'tuple' (line 221)
        tuple_32183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 221)
        # Adding element type (line 221)
        str_32184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 17), 'str', 'foo-bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 17), tuple_32183, str_32184)
        # Adding element type (line 221)
        str_32185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 28), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 17), tuple_32183, str_32185)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 15), list_32182, tuple_32183)
        
        # Assigning a type to the variable 'exts' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'exts', list_32182)
        
        # Call to assertRaises(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'DistutilsSetupError' (line 222)
        DistutilsSetupError_32188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 222)
        cmd_32189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 47), 'cmd', False)
        # Obtaining the member 'check_extensions_list' of a type (line 222)
        check_extensions_list_32190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 47), cmd_32189, 'check_extensions_list')
        # Getting the type of 'exts' (line 222)
        exts_32191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 74), 'exts', False)
        # Processing the call keyword arguments (line 222)
        kwargs_32192 = {}
        # Getting the type of 'self' (line 222)
        self_32186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 222)
        assertRaises_32187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_32186, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 222)
        assertRaises_call_result_32193 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), assertRaises_32187, *[DistutilsSetupError_32188, check_extensions_list_32190, exts_32191], **kwargs_32192)
        
        
        # Assigning a List to a Name (line 226):
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_32194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_32195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        str_32196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 17), 'str', 'foo.bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 17), tuple_32195, str_32196)
        # Adding element type (line 226)
        str_32197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 28), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 17), tuple_32195, str_32197)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 15), list_32194, tuple_32195)
        
        # Assigning a type to the variable 'exts' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'exts', list_32194)
        
        # Call to assertRaises(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'DistutilsSetupError' (line 227)
        DistutilsSetupError_32200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 227)
        cmd_32201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'cmd', False)
        # Obtaining the member 'check_extensions_list' of a type (line 227)
        check_extensions_list_32202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 47), cmd_32201, 'check_extensions_list')
        # Getting the type of 'exts' (line 227)
        exts_32203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 74), 'exts', False)
        # Processing the call keyword arguments (line 227)
        kwargs_32204 = {}
        # Getting the type of 'self' (line 227)
        self_32198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 227)
        assertRaises_32199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_32198, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 227)
        assertRaises_call_result_32205 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), assertRaises_32199, *[DistutilsSetupError_32200, check_extensions_list_32202, exts_32203], **kwargs_32204)
        
        
        # Assigning a List to a Name (line 230):
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_32206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        
        # Obtaining an instance of the builtin type 'tuple' (line 230)
        tuple_32207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 230)
        # Adding element type (line 230)
        str_32208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 17), 'str', 'foo.bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 17), tuple_32207, str_32208)
        # Adding element type (line 230)
        
        # Obtaining an instance of the builtin type 'dict' (line 230)
        dict_32209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 230)
        # Adding element type (key, value) (line 230)
        str_32210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 29), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_32211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        str_32212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 41), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 40), list_32211, str_32212)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 28), dict_32209, (str_32210, list_32211))
        # Adding element type (key, value) (line 230)
        str_32213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 46), 'str', 'libraries')
        str_32214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 59), 'str', 'foo')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 28), dict_32209, (str_32213, str_32214))
        # Adding element type (key, value) (line 230)
        str_32215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 29), 'str', 'some')
        str_32216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 37), 'str', 'bar')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 28), dict_32209, (str_32215, str_32216))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 17), tuple_32207, dict_32209)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 15), list_32206, tuple_32207)
        
        # Assigning a type to the variable 'exts' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'exts', list_32206)
        
        # Call to check_extensions_list(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'exts' (line 232)
        exts_32219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'exts', False)
        # Processing the call keyword arguments (line 232)
        kwargs_32220 = {}
        # Getting the type of 'cmd' (line 232)
        cmd_32217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'cmd', False)
        # Obtaining the member 'check_extensions_list' of a type (line 232)
        check_extensions_list_32218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), cmd_32217, 'check_extensions_list')
        # Calling check_extensions_list(args, kwargs) (line 232)
        check_extensions_list_call_result_32221 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), check_extensions_list_32218, *[exts_32219], **kwargs_32220)
        
        
        # Assigning a Subscript to a Name (line 233):
        
        # Obtaining the type of the subscript
        int_32222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 19), 'int')
        # Getting the type of 'exts' (line 233)
        exts_32223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 14), 'exts')
        # Obtaining the member '__getitem__' of a type (line 233)
        getitem___32224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 14), exts_32223, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 233)
        subscript_call_result_32225 = invoke(stypy.reporting.localization.Localization(__file__, 233, 14), getitem___32224, int_32222)
        
        # Assigning a type to the variable 'ext' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'ext', subscript_call_result_32225)
        
        # Call to assertIsInstance(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'ext' (line 234)
        ext_32228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 30), 'ext', False)
        # Getting the type of 'Extension' (line 234)
        Extension_32229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'Extension', False)
        # Processing the call keyword arguments (line 234)
        kwargs_32230 = {}
        # Getting the type of 'self' (line 234)
        self_32226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 234)
        assertIsInstance_32227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_32226, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 234)
        assertIsInstance_call_result_32231 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), assertIsInstance_32227, *[ext_32228, Extension_32229], **kwargs_32230)
        
        
        # Call to assertEqual(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'ext' (line 239)
        ext_32234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 25), 'ext', False)
        # Obtaining the member 'libraries' of a type (line 239)
        libraries_32235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 25), ext_32234, 'libraries')
        str_32236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 40), 'str', 'foo')
        # Processing the call keyword arguments (line 239)
        kwargs_32237 = {}
        # Getting the type of 'self' (line 239)
        self_32232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 239)
        assertEqual_32233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_32232, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 239)
        assertEqual_call_result_32238 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), assertEqual_32233, *[libraries_32235, str_32236], **kwargs_32237)
        
        
        # Call to assertFalse(...): (line 240)
        # Processing the call arguments (line 240)
        
        # Call to hasattr(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'ext' (line 240)
        ext_32242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 33), 'ext', False)
        str_32243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 38), 'str', 'some')
        # Processing the call keyword arguments (line 240)
        kwargs_32244 = {}
        # Getting the type of 'hasattr' (line 240)
        hasattr_32241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 25), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 240)
        hasattr_call_result_32245 = invoke(stypy.reporting.localization.Localization(__file__, 240, 25), hasattr_32241, *[ext_32242, str_32243], **kwargs_32244)
        
        # Processing the call keyword arguments (line 240)
        kwargs_32246 = {}
        # Getting the type of 'self' (line 240)
        self_32239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 240)
        assertFalse_32240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), self_32239, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 240)
        assertFalse_call_result_32247 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), assertFalse_32240, *[hasattr_call_result_32245], **kwargs_32246)
        
        
        # Assigning a List to a Name (line 243):
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_32248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'tuple' (line 243)
        tuple_32249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 243)
        # Adding element type (line 243)
        str_32250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 17), 'str', 'foo.bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 17), tuple_32249, str_32250)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'dict' (line 243)
        dict_32251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 243)
        # Adding element type (key, value) (line 243)
        str_32252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 29), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_32253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        str_32254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 41), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 40), list_32253, str_32254)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 28), dict_32251, (str_32252, list_32253))
        # Adding element type (key, value) (line 243)
        str_32255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 46), 'str', 'libraries')
        str_32256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 59), 'str', 'foo')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 28), dict_32251, (str_32255, str_32256))
        # Adding element type (key, value) (line 243)
        str_32257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 16), 'str', 'some')
        str_32258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 24), 'str', 'bar')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 28), dict_32251, (str_32257, str_32258))
        # Adding element type (key, value) (line 243)
        str_32259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 31), 'str', 'macros')
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_32260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        
        # Obtaining an instance of the builtin type 'tuple' (line 244)
        tuple_32261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 244)
        # Adding element type (line 244)
        str_32262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 43), 'str', '1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 43), tuple_32261, str_32262)
        # Adding element type (line 244)
        str_32263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 48), 'str', '2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 43), tuple_32261, str_32263)
        # Adding element type (line 244)
        str_32264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 53), 'str', '3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 43), tuple_32261, str_32264)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 41), list_32260, tuple_32261)
        # Adding element type (line 244)
        str_32265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 59), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 41), list_32260, str_32265)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 28), dict_32251, (str_32259, list_32260))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 17), tuple_32249, dict_32251)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 15), list_32248, tuple_32249)
        
        # Assigning a type to the variable 'exts' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'exts', list_32248)
        
        # Call to assertRaises(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'DistutilsSetupError' (line 245)
        DistutilsSetupError_32268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 26), 'DistutilsSetupError', False)
        # Getting the type of 'cmd' (line 245)
        cmd_32269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 47), 'cmd', False)
        # Obtaining the member 'check_extensions_list' of a type (line 245)
        check_extensions_list_32270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 47), cmd_32269, 'check_extensions_list')
        # Getting the type of 'exts' (line 245)
        exts_32271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 74), 'exts', False)
        # Processing the call keyword arguments (line 245)
        kwargs_32272 = {}
        # Getting the type of 'self' (line 245)
        self_32266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 245)
        assertRaises_32267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_32266, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 245)
        assertRaises_call_result_32273 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), assertRaises_32267, *[DistutilsSetupError_32268, check_extensions_list_32270, exts_32271], **kwargs_32272)
        
        
        # Assigning a List to a Subscript (line 247):
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_32274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        
        # Obtaining an instance of the builtin type 'tuple' (line 247)
        tuple_32275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 247)
        # Adding element type (line 247)
        str_32276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 33), 'str', '1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 33), tuple_32275, str_32276)
        # Adding element type (line 247)
        str_32277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 38), 'str', '2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 33), tuple_32275, str_32277)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 31), list_32274, tuple_32275)
        # Adding element type (line 247)
        
        # Obtaining an instance of the builtin type 'tuple' (line 247)
        tuple_32278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 247)
        # Adding element type (line 247)
        str_32279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 45), 'str', '3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 45), tuple_32278, str_32279)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 31), list_32274, tuple_32278)
        
        
        # Obtaining the type of the subscript
        int_32280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 16), 'int')
        
        # Obtaining the type of the subscript
        int_32281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 13), 'int')
        # Getting the type of 'exts' (line 247)
        exts_32282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'exts')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___32283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), exts_32282, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_32284 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), getitem___32283, int_32281)
        
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___32285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), subscript_call_result_32284, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_32286 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), getitem___32285, int_32280)
        
        str_32287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 19), 'str', 'macros')
        # Storing an element on a container (line 247)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 8), subscript_call_result_32286, (str_32287, list_32274))
        
        # Call to check_extensions_list(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'exts' (line 248)
        exts_32290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 34), 'exts', False)
        # Processing the call keyword arguments (line 248)
        kwargs_32291 = {}
        # Getting the type of 'cmd' (line 248)
        cmd_32288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'cmd', False)
        # Obtaining the member 'check_extensions_list' of a type (line 248)
        check_extensions_list_32289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), cmd_32288, 'check_extensions_list')
        # Calling check_extensions_list(args, kwargs) (line 248)
        check_extensions_list_call_result_32292 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), check_extensions_list_32289, *[exts_32290], **kwargs_32291)
        
        
        # Call to assertEqual(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Obtaining the type of the subscript
        int_32295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 30), 'int')
        # Getting the type of 'exts' (line 249)
        exts_32296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'exts', False)
        # Obtaining the member '__getitem__' of a type (line 249)
        getitem___32297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 25), exts_32296, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 249)
        subscript_call_result_32298 = invoke(stypy.reporting.localization.Localization(__file__, 249, 25), getitem___32297, int_32295)
        
        # Obtaining the member 'undef_macros' of a type (line 249)
        undef_macros_32299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 25), subscript_call_result_32298, 'undef_macros')
        
        # Obtaining an instance of the builtin type 'list' (line 249)
        list_32300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 249)
        # Adding element type (line 249)
        str_32301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 48), 'str', '3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 47), list_32300, str_32301)
        
        # Processing the call keyword arguments (line 249)
        kwargs_32302 = {}
        # Getting the type of 'self' (line 249)
        self_32293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 249)
        assertEqual_32294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_32293, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 249)
        assertEqual_call_result_32303 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), assertEqual_32294, *[undef_macros_32299, list_32300], **kwargs_32302)
        
        
        # Call to assertEqual(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Obtaining the type of the subscript
        int_32306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 30), 'int')
        # Getting the type of 'exts' (line 250)
        exts_32307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'exts', False)
        # Obtaining the member '__getitem__' of a type (line 250)
        getitem___32308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 25), exts_32307, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 250)
        subscript_call_result_32309 = invoke(stypy.reporting.localization.Localization(__file__, 250, 25), getitem___32308, int_32306)
        
        # Obtaining the member 'define_macros' of a type (line 250)
        define_macros_32310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 25), subscript_call_result_32309, 'define_macros')
        
        # Obtaining an instance of the builtin type 'list' (line 250)
        list_32311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 250)
        # Adding element type (line 250)
        
        # Obtaining an instance of the builtin type 'tuple' (line 250)
        tuple_32312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 250)
        # Adding element type (line 250)
        str_32313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 50), 'str', '1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 50), tuple_32312, str_32313)
        # Adding element type (line 250)
        str_32314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 55), 'str', '2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 50), tuple_32312, str_32314)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 48), list_32311, tuple_32312)
        
        # Processing the call keyword arguments (line 250)
        kwargs_32315 = {}
        # Getting the type of 'self' (line 250)
        self_32304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 250)
        assertEqual_32305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_32304, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 250)
        assertEqual_call_result_32316 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), assertEqual_32305, *[define_macros_32310, list_32311], **kwargs_32315)
        
        
        # ################# End of 'test_check_extensions_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_check_extensions_list' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_32317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32317)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_check_extensions_list'
        return stypy_return_type_32317


    @norecursion
    def test_get_source_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_source_files'
        module_type_store = module_type_store.open_function_context('test_get_source_files', 252, 4, False)
        # Assigning a type to the variable 'self' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_get_source_files')
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_get_source_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_get_source_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_source_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_source_files(...)' code ##################

        
        # Assigning a List to a Name (line 253):
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_32318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        
        # Call to Extension(...): (line 253)
        # Processing the call arguments (line 253)
        str_32320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 29), 'str', 'foo')
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_32321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        str_32322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 37), 'str', 'xxx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 36), list_32321, str_32322)
        
        # Processing the call keyword arguments (line 253)
        kwargs_32323 = {}
        # Getting the type of 'Extension' (line 253)
        Extension_32319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'Extension', False)
        # Calling Extension(args, kwargs) (line 253)
        Extension_call_result_32324 = invoke(stypy.reporting.localization.Localization(__file__, 253, 19), Extension_32319, *[str_32320, list_32321], **kwargs_32323)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 18), list_32318, Extension_call_result_32324)
        
        # Assigning a type to the variable 'modules' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'modules', list_32318)
        
        # Assigning a Call to a Name (line 254):
        
        # Call to Distribution(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining an instance of the builtin type 'dict' (line 254)
        dict_32326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 254)
        # Adding element type (key, value) (line 254)
        str_32327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'str', 'name')
        str_32328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 37), 'str', 'xx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 28), dict_32326, (str_32327, str_32328))
        # Adding element type (key, value) (line 254)
        str_32329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 43), 'str', 'ext_modules')
        # Getting the type of 'modules' (line 254)
        modules_32330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 58), 'modules', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 28), dict_32326, (str_32329, modules_32330))
        
        # Processing the call keyword arguments (line 254)
        kwargs_32331 = {}
        # Getting the type of 'Distribution' (line 254)
        Distribution_32325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 254)
        Distribution_call_result_32332 = invoke(stypy.reporting.localization.Localization(__file__, 254, 15), Distribution_32325, *[dict_32326], **kwargs_32331)
        
        # Assigning a type to the variable 'dist' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'dist', Distribution_call_result_32332)
        
        # Assigning a Call to a Name (line 255):
        
        # Call to build_ext(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'dist' (line 255)
        dist_32334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'dist', False)
        # Processing the call keyword arguments (line 255)
        kwargs_32335 = {}
        # Getting the type of 'build_ext' (line 255)
        build_ext_32333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 255)
        build_ext_call_result_32336 = invoke(stypy.reporting.localization.Localization(__file__, 255, 14), build_ext_32333, *[dist_32334], **kwargs_32335)
        
        # Assigning a type to the variable 'cmd' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'cmd', build_ext_call_result_32336)
        
        # Call to ensure_finalized(...): (line 256)
        # Processing the call keyword arguments (line 256)
        kwargs_32339 = {}
        # Getting the type of 'cmd' (line 256)
        cmd_32337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 256)
        ensure_finalized_32338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), cmd_32337, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 256)
        ensure_finalized_call_result_32340 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), ensure_finalized_32338, *[], **kwargs_32339)
        
        
        # Call to assertEqual(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Call to get_source_files(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_32345 = {}
        # Getting the type of 'cmd' (line 257)
        cmd_32343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 25), 'cmd', False)
        # Obtaining the member 'get_source_files' of a type (line 257)
        get_source_files_32344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 25), cmd_32343, 'get_source_files')
        # Calling get_source_files(args, kwargs) (line 257)
        get_source_files_call_result_32346 = invoke(stypy.reporting.localization.Localization(__file__, 257, 25), get_source_files_32344, *[], **kwargs_32345)
        
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_32347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        # Adding element type (line 257)
        str_32348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 50), 'str', 'xxx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 49), list_32347, str_32348)
        
        # Processing the call keyword arguments (line 257)
        kwargs_32349 = {}
        # Getting the type of 'self' (line 257)
        self_32341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 257)
        assertEqual_32342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_32341, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 257)
        assertEqual_call_result_32350 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), assertEqual_32342, *[get_source_files_call_result_32346, list_32347], **kwargs_32349)
        
        
        # ################# End of 'test_get_source_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_source_files' in the type store
        # Getting the type of 'stypy_return_type' (line 252)
        stypy_return_type_32351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_source_files'
        return stypy_return_type_32351


    @norecursion
    def test_compiler_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_compiler_option'
        module_type_store = module_type_store.open_function_context('test_compiler_option', 259, 4, False)
        # Assigning a type to the variable 'self' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_compiler_option')
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_compiler_option.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_compiler_option', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_compiler_option', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_compiler_option(...)' code ##################

        
        # Assigning a Call to a Name (line 263):
        
        # Call to Distribution(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_32353 = {}
        # Getting the type of 'Distribution' (line 263)
        Distribution_32352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 263)
        Distribution_call_result_32354 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), Distribution_32352, *[], **kwargs_32353)
        
        # Assigning a type to the variable 'dist' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'dist', Distribution_call_result_32354)
        
        # Assigning a Call to a Name (line 264):
        
        # Call to build_ext(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'dist' (line 264)
        dist_32356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'dist', False)
        # Processing the call keyword arguments (line 264)
        kwargs_32357 = {}
        # Getting the type of 'build_ext' (line 264)
        build_ext_32355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 264)
        build_ext_call_result_32358 = invoke(stypy.reporting.localization.Localization(__file__, 264, 14), build_ext_32355, *[dist_32356], **kwargs_32357)
        
        # Assigning a type to the variable 'cmd' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'cmd', build_ext_call_result_32358)
        
        # Assigning a Str to a Attribute (line 265):
        str_32359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 23), 'str', 'unix')
        # Getting the type of 'cmd' (line 265)
        cmd_32360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'cmd')
        # Setting the type of the member 'compiler' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), cmd_32360, 'compiler', str_32359)
        
        # Call to ensure_finalized(...): (line 266)
        # Processing the call keyword arguments (line 266)
        kwargs_32363 = {}
        # Getting the type of 'cmd' (line 266)
        cmd_32361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 266)
        ensure_finalized_32362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), cmd_32361, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 266)
        ensure_finalized_call_result_32364 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), ensure_finalized_32362, *[], **kwargs_32363)
        
        
        # Call to run(...): (line 267)
        # Processing the call keyword arguments (line 267)
        kwargs_32367 = {}
        # Getting the type of 'cmd' (line 267)
        cmd_32365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 267)
        run_32366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), cmd_32365, 'run')
        # Calling run(args, kwargs) (line 267)
        run_call_result_32368 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), run_32366, *[], **kwargs_32367)
        
        
        # Call to assertEqual(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'cmd' (line 268)
        cmd_32371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 25), 'cmd', False)
        # Obtaining the member 'compiler' of a type (line 268)
        compiler_32372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 25), cmd_32371, 'compiler')
        str_32373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 39), 'str', 'unix')
        # Processing the call keyword arguments (line 268)
        kwargs_32374 = {}
        # Getting the type of 'self' (line 268)
        self_32369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 268)
        assertEqual_32370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_32369, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 268)
        assertEqual_call_result_32375 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), assertEqual_32370, *[compiler_32372, str_32373], **kwargs_32374)
        
        
        # ################# End of 'test_compiler_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_compiler_option' in the type store
        # Getting the type of 'stypy_return_type' (line 259)
        stypy_return_type_32376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_compiler_option'
        return stypy_return_type_32376


    @norecursion
    def test_get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_outputs'
        module_type_store = module_type_store.open_function_context('test_get_outputs', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_get_outputs')
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_get_outputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_outputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_outputs(...)' code ##################

        
        # Assigning a Call to a Name (line 271):
        
        # Call to mkdtemp(...): (line 271)
        # Processing the call keyword arguments (line 271)
        kwargs_32379 = {}
        # Getting the type of 'self' (line 271)
        self_32377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 271)
        mkdtemp_32378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 18), self_32377, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 271)
        mkdtemp_call_result_32380 = invoke(stypy.reporting.localization.Localization(__file__, 271, 18), mkdtemp_32378, *[], **kwargs_32379)
        
        # Assigning a type to the variable 'tmp_dir' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'tmp_dir', mkdtemp_call_result_32380)
        
        # Assigning a Call to a Name (line 272):
        
        # Call to join(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'tmp_dir' (line 272)
        tmp_dir_32384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 30), 'tmp_dir', False)
        str_32385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 39), 'str', 'foo.c')
        # Processing the call keyword arguments (line 272)
        kwargs_32386 = {}
        # Getting the type of 'os' (line 272)
        os_32381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 272)
        path_32382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 17), os_32381, 'path')
        # Obtaining the member 'join' of a type (line 272)
        join_32383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 17), path_32382, 'join')
        # Calling join(args, kwargs) (line 272)
        join_call_result_32387 = invoke(stypy.reporting.localization.Localization(__file__, 272, 17), join_32383, *[tmp_dir_32384, str_32385], **kwargs_32386)
        
        # Assigning a type to the variable 'c_file' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'c_file', join_call_result_32387)
        
        # Call to write_file(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'c_file' (line 273)
        c_file_32390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'c_file', False)
        str_32391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 32), 'str', 'void initfoo(void) {};\n')
        # Processing the call keyword arguments (line 273)
        kwargs_32392 = {}
        # Getting the type of 'self' (line 273)
        self_32388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 273)
        write_file_32389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_32388, 'write_file')
        # Calling write_file(args, kwargs) (line 273)
        write_file_call_result_32393 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), write_file_32389, *[c_file_32390, str_32391], **kwargs_32392)
        
        
        # Assigning a Call to a Name (line 274):
        
        # Call to Extension(...): (line 274)
        # Processing the call arguments (line 274)
        str_32395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 24), 'str', 'foo')
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_32396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        # Getting the type of 'c_file' (line 274)
        c_file_32397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 32), 'c_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 31), list_32396, c_file_32397)
        
        # Processing the call keyword arguments (line 274)
        kwargs_32398 = {}
        # Getting the type of 'Extension' (line 274)
        Extension_32394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 14), 'Extension', False)
        # Calling Extension(args, kwargs) (line 274)
        Extension_call_result_32399 = invoke(stypy.reporting.localization.Localization(__file__, 274, 14), Extension_32394, *[str_32395, list_32396], **kwargs_32398)
        
        # Assigning a type to the variable 'ext' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'ext', Extension_call_result_32399)
        
        # Assigning a Call to a Name (line 275):
        
        # Call to Distribution(...): (line 275)
        # Processing the call arguments (line 275)
        
        # Obtaining an instance of the builtin type 'dict' (line 275)
        dict_32401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 275)
        # Adding element type (key, value) (line 275)
        str_32402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 29), 'str', 'name')
        str_32403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 37), 'str', 'xx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 28), dict_32401, (str_32402, str_32403))
        # Adding element type (key, value) (line 275)
        str_32404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 29), 'str', 'ext_modules')
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_32405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        # Getting the type of 'ext' (line 276)
        ext_32406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 45), 'ext', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 44), list_32405, ext_32406)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 28), dict_32401, (str_32404, list_32405))
        
        # Processing the call keyword arguments (line 275)
        kwargs_32407 = {}
        # Getting the type of 'Distribution' (line 275)
        Distribution_32400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 275)
        Distribution_call_result_32408 = invoke(stypy.reporting.localization.Localization(__file__, 275, 15), Distribution_32400, *[dict_32401], **kwargs_32407)
        
        # Assigning a type to the variable 'dist' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'dist', Distribution_call_result_32408)
        
        # Assigning a Call to a Name (line 277):
        
        # Call to build_ext(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'dist' (line 277)
        dist_32410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 24), 'dist', False)
        # Processing the call keyword arguments (line 277)
        kwargs_32411 = {}
        # Getting the type of 'build_ext' (line 277)
        build_ext_32409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 277)
        build_ext_call_result_32412 = invoke(stypy.reporting.localization.Localization(__file__, 277, 14), build_ext_32409, *[dist_32410], **kwargs_32411)
        
        # Assigning a type to the variable 'cmd' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'cmd', build_ext_call_result_32412)
        
        # Call to fixup_build_ext(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'cmd' (line 278)
        cmd_32415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 32), 'cmd', False)
        # Processing the call keyword arguments (line 278)
        kwargs_32416 = {}
        # Getting the type of 'support' (line 278)
        support_32413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'support', False)
        # Obtaining the member 'fixup_build_ext' of a type (line 278)
        fixup_build_ext_32414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), support_32413, 'fixup_build_ext')
        # Calling fixup_build_ext(args, kwargs) (line 278)
        fixup_build_ext_call_result_32417 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), fixup_build_ext_32414, *[cmd_32415], **kwargs_32416)
        
        
        # Call to ensure_finalized(...): (line 279)
        # Processing the call keyword arguments (line 279)
        kwargs_32420 = {}
        # Getting the type of 'cmd' (line 279)
        cmd_32418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 279)
        ensure_finalized_32419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), cmd_32418, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 279)
        ensure_finalized_call_result_32421 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), ensure_finalized_32419, *[], **kwargs_32420)
        
        
        # Call to assertEqual(...): (line 280)
        # Processing the call arguments (line 280)
        
        # Call to len(...): (line 280)
        # Processing the call arguments (line 280)
        
        # Call to get_outputs(...): (line 280)
        # Processing the call keyword arguments (line 280)
        kwargs_32427 = {}
        # Getting the type of 'cmd' (line 280)
        cmd_32425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 280)
        get_outputs_32426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 29), cmd_32425, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 280)
        get_outputs_call_result_32428 = invoke(stypy.reporting.localization.Localization(__file__, 280, 29), get_outputs_32426, *[], **kwargs_32427)
        
        # Processing the call keyword arguments (line 280)
        kwargs_32429 = {}
        # Getting the type of 'len' (line 280)
        len_32424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'len', False)
        # Calling len(args, kwargs) (line 280)
        len_call_result_32430 = invoke(stypy.reporting.localization.Localization(__file__, 280, 25), len_32424, *[get_outputs_call_result_32428], **kwargs_32429)
        
        int_32431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 49), 'int')
        # Processing the call keyword arguments (line 280)
        kwargs_32432 = {}
        # Getting the type of 'self' (line 280)
        self_32422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 280)
        assertEqual_32423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), self_32422, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 280)
        assertEqual_call_result_32433 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), assertEqual_32423, *[len_call_result_32430, int_32431], **kwargs_32432)
        
        
        # Assigning a Call to a Attribute (line 282):
        
        # Call to join(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'self' (line 282)
        self_32437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 37), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 282)
        tmp_dir_32438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 37), self_32437, 'tmp_dir')
        str_32439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 51), 'str', 'build')
        # Processing the call keyword arguments (line 282)
        kwargs_32440 = {}
        # Getting the type of 'os' (line 282)
        os_32434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 282)
        path_32435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), os_32434, 'path')
        # Obtaining the member 'join' of a type (line 282)
        join_32436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), path_32435, 'join')
        # Calling join(args, kwargs) (line 282)
        join_call_result_32441 = invoke(stypy.reporting.localization.Localization(__file__, 282, 24), join_32436, *[tmp_dir_32438, str_32439], **kwargs_32440)
        
        # Getting the type of 'cmd' (line 282)
        cmd_32442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'cmd')
        # Setting the type of the member 'build_lib' of a type (line 282)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), cmd_32442, 'build_lib', join_call_result_32441)
        
        # Assigning a Call to a Attribute (line 283):
        
        # Call to join(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'self' (line 283)
        self_32446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 38), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 283)
        tmp_dir_32447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 38), self_32446, 'tmp_dir')
        str_32448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 52), 'str', 'tempt')
        # Processing the call keyword arguments (line 283)
        kwargs_32449 = {}
        # Getting the type of 'os' (line 283)
        os_32443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 283)
        path_32444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 25), os_32443, 'path')
        # Obtaining the member 'join' of a type (line 283)
        join_32445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 25), path_32444, 'join')
        # Calling join(args, kwargs) (line 283)
        join_call_result_32450 = invoke(stypy.reporting.localization.Localization(__file__, 283, 25), join_32445, *[tmp_dir_32447, str_32448], **kwargs_32449)
        
        # Getting the type of 'cmd' (line 283)
        cmd_32451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'cmd')
        # Setting the type of the member 'build_temp' of a type (line 283)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), cmd_32451, 'build_temp', join_call_result_32450)
        
        # Assigning a Call to a Name (line 287):
        
        # Call to realpath(...): (line 287)
        # Processing the call arguments (line 287)
        
        # Call to mkdtemp(...): (line 287)
        # Processing the call keyword arguments (line 287)
        kwargs_32457 = {}
        # Getting the type of 'self' (line 287)
        self_32455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 41), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 287)
        mkdtemp_32456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 41), self_32455, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 287)
        mkdtemp_call_result_32458 = invoke(stypy.reporting.localization.Localization(__file__, 287, 41), mkdtemp_32456, *[], **kwargs_32457)
        
        # Processing the call keyword arguments (line 287)
        kwargs_32459 = {}
        # Getting the type of 'os' (line 287)
        os_32452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 287)
        path_32453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 24), os_32452, 'path')
        # Obtaining the member 'realpath' of a type (line 287)
        realpath_32454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 24), path_32453, 'realpath')
        # Calling realpath(args, kwargs) (line 287)
        realpath_call_result_32460 = invoke(stypy.reporting.localization.Localization(__file__, 287, 24), realpath_32454, *[mkdtemp_call_result_32458], **kwargs_32459)
        
        # Assigning a type to the variable 'other_tmp_dir' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'other_tmp_dir', realpath_call_result_32460)
        
        # Assigning a Call to a Name (line 288):
        
        # Call to getcwd(...): (line 288)
        # Processing the call keyword arguments (line 288)
        kwargs_32463 = {}
        # Getting the type of 'os' (line 288)
        os_32461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 17), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 288)
        getcwd_32462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 17), os_32461, 'getcwd')
        # Calling getcwd(args, kwargs) (line 288)
        getcwd_call_result_32464 = invoke(stypy.reporting.localization.Localization(__file__, 288, 17), getcwd_32462, *[], **kwargs_32463)
        
        # Assigning a type to the variable 'old_wd' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'old_wd', getcwd_call_result_32464)
        
        # Call to chdir(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'other_tmp_dir' (line 289)
        other_tmp_dir_32467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 17), 'other_tmp_dir', False)
        # Processing the call keyword arguments (line 289)
        kwargs_32468 = {}
        # Getting the type of 'os' (line 289)
        os_32465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 289)
        chdir_32466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), os_32465, 'chdir')
        # Calling chdir(args, kwargs) (line 289)
        chdir_call_result_32469 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), chdir_32466, *[other_tmp_dir_32467], **kwargs_32468)
        
        
        # Try-finally block (line 290)
        
        # Assigning a Num to a Attribute (line 291):
        int_32470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 26), 'int')
        # Getting the type of 'cmd' (line 291)
        cmd_32471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'cmd')
        # Setting the type of the member 'inplace' of a type (line 291)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), cmd_32471, 'inplace', int_32470)
        
        # Call to run(...): (line 292)
        # Processing the call keyword arguments (line 292)
        kwargs_32474 = {}
        # Getting the type of 'cmd' (line 292)
        cmd_32472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 292)
        run_32473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), cmd_32472, 'run')
        # Calling run(args, kwargs) (line 292)
        run_call_result_32475 = invoke(stypy.reporting.localization.Localization(__file__, 292, 12), run_32473, *[], **kwargs_32474)
        
        
        # Assigning a Subscript to a Name (line 293):
        
        # Obtaining the type of the subscript
        int_32476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 40), 'int')
        
        # Call to get_outputs(...): (line 293)
        # Processing the call keyword arguments (line 293)
        kwargs_32479 = {}
        # Getting the type of 'cmd' (line 293)
        cmd_32477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 22), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 293)
        get_outputs_32478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 22), cmd_32477, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 293)
        get_outputs_call_result_32480 = invoke(stypy.reporting.localization.Localization(__file__, 293, 22), get_outputs_32478, *[], **kwargs_32479)
        
        # Obtaining the member '__getitem__' of a type (line 293)
        getitem___32481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 22), get_outputs_call_result_32480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 293)
        subscript_call_result_32482 = invoke(stypy.reporting.localization.Localization(__file__, 293, 22), getitem___32481, int_32476)
        
        # Assigning a type to the variable 'so_file' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'so_file', subscript_call_result_32482)
        
        # finally branch of the try-finally block (line 290)
        
        # Call to chdir(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'old_wd' (line 295)
        old_wd_32485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'old_wd', False)
        # Processing the call keyword arguments (line 295)
        kwargs_32486 = {}
        # Getting the type of 'os' (line 295)
        os_32483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 295)
        chdir_32484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), os_32483, 'chdir')
        # Calling chdir(args, kwargs) (line 295)
        chdir_call_result_32487 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), chdir_32484, *[old_wd_32485], **kwargs_32486)
        
        
        
        # Call to assertTrue(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Call to exists(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'so_file' (line 296)
        so_file_32493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 39), 'so_file', False)
        # Processing the call keyword arguments (line 296)
        kwargs_32494 = {}
        # Getting the type of 'os' (line 296)
        os_32490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 296)
        path_32491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 24), os_32490, 'path')
        # Obtaining the member 'exists' of a type (line 296)
        exists_32492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 24), path_32491, 'exists')
        # Calling exists(args, kwargs) (line 296)
        exists_call_result_32495 = invoke(stypy.reporting.localization.Localization(__file__, 296, 24), exists_32492, *[so_file_32493], **kwargs_32494)
        
        # Processing the call keyword arguments (line 296)
        kwargs_32496 = {}
        # Getting the type of 'self' (line 296)
        self_32488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 296)
        assertTrue_32489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_32488, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 296)
        assertTrue_call_result_32497 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), assertTrue_32489, *[exists_call_result_32495], **kwargs_32496)
        
        
        # Call to assertEqual(...): (line 297)
        # Processing the call arguments (line 297)
        
        # Obtaining the type of the subscript
        int_32500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 51), 'int')
        
        # Call to splitext(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'so_file' (line 297)
        so_file_32504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 42), 'so_file', False)
        # Processing the call keyword arguments (line 297)
        kwargs_32505 = {}
        # Getting the type of 'os' (line 297)
        os_32501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 297)
        path_32502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), os_32501, 'path')
        # Obtaining the member 'splitext' of a type (line 297)
        splitext_32503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), path_32502, 'splitext')
        # Calling splitext(args, kwargs) (line 297)
        splitext_call_result_32506 = invoke(stypy.reporting.localization.Localization(__file__, 297, 25), splitext_32503, *[so_file_32504], **kwargs_32505)
        
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___32507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), splitext_call_result_32506, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_32508 = invoke(stypy.reporting.localization.Localization(__file__, 297, 25), getitem___32507, int_32500)
        
        
        # Call to get_config_var(...): (line 298)
        # Processing the call arguments (line 298)
        str_32511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 50), 'str', 'SO')
        # Processing the call keyword arguments (line 298)
        kwargs_32512 = {}
        # Getting the type of 'sysconfig' (line 298)
        sysconfig_32509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 25), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 298)
        get_config_var_32510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 25), sysconfig_32509, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 298)
        get_config_var_call_result_32513 = invoke(stypy.reporting.localization.Localization(__file__, 298, 25), get_config_var_32510, *[str_32511], **kwargs_32512)
        
        # Processing the call keyword arguments (line 297)
        kwargs_32514 = {}
        # Getting the type of 'self' (line 297)
        self_32498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 297)
        assertEqual_32499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), self_32498, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 297)
        assertEqual_call_result_32515 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), assertEqual_32499, *[subscript_call_result_32508, get_config_var_call_result_32513], **kwargs_32514)
        
        
        # Assigning a Call to a Name (line 299):
        
        # Call to dirname(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'so_file' (line 299)
        so_file_32519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 33), 'so_file', False)
        # Processing the call keyword arguments (line 299)
        kwargs_32520 = {}
        # Getting the type of 'os' (line 299)
        os_32516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 299)
        path_32517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 17), os_32516, 'path')
        # Obtaining the member 'dirname' of a type (line 299)
        dirname_32518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 17), path_32517, 'dirname')
        # Calling dirname(args, kwargs) (line 299)
        dirname_call_result_32521 = invoke(stypy.reporting.localization.Localization(__file__, 299, 17), dirname_32518, *[so_file_32519], **kwargs_32520)
        
        # Assigning a type to the variable 'so_dir' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'so_dir', dirname_call_result_32521)
        
        # Call to assertEqual(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'so_dir' (line 300)
        so_dir_32524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 25), 'so_dir', False)
        # Getting the type of 'other_tmp_dir' (line 300)
        other_tmp_dir_32525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 33), 'other_tmp_dir', False)
        # Processing the call keyword arguments (line 300)
        kwargs_32526 = {}
        # Getting the type of 'self' (line 300)
        self_32522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 300)
        assertEqual_32523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), self_32522, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 300)
        assertEqual_call_result_32527 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), assertEqual_32523, *[so_dir_32524, other_tmp_dir_32525], **kwargs_32526)
        
        
        # Assigning a Name to a Attribute (line 301):
        # Getting the type of 'None' (line 301)
        None_32528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'None')
        # Getting the type of 'cmd' (line 301)
        cmd_32529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'cmd')
        # Setting the type of the member 'compiler' of a type (line 301)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), cmd_32529, 'compiler', None_32528)
        
        # Assigning a Num to a Attribute (line 302):
        int_32530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 22), 'int')
        # Getting the type of 'cmd' (line 302)
        cmd_32531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'cmd')
        # Setting the type of the member 'inplace' of a type (line 302)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), cmd_32531, 'inplace', int_32530)
        
        # Call to run(...): (line 303)
        # Processing the call keyword arguments (line 303)
        kwargs_32534 = {}
        # Getting the type of 'cmd' (line 303)
        cmd_32532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 303)
        run_32533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 8), cmd_32532, 'run')
        # Calling run(args, kwargs) (line 303)
        run_call_result_32535 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), run_32533, *[], **kwargs_32534)
        
        
        # Assigning a Subscript to a Name (line 304):
        
        # Obtaining the type of the subscript
        int_32536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 36), 'int')
        
        # Call to get_outputs(...): (line 304)
        # Processing the call keyword arguments (line 304)
        kwargs_32539 = {}
        # Getting the type of 'cmd' (line 304)
        cmd_32537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 304)
        get_outputs_32538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 18), cmd_32537, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 304)
        get_outputs_call_result_32540 = invoke(stypy.reporting.localization.Localization(__file__, 304, 18), get_outputs_32538, *[], **kwargs_32539)
        
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___32541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 18), get_outputs_call_result_32540, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_32542 = invoke(stypy.reporting.localization.Localization(__file__, 304, 18), getitem___32541, int_32536)
        
        # Assigning a type to the variable 'so_file' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'so_file', subscript_call_result_32542)
        
        # Call to assertTrue(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Call to exists(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'so_file' (line 305)
        so_file_32548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 39), 'so_file', False)
        # Processing the call keyword arguments (line 305)
        kwargs_32549 = {}
        # Getting the type of 'os' (line 305)
        os_32545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 305)
        path_32546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 24), os_32545, 'path')
        # Obtaining the member 'exists' of a type (line 305)
        exists_32547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 24), path_32546, 'exists')
        # Calling exists(args, kwargs) (line 305)
        exists_call_result_32550 = invoke(stypy.reporting.localization.Localization(__file__, 305, 24), exists_32547, *[so_file_32548], **kwargs_32549)
        
        # Processing the call keyword arguments (line 305)
        kwargs_32551 = {}
        # Getting the type of 'self' (line 305)
        self_32543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 305)
        assertTrue_32544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), self_32543, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 305)
        assertTrue_call_result_32552 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), assertTrue_32544, *[exists_call_result_32550], **kwargs_32551)
        
        
        # Call to assertEqual(...): (line 306)
        # Processing the call arguments (line 306)
        
        # Obtaining the type of the subscript
        int_32555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 51), 'int')
        
        # Call to splitext(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'so_file' (line 306)
        so_file_32559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 42), 'so_file', False)
        # Processing the call keyword arguments (line 306)
        kwargs_32560 = {}
        # Getting the type of 'os' (line 306)
        os_32556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 306)
        path_32557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 25), os_32556, 'path')
        # Obtaining the member 'splitext' of a type (line 306)
        splitext_32558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 25), path_32557, 'splitext')
        # Calling splitext(args, kwargs) (line 306)
        splitext_call_result_32561 = invoke(stypy.reporting.localization.Localization(__file__, 306, 25), splitext_32558, *[so_file_32559], **kwargs_32560)
        
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___32562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 25), splitext_call_result_32561, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_32563 = invoke(stypy.reporting.localization.Localization(__file__, 306, 25), getitem___32562, int_32555)
        
        
        # Call to get_config_var(...): (line 307)
        # Processing the call arguments (line 307)
        str_32566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 50), 'str', 'SO')
        # Processing the call keyword arguments (line 307)
        kwargs_32567 = {}
        # Getting the type of 'sysconfig' (line 307)
        sysconfig_32564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 25), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 307)
        get_config_var_32565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 25), sysconfig_32564, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 307)
        get_config_var_call_result_32568 = invoke(stypy.reporting.localization.Localization(__file__, 307, 25), get_config_var_32565, *[str_32566], **kwargs_32567)
        
        # Processing the call keyword arguments (line 306)
        kwargs_32569 = {}
        # Getting the type of 'self' (line 306)
        self_32553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 306)
        assertEqual_32554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), self_32553, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 306)
        assertEqual_call_result_32570 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), assertEqual_32554, *[subscript_call_result_32563, get_config_var_call_result_32568], **kwargs_32569)
        
        
        # Assigning a Call to a Name (line 308):
        
        # Call to dirname(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'so_file' (line 308)
        so_file_32574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 33), 'so_file', False)
        # Processing the call keyword arguments (line 308)
        kwargs_32575 = {}
        # Getting the type of 'os' (line 308)
        os_32571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 308)
        path_32572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 17), os_32571, 'path')
        # Obtaining the member 'dirname' of a type (line 308)
        dirname_32573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 17), path_32572, 'dirname')
        # Calling dirname(args, kwargs) (line 308)
        dirname_call_result_32576 = invoke(stypy.reporting.localization.Localization(__file__, 308, 17), dirname_32573, *[so_file_32574], **kwargs_32575)
        
        # Assigning a type to the variable 'so_dir' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'so_dir', dirname_call_result_32576)
        
        # Call to assertEqual(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'so_dir' (line 309)
        so_dir_32579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 25), 'so_dir', False)
        # Getting the type of 'cmd' (line 309)
        cmd_32580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 33), 'cmd', False)
        # Obtaining the member 'build_lib' of a type (line 309)
        build_lib_32581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 33), cmd_32580, 'build_lib')
        # Processing the call keyword arguments (line 309)
        kwargs_32582 = {}
        # Getting the type of 'self' (line 309)
        self_32577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 309)
        assertEqual_32578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_32577, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 309)
        assertEqual_call_result_32583 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), assertEqual_32578, *[so_dir_32579, build_lib_32581], **kwargs_32582)
        
        
        # Assigning a Call to a Name (line 312):
        
        # Call to get_finalized_command(...): (line 312)
        # Processing the call arguments (line 312)
        str_32586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 45), 'str', 'build_py')
        # Processing the call keyword arguments (line 312)
        kwargs_32587 = {}
        # Getting the type of 'cmd' (line 312)
        cmd_32584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'cmd', False)
        # Obtaining the member 'get_finalized_command' of a type (line 312)
        get_finalized_command_32585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 19), cmd_32584, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 312)
        get_finalized_command_call_result_32588 = invoke(stypy.reporting.localization.Localization(__file__, 312, 19), get_finalized_command_32585, *[str_32586], **kwargs_32587)
        
        # Assigning a type to the variable 'build_py' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'build_py', get_finalized_command_call_result_32588)
        
        # Assigning a Dict to a Attribute (line 313):
        
        # Obtaining an instance of the builtin type 'dict' (line 313)
        dict_32589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 31), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 313)
        # Adding element type (key, value) (line 313)
        str_32590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 32), 'str', '')
        str_32591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 36), 'str', 'bar')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 31), dict_32589, (str_32590, str_32591))
        
        # Getting the type of 'build_py' (line 313)
        build_py_32592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'build_py')
        # Setting the type of the member 'package_dir' of a type (line 313)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), build_py_32592, 'package_dir', dict_32589)
        
        # Assigning a Call to a Name (line 314):
        
        # Call to get_ext_fullpath(...): (line 314)
        # Processing the call arguments (line 314)
        str_32595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 36), 'str', 'foo')
        # Processing the call keyword arguments (line 314)
        kwargs_32596 = {}
        # Getting the type of 'cmd' (line 314)
        cmd_32593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 314)
        get_ext_fullpath_32594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 15), cmd_32593, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 314)
        get_ext_fullpath_call_result_32597 = invoke(stypy.reporting.localization.Localization(__file__, 314, 15), get_ext_fullpath_32594, *[str_32595], **kwargs_32596)
        
        # Assigning a type to the variable 'path' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'path', get_ext_fullpath_call_result_32597)
        
        # Assigning a Subscript to a Name (line 316):
        
        # Obtaining the type of the subscript
        int_32598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 35), 'int')
        
        # Call to split(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'path' (line 316)
        path_32602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 29), 'path', False)
        # Processing the call keyword arguments (line 316)
        kwargs_32603 = {}
        # Getting the type of 'os' (line 316)
        os_32599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 316)
        path_32600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 15), os_32599, 'path')
        # Obtaining the member 'split' of a type (line 316)
        split_32601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 15), path_32600, 'split')
        # Calling split(args, kwargs) (line 316)
        split_call_result_32604 = invoke(stypy.reporting.localization.Localization(__file__, 316, 15), split_32601, *[path_32602], **kwargs_32603)
        
        # Obtaining the member '__getitem__' of a type (line 316)
        getitem___32605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 15), split_call_result_32604, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 316)
        subscript_call_result_32606 = invoke(stypy.reporting.localization.Localization(__file__, 316, 15), getitem___32605, int_32598)
        
        # Assigning a type to the variable 'path' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'path', subscript_call_result_32606)
        
        # Call to assertEqual(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'path' (line 317)
        path_32609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 25), 'path', False)
        # Getting the type of 'cmd' (line 317)
        cmd_32610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 31), 'cmd', False)
        # Obtaining the member 'build_lib' of a type (line 317)
        build_lib_32611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 31), cmd_32610, 'build_lib')
        # Processing the call keyword arguments (line 317)
        kwargs_32612 = {}
        # Getting the type of 'self' (line 317)
        self_32607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 317)
        assertEqual_32608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), self_32607, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 317)
        assertEqual_call_result_32613 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), assertEqual_32608, *[path_32609, build_lib_32611], **kwargs_32612)
        
        
        # Assigning a Num to a Attribute (line 320):
        int_32614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 22), 'int')
        # Getting the type of 'cmd' (line 320)
        cmd_32615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'cmd')
        # Setting the type of the member 'inplace' of a type (line 320)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), cmd_32615, 'inplace', int_32614)
        
        # Assigning a Call to a Name (line 321):
        
        # Call to realpath(...): (line 321)
        # Processing the call arguments (line 321)
        
        # Call to mkdtemp(...): (line 321)
        # Processing the call keyword arguments (line 321)
        kwargs_32621 = {}
        # Getting the type of 'self' (line 321)
        self_32619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 41), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 321)
        mkdtemp_32620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 41), self_32619, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 321)
        mkdtemp_call_result_32622 = invoke(stypy.reporting.localization.Localization(__file__, 321, 41), mkdtemp_32620, *[], **kwargs_32621)
        
        # Processing the call keyword arguments (line 321)
        kwargs_32623 = {}
        # Getting the type of 'os' (line 321)
        os_32616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 321)
        path_32617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 24), os_32616, 'path')
        # Obtaining the member 'realpath' of a type (line 321)
        realpath_32618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 24), path_32617, 'realpath')
        # Calling realpath(args, kwargs) (line 321)
        realpath_call_result_32624 = invoke(stypy.reporting.localization.Localization(__file__, 321, 24), realpath_32618, *[mkdtemp_call_result_32622], **kwargs_32623)
        
        # Assigning a type to the variable 'other_tmp_dir' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'other_tmp_dir', realpath_call_result_32624)
        
        # Assigning a Call to a Name (line 322):
        
        # Call to getcwd(...): (line 322)
        # Processing the call keyword arguments (line 322)
        kwargs_32627 = {}
        # Getting the type of 'os' (line 322)
        os_32625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 322)
        getcwd_32626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 17), os_32625, 'getcwd')
        # Calling getcwd(args, kwargs) (line 322)
        getcwd_call_result_32628 = invoke(stypy.reporting.localization.Localization(__file__, 322, 17), getcwd_32626, *[], **kwargs_32627)
        
        # Assigning a type to the variable 'old_wd' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'old_wd', getcwd_call_result_32628)
        
        # Call to chdir(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'other_tmp_dir' (line 323)
        other_tmp_dir_32631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 17), 'other_tmp_dir', False)
        # Processing the call keyword arguments (line 323)
        kwargs_32632 = {}
        # Getting the type of 'os' (line 323)
        os_32629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 323)
        chdir_32630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), os_32629, 'chdir')
        # Calling chdir(args, kwargs) (line 323)
        chdir_call_result_32633 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), chdir_32630, *[other_tmp_dir_32631], **kwargs_32632)
        
        
        # Try-finally block (line 324)
        
        # Assigning a Call to a Name (line 325):
        
        # Call to get_ext_fullpath(...): (line 325)
        # Processing the call arguments (line 325)
        str_32636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 40), 'str', 'foo')
        # Processing the call keyword arguments (line 325)
        kwargs_32637 = {}
        # Getting the type of 'cmd' (line 325)
        cmd_32634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 325)
        get_ext_fullpath_32635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 19), cmd_32634, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 325)
        get_ext_fullpath_call_result_32638 = invoke(stypy.reporting.localization.Localization(__file__, 325, 19), get_ext_fullpath_32635, *[str_32636], **kwargs_32637)
        
        # Assigning a type to the variable 'path' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'path', get_ext_fullpath_call_result_32638)
        
        # finally branch of the try-finally block (line 324)
        
        # Call to chdir(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'old_wd' (line 327)
        old_wd_32641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), 'old_wd', False)
        # Processing the call keyword arguments (line 327)
        kwargs_32642 = {}
        # Getting the type of 'os' (line 327)
        os_32639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 327)
        chdir_32640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 12), os_32639, 'chdir')
        # Calling chdir(args, kwargs) (line 327)
        chdir_call_result_32643 = invoke(stypy.reporting.localization.Localization(__file__, 327, 12), chdir_32640, *[old_wd_32641], **kwargs_32642)
        
        
        
        # Assigning a Subscript to a Name (line 329):
        
        # Obtaining the type of the subscript
        int_32644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 35), 'int')
        
        # Call to split(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'path' (line 329)
        path_32648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 29), 'path', False)
        # Processing the call keyword arguments (line 329)
        kwargs_32649 = {}
        # Getting the type of 'os' (line 329)
        os_32645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 329)
        path_32646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), os_32645, 'path')
        # Obtaining the member 'split' of a type (line 329)
        split_32647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), path_32646, 'split')
        # Calling split(args, kwargs) (line 329)
        split_call_result_32650 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), split_32647, *[path_32648], **kwargs_32649)
        
        # Obtaining the member '__getitem__' of a type (line 329)
        getitem___32651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), split_call_result_32650, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 329)
        subscript_call_result_32652 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), getitem___32651, int_32644)
        
        # Assigning a type to the variable 'path' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'path', subscript_call_result_32652)
        
        # Assigning a Subscript to a Name (line 330):
        
        # Obtaining the type of the subscript
        int_32653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 38), 'int')
        
        # Call to split(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'path' (line 330)
        path_32657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 32), 'path', False)
        # Processing the call keyword arguments (line 330)
        kwargs_32658 = {}
        # Getting the type of 'os' (line 330)
        os_32654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 330)
        path_32655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 18), os_32654, 'path')
        # Obtaining the member 'split' of a type (line 330)
        split_32656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 18), path_32655, 'split')
        # Calling split(args, kwargs) (line 330)
        split_call_result_32659 = invoke(stypy.reporting.localization.Localization(__file__, 330, 18), split_32656, *[path_32657], **kwargs_32658)
        
        # Obtaining the member '__getitem__' of a type (line 330)
        getitem___32660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 18), split_call_result_32659, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 330)
        subscript_call_result_32661 = invoke(stypy.reporting.localization.Localization(__file__, 330, 18), getitem___32660, int_32653)
        
        # Assigning a type to the variable 'lastdir' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'lastdir', subscript_call_result_32661)
        
        # Call to assertEqual(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'lastdir' (line 331)
        lastdir_32664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 25), 'lastdir', False)
        str_32665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 34), 'str', 'bar')
        # Processing the call keyword arguments (line 331)
        kwargs_32666 = {}
        # Getting the type of 'self' (line 331)
        self_32662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 331)
        assertEqual_32663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), self_32662, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 331)
        assertEqual_call_result_32667 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), assertEqual_32663, *[lastdir_32664, str_32665], **kwargs_32666)
        
        
        # ################# End of 'test_get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_32668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32668)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_outputs'
        return stypy_return_type_32668


    @norecursion
    def test_ext_fullpath(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ext_fullpath'
        module_type_store = module_type_store.open_function_context('test_ext_fullpath', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_ext_fullpath')
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_ext_fullpath.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_ext_fullpath', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ext_fullpath', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ext_fullpath(...)' code ##################

        
        # Assigning a Subscript to a Name (line 334):
        
        # Obtaining the type of the subscript
        str_32669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 42), 'str', 'SO')
        
        # Call to get_config_vars(...): (line 334)
        # Processing the call keyword arguments (line 334)
        kwargs_32672 = {}
        # Getting the type of 'sysconfig' (line 334)
        sysconfig_32670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 14), 'sysconfig', False)
        # Obtaining the member 'get_config_vars' of a type (line 334)
        get_config_vars_32671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 14), sysconfig_32670, 'get_config_vars')
        # Calling get_config_vars(args, kwargs) (line 334)
        get_config_vars_call_result_32673 = invoke(stypy.reporting.localization.Localization(__file__, 334, 14), get_config_vars_32671, *[], **kwargs_32672)
        
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___32674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 14), get_config_vars_call_result_32673, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_32675 = invoke(stypy.reporting.localization.Localization(__file__, 334, 14), getitem___32674, str_32669)
        
        # Assigning a type to the variable 'ext' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'ext', subscript_call_result_32675)
        
        # Assigning a Call to a Name (line 335):
        
        # Call to Distribution(...): (line 335)
        # Processing the call keyword arguments (line 335)
        kwargs_32677 = {}
        # Getting the type of 'Distribution' (line 335)
        Distribution_32676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 335)
        Distribution_call_result_32678 = invoke(stypy.reporting.localization.Localization(__file__, 335, 15), Distribution_32676, *[], **kwargs_32677)
        
        # Assigning a type to the variable 'dist' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'dist', Distribution_call_result_32678)
        
        # Assigning a Call to a Name (line 336):
        
        # Call to build_ext(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'dist' (line 336)
        dist_32680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'dist', False)
        # Processing the call keyword arguments (line 336)
        kwargs_32681 = {}
        # Getting the type of 'build_ext' (line 336)
        build_ext_32679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 336)
        build_ext_call_result_32682 = invoke(stypy.reporting.localization.Localization(__file__, 336, 14), build_ext_32679, *[dist_32680], **kwargs_32681)
        
        # Assigning a type to the variable 'cmd' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'cmd', build_ext_call_result_32682)
        
        # Assigning a Num to a Attribute (line 337):
        int_32683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 22), 'int')
        # Getting the type of 'cmd' (line 337)
        cmd_32684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'cmd')
        # Setting the type of the member 'inplace' of a type (line 337)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), cmd_32684, 'inplace', int_32683)
        
        # Assigning a Dict to a Attribute (line 338):
        
        # Obtaining an instance of the builtin type 'dict' (line 338)
        dict_32685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 39), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 338)
        # Adding element type (key, value) (line 338)
        str_32686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 40), 'str', '')
        str_32687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 44), 'str', 'src')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 39), dict_32685, (str_32686, str_32687))
        
        # Getting the type of 'cmd' (line 338)
        cmd_32688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 338)
        distribution_32689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), cmd_32688, 'distribution')
        # Setting the type of the member 'package_dir' of a type (line 338)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), distribution_32689, 'package_dir', dict_32685)
        
        # Assigning a List to a Attribute (line 339):
        
        # Obtaining an instance of the builtin type 'list' (line 339)
        list_32690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 339)
        # Adding element type (line 339)
        str_32691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 37), 'str', 'lxml')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 36), list_32690, str_32691)
        # Adding element type (line 339)
        str_32692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 45), 'str', 'lxml.html')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 36), list_32690, str_32692)
        
        # Getting the type of 'cmd' (line 339)
        cmd_32693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 339)
        distribution_32694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), cmd_32693, 'distribution')
        # Setting the type of the member 'packages' of a type (line 339)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), distribution_32694, 'packages', list_32690)
        
        # Assigning a Call to a Name (line 340):
        
        # Call to getcwd(...): (line 340)
        # Processing the call keyword arguments (line 340)
        kwargs_32697 = {}
        # Getting the type of 'os' (line 340)
        os_32695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 17), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 340)
        getcwd_32696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 17), os_32695, 'getcwd')
        # Calling getcwd(args, kwargs) (line 340)
        getcwd_call_result_32698 = invoke(stypy.reporting.localization.Localization(__file__, 340, 17), getcwd_32696, *[], **kwargs_32697)
        
        # Assigning a type to the variable 'curdir' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'curdir', getcwd_call_result_32698)
        
        # Assigning a Call to a Name (line 341):
        
        # Call to join(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'curdir' (line 341)
        curdir_32702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'curdir', False)
        str_32703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 38), 'str', 'src')
        str_32704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 45), 'str', 'lxml')
        str_32705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 53), 'str', 'etree')
        # Getting the type of 'ext' (line 341)
        ext_32706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 63), 'ext', False)
        # Applying the binary operator '+' (line 341)
        result_add_32707 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 53), '+', str_32705, ext_32706)
        
        # Processing the call keyword arguments (line 341)
        kwargs_32708 = {}
        # Getting the type of 'os' (line 341)
        os_32699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 341)
        path_32700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 17), os_32699, 'path')
        # Obtaining the member 'join' of a type (line 341)
        join_32701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 17), path_32700, 'join')
        # Calling join(args, kwargs) (line 341)
        join_call_result_32709 = invoke(stypy.reporting.localization.Localization(__file__, 341, 17), join_32701, *[curdir_32702, str_32703, str_32704, result_add_32707], **kwargs_32708)
        
        # Assigning a type to the variable 'wanted' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'wanted', join_call_result_32709)
        
        # Assigning a Call to a Name (line 342):
        
        # Call to get_ext_fullpath(...): (line 342)
        # Processing the call arguments (line 342)
        str_32712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 36), 'str', 'lxml.etree')
        # Processing the call keyword arguments (line 342)
        kwargs_32713 = {}
        # Getting the type of 'cmd' (line 342)
        cmd_32710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 342)
        get_ext_fullpath_32711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), cmd_32710, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 342)
        get_ext_fullpath_call_result_32714 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), get_ext_fullpath_32711, *[str_32712], **kwargs_32713)
        
        # Assigning a type to the variable 'path' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'path', get_ext_fullpath_call_result_32714)
        
        # Call to assertEqual(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'wanted' (line 343)
        wanted_32717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'wanted', False)
        # Getting the type of 'path' (line 343)
        path_32718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 33), 'path', False)
        # Processing the call keyword arguments (line 343)
        kwargs_32719 = {}
        # Getting the type of 'self' (line 343)
        self_32715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 343)
        assertEqual_32716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_32715, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 343)
        assertEqual_call_result_32720 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), assertEqual_32716, *[wanted_32717, path_32718], **kwargs_32719)
        
        
        # Assigning a Num to a Attribute (line 346):
        int_32721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 22), 'int')
        # Getting the type of 'cmd' (line 346)
        cmd_32722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'cmd')
        # Setting the type of the member 'inplace' of a type (line 346)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), cmd_32722, 'inplace', int_32721)
        
        # Assigning a Call to a Attribute (line 347):
        
        # Call to join(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'curdir' (line 347)
        curdir_32726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 37), 'curdir', False)
        str_32727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 45), 'str', 'tmpdir')
        # Processing the call keyword arguments (line 347)
        kwargs_32728 = {}
        # Getting the type of 'os' (line 347)
        os_32723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 347)
        path_32724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 24), os_32723, 'path')
        # Obtaining the member 'join' of a type (line 347)
        join_32725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 24), path_32724, 'join')
        # Calling join(args, kwargs) (line 347)
        join_call_result_32729 = invoke(stypy.reporting.localization.Localization(__file__, 347, 24), join_32725, *[curdir_32726, str_32727], **kwargs_32728)
        
        # Getting the type of 'cmd' (line 347)
        cmd_32730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'cmd')
        # Setting the type of the member 'build_lib' of a type (line 347)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), cmd_32730, 'build_lib', join_call_result_32729)
        
        # Assigning a Call to a Name (line 348):
        
        # Call to join(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'curdir' (line 348)
        curdir_32734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 30), 'curdir', False)
        str_32735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 38), 'str', 'tmpdir')
        str_32736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 48), 'str', 'lxml')
        str_32737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 56), 'str', 'etree')
        # Getting the type of 'ext' (line 348)
        ext_32738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 66), 'ext', False)
        # Applying the binary operator '+' (line 348)
        result_add_32739 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 56), '+', str_32737, ext_32738)
        
        # Processing the call keyword arguments (line 348)
        kwargs_32740 = {}
        # Getting the type of 'os' (line 348)
        os_32731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 348)
        path_32732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 17), os_32731, 'path')
        # Obtaining the member 'join' of a type (line 348)
        join_32733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 17), path_32732, 'join')
        # Calling join(args, kwargs) (line 348)
        join_call_result_32741 = invoke(stypy.reporting.localization.Localization(__file__, 348, 17), join_32733, *[curdir_32734, str_32735, str_32736, result_add_32739], **kwargs_32740)
        
        # Assigning a type to the variable 'wanted' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'wanted', join_call_result_32741)
        
        # Assigning a Call to a Name (line 349):
        
        # Call to get_ext_fullpath(...): (line 349)
        # Processing the call arguments (line 349)
        str_32744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 36), 'str', 'lxml.etree')
        # Processing the call keyword arguments (line 349)
        kwargs_32745 = {}
        # Getting the type of 'cmd' (line 349)
        cmd_32742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 349)
        get_ext_fullpath_32743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), cmd_32742, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 349)
        get_ext_fullpath_call_result_32746 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), get_ext_fullpath_32743, *[str_32744], **kwargs_32745)
        
        # Assigning a type to the variable 'path' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'path', get_ext_fullpath_call_result_32746)
        
        # Call to assertEqual(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'wanted' (line 350)
        wanted_32749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 25), 'wanted', False)
        # Getting the type of 'path' (line 350)
        path_32750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 33), 'path', False)
        # Processing the call keyword arguments (line 350)
        kwargs_32751 = {}
        # Getting the type of 'self' (line 350)
        self_32747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 350)
        assertEqual_32748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_32747, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 350)
        assertEqual_call_result_32752 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), assertEqual_32748, *[wanted_32749, path_32750], **kwargs_32751)
        
        
        # Assigning a Call to a Name (line 353):
        
        # Call to get_finalized_command(...): (line 353)
        # Processing the call arguments (line 353)
        str_32755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 45), 'str', 'build_py')
        # Processing the call keyword arguments (line 353)
        kwargs_32756 = {}
        # Getting the type of 'cmd' (line 353)
        cmd_32753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 'cmd', False)
        # Obtaining the member 'get_finalized_command' of a type (line 353)
        get_finalized_command_32754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 19), cmd_32753, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 353)
        get_finalized_command_call_result_32757 = invoke(stypy.reporting.localization.Localization(__file__, 353, 19), get_finalized_command_32754, *[str_32755], **kwargs_32756)
        
        # Assigning a type to the variable 'build_py' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'build_py', get_finalized_command_call_result_32757)
        
        # Assigning a Dict to a Attribute (line 354):
        
        # Obtaining an instance of the builtin type 'dict' (line 354)
        dict_32758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 31), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 354)
        
        # Getting the type of 'build_py' (line 354)
        build_py_32759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'build_py')
        # Setting the type of the member 'package_dir' of a type (line 354)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), build_py_32759, 'package_dir', dict_32758)
        
        # Assigning a List to a Attribute (line 355):
        
        # Obtaining an instance of the builtin type 'list' (line 355)
        list_32760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 355)
        # Adding element type (line 355)
        str_32761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 37), 'str', 'twisted')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 36), list_32760, str_32761)
        # Adding element type (line 355)
        str_32762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 48), 'str', 'twisted.runner.portmap')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 36), list_32760, str_32762)
        
        # Getting the type of 'cmd' (line 355)
        cmd_32763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 355)
        distribution_32764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), cmd_32763, 'distribution')
        # Setting the type of the member 'packages' of a type (line 355)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), distribution_32764, 'packages', list_32760)
        
        # Assigning a Call to a Name (line 356):
        
        # Call to get_ext_fullpath(...): (line 356)
        # Processing the call arguments (line 356)
        str_32767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 36), 'str', 'twisted.runner.portmap')
        # Processing the call keyword arguments (line 356)
        kwargs_32768 = {}
        # Getting the type of 'cmd' (line 356)
        cmd_32765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 356)
        get_ext_fullpath_32766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 15), cmd_32765, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 356)
        get_ext_fullpath_call_result_32769 = invoke(stypy.reporting.localization.Localization(__file__, 356, 15), get_ext_fullpath_32766, *[str_32767], **kwargs_32768)
        
        # Assigning a type to the variable 'path' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'path', get_ext_fullpath_call_result_32769)
        
        # Assigning a Call to a Name (line 357):
        
        # Call to join(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'curdir' (line 357)
        curdir_32773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 30), 'curdir', False)
        str_32774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 38), 'str', 'tmpdir')
        str_32775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 48), 'str', 'twisted')
        str_32776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 59), 'str', 'runner')
        str_32777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 30), 'str', 'portmap')
        # Getting the type of 'ext' (line 358)
        ext_32778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 42), 'ext', False)
        # Applying the binary operator '+' (line 358)
        result_add_32779 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 30), '+', str_32777, ext_32778)
        
        # Processing the call keyword arguments (line 357)
        kwargs_32780 = {}
        # Getting the type of 'os' (line 357)
        os_32770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 357)
        path_32771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 17), os_32770, 'path')
        # Obtaining the member 'join' of a type (line 357)
        join_32772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 17), path_32771, 'join')
        # Calling join(args, kwargs) (line 357)
        join_call_result_32781 = invoke(stypy.reporting.localization.Localization(__file__, 357, 17), join_32772, *[curdir_32773, str_32774, str_32775, str_32776, result_add_32779], **kwargs_32780)
        
        # Assigning a type to the variable 'wanted' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'wanted', join_call_result_32781)
        
        # Call to assertEqual(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'wanted' (line 359)
        wanted_32784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 25), 'wanted', False)
        # Getting the type of 'path' (line 359)
        path_32785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 33), 'path', False)
        # Processing the call keyword arguments (line 359)
        kwargs_32786 = {}
        # Getting the type of 'self' (line 359)
        self_32782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 359)
        assertEqual_32783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), self_32782, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 359)
        assertEqual_call_result_32787 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), assertEqual_32783, *[wanted_32784, path_32785], **kwargs_32786)
        
        
        # Assigning a Num to a Attribute (line 362):
        int_32788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 22), 'int')
        # Getting the type of 'cmd' (line 362)
        cmd_32789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'cmd')
        # Setting the type of the member 'inplace' of a type (line 362)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), cmd_32789, 'inplace', int_32788)
        
        # Assigning a Call to a Name (line 363):
        
        # Call to get_ext_fullpath(...): (line 363)
        # Processing the call arguments (line 363)
        str_32792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 36), 'str', 'twisted.runner.portmap')
        # Processing the call keyword arguments (line 363)
        kwargs_32793 = {}
        # Getting the type of 'cmd' (line 363)
        cmd_32790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 363)
        get_ext_fullpath_32791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 15), cmd_32790, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 363)
        get_ext_fullpath_call_result_32794 = invoke(stypy.reporting.localization.Localization(__file__, 363, 15), get_ext_fullpath_32791, *[str_32792], **kwargs_32793)
        
        # Assigning a type to the variable 'path' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'path', get_ext_fullpath_call_result_32794)
        
        # Assigning a Call to a Name (line 364):
        
        # Call to join(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'curdir' (line 364)
        curdir_32798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'curdir', False)
        str_32799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 38), 'str', 'twisted')
        str_32800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 49), 'str', 'runner')
        str_32801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 59), 'str', 'portmap')
        # Getting the type of 'ext' (line 364)
        ext_32802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 71), 'ext', False)
        # Applying the binary operator '+' (line 364)
        result_add_32803 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 59), '+', str_32801, ext_32802)
        
        # Processing the call keyword arguments (line 364)
        kwargs_32804 = {}
        # Getting the type of 'os' (line 364)
        os_32795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 364)
        path_32796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 17), os_32795, 'path')
        # Obtaining the member 'join' of a type (line 364)
        join_32797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 17), path_32796, 'join')
        # Calling join(args, kwargs) (line 364)
        join_call_result_32805 = invoke(stypy.reporting.localization.Localization(__file__, 364, 17), join_32797, *[curdir_32798, str_32799, str_32800, result_add_32803], **kwargs_32804)
        
        # Assigning a type to the variable 'wanted' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'wanted', join_call_result_32805)
        
        # Call to assertEqual(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'wanted' (line 365)
        wanted_32808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 25), 'wanted', False)
        # Getting the type of 'path' (line 365)
        path_32809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 33), 'path', False)
        # Processing the call keyword arguments (line 365)
        kwargs_32810 = {}
        # Getting the type of 'self' (line 365)
        self_32806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 365)
        assertEqual_32807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), self_32806, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 365)
        assertEqual_call_result_32811 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), assertEqual_32807, *[wanted_32808, path_32809], **kwargs_32810)
        
        
        # ################# End of 'test_ext_fullpath(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ext_fullpath' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_32812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32812)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ext_fullpath'
        return stypy_return_type_32812


    @norecursion
    def test_build_ext_inplace(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_build_ext_inplace'
        module_type_store = module_type_store.open_function_context('test_build_ext_inplace', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_build_ext_inplace')
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_build_ext_inplace.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_build_ext_inplace', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_build_ext_inplace', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_build_ext_inplace(...)' code ##################

        
        # Assigning a Call to a Name (line 368):
        
        # Call to join(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'self' (line 368)
        self_32816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 31), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 368)
        tmp_dir_32817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 31), self_32816, 'tmp_dir')
        str_32818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 45), 'str', 'lxml.etree.c')
        # Processing the call keyword arguments (line 368)
        kwargs_32819 = {}
        # Getting the type of 'os' (line 368)
        os_32813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 368)
        path_32814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 18), os_32813, 'path')
        # Obtaining the member 'join' of a type (line 368)
        join_32815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 18), path_32814, 'join')
        # Calling join(args, kwargs) (line 368)
        join_call_result_32820 = invoke(stypy.reporting.localization.Localization(__file__, 368, 18), join_32815, *[tmp_dir_32817, str_32818], **kwargs_32819)
        
        # Assigning a type to the variable 'etree_c' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'etree_c', join_call_result_32820)
        
        # Assigning a Call to a Name (line 369):
        
        # Call to Extension(...): (line 369)
        # Processing the call arguments (line 369)
        str_32822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 30), 'str', 'lxml.etree')
        
        # Obtaining an instance of the builtin type 'list' (line 369)
        list_32823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 369)
        # Adding element type (line 369)
        # Getting the type of 'etree_c' (line 369)
        etree_c_32824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 45), 'etree_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 44), list_32823, etree_c_32824)
        
        # Processing the call keyword arguments (line 369)
        kwargs_32825 = {}
        # Getting the type of 'Extension' (line 369)
        Extension_32821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 20), 'Extension', False)
        # Calling Extension(args, kwargs) (line 369)
        Extension_call_result_32826 = invoke(stypy.reporting.localization.Localization(__file__, 369, 20), Extension_32821, *[str_32822, list_32823], **kwargs_32825)
        
        # Assigning a type to the variable 'etree_ext' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'etree_ext', Extension_call_result_32826)
        
        # Assigning a Call to a Name (line 370):
        
        # Call to Distribution(...): (line 370)
        # Processing the call arguments (line 370)
        
        # Obtaining an instance of the builtin type 'dict' (line 370)
        dict_32828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 370)
        # Adding element type (key, value) (line 370)
        str_32829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 29), 'str', 'name')
        str_32830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 37), 'str', 'lxml')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 28), dict_32828, (str_32829, str_32830))
        # Adding element type (key, value) (line 370)
        str_32831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 45), 'str', 'ext_modules')
        
        # Obtaining an instance of the builtin type 'list' (line 370)
        list_32832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 370)
        # Adding element type (line 370)
        # Getting the type of 'etree_ext' (line 370)
        etree_ext_32833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 61), 'etree_ext', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 60), list_32832, etree_ext_32833)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 28), dict_32828, (str_32831, list_32832))
        
        # Processing the call keyword arguments (line 370)
        kwargs_32834 = {}
        # Getting the type of 'Distribution' (line 370)
        Distribution_32827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 370)
        Distribution_call_result_32835 = invoke(stypy.reporting.localization.Localization(__file__, 370, 15), Distribution_32827, *[dict_32828], **kwargs_32834)
        
        # Assigning a type to the variable 'dist' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'dist', Distribution_call_result_32835)
        
        # Assigning a Call to a Name (line 371):
        
        # Call to build_ext(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'dist' (line 371)
        dist_32837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'dist', False)
        # Processing the call keyword arguments (line 371)
        kwargs_32838 = {}
        # Getting the type of 'build_ext' (line 371)
        build_ext_32836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 371)
        build_ext_call_result_32839 = invoke(stypy.reporting.localization.Localization(__file__, 371, 14), build_ext_32836, *[dist_32837], **kwargs_32838)
        
        # Assigning a type to the variable 'cmd' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'cmd', build_ext_call_result_32839)
        
        # Call to ensure_finalized(...): (line 372)
        # Processing the call keyword arguments (line 372)
        kwargs_32842 = {}
        # Getting the type of 'cmd' (line 372)
        cmd_32840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 372)
        ensure_finalized_32841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 8), cmd_32840, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 372)
        ensure_finalized_call_result_32843 = invoke(stypy.reporting.localization.Localization(__file__, 372, 8), ensure_finalized_32841, *[], **kwargs_32842)
        
        
        # Assigning a Num to a Attribute (line 373):
        int_32844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 22), 'int')
        # Getting the type of 'cmd' (line 373)
        cmd_32845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'cmd')
        # Setting the type of the member 'inplace' of a type (line 373)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), cmd_32845, 'inplace', int_32844)
        
        # Assigning a Dict to a Attribute (line 374):
        
        # Obtaining an instance of the builtin type 'dict' (line 374)
        dict_32846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 39), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 374)
        # Adding element type (key, value) (line 374)
        str_32847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 40), 'str', '')
        str_32848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 44), 'str', 'src')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 39), dict_32846, (str_32847, str_32848))
        
        # Getting the type of 'cmd' (line 374)
        cmd_32849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 374)
        distribution_32850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), cmd_32849, 'distribution')
        # Setting the type of the member 'package_dir' of a type (line 374)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), distribution_32850, 'package_dir', dict_32846)
        
        # Assigning a List to a Attribute (line 375):
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_32851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        str_32852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 37), 'str', 'lxml')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 36), list_32851, str_32852)
        # Adding element type (line 375)
        str_32853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 45), 'str', 'lxml.html')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 36), list_32851, str_32853)
        
        # Getting the type of 'cmd' (line 375)
        cmd_32854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'cmd')
        # Obtaining the member 'distribution' of a type (line 375)
        distribution_32855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), cmd_32854, 'distribution')
        # Setting the type of the member 'packages' of a type (line 375)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 8), distribution_32855, 'packages', list_32851)
        
        # Assigning a Call to a Name (line 376):
        
        # Call to getcwd(...): (line 376)
        # Processing the call keyword arguments (line 376)
        kwargs_32858 = {}
        # Getting the type of 'os' (line 376)
        os_32856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 17), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 376)
        getcwd_32857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 17), os_32856, 'getcwd')
        # Calling getcwd(args, kwargs) (line 376)
        getcwd_call_result_32859 = invoke(stypy.reporting.localization.Localization(__file__, 376, 17), getcwd_32857, *[], **kwargs_32858)
        
        # Assigning a type to the variable 'curdir' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'curdir', getcwd_call_result_32859)
        
        # Assigning a Call to a Name (line 377):
        
        # Call to get_config_var(...): (line 377)
        # Processing the call arguments (line 377)
        str_32862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 39), 'str', 'SO')
        # Processing the call keyword arguments (line 377)
        kwargs_32863 = {}
        # Getting the type of 'sysconfig' (line 377)
        sysconfig_32860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 14), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 377)
        get_config_var_32861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 14), sysconfig_32860, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 377)
        get_config_var_call_result_32864 = invoke(stypy.reporting.localization.Localization(__file__, 377, 14), get_config_var_32861, *[str_32862], **kwargs_32863)
        
        # Assigning a type to the variable 'ext' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'ext', get_config_var_call_result_32864)
        
        # Assigning a Call to a Name (line 378):
        
        # Call to join(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'curdir' (line 378)
        curdir_32868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 30), 'curdir', False)
        str_32869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 38), 'str', 'src')
        str_32870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 45), 'str', 'lxml')
        str_32871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 53), 'str', 'etree')
        # Getting the type of 'ext' (line 378)
        ext_32872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 63), 'ext', False)
        # Applying the binary operator '+' (line 378)
        result_add_32873 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 53), '+', str_32871, ext_32872)
        
        # Processing the call keyword arguments (line 378)
        kwargs_32874 = {}
        # Getting the type of 'os' (line 378)
        os_32865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 378)
        path_32866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 17), os_32865, 'path')
        # Obtaining the member 'join' of a type (line 378)
        join_32867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 17), path_32866, 'join')
        # Calling join(args, kwargs) (line 378)
        join_call_result_32875 = invoke(stypy.reporting.localization.Localization(__file__, 378, 17), join_32867, *[curdir_32868, str_32869, str_32870, result_add_32873], **kwargs_32874)
        
        # Assigning a type to the variable 'wanted' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'wanted', join_call_result_32875)
        
        # Assigning a Call to a Name (line 379):
        
        # Call to get_ext_fullpath(...): (line 379)
        # Processing the call arguments (line 379)
        str_32878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 36), 'str', 'lxml.etree')
        # Processing the call keyword arguments (line 379)
        kwargs_32879 = {}
        # Getting the type of 'cmd' (line 379)
        cmd_32876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 379)
        get_ext_fullpath_32877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), cmd_32876, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 379)
        get_ext_fullpath_call_result_32880 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), get_ext_fullpath_32877, *[str_32878], **kwargs_32879)
        
        # Assigning a type to the variable 'path' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'path', get_ext_fullpath_call_result_32880)
        
        # Call to assertEqual(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'wanted' (line 380)
        wanted_32883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 25), 'wanted', False)
        # Getting the type of 'path' (line 380)
        path_32884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 33), 'path', False)
        # Processing the call keyword arguments (line 380)
        kwargs_32885 = {}
        # Getting the type of 'self' (line 380)
        self_32881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 380)
        assertEqual_32882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), self_32881, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 380)
        assertEqual_call_result_32886 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), assertEqual_32882, *[wanted_32883, path_32884], **kwargs_32885)
        
        
        # ################# End of 'test_build_ext_inplace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_build_ext_inplace' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_32887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_build_ext_inplace'
        return stypy_return_type_32887


    @norecursion
    def test_setuptools_compat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_setuptools_compat'
        module_type_store = module_type_store.open_function_context('test_setuptools_compat', 382, 4, False)
        # Assigning a type to the variable 'self' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_setuptools_compat')
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_setuptools_compat.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_setuptools_compat', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_setuptools_compat', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_setuptools_compat(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 383, 8))
        
        # Multiple import statement. import distutils.core (1/3) (line 383)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_32888 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.core')

        if (type(import_32888) is not StypyTypeError):

            if (import_32888 != 'pyd_module'):
                __import__(import_32888)
                sys_modules_32889 = sys.modules[import_32888]
                import_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.core', sys_modules_32889.module_type_store, module_type_store)
            else:
                import distutils.core

                import_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.core', distutils.core, module_type_store)

        else:
            # Assigning a type to the variable 'distutils.core' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.core', import_32888)

        # Multiple import statement. import distutils.extension (2/3) (line 383)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_32890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.extension')

        if (type(import_32890) is not StypyTypeError):

            if (import_32890 != 'pyd_module'):
                __import__(import_32890)
                sys_modules_32891 = sys.modules[import_32890]
                import_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.extension', sys_modules_32891.module_type_store, module_type_store)
            else:
                import distutils.extension

                import_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.extension', distutils.extension, module_type_store)

        else:
            # Assigning a type to the variable 'distutils.extension' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.extension', import_32890)

        # Multiple import statement. import distutils.command.build_ext (3/3) (line 383)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_32892 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.command.build_ext')

        if (type(import_32892) is not StypyTypeError):

            if (import_32892 != 'pyd_module'):
                __import__(import_32892)
                sys_modules_32893 = sys.modules[import_32892]
                import_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.command.build_ext', sys_modules_32893.module_type_store, module_type_store)
            else:
                import distutils.command.build_ext

                import_module(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.command.build_ext', distutils.command.build_ext, module_type_store)

        else:
            # Assigning a type to the variable 'distutils.command.build_ext' (line 383)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'distutils.command.build_ext', import_32892)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Assigning a Attribute to a Name (line 384):
        # Getting the type of 'distutils' (line 384)
        distutils_32894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'distutils')
        # Obtaining the member 'extension' of a type (line 384)
        extension_32895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 20), distutils_32894, 'extension')
        # Obtaining the member 'Extension' of a type (line 384)
        Extension_32896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 20), extension_32895, 'Extension')
        # Assigning a type to the variable 'saved_ext' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'saved_ext', Extension_32896)
        
        # Try-finally block (line 385)
        
        # Call to import_module(...): (line 387)
        # Processing the call arguments (line 387)
        str_32899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 39), 'str', 'setuptools_build_ext')
        # Processing the call keyword arguments (line 387)
        # Getting the type of 'True' (line 387)
        True_32900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 74), 'True', False)
        keyword_32901 = True_32900
        kwargs_32902 = {'deprecated': keyword_32901}
        # Getting the type of 'test_support' (line 387)
        test_support_32897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'test_support', False)
        # Obtaining the member 'import_module' of a type (line 387)
        import_module_32898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), test_support_32897, 'import_module')
        # Calling import_module(args, kwargs) (line 387)
        import_module_call_result_32903 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), import_module_32898, *[str_32899], **kwargs_32902)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 390, 12))
        
        # 'from setuptools_build_ext import setuptools_build_ext' statement (line 390)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_32904 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 390, 12), 'setuptools_build_ext')

        if (type(import_32904) is not StypyTypeError):

            if (import_32904 != 'pyd_module'):
                __import__(import_32904)
                sys_modules_32905 = sys.modules[import_32904]
                import_from_module(stypy.reporting.localization.Localization(__file__, 390, 12), 'setuptools_build_ext', sys_modules_32905.module_type_store, module_type_store, ['build_ext'])
                nest_module(stypy.reporting.localization.Localization(__file__, 390, 12), __file__, sys_modules_32905, sys_modules_32905.module_type_store, module_type_store)
            else:
                from setuptools_build_ext import build_ext as setuptools_build_ext

                import_from_module(stypy.reporting.localization.Localization(__file__, 390, 12), 'setuptools_build_ext', None, module_type_store, ['build_ext'], [setuptools_build_ext])

        else:
            # Assigning a type to the variable 'setuptools_build_ext' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'setuptools_build_ext', import_32904)

        # Adding an alias
        module_type_store.add_alias('setuptools_build_ext', 'build_ext')
        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 391, 12))
        
        # 'from setuptools_extension import Extension' statement (line 391)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_32906 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 391, 12), 'setuptools_extension')

        if (type(import_32906) is not StypyTypeError):

            if (import_32906 != 'pyd_module'):
                __import__(import_32906)
                sys_modules_32907 = sys.modules[import_32906]
                import_from_module(stypy.reporting.localization.Localization(__file__, 391, 12), 'setuptools_extension', sys_modules_32907.module_type_store, module_type_store, ['Extension'])
                nest_module(stypy.reporting.localization.Localization(__file__, 391, 12), __file__, sys_modules_32907, sys_modules_32907.module_type_store, module_type_store)
            else:
                from setuptools_extension import Extension

                import_from_module(stypy.reporting.localization.Localization(__file__, 391, 12), 'setuptools_extension', None, module_type_store, ['Extension'], [Extension])

        else:
            # Assigning a type to the variable 'setuptools_extension' (line 391)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'setuptools_extension', import_32906)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Assigning a Call to a Name (line 393):
        
        # Call to join(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'self' (line 393)
        self_32911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 35), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 393)
        tmp_dir_32912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 35), self_32911, 'tmp_dir')
        str_32913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 49), 'str', 'lxml.etree.c')
        # Processing the call keyword arguments (line 393)
        kwargs_32914 = {}
        # Getting the type of 'os' (line 393)
        os_32908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 393)
        path_32909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 22), os_32908, 'path')
        # Obtaining the member 'join' of a type (line 393)
        join_32910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 22), path_32909, 'join')
        # Calling join(args, kwargs) (line 393)
        join_call_result_32915 = invoke(stypy.reporting.localization.Localization(__file__, 393, 22), join_32910, *[tmp_dir_32912, str_32913], **kwargs_32914)
        
        # Assigning a type to the variable 'etree_c' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'etree_c', join_call_result_32915)
        
        # Assigning a Call to a Name (line 394):
        
        # Call to Extension(...): (line 394)
        # Processing the call arguments (line 394)
        str_32917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 34), 'str', 'lxml.etree')
        
        # Obtaining an instance of the builtin type 'list' (line 394)
        list_32918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 394)
        # Adding element type (line 394)
        # Getting the type of 'etree_c' (line 394)
        etree_c_32919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 49), 'etree_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 48), list_32918, etree_c_32919)
        
        # Processing the call keyword arguments (line 394)
        kwargs_32920 = {}
        # Getting the type of 'Extension' (line 394)
        Extension_32916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 24), 'Extension', False)
        # Calling Extension(args, kwargs) (line 394)
        Extension_call_result_32921 = invoke(stypy.reporting.localization.Localization(__file__, 394, 24), Extension_32916, *[str_32917, list_32918], **kwargs_32920)
        
        # Assigning a type to the variable 'etree_ext' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'etree_ext', Extension_call_result_32921)
        
        # Assigning a Call to a Name (line 395):
        
        # Call to Distribution(...): (line 395)
        # Processing the call arguments (line 395)
        
        # Obtaining an instance of the builtin type 'dict' (line 395)
        dict_32923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 32), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 395)
        # Adding element type (key, value) (line 395)
        str_32924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 33), 'str', 'name')
        str_32925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 41), 'str', 'lxml')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 32), dict_32923, (str_32924, str_32925))
        # Adding element type (key, value) (line 395)
        str_32926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 49), 'str', 'ext_modules')
        
        # Obtaining an instance of the builtin type 'list' (line 395)
        list_32927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 395)
        # Adding element type (line 395)
        # Getting the type of 'etree_ext' (line 395)
        etree_ext_32928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 65), 'etree_ext', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 64), list_32927, etree_ext_32928)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 32), dict_32923, (str_32926, list_32927))
        
        # Processing the call keyword arguments (line 395)
        kwargs_32929 = {}
        # Getting the type of 'Distribution' (line 395)
        Distribution_32922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 395)
        Distribution_call_result_32930 = invoke(stypy.reporting.localization.Localization(__file__, 395, 19), Distribution_32922, *[dict_32923], **kwargs_32929)
        
        # Assigning a type to the variable 'dist' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'dist', Distribution_call_result_32930)
        
        # Assigning a Call to a Name (line 396):
        
        # Call to setuptools_build_ext(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'dist' (line 396)
        dist_32932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 39), 'dist', False)
        # Processing the call keyword arguments (line 396)
        kwargs_32933 = {}
        # Getting the type of 'setuptools_build_ext' (line 396)
        setuptools_build_ext_32931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 18), 'setuptools_build_ext', False)
        # Calling setuptools_build_ext(args, kwargs) (line 396)
        setuptools_build_ext_call_result_32934 = invoke(stypy.reporting.localization.Localization(__file__, 396, 18), setuptools_build_ext_32931, *[dist_32932], **kwargs_32933)
        
        # Assigning a type to the variable 'cmd' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'cmd', setuptools_build_ext_call_result_32934)
        
        # Call to ensure_finalized(...): (line 397)
        # Processing the call keyword arguments (line 397)
        kwargs_32937 = {}
        # Getting the type of 'cmd' (line 397)
        cmd_32935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 397)
        ensure_finalized_32936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), cmd_32935, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 397)
        ensure_finalized_call_result_32938 = invoke(stypy.reporting.localization.Localization(__file__, 397, 12), ensure_finalized_32936, *[], **kwargs_32937)
        
        
        # Assigning a Num to a Attribute (line 398):
        int_32939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 26), 'int')
        # Getting the type of 'cmd' (line 398)
        cmd_32940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'cmd')
        # Setting the type of the member 'inplace' of a type (line 398)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), cmd_32940, 'inplace', int_32939)
        
        # Assigning a Dict to a Attribute (line 399):
        
        # Obtaining an instance of the builtin type 'dict' (line 399)
        dict_32941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 43), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 399)
        # Adding element type (key, value) (line 399)
        str_32942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 44), 'str', '')
        str_32943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 48), 'str', 'src')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 43), dict_32941, (str_32942, str_32943))
        
        # Getting the type of 'cmd' (line 399)
        cmd_32944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'cmd')
        # Obtaining the member 'distribution' of a type (line 399)
        distribution_32945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), cmd_32944, 'distribution')
        # Setting the type of the member 'package_dir' of a type (line 399)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), distribution_32945, 'package_dir', dict_32941)
        
        # Assigning a List to a Attribute (line 400):
        
        # Obtaining an instance of the builtin type 'list' (line 400)
        list_32946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 400)
        # Adding element type (line 400)
        str_32947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 41), 'str', 'lxml')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 40), list_32946, str_32947)
        # Adding element type (line 400)
        str_32948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 49), 'str', 'lxml.html')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 40), list_32946, str_32948)
        
        # Getting the type of 'cmd' (line 400)
        cmd_32949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'cmd')
        # Obtaining the member 'distribution' of a type (line 400)
        distribution_32950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), cmd_32949, 'distribution')
        # Setting the type of the member 'packages' of a type (line 400)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), distribution_32950, 'packages', list_32946)
        
        # Assigning a Call to a Name (line 401):
        
        # Call to getcwd(...): (line 401)
        # Processing the call keyword arguments (line 401)
        kwargs_32953 = {}
        # Getting the type of 'os' (line 401)
        os_32951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 21), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 401)
        getcwd_32952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 21), os_32951, 'getcwd')
        # Calling getcwd(args, kwargs) (line 401)
        getcwd_call_result_32954 = invoke(stypy.reporting.localization.Localization(__file__, 401, 21), getcwd_32952, *[], **kwargs_32953)
        
        # Assigning a type to the variable 'curdir' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'curdir', getcwd_call_result_32954)
        
        # Assigning a Call to a Name (line 402):
        
        # Call to get_config_var(...): (line 402)
        # Processing the call arguments (line 402)
        str_32957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 43), 'str', 'SO')
        # Processing the call keyword arguments (line 402)
        kwargs_32958 = {}
        # Getting the type of 'sysconfig' (line 402)
        sysconfig_32955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 18), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 402)
        get_config_var_32956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 18), sysconfig_32955, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 402)
        get_config_var_call_result_32959 = invoke(stypy.reporting.localization.Localization(__file__, 402, 18), get_config_var_32956, *[str_32957], **kwargs_32958)
        
        # Assigning a type to the variable 'ext' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'ext', get_config_var_call_result_32959)
        
        # Assigning a Call to a Name (line 403):
        
        # Call to join(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'curdir' (line 403)
        curdir_32963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 34), 'curdir', False)
        str_32964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 42), 'str', 'src')
        str_32965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 49), 'str', 'lxml')
        str_32966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 57), 'str', 'etree')
        # Getting the type of 'ext' (line 403)
        ext_32967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 67), 'ext', False)
        # Applying the binary operator '+' (line 403)
        result_add_32968 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 57), '+', str_32966, ext_32967)
        
        # Processing the call keyword arguments (line 403)
        kwargs_32969 = {}
        # Getting the type of 'os' (line 403)
        os_32960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 403)
        path_32961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 21), os_32960, 'path')
        # Obtaining the member 'join' of a type (line 403)
        join_32962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 21), path_32961, 'join')
        # Calling join(args, kwargs) (line 403)
        join_call_result_32970 = invoke(stypy.reporting.localization.Localization(__file__, 403, 21), join_32962, *[curdir_32963, str_32964, str_32965, result_add_32968], **kwargs_32969)
        
        # Assigning a type to the variable 'wanted' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'wanted', join_call_result_32970)
        
        # Assigning a Call to a Name (line 404):
        
        # Call to get_ext_fullpath(...): (line 404)
        # Processing the call arguments (line 404)
        str_32973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 40), 'str', 'lxml.etree')
        # Processing the call keyword arguments (line 404)
        kwargs_32974 = {}
        # Getting the type of 'cmd' (line 404)
        cmd_32971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 404)
        get_ext_fullpath_32972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 19), cmd_32971, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 404)
        get_ext_fullpath_call_result_32975 = invoke(stypy.reporting.localization.Localization(__file__, 404, 19), get_ext_fullpath_32972, *[str_32973], **kwargs_32974)
        
        # Assigning a type to the variable 'path' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'path', get_ext_fullpath_call_result_32975)
        
        # Call to assertEqual(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'wanted' (line 405)
        wanted_32978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 29), 'wanted', False)
        # Getting the type of 'path' (line 405)
        path_32979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 37), 'path', False)
        # Processing the call keyword arguments (line 405)
        kwargs_32980 = {}
        # Getting the type of 'self' (line 405)
        self_32976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 405)
        assertEqual_32977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 12), self_32976, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 405)
        assertEqual_call_result_32981 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), assertEqual_32977, *[wanted_32978, path_32979], **kwargs_32980)
        
        
        # finally branch of the try-finally block (line 385)
        
        # Assigning a Name to a Attribute (line 408):
        # Getting the type of 'saved_ext' (line 408)
        saved_ext_32982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 44), 'saved_ext')
        # Getting the type of 'distutils' (line 408)
        distutils_32983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'distutils')
        # Obtaining the member 'extension' of a type (line 408)
        extension_32984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), distutils_32983, 'extension')
        # Setting the type of the member 'Extension' of a type (line 408)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), extension_32984, 'Extension', saved_ext_32982)
        
        # Assigning a Name to a Attribute (line 409):
        # Getting the type of 'saved_ext' (line 409)
        saved_ext_32985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 39), 'saved_ext')
        # Getting the type of 'distutils' (line 409)
        distutils_32986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'distutils')
        # Obtaining the member 'core' of a type (line 409)
        core_32987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 12), distutils_32986, 'core')
        # Setting the type of the member 'Extension' of a type (line 409)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 12), core_32987, 'Extension', saved_ext_32985)
        
        # Assigning a Name to a Attribute (line 410):
        # Getting the type of 'saved_ext' (line 410)
        saved_ext_32988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 52), 'saved_ext')
        # Getting the type of 'distutils' (line 410)
        distutils_32989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'distutils')
        # Obtaining the member 'command' of a type (line 410)
        command_32990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), distutils_32989, 'command')
        # Obtaining the member 'build_ext' of a type (line 410)
        build_ext_32991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), command_32990, 'build_ext')
        # Setting the type of the member 'Extension' of a type (line 410)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), build_ext_32991, 'Extension', saved_ext_32988)
        
        
        # ################# End of 'test_setuptools_compat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_setuptools_compat' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_32992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_setuptools_compat'
        return stypy_return_type_32992


    @norecursion
    def test_build_ext_path_with_os_sep(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_build_ext_path_with_os_sep'
        module_type_store = module_type_store.open_function_context('test_build_ext_path_with_os_sep', 412, 4, False)
        # Assigning a type to the variable 'self' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_build_ext_path_with_os_sep')
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_build_ext_path_with_os_sep.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_build_ext_path_with_os_sep', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_build_ext_path_with_os_sep', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_build_ext_path_with_os_sep(...)' code ##################

        
        # Assigning a Call to a Name (line 413):
        
        # Call to Distribution(...): (line 413)
        # Processing the call arguments (line 413)
        
        # Obtaining an instance of the builtin type 'dict' (line 413)
        dict_32994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 413)
        # Adding element type (key, value) (line 413)
        str_32995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 29), 'str', 'name')
        str_32996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 37), 'str', 'UpdateManager')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 28), dict_32994, (str_32995, str_32996))
        
        # Processing the call keyword arguments (line 413)
        kwargs_32997 = {}
        # Getting the type of 'Distribution' (line 413)
        Distribution_32993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 413)
        Distribution_call_result_32998 = invoke(stypy.reporting.localization.Localization(__file__, 413, 15), Distribution_32993, *[dict_32994], **kwargs_32997)
        
        # Assigning a type to the variable 'dist' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'dist', Distribution_call_result_32998)
        
        # Assigning a Call to a Name (line 414):
        
        # Call to build_ext(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'dist' (line 414)
        dist_33000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 24), 'dist', False)
        # Processing the call keyword arguments (line 414)
        kwargs_33001 = {}
        # Getting the type of 'build_ext' (line 414)
        build_ext_32999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 414)
        build_ext_call_result_33002 = invoke(stypy.reporting.localization.Localization(__file__, 414, 14), build_ext_32999, *[dist_33000], **kwargs_33001)
        
        # Assigning a type to the variable 'cmd' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'cmd', build_ext_call_result_33002)
        
        # Call to ensure_finalized(...): (line 415)
        # Processing the call keyword arguments (line 415)
        kwargs_33005 = {}
        # Getting the type of 'cmd' (line 415)
        cmd_33003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 415)
        ensure_finalized_33004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), cmd_33003, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 415)
        ensure_finalized_call_result_33006 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), ensure_finalized_33004, *[], **kwargs_33005)
        
        
        # Assigning a Call to a Name (line 416):
        
        # Call to get_config_var(...): (line 416)
        # Processing the call arguments (line 416)
        str_33009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 39), 'str', 'SO')
        # Processing the call keyword arguments (line 416)
        kwargs_33010 = {}
        # Getting the type of 'sysconfig' (line 416)
        sysconfig_33007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 14), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 416)
        get_config_var_33008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 14), sysconfig_33007, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 416)
        get_config_var_call_result_33011 = invoke(stypy.reporting.localization.Localization(__file__, 416, 14), get_config_var_33008, *[str_33009], **kwargs_33010)
        
        # Assigning a type to the variable 'ext' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'ext', get_config_var_call_result_33011)
        
        # Assigning a Call to a Name (line 417):
        
        # Call to join(...): (line 417)
        # Processing the call arguments (line 417)
        str_33015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 32), 'str', 'UpdateManager')
        str_33016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 49), 'str', 'fdsend')
        # Processing the call keyword arguments (line 417)
        kwargs_33017 = {}
        # Getting the type of 'os' (line 417)
        os_33012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 417)
        path_33013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 19), os_33012, 'path')
        # Obtaining the member 'join' of a type (line 417)
        join_33014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 19), path_33013, 'join')
        # Calling join(args, kwargs) (line 417)
        join_call_result_33018 = invoke(stypy.reporting.localization.Localization(__file__, 417, 19), join_33014, *[str_33015, str_33016], **kwargs_33017)
        
        # Assigning a type to the variable 'ext_name' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'ext_name', join_call_result_33018)
        
        # Assigning a Call to a Name (line 418):
        
        # Call to get_ext_fullpath(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'ext_name' (line 418)
        ext_name_33021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 40), 'ext_name', False)
        # Processing the call keyword arguments (line 418)
        kwargs_33022 = {}
        # Getting the type of 'cmd' (line 418)
        cmd_33019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 19), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 418)
        get_ext_fullpath_33020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 19), cmd_33019, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 418)
        get_ext_fullpath_call_result_33023 = invoke(stypy.reporting.localization.Localization(__file__, 418, 19), get_ext_fullpath_33020, *[ext_name_33021], **kwargs_33022)
        
        # Assigning a type to the variable 'ext_path' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'ext_path', get_ext_fullpath_call_result_33023)
        
        # Assigning a Call to a Name (line 419):
        
        # Call to join(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 'cmd' (line 419)
        cmd_33027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 30), 'cmd', False)
        # Obtaining the member 'build_lib' of a type (line 419)
        build_lib_33028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 30), cmd_33027, 'build_lib')
        str_33029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 45), 'str', 'UpdateManager')
        str_33030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 62), 'str', 'fdsend')
        # Getting the type of 'ext' (line 419)
        ext_33031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 73), 'ext', False)
        # Applying the binary operator '+' (line 419)
        result_add_33032 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 62), '+', str_33030, ext_33031)
        
        # Processing the call keyword arguments (line 419)
        kwargs_33033 = {}
        # Getting the type of 'os' (line 419)
        os_33024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 419)
        path_33025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 17), os_33024, 'path')
        # Obtaining the member 'join' of a type (line 419)
        join_33026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 17), path_33025, 'join')
        # Calling join(args, kwargs) (line 419)
        join_call_result_33034 = invoke(stypy.reporting.localization.Localization(__file__, 419, 17), join_33026, *[build_lib_33028, str_33029, result_add_33032], **kwargs_33033)
        
        # Assigning a type to the variable 'wanted' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'wanted', join_call_result_33034)
        
        # Call to assertEqual(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'ext_path' (line 420)
        ext_path_33037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 25), 'ext_path', False)
        # Getting the type of 'wanted' (line 420)
        wanted_33038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 35), 'wanted', False)
        # Processing the call keyword arguments (line 420)
        kwargs_33039 = {}
        # Getting the type of 'self' (line 420)
        self_33035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 420)
        assertEqual_33036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 8), self_33035, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 420)
        assertEqual_call_result_33040 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), assertEqual_33036, *[ext_path_33037, wanted_33038], **kwargs_33039)
        
        
        # ################# End of 'test_build_ext_path_with_os_sep(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_build_ext_path_with_os_sep' in the type store
        # Getting the type of 'stypy_return_type' (line 412)
        stypy_return_type_33041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_build_ext_path_with_os_sep'
        return stypy_return_type_33041


    @norecursion
    def test_build_ext_path_cross_platform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_build_ext_path_cross_platform'
        module_type_store = module_type_store.open_function_context('test_build_ext_path_cross_platform', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_build_ext_path_cross_platform')
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_build_ext_path_cross_platform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_build_ext_path_cross_platform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_build_ext_path_cross_platform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_build_ext_path_cross_platform(...)' code ##################

        
        # Assigning a Call to a Name (line 424):
        
        # Call to Distribution(...): (line 424)
        # Processing the call arguments (line 424)
        
        # Obtaining an instance of the builtin type 'dict' (line 424)
        dict_33043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 424)
        # Adding element type (key, value) (line 424)
        str_33044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 29), 'str', 'name')
        str_33045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 37), 'str', 'UpdateManager')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 28), dict_33043, (str_33044, str_33045))
        
        # Processing the call keyword arguments (line 424)
        kwargs_33046 = {}
        # Getting the type of 'Distribution' (line 424)
        Distribution_33042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 424)
        Distribution_call_result_33047 = invoke(stypy.reporting.localization.Localization(__file__, 424, 15), Distribution_33042, *[dict_33043], **kwargs_33046)
        
        # Assigning a type to the variable 'dist' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'dist', Distribution_call_result_33047)
        
        # Assigning a Call to a Name (line 425):
        
        # Call to build_ext(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'dist' (line 425)
        dist_33049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 24), 'dist', False)
        # Processing the call keyword arguments (line 425)
        kwargs_33050 = {}
        # Getting the type of 'build_ext' (line 425)
        build_ext_33048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 425)
        build_ext_call_result_33051 = invoke(stypy.reporting.localization.Localization(__file__, 425, 14), build_ext_33048, *[dist_33049], **kwargs_33050)
        
        # Assigning a type to the variable 'cmd' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'cmd', build_ext_call_result_33051)
        
        # Call to ensure_finalized(...): (line 426)
        # Processing the call keyword arguments (line 426)
        kwargs_33054 = {}
        # Getting the type of 'cmd' (line 426)
        cmd_33052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 426)
        ensure_finalized_33053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), cmd_33052, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 426)
        ensure_finalized_call_result_33055 = invoke(stypy.reporting.localization.Localization(__file__, 426, 8), ensure_finalized_33053, *[], **kwargs_33054)
        
        
        # Assigning a Call to a Name (line 427):
        
        # Call to get_config_var(...): (line 427)
        # Processing the call arguments (line 427)
        str_33058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 39), 'str', 'SO')
        # Processing the call keyword arguments (line 427)
        kwargs_33059 = {}
        # Getting the type of 'sysconfig' (line 427)
        sysconfig_33056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 14), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 427)
        get_config_var_33057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 14), sysconfig_33056, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 427)
        get_config_var_call_result_33060 = invoke(stypy.reporting.localization.Localization(__file__, 427, 14), get_config_var_33057, *[str_33058], **kwargs_33059)
        
        # Assigning a type to the variable 'ext' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'ext', get_config_var_call_result_33060)
        
        # Assigning a Str to a Name (line 429):
        str_33061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 19), 'str', 'UpdateManager/fdsend')
        # Assigning a type to the variable 'ext_name' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'ext_name', str_33061)
        
        # Assigning a Call to a Name (line 430):
        
        # Call to get_ext_fullpath(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'ext_name' (line 430)
        ext_name_33064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 40), 'ext_name', False)
        # Processing the call keyword arguments (line 430)
        kwargs_33065 = {}
        # Getting the type of 'cmd' (line 430)
        cmd_33062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 'cmd', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 430)
        get_ext_fullpath_33063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 19), cmd_33062, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 430)
        get_ext_fullpath_call_result_33066 = invoke(stypy.reporting.localization.Localization(__file__, 430, 19), get_ext_fullpath_33063, *[ext_name_33064], **kwargs_33065)
        
        # Assigning a type to the variable 'ext_path' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'ext_path', get_ext_fullpath_call_result_33066)
        
        # Assigning a Call to a Name (line 431):
        
        # Call to join(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'cmd' (line 431)
        cmd_33070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 30), 'cmd', False)
        # Obtaining the member 'build_lib' of a type (line 431)
        build_lib_33071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 30), cmd_33070, 'build_lib')
        str_33072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 45), 'str', 'UpdateManager')
        str_33073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 62), 'str', 'fdsend')
        # Getting the type of 'ext' (line 431)
        ext_33074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 73), 'ext', False)
        # Applying the binary operator '+' (line 431)
        result_add_33075 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 62), '+', str_33073, ext_33074)
        
        # Processing the call keyword arguments (line 431)
        kwargs_33076 = {}
        # Getting the type of 'os' (line 431)
        os_33067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 431)
        path_33068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 17), os_33067, 'path')
        # Obtaining the member 'join' of a type (line 431)
        join_33069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 17), path_33068, 'join')
        # Calling join(args, kwargs) (line 431)
        join_call_result_33077 = invoke(stypy.reporting.localization.Localization(__file__, 431, 17), join_33069, *[build_lib_33071, str_33072, result_add_33075], **kwargs_33076)
        
        # Assigning a type to the variable 'wanted' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'wanted', join_call_result_33077)
        
        # Call to assertEqual(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'ext_path' (line 432)
        ext_path_33080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 25), 'ext_path', False)
        # Getting the type of 'wanted' (line 432)
        wanted_33081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 35), 'wanted', False)
        # Processing the call keyword arguments (line 432)
        kwargs_33082 = {}
        # Getting the type of 'self' (line 432)
        self_33078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 432)
        assertEqual_33079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), self_33078, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 432)
        assertEqual_call_result_33083 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), assertEqual_33079, *[ext_path_33080, wanted_33081], **kwargs_33082)
        
        
        # ################# End of 'test_build_ext_path_cross_platform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_build_ext_path_cross_platform' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_33084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_build_ext_path_cross_platform'
        return stypy_return_type_33084


    @norecursion
    def test_deployment_target_default(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_deployment_target_default'
        module_type_store = module_type_store.open_function_context('test_deployment_target_default', 434, 4, False)
        # Assigning a type to the variable 'self' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_deployment_target_default')
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_deployment_target_default.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_deployment_target_default', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_deployment_target_default', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_deployment_target_default(...)' code ##################

        
        # Call to _try_compile_deployment_target(...): (line 439)
        # Processing the call arguments (line 439)
        str_33087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 44), 'str', '==')
        # Getting the type of 'None' (line 439)
        None_33088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 50), 'None', False)
        # Processing the call keyword arguments (line 439)
        kwargs_33089 = {}
        # Getting the type of 'self' (line 439)
        self_33085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'self', False)
        # Obtaining the member '_try_compile_deployment_target' of a type (line 439)
        _try_compile_deployment_target_33086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), self_33085, '_try_compile_deployment_target')
        # Calling _try_compile_deployment_target(args, kwargs) (line 439)
        _try_compile_deployment_target_call_result_33090 = invoke(stypy.reporting.localization.Localization(__file__, 439, 8), _try_compile_deployment_target_33086, *[str_33087, None_33088], **kwargs_33089)
        
        
        # ################# End of 'test_deployment_target_default(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_deployment_target_default' in the type store
        # Getting the type of 'stypy_return_type' (line 434)
        stypy_return_type_33091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33091)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_deployment_target_default'
        return stypy_return_type_33091


    @norecursion
    def test_deployment_target_too_low(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_deployment_target_too_low'
        module_type_store = module_type_store.open_function_context('test_deployment_target_too_low', 441, 4, False)
        # Assigning a type to the variable 'self' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_deployment_target_too_low')
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_deployment_target_too_low.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_deployment_target_too_low', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_deployment_target_too_low', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_deployment_target_too_low(...)' code ##################

        
        # Call to assertRaises(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'DistutilsPlatformError' (line 445)
        DistutilsPlatformError_33094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 26), 'DistutilsPlatformError', False)
        # Getting the type of 'self' (line 446)
        self_33095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'self', False)
        # Obtaining the member '_try_compile_deployment_target' of a type (line 446)
        _try_compile_deployment_target_33096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 12), self_33095, '_try_compile_deployment_target')
        str_33097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 49), 'str', '>')
        str_33098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 54), 'str', '10.1')
        # Processing the call keyword arguments (line 445)
        kwargs_33099 = {}
        # Getting the type of 'self' (line 445)
        self_33092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 445)
        assertRaises_33093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), self_33092, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 445)
        assertRaises_call_result_33100 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), assertRaises_33093, *[DistutilsPlatformError_33094, _try_compile_deployment_target_33096, str_33097, str_33098], **kwargs_33099)
        
        
        # ################# End of 'test_deployment_target_too_low(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_deployment_target_too_low' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_33101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33101)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_deployment_target_too_low'
        return stypy_return_type_33101


    @norecursion
    def test_deployment_target_higher_ok(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_deployment_target_higher_ok'
        module_type_store = module_type_store.open_function_context('test_deployment_target_higher_ok', 448, 4, False)
        # Assigning a type to the variable 'self' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase.test_deployment_target_higher_ok')
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_param_names_list', [])
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase.test_deployment_target_higher_ok.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.test_deployment_target_higher_ok', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_deployment_target_higher_ok', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_deployment_target_higher_ok(...)' code ##################

        
        # Assigning a Call to a Name (line 453):
        
        # Call to get_config_var(...): (line 453)
        # Processing the call arguments (line 453)
        str_33104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 45), 'str', 'MACOSX_DEPLOYMENT_TARGET')
        # Processing the call keyword arguments (line 453)
        kwargs_33105 = {}
        # Getting the type of 'sysconfig' (line 453)
        sysconfig_33102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 20), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 453)
        get_config_var_33103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 20), sysconfig_33102, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 453)
        get_config_var_call_result_33106 = invoke(stypy.reporting.localization.Localization(__file__, 453, 20), get_config_var_33103, *[str_33104], **kwargs_33105)
        
        # Assigning a type to the variable 'deptarget' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'deptarget', get_config_var_call_result_33106)
        
        # Getting the type of 'deptarget' (line 454)
        deptarget_33107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'deptarget')
        # Testing the type of an if condition (line 454)
        if_condition_33108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 8), deptarget_33107)
        # Assigning a type to the variable 'if_condition_33108' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'if_condition_33108', if_condition_33108)
        # SSA begins for if statement (line 454)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Name (line 456):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 456)
        # Processing the call arguments (line 456)
        str_33115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'str', '.')
        # Processing the call keyword arguments (line 456)
        kwargs_33116 = {}
        # Getting the type of 'deptarget' (line 456)
        deptarget_33113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 41), 'deptarget', False)
        # Obtaining the member 'split' of a type (line 456)
        split_33114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 41), deptarget_33113, 'split')
        # Calling split(args, kwargs) (line 456)
        split_call_result_33117 = invoke(stypy.reporting.localization.Localization(__file__, 456, 41), split_33114, *[str_33115], **kwargs_33116)
        
        comprehension_33118 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 25), split_call_result_33117)
        # Assigning a type to the variable 'x' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 25), 'x', comprehension_33118)
        
        # Call to int(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'x' (line 456)
        x_33110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'x', False)
        # Processing the call keyword arguments (line 456)
        kwargs_33111 = {}
        # Getting the type of 'int' (line 456)
        int_33109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 25), 'int', False)
        # Calling int(args, kwargs) (line 456)
        int_call_result_33112 = invoke(stypy.reporting.localization.Localization(__file__, 456, 25), int_33109, *[x_33110], **kwargs_33111)
        
        list_33119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 25), list_33119, int_call_result_33112)
        # Assigning a type to the variable 'deptarget' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'deptarget', list_33119)
        
        # Getting the type of 'deptarget' (line 457)
        deptarget_33120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'deptarget')
        
        # Obtaining the type of the subscript
        int_33121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 22), 'int')
        # Getting the type of 'deptarget' (line 457)
        deptarget_33122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'deptarget')
        # Obtaining the member '__getitem__' of a type (line 457)
        getitem___33123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 12), deptarget_33122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 457)
        subscript_call_result_33124 = invoke(stypy.reporting.localization.Localization(__file__, 457, 12), getitem___33123, int_33121)
        
        int_33125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 29), 'int')
        # Applying the binary operator '+=' (line 457)
        result_iadd_33126 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 12), '+=', subscript_call_result_33124, int_33125)
        # Getting the type of 'deptarget' (line 457)
        deptarget_33127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'deptarget')
        int_33128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 22), 'int')
        # Storing an element on a container (line 457)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 12), deptarget_33127, (int_33128, result_iadd_33126))
        
        
        # Assigning a Call to a Name (line 458):
        
        # Call to join(...): (line 458)
        # Processing the call arguments (line 458)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 458, 33, True)
        # Calculating comprehension expression
        # Getting the type of 'deptarget' (line 458)
        deptarget_33135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 49), 'deptarget', False)
        comprehension_33136 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 33), deptarget_33135)
        # Assigning a type to the variable 'i' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 33), 'i', comprehension_33136)
        
        # Call to str(...): (line 458)
        # Processing the call arguments (line 458)
        # Getting the type of 'i' (line 458)
        i_33132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 37), 'i', False)
        # Processing the call keyword arguments (line 458)
        kwargs_33133 = {}
        # Getting the type of 'str' (line 458)
        str_33131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 33), 'str', False)
        # Calling str(args, kwargs) (line 458)
        str_call_result_33134 = invoke(stypy.reporting.localization.Localization(__file__, 458, 33), str_33131, *[i_33132], **kwargs_33133)
        
        list_33137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 33), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 33), list_33137, str_call_result_33134)
        # Processing the call keyword arguments (line 458)
        kwargs_33138 = {}
        str_33129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 24), 'str', '.')
        # Obtaining the member 'join' of a type (line 458)
        join_33130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 24), str_33129, 'join')
        # Calling join(args, kwargs) (line 458)
        join_call_result_33139 = invoke(stypy.reporting.localization.Localization(__file__, 458, 24), join_33130, *[list_33137], **kwargs_33138)
        
        # Assigning a type to the variable 'deptarget' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'deptarget', join_call_result_33139)
        
        # Call to _try_compile_deployment_target(...): (line 459)
        # Processing the call arguments (line 459)
        str_33142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 48), 'str', '<')
        # Getting the type of 'deptarget' (line 459)
        deptarget_33143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 53), 'deptarget', False)
        # Processing the call keyword arguments (line 459)
        kwargs_33144 = {}
        # Getting the type of 'self' (line 459)
        self_33140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'self', False)
        # Obtaining the member '_try_compile_deployment_target' of a type (line 459)
        _try_compile_deployment_target_33141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 12), self_33140, '_try_compile_deployment_target')
        # Calling _try_compile_deployment_target(args, kwargs) (line 459)
        _try_compile_deployment_target_call_result_33145 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), _try_compile_deployment_target_33141, *[str_33142, deptarget_33143], **kwargs_33144)
        
        # SSA join for if statement (line 454)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_deployment_target_higher_ok(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_deployment_target_higher_ok' in the type store
        # Getting the type of 'stypy_return_type' (line 448)
        stypy_return_type_33146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33146)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_deployment_target_higher_ok'
        return stypy_return_type_33146


    @norecursion
    def _try_compile_deployment_target(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_try_compile_deployment_target'
        module_type_store = module_type_store.open_function_context('_try_compile_deployment_target', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_localization', localization)
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_type_store', module_type_store)
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_function_name', 'BuildExtTestCase._try_compile_deployment_target')
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_param_names_list', ['operator', 'target'])
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_varargs_param_name', None)
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_call_defaults', defaults)
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_call_varargs', varargs)
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BuildExtTestCase._try_compile_deployment_target.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase._try_compile_deployment_target', ['operator', 'target'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_try_compile_deployment_target', localization, ['operator', 'target'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_try_compile_deployment_target(...)' code ##################

        
        # Assigning a Attribute to a Name (line 462):
        # Getting the type of 'os' (line 462)
        os_33147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 23), 'os')
        # Obtaining the member 'environ' of a type (line 462)
        environ_33148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 23), os_33147, 'environ')
        # Assigning a type to the variable 'orig_environ' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'orig_environ', environ_33148)
        
        # Assigning a Call to a Attribute (line 463):
        
        # Call to copy(...): (line 463)
        # Processing the call keyword arguments (line 463)
        kwargs_33151 = {}
        # Getting the type of 'orig_environ' (line 463)
        orig_environ_33149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 21), 'orig_environ', False)
        # Obtaining the member 'copy' of a type (line 463)
        copy_33150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 21), orig_environ_33149, 'copy')
        # Calling copy(args, kwargs) (line 463)
        copy_call_result_33152 = invoke(stypy.reporting.localization.Localization(__file__, 463, 21), copy_33150, *[], **kwargs_33151)
        
        # Getting the type of 'os' (line 463)
        os_33153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'os')
        # Setting the type of the member 'environ' of a type (line 463)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), os_33153, 'environ', copy_call_result_33152)
        
        # Call to addCleanup(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'setattr' (line 464)
        setattr_33156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'setattr', False)
        # Getting the type of 'os' (line 464)
        os_33157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 33), 'os', False)
        str_33158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 37), 'str', 'environ')
        # Getting the type of 'orig_environ' (line 464)
        orig_environ_33159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 48), 'orig_environ', False)
        # Processing the call keyword arguments (line 464)
        kwargs_33160 = {}
        # Getting the type of 'self' (line 464)
        self_33154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 464)
        addCleanup_33155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 8), self_33154, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 464)
        addCleanup_call_result_33161 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), addCleanup_33155, *[setattr_33156, os_33157, str_33158, orig_environ_33159], **kwargs_33160)
        
        
        # Type idiom detected: calculating its left and rigth part (line 466)
        # Getting the type of 'target' (line 466)
        target_33162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 11), 'target')
        # Getting the type of 'None' (line 466)
        None_33163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 21), 'None')
        
        (may_be_33164, more_types_in_union_33165) = may_be_none(target_33162, None_33163)

        if may_be_33164:

            if more_types_in_union_33165:
                # Runtime conditional SSA (line 466)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to get(...): (line 467)
            # Processing the call arguments (line 467)
            str_33169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 30), 'str', 'MACOSX_DEPLOYMENT_TARGET')
            # Processing the call keyword arguments (line 467)
            kwargs_33170 = {}
            # Getting the type of 'os' (line 467)
            os_33166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'os', False)
            # Obtaining the member 'environ' of a type (line 467)
            environ_33167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 15), os_33166, 'environ')
            # Obtaining the member 'get' of a type (line 467)
            get_33168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 15), environ_33167, 'get')
            # Calling get(args, kwargs) (line 467)
            get_call_result_33171 = invoke(stypy.reporting.localization.Localization(__file__, 467, 15), get_33168, *[str_33169], **kwargs_33170)
            
            # Testing the type of an if condition (line 467)
            if_condition_33172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 467, 12), get_call_result_33171)
            # Assigning a type to the variable 'if_condition_33172' (line 467)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'if_condition_33172', if_condition_33172)
            # SSA begins for if statement (line 467)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Deleting a member
            # Getting the type of 'os' (line 468)
            os_33173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 20), 'os')
            # Obtaining the member 'environ' of a type (line 468)
            environ_33174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 20), os_33173, 'environ')
            
            # Obtaining the type of the subscript
            str_33175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 31), 'str', 'MACOSX_DEPLOYMENT_TARGET')
            # Getting the type of 'os' (line 468)
            os_33176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 20), 'os')
            # Obtaining the member 'environ' of a type (line 468)
            environ_33177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 20), os_33176, 'environ')
            # Obtaining the member '__getitem__' of a type (line 468)
            getitem___33178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 20), environ_33177, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 468)
            subscript_call_result_33179 = invoke(stypy.reporting.localization.Localization(__file__, 468, 20), getitem___33178, str_33175)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 16), environ_33174, subscript_call_result_33179)
            # SSA join for if statement (line 467)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_33165:
                # Runtime conditional SSA for else branch (line 466)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_33164) or more_types_in_union_33165):
            
            # Assigning a Name to a Subscript (line 470):
            # Getting the type of 'target' (line 470)
            target_33180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 53), 'target')
            # Getting the type of 'os' (line 470)
            os_33181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'os')
            # Obtaining the member 'environ' of a type (line 470)
            environ_33182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 12), os_33181, 'environ')
            str_33183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 23), 'str', 'MACOSX_DEPLOYMENT_TARGET')
            # Storing an element on a container (line 470)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 12), environ_33182, (str_33183, target_33180))

            if (may_be_33164 and more_types_in_union_33165):
                # SSA join for if statement (line 466)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 472):
        
        # Call to join(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'self' (line 472)
        self_33187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 35), 'self', False)
        # Obtaining the member 'tmp_dir' of a type (line 472)
        tmp_dir_33188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 35), self_33187, 'tmp_dir')
        str_33189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 49), 'str', 'deptargetmodule.c')
        # Processing the call keyword arguments (line 472)
        kwargs_33190 = {}
        # Getting the type of 'os' (line 472)
        os_33184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 472)
        path_33185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 22), os_33184, 'path')
        # Obtaining the member 'join' of a type (line 472)
        join_33186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 22), path_33185, 'join')
        # Calling join(args, kwargs) (line 472)
        join_call_result_33191 = invoke(stypy.reporting.localization.Localization(__file__, 472, 22), join_33186, *[tmp_dir_33188, str_33189], **kwargs_33190)
        
        # Assigning a type to the variable 'deptarget_c' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'deptarget_c', join_call_result_33191)
        
        # Call to open(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'deptarget_c' (line 474)
        deptarget_c_33193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 18), 'deptarget_c', False)
        str_33194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 31), 'str', 'w')
        # Processing the call keyword arguments (line 474)
        kwargs_33195 = {}
        # Getting the type of 'open' (line 474)
        open_33192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 13), 'open', False)
        # Calling open(args, kwargs) (line 474)
        open_call_result_33196 = invoke(stypy.reporting.localization.Localization(__file__, 474, 13), open_33192, *[deptarget_c_33193, str_33194], **kwargs_33195)
        
        with_33197 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 474, 13), open_call_result_33196, 'with parameter', '__enter__', '__exit__')

        if with_33197:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 474)
            enter___33198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 13), open_call_result_33196, '__enter__')
            with_enter_33199 = invoke(stypy.reporting.localization.Localization(__file__, 474, 13), enter___33198)
            # Assigning a type to the variable 'fp' (line 474)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 13), 'fp', with_enter_33199)
            
            # Call to write(...): (line 475)
            # Processing the call arguments (line 475)
            
            # Call to dedent(...): (line 475)
            # Processing the call arguments (line 475)
            str_33204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, (-1)), 'str', '                #include <AvailabilityMacros.h>\n\n                int dummy;\n\n                #if TARGET %s MAC_OS_X_VERSION_MIN_REQUIRED\n                #else\n                #error "Unexpected target"\n                #endif\n\n            ')
            # Getting the type of 'operator' (line 485)
            operator_33205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 18), 'operator', False)
            # Applying the binary operator '%' (line 485)
            result_mod_33206 = python_operator(stypy.reporting.localization.Localization(__file__, 485, (-1)), '%', str_33204, operator_33205)
            
            # Processing the call keyword arguments (line 475)
            kwargs_33207 = {}
            # Getting the type of 'textwrap' (line 475)
            textwrap_33202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 21), 'textwrap', False)
            # Obtaining the member 'dedent' of a type (line 475)
            dedent_33203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 21), textwrap_33202, 'dedent')
            # Calling dedent(args, kwargs) (line 475)
            dedent_call_result_33208 = invoke(stypy.reporting.localization.Localization(__file__, 475, 21), dedent_33203, *[result_mod_33206], **kwargs_33207)
            
            # Processing the call keyword arguments (line 475)
            kwargs_33209 = {}
            # Getting the type of 'fp' (line 475)
            fp_33200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'fp', False)
            # Obtaining the member 'write' of a type (line 475)
            write_33201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), fp_33200, 'write')
            # Calling write(args, kwargs) (line 475)
            write_call_result_33210 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), write_33201, *[dedent_call_result_33208], **kwargs_33209)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 474)
            exit___33211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 13), open_call_result_33196, '__exit__')
            with_exit_33212 = invoke(stypy.reporting.localization.Localization(__file__, 474, 13), exit___33211, None, None, None)

        
        # Assigning a Call to a Name (line 488):
        
        # Call to get_config_var(...): (line 488)
        # Processing the call arguments (line 488)
        str_33215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 42), 'str', 'MACOSX_DEPLOYMENT_TARGET')
        # Processing the call keyword arguments (line 488)
        kwargs_33216 = {}
        # Getting the type of 'sysconfig' (line 488)
        sysconfig_33213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 17), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 488)
        get_config_var_33214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 17), sysconfig_33213, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 488)
        get_config_var_call_result_33217 = invoke(stypy.reporting.localization.Localization(__file__, 488, 17), get_config_var_33214, *[str_33215], **kwargs_33216)
        
        # Assigning a type to the variable 'target' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'target', get_config_var_call_result_33217)
        
        # Assigning a Call to a Name (line 489):
        
        # Call to tuple(...): (line 489)
        # Processing the call arguments (line 489)
        
        # Call to map(...): (line 489)
        # Processing the call arguments (line 489)
        # Getting the type of 'int' (line 489)
        int_33220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 27), 'int', False)
        
        # Obtaining the type of the subscript
        int_33221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 50), 'int')
        int_33222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 52), 'int')
        slice_33223 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 489, 32), int_33221, int_33222, None)
        
        # Call to split(...): (line 489)
        # Processing the call arguments (line 489)
        str_33226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 45), 'str', '.')
        # Processing the call keyword arguments (line 489)
        kwargs_33227 = {}
        # Getting the type of 'target' (line 489)
        target_33224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 32), 'target', False)
        # Obtaining the member 'split' of a type (line 489)
        split_33225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 32), target_33224, 'split')
        # Calling split(args, kwargs) (line 489)
        split_call_result_33228 = invoke(stypy.reporting.localization.Localization(__file__, 489, 32), split_33225, *[str_33226], **kwargs_33227)
        
        # Obtaining the member '__getitem__' of a type (line 489)
        getitem___33229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 32), split_call_result_33228, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 489)
        subscript_call_result_33230 = invoke(stypy.reporting.localization.Localization(__file__, 489, 32), getitem___33229, slice_33223)
        
        # Processing the call keyword arguments (line 489)
        kwargs_33231 = {}
        # Getting the type of 'map' (line 489)
        map_33219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 23), 'map', False)
        # Calling map(args, kwargs) (line 489)
        map_call_result_33232 = invoke(stypy.reporting.localization.Localization(__file__, 489, 23), map_33219, *[int_33220, subscript_call_result_33230], **kwargs_33231)
        
        # Processing the call keyword arguments (line 489)
        kwargs_33233 = {}
        # Getting the type of 'tuple' (line 489)
        tuple_33218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'tuple', False)
        # Calling tuple(args, kwargs) (line 489)
        tuple_call_result_33234 = invoke(stypy.reporting.localization.Localization(__file__, 489, 17), tuple_33218, *[map_call_result_33232], **kwargs_33233)
        
        # Assigning a type to the variable 'target' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'target', tuple_call_result_33234)
        
        
        
        # Obtaining the type of the subscript
        int_33235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 18), 'int')
        # Getting the type of 'target' (line 493)
        target_33236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'target')
        # Obtaining the member '__getitem__' of a type (line 493)
        getitem___33237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 11), target_33236, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 493)
        subscript_call_result_33238 = invoke(stypy.reporting.localization.Localization(__file__, 493, 11), getitem___33237, int_33235)
        
        int_33239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 23), 'int')
        # Applying the binary operator '<' (line 493)
        result_lt_33240 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 11), '<', subscript_call_result_33238, int_33239)
        
        # Testing the type of an if condition (line 493)
        if_condition_33241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 493, 8), result_lt_33240)
        # Assigning a type to the variable 'if_condition_33241' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'if_condition_33241', if_condition_33241)
        # SSA begins for if statement (line 493)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 495):
        str_33242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 21), 'str', '%02d%01d0')
        # Getting the type of 'target' (line 495)
        target_33243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 35), 'target')
        # Applying the binary operator '%' (line 495)
        result_mod_33244 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 21), '%', str_33242, target_33243)
        
        # Assigning a type to the variable 'target' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'target', result_mod_33244)
        # SSA branch for the else part of an if statement (line 493)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 498):
        str_33245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 21), 'str', '%02d%02d00')
        # Getting the type of 'target' (line 498)
        target_33246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 36), 'target')
        # Applying the binary operator '%' (line 498)
        result_mod_33247 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 21), '%', str_33245, target_33246)
        
        # Assigning a type to the variable 'target' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'target', result_mod_33247)
        # SSA join for if statement (line 493)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 499):
        
        # Call to Extension(...): (line 499)
        # Processing the call arguments (line 499)
        str_33249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 12), 'str', 'deptarget')
        
        # Obtaining an instance of the builtin type 'list' (line 501)
        list_33250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 501)
        # Adding element type (line 501)
        # Getting the type of 'deptarget_c' (line 501)
        deptarget_c_33251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 13), 'deptarget_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 12), list_33250, deptarget_c_33251)
        
        # Processing the call keyword arguments (line 499)
        
        # Obtaining an instance of the builtin type 'list' (line 502)
        list_33252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 502)
        # Adding element type (line 502)
        str_33253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 32), 'str', '-DTARGET=%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 502)
        tuple_33254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 502)
        # Adding element type (line 502)
        # Getting the type of 'target' (line 502)
        target_33255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 47), 'target', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 47), tuple_33254, target_33255)
        
        # Applying the binary operator '%' (line 502)
        result_mod_33256 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 32), '%', str_33253, tuple_33254)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 502, 31), list_33252, result_mod_33256)
        
        keyword_33257 = list_33252
        kwargs_33258 = {'extra_compile_args': keyword_33257}
        # Getting the type of 'Extension' (line 499)
        Extension_33248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 'Extension', False)
        # Calling Extension(args, kwargs) (line 499)
        Extension_call_result_33259 = invoke(stypy.reporting.localization.Localization(__file__, 499, 24), Extension_33248, *[str_33249, list_33250], **kwargs_33258)
        
        # Assigning a type to the variable 'deptarget_ext' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'deptarget_ext', Extension_call_result_33259)
        
        # Assigning a Call to a Name (line 504):
        
        # Call to Distribution(...): (line 504)
        # Processing the call arguments (line 504)
        
        # Obtaining an instance of the builtin type 'dict' (line 504)
        dict_33261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 504)
        # Adding element type (key, value) (line 504)
        str_33262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 12), 'str', 'name')
        str_33263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 20), 'str', 'deptarget')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 28), dict_33261, (str_33262, str_33263))
        # Adding element type (key, value) (line 504)
        str_33264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 12), 'str', 'ext_modules')
        
        # Obtaining an instance of the builtin type 'list' (line 506)
        list_33265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 506)
        # Adding element type (line 506)
        # Getting the type of 'deptarget_ext' (line 506)
        deptarget_ext_33266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 28), 'deptarget_ext', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 27), list_33265, deptarget_ext_33266)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 28), dict_33261, (str_33264, list_33265))
        
        # Processing the call keyword arguments (line 504)
        kwargs_33267 = {}
        # Getting the type of 'Distribution' (line 504)
        Distribution_33260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 504)
        Distribution_call_result_33268 = invoke(stypy.reporting.localization.Localization(__file__, 504, 15), Distribution_33260, *[dict_33261], **kwargs_33267)
        
        # Assigning a type to the variable 'dist' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'dist', Distribution_call_result_33268)
        
        # Assigning a Attribute to a Attribute (line 508):
        # Getting the type of 'self' (line 508)
        self_33269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 27), 'self')
        # Obtaining the member 'tmp_dir' of a type (line 508)
        tmp_dir_33270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 27), self_33269, 'tmp_dir')
        # Getting the type of 'dist' (line 508)
        dist_33271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'dist')
        # Setting the type of the member 'package_dir' of a type (line 508)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), dist_33271, 'package_dir', tmp_dir_33270)
        
        # Assigning a Call to a Name (line 509):
        
        # Call to build_ext(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'dist' (line 509)
        dist_33273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 24), 'dist', False)
        # Processing the call keyword arguments (line 509)
        kwargs_33274 = {}
        # Getting the type of 'build_ext' (line 509)
        build_ext_33272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 14), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 509)
        build_ext_call_result_33275 = invoke(stypy.reporting.localization.Localization(__file__, 509, 14), build_ext_33272, *[dist_33273], **kwargs_33274)
        
        # Assigning a type to the variable 'cmd' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'cmd', build_ext_call_result_33275)
        
        # Assigning a Attribute to a Attribute (line 510):
        # Getting the type of 'self' (line 510)
        self_33276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 24), 'self')
        # Obtaining the member 'tmp_dir' of a type (line 510)
        tmp_dir_33277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 24), self_33276, 'tmp_dir')
        # Getting the type of 'cmd' (line 510)
        cmd_33278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'cmd')
        # Setting the type of the member 'build_lib' of a type (line 510)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), cmd_33278, 'build_lib', tmp_dir_33277)
        
        # Assigning a Attribute to a Attribute (line 511):
        # Getting the type of 'self' (line 511)
        self_33279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 25), 'self')
        # Obtaining the member 'tmp_dir' of a type (line 511)
        tmp_dir_33280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 25), self_33279, 'tmp_dir')
        # Getting the type of 'cmd' (line 511)
        cmd_33281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'cmd')
        # Setting the type of the member 'build_temp' of a type (line 511)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 8), cmd_33281, 'build_temp', tmp_dir_33280)
        
        
        # SSA begins for try-except statement (line 513)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to ensure_finalized(...): (line 514)
        # Processing the call keyword arguments (line 514)
        kwargs_33284 = {}
        # Getting the type of 'cmd' (line 514)
        cmd_33282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 514)
        ensure_finalized_33283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 12), cmd_33282, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 514)
        ensure_finalized_call_result_33285 = invoke(stypy.reporting.localization.Localization(__file__, 514, 12), ensure_finalized_33283, *[], **kwargs_33284)
        
        
        # Call to run(...): (line 515)
        # Processing the call keyword arguments (line 515)
        kwargs_33288 = {}
        # Getting the type of 'cmd' (line 515)
        cmd_33286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'cmd', False)
        # Obtaining the member 'run' of a type (line 515)
        run_33287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 12), cmd_33286, 'run')
        # Calling run(args, kwargs) (line 515)
        run_call_result_33289 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), run_33287, *[], **kwargs_33288)
        
        # SSA branch for the except part of a try statement (line 513)
        # SSA branch for the except 'CompileError' branch of a try statement (line 513)
        module_type_store.open_ssa_branch('except')
        
        # Call to fail(...): (line 517)
        # Processing the call arguments (line 517)
        str_33292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 22), 'str', 'Wrong deployment target during compilation')
        # Processing the call keyword arguments (line 517)
        kwargs_33293 = {}
        # Getting the type of 'self' (line 517)
        self_33290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 517)
        fail_33291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 12), self_33290, 'fail')
        # Calling fail(args, kwargs) (line 517)
        fail_call_result_33294 = invoke(stypy.reporting.localization.Localization(__file__, 517, 12), fail_33291, *[str_33292], **kwargs_33293)
        
        # SSA join for try-except statement (line 513)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_try_compile_deployment_target(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_try_compile_deployment_target' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_33295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33295)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_try_compile_deployment_target'
        return stypy_return_type_33295


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 0, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BuildExtTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BuildExtTestCase' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'BuildExtTestCase', BuildExtTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 519, 0, False)
    
    # Passed parameters checking function
    test_suite.stypy_localization = localization
    test_suite.stypy_type_of_self = None
    test_suite.stypy_type_store = module_type_store
    test_suite.stypy_function_name = 'test_suite'
    test_suite.stypy_param_names_list = []
    test_suite.stypy_varargs_param_name = None
    test_suite.stypy_kwargs_param_name = None
    test_suite.stypy_call_defaults = defaults
    test_suite.stypy_call_varargs = varargs
    test_suite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_suite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_suite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_suite(...)' code ##################

    
    # Call to makeSuite(...): (line 520)
    # Processing the call arguments (line 520)
    # Getting the type of 'BuildExtTestCase' (line 520)
    BuildExtTestCase_33298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 30), 'BuildExtTestCase', False)
    # Processing the call keyword arguments (line 520)
    kwargs_33299 = {}
    # Getting the type of 'unittest' (line 520)
    unittest_33296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 520)
    makeSuite_33297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 11), unittest_33296, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 520)
    makeSuite_call_result_33300 = invoke(stypy.reporting.localization.Localization(__file__, 520, 11), makeSuite_33297, *[BuildExtTestCase_33298], **kwargs_33299)
    
    # Assigning a type to the variable 'stypy_return_type' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'stypy_return_type', makeSuite_call_result_33300)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 519)
    stypy_return_type_33301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_33301)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_33301

# Assigning a type to the variable 'test_suite' (line 519)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 523)
    # Processing the call arguments (line 523)
    
    # Call to test_suite(...): (line 523)
    # Processing the call keyword arguments (line 523)
    kwargs_33305 = {}
    # Getting the type of 'test_suite' (line 523)
    test_suite_33304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 30), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 523)
    test_suite_call_result_33306 = invoke(stypy.reporting.localization.Localization(__file__, 523, 30), test_suite_33304, *[], **kwargs_33305)
    
    # Processing the call keyword arguments (line 523)
    kwargs_33307 = {}
    # Getting the type of 'test_support' (line 523)
    test_support_33302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'test_support', False)
    # Obtaining the member 'run_unittest' of a type (line 523)
    run_unittest_33303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 4), test_support_33302, 'run_unittest')
    # Calling run_unittest(args, kwargs) (line 523)
    run_unittest_call_result_33308 = invoke(stypy.reporting.localization.Localization(__file__, 523, 4), run_unittest_33303, *[test_suite_call_result_33306], **kwargs_33307)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
