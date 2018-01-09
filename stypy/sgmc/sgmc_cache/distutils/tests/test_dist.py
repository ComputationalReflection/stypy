
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf8 -*-
2: 
3: '''Tests for distutils.dist.'''
4: import os
5: import StringIO
6: import sys
7: import unittest
8: import warnings
9: import textwrap
10: 
11: from distutils.dist import Distribution, fix_help_options
12: from distutils.cmd import Command
13: import distutils.dist
14: from test.test_support import TESTFN, captured_stdout, run_unittest, unlink
15: from distutils.tests import support
16: from distutils import log
17: 
18: 
19: class test_dist(Command):
20:     '''Sample distutils extension command.'''
21: 
22:     user_options = [
23:         ("sample-option=", "S", "help text"),
24:         ]
25: 
26:     def initialize_options(self):
27:         self.sample_option = None
28: 
29: 
30: class TestDistribution(Distribution):
31:     '''Distribution subclasses that avoids the default search for
32:     configuration files.
33: 
34:     The ._config_files attribute must be set before
35:     .parse_config_files() is called.
36:     '''
37: 
38:     def find_config_files(self):
39:         return self._config_files
40: 
41: 
42: class DistributionTestCase(support.TempdirManager,
43:                            support.LoggingSilencer,
44:                            support.EnvironGuard,
45:                            unittest.TestCase):
46: 
47:     def setUp(self):
48:         super(DistributionTestCase, self).setUp()
49:         self.argv = sys.argv, sys.argv[:]
50:         del sys.argv[1:]
51: 
52:     def tearDown(self):
53:         sys.argv = self.argv[0]
54:         sys.argv[:] = self.argv[1]
55:         super(DistributionTestCase, self).tearDown()
56: 
57:     def create_distribution(self, configfiles=()):
58:         d = TestDistribution()
59:         d._config_files = configfiles
60:         d.parse_config_files()
61:         d.parse_command_line()
62:         return d
63: 
64:     def test_debug_mode(self):
65:         with open(TESTFN, "w") as f:
66:             f.write("[global]\n")
67:             f.write("command_packages = foo.bar, splat")
68:         self.addCleanup(unlink, TESTFN)
69: 
70:         files = [TESTFN]
71:         sys.argv.append("build")
72: 
73:         with captured_stdout() as stdout:
74:             self.create_distribution(files)
75:         stdout.seek(0)
76:         self.assertEqual(stdout.read(), '')
77:         distutils.dist.DEBUG = True
78:         try:
79:             with captured_stdout() as stdout:
80:                 self.create_distribution(files)
81:             stdout.seek(0)
82:             self.assertEqual(stdout.read(), '')
83:         finally:
84:             distutils.dist.DEBUG = False
85: 
86:     def test_command_packages_unspecified(self):
87:         sys.argv.append("build")
88:         d = self.create_distribution()
89:         self.assertEqual(d.get_command_packages(), ["distutils.command"])
90: 
91:     def test_command_packages_cmdline(self):
92:         from distutils.tests.test_dist import test_dist
93:         sys.argv.extend(["--command-packages",
94:                          "foo.bar,distutils.tests",
95:                          "test_dist",
96:                          "-Ssometext",
97:                          ])
98:         d = self.create_distribution()
99:         # let's actually try to load our test command:
100:         self.assertEqual(d.get_command_packages(),
101:                          ["distutils.command", "foo.bar", "distutils.tests"])
102:         cmd = d.get_command_obj("test_dist")
103:         self.assertIsInstance(cmd, test_dist)
104:         self.assertEqual(cmd.sample_option, "sometext")
105: 
106:     def test_command_packages_configfile(self):
107:         sys.argv.append("build")
108:         self.addCleanup(os.unlink, TESTFN)
109:         f = open(TESTFN, "w")
110:         try:
111:             print >> f, "[global]"
112:             print >> f, "command_packages = foo.bar, splat"
113:         finally:
114:             f.close()
115: 
116:         d = self.create_distribution([TESTFN])
117:         self.assertEqual(d.get_command_packages(),
118:                          ["distutils.command", "foo.bar", "splat"])
119: 
120:         # ensure command line overrides config:
121:         sys.argv[1:] = ["--command-packages", "spork", "build"]
122:         d = self.create_distribution([TESTFN])
123:         self.assertEqual(d.get_command_packages(),
124:                          ["distutils.command", "spork"])
125: 
126:         # Setting --command-packages to '' should cause the default to
127:         # be used even if a config file specified something else:
128:         sys.argv[1:] = ["--command-packages", "", "build"]
129:         d = self.create_distribution([TESTFN])
130:         self.assertEqual(d.get_command_packages(), ["distutils.command"])
131: 
132:     def test_write_pkg_file(self):
133:         # Check DistributionMetadata handling of Unicode fields
134:         tmp_dir = self.mkdtemp()
135:         my_file = os.path.join(tmp_dir, 'f')
136:         klass = Distribution
137: 
138:         dist = klass(attrs={'author': u'Mister Café',
139:                             'name': 'my.package',
140:                             'maintainer': u'Café Junior',
141:                             'description': u'Café torréfié',
142:                             'long_description': u'Héhéhé'})
143: 
144:         # let's make sure the file can be written
145:         # with Unicode fields. they are encoded with
146:         # PKG_INFO_ENCODING
147:         dist.metadata.write_pkg_file(open(my_file, 'w'))
148: 
149:         # regular ascii is of course always usable
150:         dist = klass(attrs={'author': 'Mister Cafe',
151:                             'name': 'my.package',
152:                             'maintainer': 'Cafe Junior',
153:                             'description': 'Cafe torrefie',
154:                             'long_description': 'Hehehe'})
155: 
156:         my_file2 = os.path.join(tmp_dir, 'f2')
157:         dist.metadata.write_pkg_file(open(my_file2, 'w'))
158: 
159:     def test_empty_options(self):
160:         # an empty options dictionary should not stay in the
161:         # list of attributes
162: 
163:         # catching warnings
164:         warns = []
165: 
166:         def _warn(msg):
167:             warns.append(msg)
168: 
169:         self.addCleanup(setattr, warnings, 'warn', warnings.warn)
170:         warnings.warn = _warn
171:         dist = Distribution(attrs={'author': 'xxx', 'name': 'xxx',
172:                                    'version': 'xxx', 'url': 'xxxx',
173:                                    'options': {}})
174: 
175:         self.assertEqual(len(warns), 0)
176:         self.assertNotIn('options', dir(dist))
177: 
178:     def test_finalize_options(self):
179:         attrs = {'keywords': 'one,two',
180:                  'platforms': 'one,two'}
181: 
182:         dist = Distribution(attrs=attrs)
183:         dist.finalize_options()
184: 
185:         # finalize_option splits platforms and keywords
186:         self.assertEqual(dist.metadata.platforms, ['one', 'two'])
187:         self.assertEqual(dist.metadata.keywords, ['one', 'two'])
188: 
189:     def test_get_command_packages(self):
190:         dist = Distribution()
191:         self.assertEqual(dist.command_packages, None)
192:         cmds = dist.get_command_packages()
193:         self.assertEqual(cmds, ['distutils.command'])
194:         self.assertEqual(dist.command_packages,
195:                          ['distutils.command'])
196: 
197:         dist.command_packages = 'one,two'
198:         cmds = dist.get_command_packages()
199:         self.assertEqual(cmds, ['distutils.command', 'one', 'two'])
200: 
201:     def test_announce(self):
202:         # make sure the level is known
203:         dist = Distribution()
204:         args = ('ok',)
205:         kwargs = {'level': 'ok2'}
206:         self.assertRaises(ValueError, dist.announce, args, kwargs)
207: 
208:     def test_find_config_files_disable(self):
209:         # Ticket #1180: Allow user to disable their home config file.
210:         temp_home = self.mkdtemp()
211:         if os.name == 'posix':
212:             user_filename = os.path.join(temp_home, ".pydistutils.cfg")
213:         else:
214:             user_filename = os.path.join(temp_home, "pydistutils.cfg")
215: 
216:         with open(user_filename, 'w') as f:
217:             f.write('[distutils]\n')
218: 
219:         def _expander(path):
220:             return temp_home
221: 
222:         old_expander = os.path.expanduser
223:         os.path.expanduser = _expander
224:         try:
225:             d = distutils.dist.Distribution()
226:             all_files = d.find_config_files()
227: 
228:             d = distutils.dist.Distribution(attrs={'script_args':
229:                                             ['--no-user-cfg']})
230:             files = d.find_config_files()
231:         finally:
232:             os.path.expanduser = old_expander
233: 
234:         # make sure --no-user-cfg disables the user cfg file
235:         self.assertEqual(len(all_files)-1, len(files))
236: 
237: 
238: class MetadataTestCase(support.TempdirManager, support.EnvironGuard,
239:                        unittest.TestCase):
240: 
241:     def setUp(self):
242:         super(MetadataTestCase, self).setUp()
243:         self.argv = sys.argv, sys.argv[:]
244: 
245:     def tearDown(self):
246:         sys.argv = self.argv[0]
247:         sys.argv[:] = self.argv[1]
248:         super(MetadataTestCase, self).tearDown()
249: 
250:     def test_classifier(self):
251:         attrs = {'name': 'Boa', 'version': '3.0',
252:                  'classifiers': ['Programming Language :: Python :: 3']}
253:         dist = Distribution(attrs)
254:         meta = self.format_metadata(dist)
255:         self.assertIn('Metadata-Version: 1.1', meta)
256: 
257:     def test_download_url(self):
258:         attrs = {'name': 'Boa', 'version': '3.0',
259:                  'download_url': 'http://example.org/boa'}
260:         dist = Distribution(attrs)
261:         meta = self.format_metadata(dist)
262:         self.assertIn('Metadata-Version: 1.1', meta)
263: 
264:     def test_long_description(self):
265:         long_desc = textwrap.dedent('''\
266:         example::
267:               We start here
268:             and continue here
269:           and end here.''')
270:         attrs = {"name": "package",
271:                  "version": "1.0",
272:                  "long_description": long_desc}
273: 
274:         dist = Distribution(attrs)
275:         meta = self.format_metadata(dist)
276:         meta = meta.replace('\n' + 8 * ' ', '\n')
277:         self.assertIn(long_desc, meta)
278: 
279:     def test_simple_metadata(self):
280:         attrs = {"name": "package",
281:                  "version": "1.0"}
282:         dist = Distribution(attrs)
283:         meta = self.format_metadata(dist)
284:         self.assertIn("Metadata-Version: 1.0", meta)
285:         self.assertNotIn("provides:", meta.lower())
286:         self.assertNotIn("requires:", meta.lower())
287:         self.assertNotIn("obsoletes:", meta.lower())
288: 
289:     def test_provides(self):
290:         attrs = {"name": "package",
291:                  "version": "1.0",
292:                  "provides": ["package", "package.sub"]}
293:         dist = Distribution(attrs)
294:         self.assertEqual(dist.metadata.get_provides(),
295:                          ["package", "package.sub"])
296:         self.assertEqual(dist.get_provides(),
297:                          ["package", "package.sub"])
298:         meta = self.format_metadata(dist)
299:         self.assertIn("Metadata-Version: 1.1", meta)
300:         self.assertNotIn("requires:", meta.lower())
301:         self.assertNotIn("obsoletes:", meta.lower())
302: 
303:     def test_provides_illegal(self):
304:         self.assertRaises(ValueError, Distribution,
305:                           {"name": "package",
306:                            "version": "1.0",
307:                            "provides": ["my.pkg (splat)"]})
308: 
309:     def test_requires(self):
310:         attrs = {"name": "package",
311:                  "version": "1.0",
312:                  "requires": ["other", "another (==1.0)"]}
313:         dist = Distribution(attrs)
314:         self.assertEqual(dist.metadata.get_requires(),
315:                          ["other", "another (==1.0)"])
316:         self.assertEqual(dist.get_requires(),
317:                          ["other", "another (==1.0)"])
318:         meta = self.format_metadata(dist)
319:         self.assertIn("Metadata-Version: 1.1", meta)
320:         self.assertNotIn("provides:", meta.lower())
321:         self.assertIn("Requires: other", meta)
322:         self.assertIn("Requires: another (==1.0)", meta)
323:         self.assertNotIn("obsoletes:", meta.lower())
324: 
325:     def test_requires_illegal(self):
326:         self.assertRaises(ValueError, Distribution,
327:                           {"name": "package",
328:                            "version": "1.0",
329:                            "requires": ["my.pkg (splat)"]})
330: 
331:     def test_obsoletes(self):
332:         attrs = {"name": "package",
333:                  "version": "1.0",
334:                  "obsoletes": ["other", "another (<1.0)"]}
335:         dist = Distribution(attrs)
336:         self.assertEqual(dist.metadata.get_obsoletes(),
337:                          ["other", "another (<1.0)"])
338:         self.assertEqual(dist.get_obsoletes(),
339:                          ["other", "another (<1.0)"])
340:         meta = self.format_metadata(dist)
341:         self.assertIn("Metadata-Version: 1.1", meta)
342:         self.assertNotIn("provides:", meta.lower())
343:         self.assertNotIn("requires:", meta.lower())
344:         self.assertIn("Obsoletes: other", meta)
345:         self.assertIn("Obsoletes: another (<1.0)", meta)
346: 
347:     def test_obsoletes_illegal(self):
348:         self.assertRaises(ValueError, Distribution,
349:                           {"name": "package",
350:                            "version": "1.0",
351:                            "obsoletes": ["my.pkg (splat)"]})
352: 
353:     def format_metadata(self, dist):
354:         sio = StringIO.StringIO()
355:         dist.metadata.write_pkg_file(sio)
356:         return sio.getvalue()
357: 
358:     def test_custom_pydistutils(self):
359:         # fixes #2166
360:         # make sure pydistutils.cfg is found
361:         if os.name == 'posix':
362:             user_filename = ".pydistutils.cfg"
363:         else:
364:             user_filename = "pydistutils.cfg"
365: 
366:         temp_dir = self.mkdtemp()
367:         user_filename = os.path.join(temp_dir, user_filename)
368:         f = open(user_filename, 'w')
369:         try:
370:             f.write('.')
371:         finally:
372:             f.close()
373: 
374:         try:
375:             dist = Distribution()
376: 
377:             # linux-style
378:             if sys.platform in ('linux', 'darwin'):
379:                 os.environ['HOME'] = temp_dir
380:                 files = dist.find_config_files()
381:                 self.assertIn(user_filename, files)
382: 
383:             # win32-style
384:             if sys.platform == 'win32':
385:                 # home drive should be found
386:                 os.environ['HOME'] = temp_dir
387:                 files = dist.find_config_files()
388:                 self.assertIn(user_filename, files,
389:                              '%r not found in %r' % (user_filename, files))
390:         finally:
391:             os.remove(user_filename)
392: 
393:     def test_fix_help_options(self):
394:         help_tuples = [('a', 'b', 'c', 'd'), (1, 2, 3, 4)]
395:         fancy_options = fix_help_options(help_tuples)
396:         self.assertEqual(fancy_options[0], ('a', 'b', 'c'))
397:         self.assertEqual(fancy_options[1], (1, 2, 3))
398: 
399:     def test_show_help(self):
400:         # smoke test, just makes sure some help is displayed
401:         self.addCleanup(log.set_threshold, log._global_log.threshold)
402:         dist = Distribution()
403:         sys.argv = []
404:         dist.help = 1
405:         dist.script_name = 'setup.py'
406:         with captured_stdout() as s:
407:             dist.parse_command_line()
408: 
409:         output = [line for line in s.getvalue().split('\n')
410:                   if line.strip() != '']
411:         self.assertTrue(output)
412: 
413:     def test_read_metadata(self):
414:         attrs = {"name": "package",
415:                  "version": "1.0",
416:                  "long_description": "desc",
417:                  "description": "xxx",
418:                  "download_url": "http://example.com",
419:                  "keywords": ['one', 'two'],
420:                  "requires": ['foo']}
421: 
422:         dist = Distribution(attrs)
423:         metadata = dist.metadata
424: 
425:         # write it then reloads it
426:         PKG_INFO = StringIO.StringIO()
427:         metadata.write_pkg_file(PKG_INFO)
428:         PKG_INFO.seek(0)
429:         metadata.read_pkg_file(PKG_INFO)
430: 
431:         self.assertEqual(metadata.name, "package")
432:         self.assertEqual(metadata.version, "1.0")
433:         self.assertEqual(metadata.description, "xxx")
434:         self.assertEqual(metadata.download_url, 'http://example.com')
435:         self.assertEqual(metadata.keywords, ['one', 'two'])
436:         self.assertEqual(metadata.platforms, ['UNKNOWN'])
437:         self.assertEqual(metadata.obsoletes, None)
438:         self.assertEqual(metadata.requires, ['foo'])
439: 
440: 
441: def test_suite():
442:     suite = unittest.TestSuite()
443:     suite.addTest(unittest.makeSuite(DistributionTestCase))
444:     suite.addTest(unittest.makeSuite(MetadataTestCase))
445:     return suite
446: 
447: if __name__ == "__main__":
448:     run_unittest(test_suite())
449: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_36725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 0), 'str', 'Tests for distutils.dist.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import StringIO' statement (line 5)
import StringIO

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'StringIO', StringIO, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import sys' statement (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import unittest' statement (line 7)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import warnings' statement (line 8)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import textwrap' statement (line 9)
import textwrap

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'textwrap', textwrap, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.dist import Distribution, fix_help_options' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_36726 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dist')

if (type(import_36726) is not StypyTypeError):

    if (import_36726 != 'pyd_module'):
        __import__(import_36726)
        sys_modules_36727 = sys.modules[import_36726]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dist', sys_modules_36727.module_type_store, module_type_store, ['Distribution', 'fix_help_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_36727, sys_modules_36727.module_type_store, module_type_store)
    else:
        from distutils.dist import Distribution, fix_help_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dist', None, module_type_store, ['Distribution', 'fix_help_options'], [Distribution, fix_help_options])

else:
    # Assigning a type to the variable 'distutils.dist' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dist', import_36726)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.cmd import Command' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_36728 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.cmd')

if (type(import_36728) is not StypyTypeError):

    if (import_36728 != 'pyd_module'):
        __import__(import_36728)
        sys_modules_36729 = sys.modules[import_36728]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.cmd', sys_modules_36729.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_36729, sys_modules_36729.module_type_store, module_type_store)
    else:
        from distutils.cmd import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.cmd', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.cmd' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.cmd', import_36728)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import distutils.dist' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_36730 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dist')

if (type(import_36730) is not StypyTypeError):

    if (import_36730 != 'pyd_module'):
        __import__(import_36730)
        sys_modules_36731 = sys.modules[import_36730]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dist', sys_modules_36731.module_type_store, module_type_store)
    else:
        import distutils.dist

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dist', distutils.dist, module_type_store)

else:
    # Assigning a type to the variable 'distutils.dist' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dist', import_36730)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from test.test_support import TESTFN, captured_stdout, run_unittest, unlink' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_36732 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'test.test_support')

if (type(import_36732) is not StypyTypeError):

    if (import_36732 != 'pyd_module'):
        __import__(import_36732)
        sys_modules_36733 = sys.modules[import_36732]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'test.test_support', sys_modules_36733.module_type_store, module_type_store, ['TESTFN', 'captured_stdout', 'run_unittest', 'unlink'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_36733, sys_modules_36733.module_type_store, module_type_store)
    else:
        from test.test_support import TESTFN, captured_stdout, run_unittest, unlink

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'test.test_support', None, module_type_store, ['TESTFN', 'captured_stdout', 'run_unittest', 'unlink'], [TESTFN, captured_stdout, run_unittest, unlink])

else:
    # Assigning a type to the variable 'test.test_support' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'test.test_support', import_36732)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.tests import support' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_36734 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.tests')

if (type(import_36734) is not StypyTypeError):

    if (import_36734 != 'pyd_module'):
        __import__(import_36734)
        sys_modules_36735 = sys.modules[import_36734]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.tests', sys_modules_36735.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_36735, sys_modules_36735.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.tests', import_36734)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils import log' statement (line 16)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'test_dist' class
# Getting the type of 'Command' (line 19)
Command_36736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'Command')

class test_dist(Command_36736, ):
    str_36737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'str', 'Sample distutils extension command.')

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        test_dist.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        test_dist.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        test_dist.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        test_dist.initialize_options.__dict__.__setitem__('stypy_function_name', 'test_dist.initialize_options')
        test_dist.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        test_dist.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        test_dist.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        test_dist.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        test_dist.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        test_dist.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        test_dist.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'test_dist.initialize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize_options(...)' code ##################

        
        # Assigning a Name to a Attribute (line 27):
        # Getting the type of 'None' (line 27)
        None_36738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'None')
        # Getting the type of 'self' (line 27)
        self_36739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'sample_option' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_36739, 'sample_option', None_36738)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_36740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36740)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_36740


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 0, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'test_dist.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'test_dist' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'test_dist', test_dist)

# Assigning a List to a Name (line 22):

# Obtaining an instance of the builtin type 'list' (line 22)
list_36741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_36742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
str_36743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'sample-option=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_36742, str_36743)
# Adding element type (line 23)
str_36744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 27), 'str', 'S')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_36742, str_36744)
# Adding element type (line 23)
str_36745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'str', 'help text')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_36742, str_36745)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 19), list_36741, tuple_36742)

# Getting the type of 'test_dist'
test_dist_36746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'test_dist')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), test_dist_36746, 'user_options', list_36741)
# Declaration of the 'TestDistribution' class
# Getting the type of 'Distribution' (line 30)
Distribution_36747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'Distribution')

class TestDistribution(Distribution_36747, ):
    str_36748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', 'Distribution subclasses that avoids the default search for\n    configuration files.\n\n    The ._config_files attribute must be set before\n    .parse_config_files() is called.\n    ')

    @norecursion
    def find_config_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_config_files'
        module_type_store = module_type_store.open_function_context('find_config_files', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_localization', localization)
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_function_name', 'TestDistribution.find_config_files')
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_param_names_list', [])
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDistribution.find_config_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDistribution.find_config_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_config_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_config_files(...)' code ##################

        # Getting the type of 'self' (line 39)
        self_36749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'self')
        # Obtaining the member '_config_files' of a type (line 39)
        _config_files_36750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), self_36749, '_config_files')
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', _config_files_36750)
        
        # ################# End of 'find_config_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_config_files' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_36751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_config_files'
        return stypy_return_type_36751


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 0, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDistribution.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDistribution' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'TestDistribution', TestDistribution)
# Declaration of the 'DistributionTestCase' class
# Getting the type of 'support' (line 42)
support_36752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'support')
# Obtaining the member 'TempdirManager' of a type (line 42)
TempdirManager_36753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), support_36752, 'TempdirManager')
# Getting the type of 'support' (line 43)
support_36754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 43)
LoggingSilencer_36755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 27), support_36754, 'LoggingSilencer')
# Getting the type of 'support' (line 44)
support_36756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 44)
EnvironGuard_36757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 27), support_36756, 'EnvironGuard')
# Getting the type of 'unittest' (line 45)
unittest_36758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'unittest')
# Obtaining the member 'TestCase' of a type (line 45)
TestCase_36759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 27), unittest_36758, 'TestCase')

class DistributionTestCase(TempdirManager_36753, LoggingSilencer_36755, EnvironGuard_36757, TestCase_36759, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.setUp')
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_36766 = {}
        
        # Call to super(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'DistributionTestCase' (line 48)
        DistributionTestCase_36761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'DistributionTestCase', False)
        # Getting the type of 'self' (line 48)
        self_36762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 36), 'self', False)
        # Processing the call keyword arguments (line 48)
        kwargs_36763 = {}
        # Getting the type of 'super' (line 48)
        super_36760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'super', False)
        # Calling super(args, kwargs) (line 48)
        super_call_result_36764 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), super_36760, *[DistributionTestCase_36761, self_36762], **kwargs_36763)
        
        # Obtaining the member 'setUp' of a type (line 48)
        setUp_36765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), super_call_result_36764, 'setUp')
        # Calling setUp(args, kwargs) (line 48)
        setUp_call_result_36767 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), setUp_36765, *[], **kwargs_36766)
        
        
        # Assigning a Tuple to a Attribute (line 49):
        
        # Obtaining an instance of the builtin type 'tuple' (line 49)
        tuple_36768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 49)
        # Adding element type (line 49)
        # Getting the type of 'sys' (line 49)
        sys_36769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'sys')
        # Obtaining the member 'argv' of a type (line 49)
        argv_36770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 20), sys_36769, 'argv')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 20), tuple_36768, argv_36770)
        # Adding element type (line 49)
        
        # Obtaining the type of the subscript
        slice_36771 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 49, 30), None, None, None)
        # Getting the type of 'sys' (line 49)
        sys_36772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'sys')
        # Obtaining the member 'argv' of a type (line 49)
        argv_36773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 30), sys_36772, 'argv')
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___36774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 30), argv_36773, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_36775 = invoke(stypy.reporting.localization.Localization(__file__, 49, 30), getitem___36774, slice_36771)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 20), tuple_36768, subscript_call_result_36775)
        
        # Getting the type of 'self' (line 49)
        self_36776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self')
        # Setting the type of the member 'argv' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_36776, 'argv', tuple_36768)
        # Deleting a member
        # Getting the type of 'sys' (line 50)
        sys_36777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'sys')
        # Obtaining the member 'argv' of a type (line 50)
        argv_36778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), sys_36777, 'argv')
        
        # Obtaining the type of the subscript
        int_36779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 21), 'int')
        slice_36780 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 50, 12), int_36779, None, None)
        # Getting the type of 'sys' (line 50)
        sys_36781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'sys')
        # Obtaining the member 'argv' of a type (line 50)
        argv_36782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), sys_36781, 'argv')
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___36783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), argv_36782, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_36784 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), getitem___36783, slice_36780)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), argv_36778, subscript_call_result_36784)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_36785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36785)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_36785


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.tearDown')
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Subscript to a Attribute (line 53):
        
        # Obtaining the type of the subscript
        int_36786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'int')
        # Getting the type of 'self' (line 53)
        self_36787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'self')
        # Obtaining the member 'argv' of a type (line 53)
        argv_36788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 19), self_36787, 'argv')
        # Obtaining the member '__getitem__' of a type (line 53)
        getitem___36789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 19), argv_36788, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 53)
        subscript_call_result_36790 = invoke(stypy.reporting.localization.Localization(__file__, 53, 19), getitem___36789, int_36786)
        
        # Getting the type of 'sys' (line 53)
        sys_36791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), sys_36791, 'argv', subscript_call_result_36790)
        
        # Assigning a Subscript to a Subscript (line 54):
        
        # Obtaining the type of the subscript
        int_36792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 32), 'int')
        # Getting the type of 'self' (line 54)
        self_36793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'self')
        # Obtaining the member 'argv' of a type (line 54)
        argv_36794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 22), self_36793, 'argv')
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___36795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 22), argv_36794, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_36796 = invoke(stypy.reporting.localization.Localization(__file__, 54, 22), getitem___36795, int_36792)
        
        # Getting the type of 'sys' (line 54)
        sys_36797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'sys')
        # Obtaining the member 'argv' of a type (line 54)
        argv_36798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), sys_36797, 'argv')
        slice_36799 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 54, 8), None, None, None)
        # Storing an element on a container (line 54)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), argv_36798, (slice_36799, subscript_call_result_36796))
        
        # Call to tearDown(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_36806 = {}
        
        # Call to super(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'DistributionTestCase' (line 55)
        DistributionTestCase_36801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 14), 'DistributionTestCase', False)
        # Getting the type of 'self' (line 55)
        self_36802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'self', False)
        # Processing the call keyword arguments (line 55)
        kwargs_36803 = {}
        # Getting the type of 'super' (line 55)
        super_36800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'super', False)
        # Calling super(args, kwargs) (line 55)
        super_call_result_36804 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), super_36800, *[DistributionTestCase_36801, self_36802], **kwargs_36803)
        
        # Obtaining the member 'tearDown' of a type (line 55)
        tearDown_36805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), super_call_result_36804, 'tearDown')
        # Calling tearDown(args, kwargs) (line 55)
        tearDown_call_result_36807 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), tearDown_36805, *[], **kwargs_36806)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_36808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36808)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_36808


    @norecursion
    def create_distribution(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 57)
        tuple_36809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 57)
        
        defaults = [tuple_36809]
        # Create a new context for function 'create_distribution'
        module_type_store = module_type_store.open_function_context('create_distribution', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.create_distribution')
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_param_names_list', ['configfiles'])
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.create_distribution.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.create_distribution', ['configfiles'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_distribution', localization, ['configfiles'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_distribution(...)' code ##################

        
        # Assigning a Call to a Name (line 58):
        
        # Call to TestDistribution(...): (line 58)
        # Processing the call keyword arguments (line 58)
        kwargs_36811 = {}
        # Getting the type of 'TestDistribution' (line 58)
        TestDistribution_36810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'TestDistribution', False)
        # Calling TestDistribution(args, kwargs) (line 58)
        TestDistribution_call_result_36812 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), TestDistribution_36810, *[], **kwargs_36811)
        
        # Assigning a type to the variable 'd' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'd', TestDistribution_call_result_36812)
        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'configfiles' (line 59)
        configfiles_36813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'configfiles')
        # Getting the type of 'd' (line 59)
        d_36814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'd')
        # Setting the type of the member '_config_files' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), d_36814, '_config_files', configfiles_36813)
        
        # Call to parse_config_files(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_36817 = {}
        # Getting the type of 'd' (line 60)
        d_36815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'd', False)
        # Obtaining the member 'parse_config_files' of a type (line 60)
        parse_config_files_36816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), d_36815, 'parse_config_files')
        # Calling parse_config_files(args, kwargs) (line 60)
        parse_config_files_call_result_36818 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), parse_config_files_36816, *[], **kwargs_36817)
        
        
        # Call to parse_command_line(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_36821 = {}
        # Getting the type of 'd' (line 61)
        d_36819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'd', False)
        # Obtaining the member 'parse_command_line' of a type (line 61)
        parse_command_line_36820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), d_36819, 'parse_command_line')
        # Calling parse_command_line(args, kwargs) (line 61)
        parse_command_line_call_result_36822 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), parse_command_line_36820, *[], **kwargs_36821)
        
        # Getting the type of 'd' (line 62)
        d_36823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', d_36823)
        
        # ################# End of 'create_distribution(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_distribution' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_36824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36824)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_distribution'
        return stypy_return_type_36824


    @norecursion
    def test_debug_mode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_debug_mode'
        module_type_store = module_type_store.open_function_context('test_debug_mode', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_debug_mode')
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_debug_mode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_debug_mode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_debug_mode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_debug_mode(...)' code ##################

        
        # Call to open(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'TESTFN' (line 65)
        TESTFN_36826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 18), 'TESTFN', False)
        str_36827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'str', 'w')
        # Processing the call keyword arguments (line 65)
        kwargs_36828 = {}
        # Getting the type of 'open' (line 65)
        open_36825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'open', False)
        # Calling open(args, kwargs) (line 65)
        open_call_result_36829 = invoke(stypy.reporting.localization.Localization(__file__, 65, 13), open_36825, *[TESTFN_36826, str_36827], **kwargs_36828)
        
        with_36830 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 65, 13), open_call_result_36829, 'with parameter', '__enter__', '__exit__')

        if with_36830:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 65)
            enter___36831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), open_call_result_36829, '__enter__')
            with_enter_36832 = invoke(stypy.reporting.localization.Localization(__file__, 65, 13), enter___36831)
            # Assigning a type to the variable 'f' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'f', with_enter_36832)
            
            # Call to write(...): (line 66)
            # Processing the call arguments (line 66)
            str_36835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'str', '[global]\n')
            # Processing the call keyword arguments (line 66)
            kwargs_36836 = {}
            # Getting the type of 'f' (line 66)
            f_36833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'f', False)
            # Obtaining the member 'write' of a type (line 66)
            write_36834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), f_36833, 'write')
            # Calling write(args, kwargs) (line 66)
            write_call_result_36837 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), write_36834, *[str_36835], **kwargs_36836)
            
            
            # Call to write(...): (line 67)
            # Processing the call arguments (line 67)
            str_36840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'str', 'command_packages = foo.bar, splat')
            # Processing the call keyword arguments (line 67)
            kwargs_36841 = {}
            # Getting the type of 'f' (line 67)
            f_36838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'f', False)
            # Obtaining the member 'write' of a type (line 67)
            write_36839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), f_36838, 'write')
            # Calling write(args, kwargs) (line 67)
            write_call_result_36842 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), write_36839, *[str_36840], **kwargs_36841)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 65)
            exit___36843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), open_call_result_36829, '__exit__')
            with_exit_36844 = invoke(stypy.reporting.localization.Localization(__file__, 65, 13), exit___36843, None, None, None)

        
        # Call to addCleanup(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'unlink' (line 68)
        unlink_36847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'unlink', False)
        # Getting the type of 'TESTFN' (line 68)
        TESTFN_36848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), 'TESTFN', False)
        # Processing the call keyword arguments (line 68)
        kwargs_36849 = {}
        # Getting the type of 'self' (line 68)
        self_36845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 68)
        addCleanup_36846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_36845, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 68)
        addCleanup_call_result_36850 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), addCleanup_36846, *[unlink_36847, TESTFN_36848], **kwargs_36849)
        
        
        # Assigning a List to a Name (line 70):
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_36851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        # Getting the type of 'TESTFN' (line 70)
        TESTFN_36852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'TESTFN')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 16), list_36851, TESTFN_36852)
        
        # Assigning a type to the variable 'files' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'files', list_36851)
        
        # Call to append(...): (line 71)
        # Processing the call arguments (line 71)
        str_36856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 24), 'str', 'build')
        # Processing the call keyword arguments (line 71)
        kwargs_36857 = {}
        # Getting the type of 'sys' (line 71)
        sys_36853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'sys', False)
        # Obtaining the member 'argv' of a type (line 71)
        argv_36854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), sys_36853, 'argv')
        # Obtaining the member 'append' of a type (line 71)
        append_36855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), argv_36854, 'append')
        # Calling append(args, kwargs) (line 71)
        append_call_result_36858 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), append_36855, *[str_36856], **kwargs_36857)
        
        
        # Call to captured_stdout(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_36860 = {}
        # Getting the type of 'captured_stdout' (line 73)
        captured_stdout_36859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 73)
        captured_stdout_call_result_36861 = invoke(stypy.reporting.localization.Localization(__file__, 73, 13), captured_stdout_36859, *[], **kwargs_36860)
        
        with_36862 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 73, 13), captured_stdout_call_result_36861, 'with parameter', '__enter__', '__exit__')

        if with_36862:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 73)
            enter___36863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 13), captured_stdout_call_result_36861, '__enter__')
            with_enter_36864 = invoke(stypy.reporting.localization.Localization(__file__, 73, 13), enter___36863)
            # Assigning a type to the variable 'stdout' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'stdout', with_enter_36864)
            
            # Call to create_distribution(...): (line 74)
            # Processing the call arguments (line 74)
            # Getting the type of 'files' (line 74)
            files_36867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 37), 'files', False)
            # Processing the call keyword arguments (line 74)
            kwargs_36868 = {}
            # Getting the type of 'self' (line 74)
            self_36865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'self', False)
            # Obtaining the member 'create_distribution' of a type (line 74)
            create_distribution_36866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), self_36865, 'create_distribution')
            # Calling create_distribution(args, kwargs) (line 74)
            create_distribution_call_result_36869 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), create_distribution_36866, *[files_36867], **kwargs_36868)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 73)
            exit___36870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 13), captured_stdout_call_result_36861, '__exit__')
            with_exit_36871 = invoke(stypy.reporting.localization.Localization(__file__, 73, 13), exit___36870, None, None, None)

        
        # Call to seek(...): (line 75)
        # Processing the call arguments (line 75)
        int_36874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'int')
        # Processing the call keyword arguments (line 75)
        kwargs_36875 = {}
        # Getting the type of 'stdout' (line 75)
        stdout_36872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stdout', False)
        # Obtaining the member 'seek' of a type (line 75)
        seek_36873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), stdout_36872, 'seek')
        # Calling seek(args, kwargs) (line 75)
        seek_call_result_36876 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), seek_36873, *[int_36874], **kwargs_36875)
        
        
        # Call to assertEqual(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to read(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_36881 = {}
        # Getting the type of 'stdout' (line 76)
        stdout_36879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'stdout', False)
        # Obtaining the member 'read' of a type (line 76)
        read_36880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 25), stdout_36879, 'read')
        # Calling read(args, kwargs) (line 76)
        read_call_result_36882 = invoke(stypy.reporting.localization.Localization(__file__, 76, 25), read_36880, *[], **kwargs_36881)
        
        str_36883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 40), 'str', '')
        # Processing the call keyword arguments (line 76)
        kwargs_36884 = {}
        # Getting the type of 'self' (line 76)
        self_36877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 76)
        assertEqual_36878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_36877, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 76)
        assertEqual_call_result_36885 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assertEqual_36878, *[read_call_result_36882, str_36883], **kwargs_36884)
        
        
        # Assigning a Name to a Attribute (line 77):
        # Getting the type of 'True' (line 77)
        True_36886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'True')
        # Getting the type of 'distutils' (line 77)
        distutils_36887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'distutils')
        # Obtaining the member 'dist' of a type (line 77)
        dist_36888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), distutils_36887, 'dist')
        # Setting the type of the member 'DEBUG' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), dist_36888, 'DEBUG', True_36886)
        
        # Try-finally block (line 78)
        
        # Call to captured_stdout(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_36890 = {}
        # Getting the type of 'captured_stdout' (line 79)
        captured_stdout_36889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 79)
        captured_stdout_call_result_36891 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), captured_stdout_36889, *[], **kwargs_36890)
        
        with_36892 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 79, 17), captured_stdout_call_result_36891, 'with parameter', '__enter__', '__exit__')

        if with_36892:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 79)
            enter___36893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 17), captured_stdout_call_result_36891, '__enter__')
            with_enter_36894 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), enter___36893)
            # Assigning a type to the variable 'stdout' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'stdout', with_enter_36894)
            
            # Call to create_distribution(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'files' (line 80)
            files_36897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 41), 'files', False)
            # Processing the call keyword arguments (line 80)
            kwargs_36898 = {}
            # Getting the type of 'self' (line 80)
            self_36895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'self', False)
            # Obtaining the member 'create_distribution' of a type (line 80)
            create_distribution_36896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 16), self_36895, 'create_distribution')
            # Calling create_distribution(args, kwargs) (line 80)
            create_distribution_call_result_36899 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), create_distribution_36896, *[files_36897], **kwargs_36898)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 79)
            exit___36900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 17), captured_stdout_call_result_36891, '__exit__')
            with_exit_36901 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), exit___36900, None, None, None)

        
        # Call to seek(...): (line 81)
        # Processing the call arguments (line 81)
        int_36904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'int')
        # Processing the call keyword arguments (line 81)
        kwargs_36905 = {}
        # Getting the type of 'stdout' (line 81)
        stdout_36902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stdout', False)
        # Obtaining the member 'seek' of a type (line 81)
        seek_36903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), stdout_36902, 'seek')
        # Calling seek(args, kwargs) (line 81)
        seek_call_result_36906 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), seek_36903, *[int_36904], **kwargs_36905)
        
        
        # Call to assertEqual(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Call to read(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_36911 = {}
        # Getting the type of 'stdout' (line 82)
        stdout_36909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'stdout', False)
        # Obtaining the member 'read' of a type (line 82)
        read_36910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 29), stdout_36909, 'read')
        # Calling read(args, kwargs) (line 82)
        read_call_result_36912 = invoke(stypy.reporting.localization.Localization(__file__, 82, 29), read_36910, *[], **kwargs_36911)
        
        str_36913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 44), 'str', '')
        # Processing the call keyword arguments (line 82)
        kwargs_36914 = {}
        # Getting the type of 'self' (line 82)
        self_36907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 82)
        assertEqual_36908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), self_36907, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 82)
        assertEqual_call_result_36915 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), assertEqual_36908, *[read_call_result_36912, str_36913], **kwargs_36914)
        
        
        # finally branch of the try-finally block (line 78)
        
        # Assigning a Name to a Attribute (line 84):
        # Getting the type of 'False' (line 84)
        False_36916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 35), 'False')
        # Getting the type of 'distutils' (line 84)
        distutils_36917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'distutils')
        # Obtaining the member 'dist' of a type (line 84)
        dist_36918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), distutils_36917, 'dist')
        # Setting the type of the member 'DEBUG' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), dist_36918, 'DEBUG', False_36916)
        
        
        # ################# End of 'test_debug_mode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_debug_mode' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_36919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36919)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_debug_mode'
        return stypy_return_type_36919


    @norecursion
    def test_command_packages_unspecified(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_command_packages_unspecified'
        module_type_store = module_type_store.open_function_context('test_command_packages_unspecified', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_command_packages_unspecified')
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_command_packages_unspecified.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_command_packages_unspecified', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_command_packages_unspecified', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_command_packages_unspecified(...)' code ##################

        
        # Call to append(...): (line 87)
        # Processing the call arguments (line 87)
        str_36923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'str', 'build')
        # Processing the call keyword arguments (line 87)
        kwargs_36924 = {}
        # Getting the type of 'sys' (line 87)
        sys_36920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'sys', False)
        # Obtaining the member 'argv' of a type (line 87)
        argv_36921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), sys_36920, 'argv')
        # Obtaining the member 'append' of a type (line 87)
        append_36922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), argv_36921, 'append')
        # Calling append(args, kwargs) (line 87)
        append_call_result_36925 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), append_36922, *[str_36923], **kwargs_36924)
        
        
        # Assigning a Call to a Name (line 88):
        
        # Call to create_distribution(...): (line 88)
        # Processing the call keyword arguments (line 88)
        kwargs_36928 = {}
        # Getting the type of 'self' (line 88)
        self_36926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self', False)
        # Obtaining the member 'create_distribution' of a type (line 88)
        create_distribution_36927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), self_36926, 'create_distribution')
        # Calling create_distribution(args, kwargs) (line 88)
        create_distribution_call_result_36929 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), create_distribution_36927, *[], **kwargs_36928)
        
        # Assigning a type to the variable 'd' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'd', create_distribution_call_result_36929)
        
        # Call to assertEqual(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to get_command_packages(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_36934 = {}
        # Getting the type of 'd' (line 89)
        d_36932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'd', False)
        # Obtaining the member 'get_command_packages' of a type (line 89)
        get_command_packages_36933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 25), d_36932, 'get_command_packages')
        # Calling get_command_packages(args, kwargs) (line 89)
        get_command_packages_call_result_36935 = invoke(stypy.reporting.localization.Localization(__file__, 89, 25), get_command_packages_36933, *[], **kwargs_36934)
        
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_36936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        str_36937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 52), 'str', 'distutils.command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 51), list_36936, str_36937)
        
        # Processing the call keyword arguments (line 89)
        kwargs_36938 = {}
        # Getting the type of 'self' (line 89)
        self_36930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 89)
        assertEqual_36931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_36930, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 89)
        assertEqual_call_result_36939 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assertEqual_36931, *[get_command_packages_call_result_36935, list_36936], **kwargs_36938)
        
        
        # ################# End of 'test_command_packages_unspecified(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_command_packages_unspecified' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_36940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36940)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_command_packages_unspecified'
        return stypy_return_type_36940


    @norecursion
    def test_command_packages_cmdline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_command_packages_cmdline'
        module_type_store = module_type_store.open_function_context('test_command_packages_cmdline', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_command_packages_cmdline')
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_command_packages_cmdline.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_command_packages_cmdline', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_command_packages_cmdline', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_command_packages_cmdline(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 92, 8))
        
        # 'from distutils.tests.test_dist import test_dist' statement (line 92)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_36941 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 92, 8), 'distutils.tests.test_dist')

        if (type(import_36941) is not StypyTypeError):

            if (import_36941 != 'pyd_module'):
                __import__(import_36941)
                sys_modules_36942 = sys.modules[import_36941]
                import_from_module(stypy.reporting.localization.Localization(__file__, 92, 8), 'distutils.tests.test_dist', sys_modules_36942.module_type_store, module_type_store, ['test_dist'])
                nest_module(stypy.reporting.localization.Localization(__file__, 92, 8), __file__, sys_modules_36942, sys_modules_36942.module_type_store, module_type_store)
            else:
                from distutils.tests.test_dist import test_dist

                import_from_module(stypy.reporting.localization.Localization(__file__, 92, 8), 'distutils.tests.test_dist', None, module_type_store, ['test_dist'], [test_dist])

        else:
            # Assigning a type to the variable 'distutils.tests.test_dist' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'distutils.tests.test_dist', import_36941)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Call to extend(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_36946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        str_36947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'str', '--command-packages')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 24), list_36946, str_36947)
        # Adding element type (line 93)
        str_36948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'str', 'foo.bar,distutils.tests')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 24), list_36946, str_36948)
        # Adding element type (line 93)
        str_36949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'str', 'test_dist')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 24), list_36946, str_36949)
        # Adding element type (line 93)
        str_36950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'str', '-Ssometext')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 24), list_36946, str_36950)
        
        # Processing the call keyword arguments (line 93)
        kwargs_36951 = {}
        # Getting the type of 'sys' (line 93)
        sys_36943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'sys', False)
        # Obtaining the member 'argv' of a type (line 93)
        argv_36944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), sys_36943, 'argv')
        # Obtaining the member 'extend' of a type (line 93)
        extend_36945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), argv_36944, 'extend')
        # Calling extend(args, kwargs) (line 93)
        extend_call_result_36952 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), extend_36945, *[list_36946], **kwargs_36951)
        
        
        # Assigning a Call to a Name (line 98):
        
        # Call to create_distribution(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_36955 = {}
        # Getting the type of 'self' (line 98)
        self_36953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self', False)
        # Obtaining the member 'create_distribution' of a type (line 98)
        create_distribution_36954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_36953, 'create_distribution')
        # Calling create_distribution(args, kwargs) (line 98)
        create_distribution_call_result_36956 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), create_distribution_36954, *[], **kwargs_36955)
        
        # Assigning a type to the variable 'd' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'd', create_distribution_call_result_36956)
        
        # Call to assertEqual(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to get_command_packages(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_36961 = {}
        # Getting the type of 'd' (line 100)
        d_36959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'd', False)
        # Obtaining the member 'get_command_packages' of a type (line 100)
        get_command_packages_36960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 25), d_36959, 'get_command_packages')
        # Calling get_command_packages(args, kwargs) (line 100)
        get_command_packages_call_result_36962 = invoke(stypy.reporting.localization.Localization(__file__, 100, 25), get_command_packages_36960, *[], **kwargs_36961)
        
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_36963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        str_36964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'str', 'distutils.command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 25), list_36963, str_36964)
        # Adding element type (line 101)
        str_36965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 47), 'str', 'foo.bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 25), list_36963, str_36965)
        # Adding element type (line 101)
        str_36966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 58), 'str', 'distutils.tests')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 25), list_36963, str_36966)
        
        # Processing the call keyword arguments (line 100)
        kwargs_36967 = {}
        # Getting the type of 'self' (line 100)
        self_36957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 100)
        assertEqual_36958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_36957, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 100)
        assertEqual_call_result_36968 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), assertEqual_36958, *[get_command_packages_call_result_36962, list_36963], **kwargs_36967)
        
        
        # Assigning a Call to a Name (line 102):
        
        # Call to get_command_obj(...): (line 102)
        # Processing the call arguments (line 102)
        str_36971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 32), 'str', 'test_dist')
        # Processing the call keyword arguments (line 102)
        kwargs_36972 = {}
        # Getting the type of 'd' (line 102)
        d_36969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'd', False)
        # Obtaining the member 'get_command_obj' of a type (line 102)
        get_command_obj_36970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 14), d_36969, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 102)
        get_command_obj_call_result_36973 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), get_command_obj_36970, *[str_36971], **kwargs_36972)
        
        # Assigning a type to the variable 'cmd' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'cmd', get_command_obj_call_result_36973)
        
        # Call to assertIsInstance(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'cmd' (line 103)
        cmd_36976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'cmd', False)
        # Getting the type of 'test_dist' (line 103)
        test_dist_36977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 35), 'test_dist', False)
        # Processing the call keyword arguments (line 103)
        kwargs_36978 = {}
        # Getting the type of 'self' (line 103)
        self_36974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self', False)
        # Obtaining the member 'assertIsInstance' of a type (line 103)
        assertIsInstance_36975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_36974, 'assertIsInstance')
        # Calling assertIsInstance(args, kwargs) (line 103)
        assertIsInstance_call_result_36979 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assertIsInstance_36975, *[cmd_36976, test_dist_36977], **kwargs_36978)
        
        
        # Call to assertEqual(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'cmd' (line 104)
        cmd_36982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'cmd', False)
        # Obtaining the member 'sample_option' of a type (line 104)
        sample_option_36983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 25), cmd_36982, 'sample_option')
        str_36984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 44), 'str', 'sometext')
        # Processing the call keyword arguments (line 104)
        kwargs_36985 = {}
        # Getting the type of 'self' (line 104)
        self_36980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 104)
        assertEqual_36981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_36980, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 104)
        assertEqual_call_result_36986 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assertEqual_36981, *[sample_option_36983, str_36984], **kwargs_36985)
        
        
        # ################# End of 'test_command_packages_cmdline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_command_packages_cmdline' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_36987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36987)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_command_packages_cmdline'
        return stypy_return_type_36987


    @norecursion
    def test_command_packages_configfile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_command_packages_configfile'
        module_type_store = module_type_store.open_function_context('test_command_packages_configfile', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_command_packages_configfile')
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_command_packages_configfile.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_command_packages_configfile', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_command_packages_configfile', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_command_packages_configfile(...)' code ##################

        
        # Call to append(...): (line 107)
        # Processing the call arguments (line 107)
        str_36991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'str', 'build')
        # Processing the call keyword arguments (line 107)
        kwargs_36992 = {}
        # Getting the type of 'sys' (line 107)
        sys_36988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'sys', False)
        # Obtaining the member 'argv' of a type (line 107)
        argv_36989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), sys_36988, 'argv')
        # Obtaining the member 'append' of a type (line 107)
        append_36990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), argv_36989, 'append')
        # Calling append(args, kwargs) (line 107)
        append_call_result_36993 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), append_36990, *[str_36991], **kwargs_36992)
        
        
        # Call to addCleanup(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'os' (line 108)
        os_36996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'os', False)
        # Obtaining the member 'unlink' of a type (line 108)
        unlink_36997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 24), os_36996, 'unlink')
        # Getting the type of 'TESTFN' (line 108)
        TESTFN_36998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 35), 'TESTFN', False)
        # Processing the call keyword arguments (line 108)
        kwargs_36999 = {}
        # Getting the type of 'self' (line 108)
        self_36994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 108)
        addCleanup_36995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_36994, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 108)
        addCleanup_call_result_37000 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), addCleanup_36995, *[unlink_36997, TESTFN_36998], **kwargs_36999)
        
        
        # Assigning a Call to a Name (line 109):
        
        # Call to open(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'TESTFN' (line 109)
        TESTFN_37002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'TESTFN', False)
        str_37003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'str', 'w')
        # Processing the call keyword arguments (line 109)
        kwargs_37004 = {}
        # Getting the type of 'open' (line 109)
        open_37001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'open', False)
        # Calling open(args, kwargs) (line 109)
        open_call_result_37005 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), open_37001, *[TESTFN_37002, str_37003], **kwargs_37004)
        
        # Assigning a type to the variable 'f' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'f', open_call_result_37005)
        
        # Try-finally block (line 110)
        str_37006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 24), 'str', '[global]')
        str_37007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 24), 'str', 'command_packages = foo.bar, splat')
        
        # finally branch of the try-finally block (line 110)
        
        # Call to close(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_37010 = {}
        # Getting the type of 'f' (line 114)
        f_37008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 114)
        close_37009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), f_37008, 'close')
        # Calling close(args, kwargs) (line 114)
        close_call_result_37011 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), close_37009, *[], **kwargs_37010)
        
        
        
        # Assigning a Call to a Name (line 116):
        
        # Call to create_distribution(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_37014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        # Getting the type of 'TESTFN' (line 116)
        TESTFN_37015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 38), 'TESTFN', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 37), list_37014, TESTFN_37015)
        
        # Processing the call keyword arguments (line 116)
        kwargs_37016 = {}
        # Getting the type of 'self' (line 116)
        self_37012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'self', False)
        # Obtaining the member 'create_distribution' of a type (line 116)
        create_distribution_37013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), self_37012, 'create_distribution')
        # Calling create_distribution(args, kwargs) (line 116)
        create_distribution_call_result_37017 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), create_distribution_37013, *[list_37014], **kwargs_37016)
        
        # Assigning a type to the variable 'd' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'd', create_distribution_call_result_37017)
        
        # Call to assertEqual(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Call to get_command_packages(...): (line 117)
        # Processing the call keyword arguments (line 117)
        kwargs_37022 = {}
        # Getting the type of 'd' (line 117)
        d_37020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'd', False)
        # Obtaining the member 'get_command_packages' of a type (line 117)
        get_command_packages_37021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), d_37020, 'get_command_packages')
        # Calling get_command_packages(args, kwargs) (line 117)
        get_command_packages_call_result_37023 = invoke(stypy.reporting.localization.Localization(__file__, 117, 25), get_command_packages_37021, *[], **kwargs_37022)
        
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_37024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        str_37025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 26), 'str', 'distutils.command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 25), list_37024, str_37025)
        # Adding element type (line 118)
        str_37026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 47), 'str', 'foo.bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 25), list_37024, str_37026)
        # Adding element type (line 118)
        str_37027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 58), 'str', 'splat')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 25), list_37024, str_37027)
        
        # Processing the call keyword arguments (line 117)
        kwargs_37028 = {}
        # Getting the type of 'self' (line 117)
        self_37018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 117)
        assertEqual_37019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_37018, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 117)
        assertEqual_call_result_37029 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), assertEqual_37019, *[get_command_packages_call_result_37023, list_37024], **kwargs_37028)
        
        
        # Assigning a List to a Subscript (line 121):
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_37030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        str_37031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 24), 'str', '--command-packages')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 23), list_37030, str_37031)
        # Adding element type (line 121)
        str_37032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 46), 'str', 'spork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 23), list_37030, str_37032)
        # Adding element type (line 121)
        str_37033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 55), 'str', 'build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 23), list_37030, str_37033)
        
        # Getting the type of 'sys' (line 121)
        sys_37034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'sys')
        # Obtaining the member 'argv' of a type (line 121)
        argv_37035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), sys_37034, 'argv')
        int_37036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 17), 'int')
        slice_37037 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 121, 8), int_37036, None, None)
        # Storing an element on a container (line 121)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), argv_37035, (slice_37037, list_37030))
        
        # Assigning a Call to a Name (line 122):
        
        # Call to create_distribution(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_37040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        # Getting the type of 'TESTFN' (line 122)
        TESTFN_37041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'TESTFN', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 37), list_37040, TESTFN_37041)
        
        # Processing the call keyword arguments (line 122)
        kwargs_37042 = {}
        # Getting the type of 'self' (line 122)
        self_37038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'self', False)
        # Obtaining the member 'create_distribution' of a type (line 122)
        create_distribution_37039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), self_37038, 'create_distribution')
        # Calling create_distribution(args, kwargs) (line 122)
        create_distribution_call_result_37043 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), create_distribution_37039, *[list_37040], **kwargs_37042)
        
        # Assigning a type to the variable 'd' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'd', create_distribution_call_result_37043)
        
        # Call to assertEqual(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to get_command_packages(...): (line 123)
        # Processing the call keyword arguments (line 123)
        kwargs_37048 = {}
        # Getting the type of 'd' (line 123)
        d_37046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'd', False)
        # Obtaining the member 'get_command_packages' of a type (line 123)
        get_command_packages_37047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 25), d_37046, 'get_command_packages')
        # Calling get_command_packages(args, kwargs) (line 123)
        get_command_packages_call_result_37049 = invoke(stypy.reporting.localization.Localization(__file__, 123, 25), get_command_packages_37047, *[], **kwargs_37048)
        
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_37050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        # Adding element type (line 124)
        str_37051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 26), 'str', 'distutils.command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 25), list_37050, str_37051)
        # Adding element type (line 124)
        str_37052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 47), 'str', 'spork')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 25), list_37050, str_37052)
        
        # Processing the call keyword arguments (line 123)
        kwargs_37053 = {}
        # Getting the type of 'self' (line 123)
        self_37044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 123)
        assertEqual_37045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_37044, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 123)
        assertEqual_call_result_37054 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), assertEqual_37045, *[get_command_packages_call_result_37049, list_37050], **kwargs_37053)
        
        
        # Assigning a List to a Subscript (line 128):
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_37055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        # Adding element type (line 128)
        str_37056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'str', '--command-packages')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 23), list_37055, str_37056)
        # Adding element type (line 128)
        str_37057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 46), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 23), list_37055, str_37057)
        # Adding element type (line 128)
        str_37058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 50), 'str', 'build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 23), list_37055, str_37058)
        
        # Getting the type of 'sys' (line 128)
        sys_37059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'sys')
        # Obtaining the member 'argv' of a type (line 128)
        argv_37060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), sys_37059, 'argv')
        int_37061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 17), 'int')
        slice_37062 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 128, 8), int_37061, None, None)
        # Storing an element on a container (line 128)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 8), argv_37060, (slice_37062, list_37055))
        
        # Assigning a Call to a Name (line 129):
        
        # Call to create_distribution(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_37065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        # Getting the type of 'TESTFN' (line 129)
        TESTFN_37066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), 'TESTFN', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 37), list_37065, TESTFN_37066)
        
        # Processing the call keyword arguments (line 129)
        kwargs_37067 = {}
        # Getting the type of 'self' (line 129)
        self_37063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'self', False)
        # Obtaining the member 'create_distribution' of a type (line 129)
        create_distribution_37064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), self_37063, 'create_distribution')
        # Calling create_distribution(args, kwargs) (line 129)
        create_distribution_call_result_37068 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), create_distribution_37064, *[list_37065], **kwargs_37067)
        
        # Assigning a type to the variable 'd' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'd', create_distribution_call_result_37068)
        
        # Call to assertEqual(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Call to get_command_packages(...): (line 130)
        # Processing the call keyword arguments (line 130)
        kwargs_37073 = {}
        # Getting the type of 'd' (line 130)
        d_37071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'd', False)
        # Obtaining the member 'get_command_packages' of a type (line 130)
        get_command_packages_37072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 25), d_37071, 'get_command_packages')
        # Calling get_command_packages(args, kwargs) (line 130)
        get_command_packages_call_result_37074 = invoke(stypy.reporting.localization.Localization(__file__, 130, 25), get_command_packages_37072, *[], **kwargs_37073)
        
        
        # Obtaining an instance of the builtin type 'list' (line 130)
        list_37075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 130)
        # Adding element type (line 130)
        str_37076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 52), 'str', 'distutils.command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 51), list_37075, str_37076)
        
        # Processing the call keyword arguments (line 130)
        kwargs_37077 = {}
        # Getting the type of 'self' (line 130)
        self_37069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 130)
        assertEqual_37070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_37069, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 130)
        assertEqual_call_result_37078 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), assertEqual_37070, *[get_command_packages_call_result_37074, list_37075], **kwargs_37077)
        
        
        # ################# End of 'test_command_packages_configfile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_command_packages_configfile' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_37079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37079)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_command_packages_configfile'
        return stypy_return_type_37079


    @norecursion
    def test_write_pkg_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_write_pkg_file'
        module_type_store = module_type_store.open_function_context('test_write_pkg_file', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_write_pkg_file')
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_write_pkg_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_write_pkg_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_write_pkg_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_write_pkg_file(...)' code ##################

        
        # Assigning a Call to a Name (line 134):
        
        # Call to mkdtemp(...): (line 134)
        # Processing the call keyword arguments (line 134)
        kwargs_37082 = {}
        # Getting the type of 'self' (line 134)
        self_37080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 134)
        mkdtemp_37081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 18), self_37080, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 134)
        mkdtemp_call_result_37083 = invoke(stypy.reporting.localization.Localization(__file__, 134, 18), mkdtemp_37081, *[], **kwargs_37082)
        
        # Assigning a type to the variable 'tmp_dir' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tmp_dir', mkdtemp_call_result_37083)
        
        # Assigning a Call to a Name (line 135):
        
        # Call to join(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'tmp_dir' (line 135)
        tmp_dir_37087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'tmp_dir', False)
        str_37088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 40), 'str', 'f')
        # Processing the call keyword arguments (line 135)
        kwargs_37089 = {}
        # Getting the type of 'os' (line 135)
        os_37084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 135)
        path_37085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 18), os_37084, 'path')
        # Obtaining the member 'join' of a type (line 135)
        join_37086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 18), path_37085, 'join')
        # Calling join(args, kwargs) (line 135)
        join_call_result_37090 = invoke(stypy.reporting.localization.Localization(__file__, 135, 18), join_37086, *[tmp_dir_37087, str_37088], **kwargs_37089)
        
        # Assigning a type to the variable 'my_file' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'my_file', join_call_result_37090)
        
        # Assigning a Name to a Name (line 136):
        # Getting the type of 'Distribution' (line 136)
        Distribution_37091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'Distribution')
        # Assigning a type to the variable 'klass' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'klass', Distribution_37091)
        
        # Assigning a Call to a Name (line 138):
        
        # Call to klass(...): (line 138)
        # Processing the call keyword arguments (line 138)
        
        # Obtaining an instance of the builtin type 'dict' (line 138)
        dict_37093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 138)
        # Adding element type (key, value) (line 138)
        str_37094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 28), 'str', 'author')
        unicode_37095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 38), 'unicode', u'Mister Caf\xe9')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 27), dict_37093, (str_37094, unicode_37095))
        # Adding element type (key, value) (line 138)
        str_37096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 28), 'str', 'name')
        str_37097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 36), 'str', 'my.package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 27), dict_37093, (str_37096, str_37097))
        # Adding element type (key, value) (line 138)
        str_37098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 28), 'str', 'maintainer')
        unicode_37099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 42), 'unicode', u'Caf\xe9 Junior')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 27), dict_37093, (str_37098, unicode_37099))
        # Adding element type (key, value) (line 138)
        str_37100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 28), 'str', 'description')
        unicode_37101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 43), 'unicode', u'Caf\xe9 torr\xe9fi\xe9')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 27), dict_37093, (str_37100, unicode_37101))
        # Adding element type (key, value) (line 138)
        str_37102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 28), 'str', 'long_description')
        unicode_37103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 48), 'unicode', u'H\xe9h\xe9h\xe9')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 27), dict_37093, (str_37102, unicode_37103))
        
        keyword_37104 = dict_37093
        kwargs_37105 = {'attrs': keyword_37104}
        # Getting the type of 'klass' (line 138)
        klass_37092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'klass', False)
        # Calling klass(args, kwargs) (line 138)
        klass_call_result_37106 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), klass_37092, *[], **kwargs_37105)
        
        # Assigning a type to the variable 'dist' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'dist', klass_call_result_37106)
        
        # Call to write_pkg_file(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Call to open(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'my_file' (line 147)
        my_file_37111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 42), 'my_file', False)
        str_37112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 51), 'str', 'w')
        # Processing the call keyword arguments (line 147)
        kwargs_37113 = {}
        # Getting the type of 'open' (line 147)
        open_37110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 37), 'open', False)
        # Calling open(args, kwargs) (line 147)
        open_call_result_37114 = invoke(stypy.reporting.localization.Localization(__file__, 147, 37), open_37110, *[my_file_37111, str_37112], **kwargs_37113)
        
        # Processing the call keyword arguments (line 147)
        kwargs_37115 = {}
        # Getting the type of 'dist' (line 147)
        dist_37107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'dist', False)
        # Obtaining the member 'metadata' of a type (line 147)
        metadata_37108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), dist_37107, 'metadata')
        # Obtaining the member 'write_pkg_file' of a type (line 147)
        write_pkg_file_37109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), metadata_37108, 'write_pkg_file')
        # Calling write_pkg_file(args, kwargs) (line 147)
        write_pkg_file_call_result_37116 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), write_pkg_file_37109, *[open_call_result_37114], **kwargs_37115)
        
        
        # Assigning a Call to a Name (line 150):
        
        # Call to klass(...): (line 150)
        # Processing the call keyword arguments (line 150)
        
        # Obtaining an instance of the builtin type 'dict' (line 150)
        dict_37118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 150)
        # Adding element type (key, value) (line 150)
        str_37119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 28), 'str', 'author')
        str_37120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 38), 'str', 'Mister Cafe')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 27), dict_37118, (str_37119, str_37120))
        # Adding element type (key, value) (line 150)
        str_37121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 28), 'str', 'name')
        str_37122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 36), 'str', 'my.package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 27), dict_37118, (str_37121, str_37122))
        # Adding element type (key, value) (line 150)
        str_37123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'str', 'maintainer')
        str_37124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 42), 'str', 'Cafe Junior')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 27), dict_37118, (str_37123, str_37124))
        # Adding element type (key, value) (line 150)
        str_37125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 28), 'str', 'description')
        str_37126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 43), 'str', 'Cafe torrefie')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 27), dict_37118, (str_37125, str_37126))
        # Adding element type (key, value) (line 150)
        str_37127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 28), 'str', 'long_description')
        str_37128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 48), 'str', 'Hehehe')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 27), dict_37118, (str_37127, str_37128))
        
        keyword_37129 = dict_37118
        kwargs_37130 = {'attrs': keyword_37129}
        # Getting the type of 'klass' (line 150)
        klass_37117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'klass', False)
        # Calling klass(args, kwargs) (line 150)
        klass_call_result_37131 = invoke(stypy.reporting.localization.Localization(__file__, 150, 15), klass_37117, *[], **kwargs_37130)
        
        # Assigning a type to the variable 'dist' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'dist', klass_call_result_37131)
        
        # Assigning a Call to a Name (line 156):
        
        # Call to join(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'tmp_dir' (line 156)
        tmp_dir_37135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'tmp_dir', False)
        str_37136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 41), 'str', 'f2')
        # Processing the call keyword arguments (line 156)
        kwargs_37137 = {}
        # Getting the type of 'os' (line 156)
        os_37132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 156)
        path_37133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 19), os_37132, 'path')
        # Obtaining the member 'join' of a type (line 156)
        join_37134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 19), path_37133, 'join')
        # Calling join(args, kwargs) (line 156)
        join_call_result_37138 = invoke(stypy.reporting.localization.Localization(__file__, 156, 19), join_37134, *[tmp_dir_37135, str_37136], **kwargs_37137)
        
        # Assigning a type to the variable 'my_file2' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'my_file2', join_call_result_37138)
        
        # Call to write_pkg_file(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Call to open(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'my_file2' (line 157)
        my_file2_37143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 42), 'my_file2', False)
        str_37144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 52), 'str', 'w')
        # Processing the call keyword arguments (line 157)
        kwargs_37145 = {}
        # Getting the type of 'open' (line 157)
        open_37142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'open', False)
        # Calling open(args, kwargs) (line 157)
        open_call_result_37146 = invoke(stypy.reporting.localization.Localization(__file__, 157, 37), open_37142, *[my_file2_37143, str_37144], **kwargs_37145)
        
        # Processing the call keyword arguments (line 157)
        kwargs_37147 = {}
        # Getting the type of 'dist' (line 157)
        dist_37139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'dist', False)
        # Obtaining the member 'metadata' of a type (line 157)
        metadata_37140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), dist_37139, 'metadata')
        # Obtaining the member 'write_pkg_file' of a type (line 157)
        write_pkg_file_37141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), metadata_37140, 'write_pkg_file')
        # Calling write_pkg_file(args, kwargs) (line 157)
        write_pkg_file_call_result_37148 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), write_pkg_file_37141, *[open_call_result_37146], **kwargs_37147)
        
        
        # ################# End of 'test_write_pkg_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_write_pkg_file' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_37149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_write_pkg_file'
        return stypy_return_type_37149


    @norecursion
    def test_empty_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_empty_options'
        module_type_store = module_type_store.open_function_context('test_empty_options', 159, 4, False)
        # Assigning a type to the variable 'self' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_empty_options')
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_empty_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_empty_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_empty_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_empty_options(...)' code ##################

        
        # Assigning a List to a Name (line 164):
        
        # Obtaining an instance of the builtin type 'list' (line 164)
        list_37150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 164)
        
        # Assigning a type to the variable 'warns' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'warns', list_37150)

        @norecursion
        def _warn(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_warn'
            module_type_store = module_type_store.open_function_context('_warn', 166, 8, False)
            
            # Passed parameters checking function
            _warn.stypy_localization = localization
            _warn.stypy_type_of_self = None
            _warn.stypy_type_store = module_type_store
            _warn.stypy_function_name = '_warn'
            _warn.stypy_param_names_list = ['msg']
            _warn.stypy_varargs_param_name = None
            _warn.stypy_kwargs_param_name = None
            _warn.stypy_call_defaults = defaults
            _warn.stypy_call_varargs = varargs
            _warn.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_warn', ['msg'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_warn', localization, ['msg'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_warn(...)' code ##################

            
            # Call to append(...): (line 167)
            # Processing the call arguments (line 167)
            # Getting the type of 'msg' (line 167)
            msg_37153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'msg', False)
            # Processing the call keyword arguments (line 167)
            kwargs_37154 = {}
            # Getting the type of 'warns' (line 167)
            warns_37151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'warns', False)
            # Obtaining the member 'append' of a type (line 167)
            append_37152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), warns_37151, 'append')
            # Calling append(args, kwargs) (line 167)
            append_call_result_37155 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), append_37152, *[msg_37153], **kwargs_37154)
            
            
            # ################# End of '_warn(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_warn' in the type store
            # Getting the type of 'stypy_return_type' (line 166)
            stypy_return_type_37156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_37156)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_warn'
            return stypy_return_type_37156

        # Assigning a type to the variable '_warn' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), '_warn', _warn)
        
        # Call to addCleanup(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'setattr' (line 169)
        setattr_37159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'setattr', False)
        # Getting the type of 'warnings' (line 169)
        warnings_37160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), 'warnings', False)
        str_37161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 43), 'str', 'warn')
        # Getting the type of 'warnings' (line 169)
        warnings_37162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 51), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 169)
        warn_37163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 51), warnings_37162, 'warn')
        # Processing the call keyword arguments (line 169)
        kwargs_37164 = {}
        # Getting the type of 'self' (line 169)
        self_37157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 169)
        addCleanup_37158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_37157, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 169)
        addCleanup_call_result_37165 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), addCleanup_37158, *[setattr_37159, warnings_37160, str_37161, warn_37163], **kwargs_37164)
        
        
        # Assigning a Name to a Attribute (line 170):
        # Getting the type of '_warn' (line 170)
        _warn_37166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), '_warn')
        # Getting the type of 'warnings' (line 170)
        warnings_37167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'warnings')
        # Setting the type of the member 'warn' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), warnings_37167, 'warn', _warn_37166)
        
        # Assigning a Call to a Name (line 171):
        
        # Call to Distribution(...): (line 171)
        # Processing the call keyword arguments (line 171)
        
        # Obtaining an instance of the builtin type 'dict' (line 171)
        dict_37169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 34), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 171)
        # Adding element type (key, value) (line 171)
        str_37170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 35), 'str', 'author')
        str_37171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 45), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 34), dict_37169, (str_37170, str_37171))
        # Adding element type (key, value) (line 171)
        str_37172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 52), 'str', 'name')
        str_37173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 60), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 34), dict_37169, (str_37172, str_37173))
        # Adding element type (key, value) (line 171)
        str_37174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 35), 'str', 'version')
        str_37175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 46), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 34), dict_37169, (str_37174, str_37175))
        # Adding element type (key, value) (line 171)
        str_37176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 53), 'str', 'url')
        str_37177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 60), 'str', 'xxxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 34), dict_37169, (str_37176, str_37177))
        # Adding element type (key, value) (line 171)
        str_37178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 35), 'str', 'options')
        
        # Obtaining an instance of the builtin type 'dict' (line 173)
        dict_37179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 46), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 173)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 34), dict_37169, (str_37178, dict_37179))
        
        keyword_37180 = dict_37169
        kwargs_37181 = {'attrs': keyword_37180}
        # Getting the type of 'Distribution' (line 171)
        Distribution_37168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 171)
        Distribution_call_result_37182 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), Distribution_37168, *[], **kwargs_37181)
        
        # Assigning a type to the variable 'dist' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'dist', Distribution_call_result_37182)
        
        # Call to assertEqual(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Call to len(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'warns' (line 175)
        warns_37186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 29), 'warns', False)
        # Processing the call keyword arguments (line 175)
        kwargs_37187 = {}
        # Getting the type of 'len' (line 175)
        len_37185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'len', False)
        # Calling len(args, kwargs) (line 175)
        len_call_result_37188 = invoke(stypy.reporting.localization.Localization(__file__, 175, 25), len_37185, *[warns_37186], **kwargs_37187)
        
        int_37189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 37), 'int')
        # Processing the call keyword arguments (line 175)
        kwargs_37190 = {}
        # Getting the type of 'self' (line 175)
        self_37183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 175)
        assertEqual_37184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_37183, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 175)
        assertEqual_call_result_37191 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), assertEqual_37184, *[len_call_result_37188, int_37189], **kwargs_37190)
        
        
        # Call to assertNotIn(...): (line 176)
        # Processing the call arguments (line 176)
        str_37194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 25), 'str', 'options')
        
        # Call to dir(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'dist' (line 176)
        dist_37196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 40), 'dist', False)
        # Processing the call keyword arguments (line 176)
        kwargs_37197 = {}
        # Getting the type of 'dir' (line 176)
        dir_37195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'dir', False)
        # Calling dir(args, kwargs) (line 176)
        dir_call_result_37198 = invoke(stypy.reporting.localization.Localization(__file__, 176, 36), dir_37195, *[dist_37196], **kwargs_37197)
        
        # Processing the call keyword arguments (line 176)
        kwargs_37199 = {}
        # Getting the type of 'self' (line 176)
        self_37192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 176)
        assertNotIn_37193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_37192, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 176)
        assertNotIn_call_result_37200 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assertNotIn_37193, *[str_37194, dir_call_result_37198], **kwargs_37199)
        
        
        # ################# End of 'test_empty_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_empty_options' in the type store
        # Getting the type of 'stypy_return_type' (line 159)
        stypy_return_type_37201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37201)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_empty_options'
        return stypy_return_type_37201


    @norecursion
    def test_finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_finalize_options'
        module_type_store = module_type_store.open_function_context('test_finalize_options', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_finalize_options')
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Dict to a Name (line 179):
        
        # Obtaining an instance of the builtin type 'dict' (line 179)
        dict_37202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 179)
        # Adding element type (key, value) (line 179)
        str_37203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 17), 'str', 'keywords')
        str_37204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'str', 'one,two')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 16), dict_37202, (str_37203, str_37204))
        # Adding element type (key, value) (line 179)
        str_37205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 17), 'str', 'platforms')
        str_37206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 30), 'str', 'one,two')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 16), dict_37202, (str_37205, str_37206))
        
        # Assigning a type to the variable 'attrs' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'attrs', dict_37202)
        
        # Assigning a Call to a Name (line 182):
        
        # Call to Distribution(...): (line 182)
        # Processing the call keyword arguments (line 182)
        # Getting the type of 'attrs' (line 182)
        attrs_37208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 34), 'attrs', False)
        keyword_37209 = attrs_37208
        kwargs_37210 = {'attrs': keyword_37209}
        # Getting the type of 'Distribution' (line 182)
        Distribution_37207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 182)
        Distribution_call_result_37211 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), Distribution_37207, *[], **kwargs_37210)
        
        # Assigning a type to the variable 'dist' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'dist', Distribution_call_result_37211)
        
        # Call to finalize_options(...): (line 183)
        # Processing the call keyword arguments (line 183)
        kwargs_37214 = {}
        # Getting the type of 'dist' (line 183)
        dist_37212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'dist', False)
        # Obtaining the member 'finalize_options' of a type (line 183)
        finalize_options_37213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), dist_37212, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 183)
        finalize_options_call_result_37215 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), finalize_options_37213, *[], **kwargs_37214)
        
        
        # Call to assertEqual(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'dist' (line 186)
        dist_37218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'dist', False)
        # Obtaining the member 'metadata' of a type (line 186)
        metadata_37219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 25), dist_37218, 'metadata')
        # Obtaining the member 'platforms' of a type (line 186)
        platforms_37220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 25), metadata_37219, 'platforms')
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_37221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        # Adding element type (line 186)
        str_37222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 51), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 50), list_37221, str_37222)
        # Adding element type (line 186)
        str_37223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 58), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 50), list_37221, str_37223)
        
        # Processing the call keyword arguments (line 186)
        kwargs_37224 = {}
        # Getting the type of 'self' (line 186)
        self_37216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 186)
        assertEqual_37217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_37216, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 186)
        assertEqual_call_result_37225 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), assertEqual_37217, *[platforms_37220, list_37221], **kwargs_37224)
        
        
        # Call to assertEqual(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'dist' (line 187)
        dist_37228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'dist', False)
        # Obtaining the member 'metadata' of a type (line 187)
        metadata_37229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 25), dist_37228, 'metadata')
        # Obtaining the member 'keywords' of a type (line 187)
        keywords_37230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 25), metadata_37229, 'keywords')
        
        # Obtaining an instance of the builtin type 'list' (line 187)
        list_37231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 187)
        # Adding element type (line 187)
        str_37232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 50), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 49), list_37231, str_37232)
        # Adding element type (line 187)
        str_37233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 57), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 49), list_37231, str_37233)
        
        # Processing the call keyword arguments (line 187)
        kwargs_37234 = {}
        # Getting the type of 'self' (line 187)
        self_37226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 187)
        assertEqual_37227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_37226, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 187)
        assertEqual_call_result_37235 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), assertEqual_37227, *[keywords_37230, list_37231], **kwargs_37234)
        
        
        # ################# End of 'test_finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_37236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_finalize_options'
        return stypy_return_type_37236


    @norecursion
    def test_get_command_packages(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_get_command_packages'
        module_type_store = module_type_store.open_function_context('test_get_command_packages', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_get_command_packages')
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_get_command_packages.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_get_command_packages', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_get_command_packages', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_get_command_packages(...)' code ##################

        
        # Assigning a Call to a Name (line 190):
        
        # Call to Distribution(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_37238 = {}
        # Getting the type of 'Distribution' (line 190)
        Distribution_37237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 190)
        Distribution_call_result_37239 = invoke(stypy.reporting.localization.Localization(__file__, 190, 15), Distribution_37237, *[], **kwargs_37238)
        
        # Assigning a type to the variable 'dist' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'dist', Distribution_call_result_37239)
        
        # Call to assertEqual(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'dist' (line 191)
        dist_37242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'dist', False)
        # Obtaining the member 'command_packages' of a type (line 191)
        command_packages_37243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 25), dist_37242, 'command_packages')
        # Getting the type of 'None' (line 191)
        None_37244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 48), 'None', False)
        # Processing the call keyword arguments (line 191)
        kwargs_37245 = {}
        # Getting the type of 'self' (line 191)
        self_37240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 191)
        assertEqual_37241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_37240, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 191)
        assertEqual_call_result_37246 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), assertEqual_37241, *[command_packages_37243, None_37244], **kwargs_37245)
        
        
        # Assigning a Call to a Name (line 192):
        
        # Call to get_command_packages(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_37249 = {}
        # Getting the type of 'dist' (line 192)
        dist_37247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'dist', False)
        # Obtaining the member 'get_command_packages' of a type (line 192)
        get_command_packages_37248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 15), dist_37247, 'get_command_packages')
        # Calling get_command_packages(args, kwargs) (line 192)
        get_command_packages_call_result_37250 = invoke(stypy.reporting.localization.Localization(__file__, 192, 15), get_command_packages_37248, *[], **kwargs_37249)
        
        # Assigning a type to the variable 'cmds' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'cmds', get_command_packages_call_result_37250)
        
        # Call to assertEqual(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'cmds' (line 193)
        cmds_37253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 25), 'cmds', False)
        
        # Obtaining an instance of the builtin type 'list' (line 193)
        list_37254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 193)
        # Adding element type (line 193)
        str_37255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 32), 'str', 'distutils.command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 31), list_37254, str_37255)
        
        # Processing the call keyword arguments (line 193)
        kwargs_37256 = {}
        # Getting the type of 'self' (line 193)
        self_37251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 193)
        assertEqual_37252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_37251, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 193)
        assertEqual_call_result_37257 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), assertEqual_37252, *[cmds_37253, list_37254], **kwargs_37256)
        
        
        # Call to assertEqual(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'dist' (line 194)
        dist_37260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 25), 'dist', False)
        # Obtaining the member 'command_packages' of a type (line 194)
        command_packages_37261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 25), dist_37260, 'command_packages')
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_37262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        str_37263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 26), 'str', 'distutils.command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 25), list_37262, str_37263)
        
        # Processing the call keyword arguments (line 194)
        kwargs_37264 = {}
        # Getting the type of 'self' (line 194)
        self_37258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 194)
        assertEqual_37259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_37258, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 194)
        assertEqual_call_result_37265 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), assertEqual_37259, *[command_packages_37261, list_37262], **kwargs_37264)
        
        
        # Assigning a Str to a Attribute (line 197):
        str_37266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 32), 'str', 'one,two')
        # Getting the type of 'dist' (line 197)
        dist_37267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'dist')
        # Setting the type of the member 'command_packages' of a type (line 197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), dist_37267, 'command_packages', str_37266)
        
        # Assigning a Call to a Name (line 198):
        
        # Call to get_command_packages(...): (line 198)
        # Processing the call keyword arguments (line 198)
        kwargs_37270 = {}
        # Getting the type of 'dist' (line 198)
        dist_37268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'dist', False)
        # Obtaining the member 'get_command_packages' of a type (line 198)
        get_command_packages_37269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 15), dist_37268, 'get_command_packages')
        # Calling get_command_packages(args, kwargs) (line 198)
        get_command_packages_call_result_37271 = invoke(stypy.reporting.localization.Localization(__file__, 198, 15), get_command_packages_37269, *[], **kwargs_37270)
        
        # Assigning a type to the variable 'cmds' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'cmds', get_command_packages_call_result_37271)
        
        # Call to assertEqual(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'cmds' (line 199)
        cmds_37274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 25), 'cmds', False)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_37275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        str_37276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 32), 'str', 'distutils.command')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 31), list_37275, str_37276)
        # Adding element type (line 199)
        str_37277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 53), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 31), list_37275, str_37277)
        # Adding element type (line 199)
        str_37278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 60), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 31), list_37275, str_37278)
        
        # Processing the call keyword arguments (line 199)
        kwargs_37279 = {}
        # Getting the type of 'self' (line 199)
        self_37272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 199)
        assertEqual_37273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_37272, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 199)
        assertEqual_call_result_37280 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), assertEqual_37273, *[cmds_37274, list_37275], **kwargs_37279)
        
        
        # ################# End of 'test_get_command_packages(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_get_command_packages' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_37281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37281)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_get_command_packages'
        return stypy_return_type_37281


    @norecursion
    def test_announce(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_announce'
        module_type_store = module_type_store.open_function_context('test_announce', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_announce')
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_announce.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_announce', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_announce', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_announce(...)' code ##################

        
        # Assigning a Call to a Name (line 203):
        
        # Call to Distribution(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_37283 = {}
        # Getting the type of 'Distribution' (line 203)
        Distribution_37282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 203)
        Distribution_call_result_37284 = invoke(stypy.reporting.localization.Localization(__file__, 203, 15), Distribution_37282, *[], **kwargs_37283)
        
        # Assigning a type to the variable 'dist' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'dist', Distribution_call_result_37284)
        
        # Assigning a Tuple to a Name (line 204):
        
        # Obtaining an instance of the builtin type 'tuple' (line 204)
        tuple_37285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 204)
        # Adding element type (line 204)
        str_37286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 16), 'str', 'ok')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 16), tuple_37285, str_37286)
        
        # Assigning a type to the variable 'args' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'args', tuple_37285)
        
        # Assigning a Dict to a Name (line 205):
        
        # Obtaining an instance of the builtin type 'dict' (line 205)
        dict_37287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 205)
        # Adding element type (key, value) (line 205)
        str_37288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 18), 'str', 'level')
        str_37289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 27), 'str', 'ok2')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 17), dict_37287, (str_37288, str_37289))
        
        # Assigning a type to the variable 'kwargs' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'kwargs', dict_37287)
        
        # Call to assertRaises(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'ValueError' (line 206)
        ValueError_37292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'ValueError', False)
        # Getting the type of 'dist' (line 206)
        dist_37293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 38), 'dist', False)
        # Obtaining the member 'announce' of a type (line 206)
        announce_37294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 38), dist_37293, 'announce')
        # Getting the type of 'args' (line 206)
        args_37295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 53), 'args', False)
        # Getting the type of 'kwargs' (line 206)
        kwargs_37296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 59), 'kwargs', False)
        # Processing the call keyword arguments (line 206)
        kwargs_37297 = {}
        # Getting the type of 'self' (line 206)
        self_37290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 206)
        assertRaises_37291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_37290, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 206)
        assertRaises_call_result_37298 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), assertRaises_37291, *[ValueError_37292, announce_37294, args_37295, kwargs_37296], **kwargs_37297)
        
        
        # ################# End of 'test_announce(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_announce' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_37299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37299)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_announce'
        return stypy_return_type_37299


    @norecursion
    def test_find_config_files_disable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_find_config_files_disable'
        module_type_store = module_type_store.open_function_context('test_find_config_files_disable', 208, 4, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_localization', localization)
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_type_store', module_type_store)
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_function_name', 'DistributionTestCase.test_find_config_files_disable')
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_param_names_list', [])
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_varargs_param_name', None)
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_call_defaults', defaults)
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_call_varargs', varargs)
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DistributionTestCase.test_find_config_files_disable.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.test_find_config_files_disable', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_find_config_files_disable', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_find_config_files_disable(...)' code ##################

        
        # Assigning a Call to a Name (line 210):
        
        # Call to mkdtemp(...): (line 210)
        # Processing the call keyword arguments (line 210)
        kwargs_37302 = {}
        # Getting the type of 'self' (line 210)
        self_37300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 210)
        mkdtemp_37301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 20), self_37300, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 210)
        mkdtemp_call_result_37303 = invoke(stypy.reporting.localization.Localization(__file__, 210, 20), mkdtemp_37301, *[], **kwargs_37302)
        
        # Assigning a type to the variable 'temp_home' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'temp_home', mkdtemp_call_result_37303)
        
        
        # Getting the type of 'os' (line 211)
        os_37304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'os')
        # Obtaining the member 'name' of a type (line 211)
        name_37305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), os_37304, 'name')
        str_37306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 22), 'str', 'posix')
        # Applying the binary operator '==' (line 211)
        result_eq_37307 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 11), '==', name_37305, str_37306)
        
        # Testing the type of an if condition (line 211)
        if_condition_37308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), result_eq_37307)
        # Assigning a type to the variable 'if_condition_37308' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_37308', if_condition_37308)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 212):
        
        # Call to join(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'temp_home' (line 212)
        temp_home_37312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 41), 'temp_home', False)
        str_37313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 52), 'str', '.pydistutils.cfg')
        # Processing the call keyword arguments (line 212)
        kwargs_37314 = {}
        # Getting the type of 'os' (line 212)
        os_37309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 212)
        path_37310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 28), os_37309, 'path')
        # Obtaining the member 'join' of a type (line 212)
        join_37311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 28), path_37310, 'join')
        # Calling join(args, kwargs) (line 212)
        join_call_result_37315 = invoke(stypy.reporting.localization.Localization(__file__, 212, 28), join_37311, *[temp_home_37312, str_37313], **kwargs_37314)
        
        # Assigning a type to the variable 'user_filename' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'user_filename', join_call_result_37315)
        # SSA branch for the else part of an if statement (line 211)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 214):
        
        # Call to join(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'temp_home' (line 214)
        temp_home_37319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 41), 'temp_home', False)
        str_37320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 52), 'str', 'pydistutils.cfg')
        # Processing the call keyword arguments (line 214)
        kwargs_37321 = {}
        # Getting the type of 'os' (line 214)
        os_37316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 214)
        path_37317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 28), os_37316, 'path')
        # Obtaining the member 'join' of a type (line 214)
        join_37318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 28), path_37317, 'join')
        # Calling join(args, kwargs) (line 214)
        join_call_result_37322 = invoke(stypy.reporting.localization.Localization(__file__, 214, 28), join_37318, *[temp_home_37319, str_37320], **kwargs_37321)
        
        # Assigning a type to the variable 'user_filename' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'user_filename', join_call_result_37322)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to open(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'user_filename' (line 216)
        user_filename_37324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'user_filename', False)
        str_37325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 33), 'str', 'w')
        # Processing the call keyword arguments (line 216)
        kwargs_37326 = {}
        # Getting the type of 'open' (line 216)
        open_37323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'open', False)
        # Calling open(args, kwargs) (line 216)
        open_call_result_37327 = invoke(stypy.reporting.localization.Localization(__file__, 216, 13), open_37323, *[user_filename_37324, str_37325], **kwargs_37326)
        
        with_37328 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 216, 13), open_call_result_37327, 'with parameter', '__enter__', '__exit__')

        if with_37328:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 216)
            enter___37329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 13), open_call_result_37327, '__enter__')
            with_enter_37330 = invoke(stypy.reporting.localization.Localization(__file__, 216, 13), enter___37329)
            # Assigning a type to the variable 'f' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'f', with_enter_37330)
            
            # Call to write(...): (line 217)
            # Processing the call arguments (line 217)
            str_37333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 20), 'str', '[distutils]\n')
            # Processing the call keyword arguments (line 217)
            kwargs_37334 = {}
            # Getting the type of 'f' (line 217)
            f_37331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'f', False)
            # Obtaining the member 'write' of a type (line 217)
            write_37332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), f_37331, 'write')
            # Calling write(args, kwargs) (line 217)
            write_call_result_37335 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), write_37332, *[str_37333], **kwargs_37334)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 216)
            exit___37336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 13), open_call_result_37327, '__exit__')
            with_exit_37337 = invoke(stypy.reporting.localization.Localization(__file__, 216, 13), exit___37336, None, None, None)


        @norecursion
        def _expander(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_expander'
            module_type_store = module_type_store.open_function_context('_expander', 219, 8, False)
            
            # Passed parameters checking function
            _expander.stypy_localization = localization
            _expander.stypy_type_of_self = None
            _expander.stypy_type_store = module_type_store
            _expander.stypy_function_name = '_expander'
            _expander.stypy_param_names_list = ['path']
            _expander.stypy_varargs_param_name = None
            _expander.stypy_kwargs_param_name = None
            _expander.stypy_call_defaults = defaults
            _expander.stypy_call_varargs = varargs
            _expander.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_expander', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_expander', localization, ['path'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_expander(...)' code ##################

            # Getting the type of 'temp_home' (line 220)
            temp_home_37338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'temp_home')
            # Assigning a type to the variable 'stypy_return_type' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'stypy_return_type', temp_home_37338)
            
            # ################# End of '_expander(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_expander' in the type store
            # Getting the type of 'stypy_return_type' (line 219)
            stypy_return_type_37339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_37339)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_expander'
            return stypy_return_type_37339

        # Assigning a type to the variable '_expander' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), '_expander', _expander)
        
        # Assigning a Attribute to a Name (line 222):
        # Getting the type of 'os' (line 222)
        os_37340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'os')
        # Obtaining the member 'path' of a type (line 222)
        path_37341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 23), os_37340, 'path')
        # Obtaining the member 'expanduser' of a type (line 222)
        expanduser_37342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 23), path_37341, 'expanduser')
        # Assigning a type to the variable 'old_expander' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'old_expander', expanduser_37342)
        
        # Assigning a Name to a Attribute (line 223):
        # Getting the type of '_expander' (line 223)
        _expander_37343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), '_expander')
        # Getting the type of 'os' (line 223)
        os_37344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'os')
        # Obtaining the member 'path' of a type (line 223)
        path_37345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), os_37344, 'path')
        # Setting the type of the member 'expanduser' of a type (line 223)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), path_37345, 'expanduser', _expander_37343)
        
        # Try-finally block (line 224)
        
        # Assigning a Call to a Name (line 225):
        
        # Call to Distribution(...): (line 225)
        # Processing the call keyword arguments (line 225)
        kwargs_37349 = {}
        # Getting the type of 'distutils' (line 225)
        distutils_37346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'distutils', False)
        # Obtaining the member 'dist' of a type (line 225)
        dist_37347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), distutils_37346, 'dist')
        # Obtaining the member 'Distribution' of a type (line 225)
        Distribution_37348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), dist_37347, 'Distribution')
        # Calling Distribution(args, kwargs) (line 225)
        Distribution_call_result_37350 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), Distribution_37348, *[], **kwargs_37349)
        
        # Assigning a type to the variable 'd' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'd', Distribution_call_result_37350)
        
        # Assigning a Call to a Name (line 226):
        
        # Call to find_config_files(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_37353 = {}
        # Getting the type of 'd' (line 226)
        d_37351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'd', False)
        # Obtaining the member 'find_config_files' of a type (line 226)
        find_config_files_37352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 24), d_37351, 'find_config_files')
        # Calling find_config_files(args, kwargs) (line 226)
        find_config_files_call_result_37354 = invoke(stypy.reporting.localization.Localization(__file__, 226, 24), find_config_files_37352, *[], **kwargs_37353)
        
        # Assigning a type to the variable 'all_files' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'all_files', find_config_files_call_result_37354)
        
        # Assigning a Call to a Name (line 228):
        
        # Call to Distribution(...): (line 228)
        # Processing the call keyword arguments (line 228)
        
        # Obtaining an instance of the builtin type 'dict' (line 228)
        dict_37358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 50), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 228)
        # Adding element type (key, value) (line 228)
        str_37359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 51), 'str', 'script_args')
        
        # Obtaining an instance of the builtin type 'list' (line 229)
        list_37360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 229)
        # Adding element type (line 229)
        str_37361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 45), 'str', '--no-user-cfg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 44), list_37360, str_37361)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 50), dict_37358, (str_37359, list_37360))
        
        keyword_37362 = dict_37358
        kwargs_37363 = {'attrs': keyword_37362}
        # Getting the type of 'distutils' (line 228)
        distutils_37355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'distutils', False)
        # Obtaining the member 'dist' of a type (line 228)
        dist_37356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), distutils_37355, 'dist')
        # Obtaining the member 'Distribution' of a type (line 228)
        Distribution_37357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), dist_37356, 'Distribution')
        # Calling Distribution(args, kwargs) (line 228)
        Distribution_call_result_37364 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), Distribution_37357, *[], **kwargs_37363)
        
        # Assigning a type to the variable 'd' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'd', Distribution_call_result_37364)
        
        # Assigning a Call to a Name (line 230):
        
        # Call to find_config_files(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_37367 = {}
        # Getting the type of 'd' (line 230)
        d_37365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'd', False)
        # Obtaining the member 'find_config_files' of a type (line 230)
        find_config_files_37366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 20), d_37365, 'find_config_files')
        # Calling find_config_files(args, kwargs) (line 230)
        find_config_files_call_result_37368 = invoke(stypy.reporting.localization.Localization(__file__, 230, 20), find_config_files_37366, *[], **kwargs_37367)
        
        # Assigning a type to the variable 'files' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'files', find_config_files_call_result_37368)
        
        # finally branch of the try-finally block (line 224)
        
        # Assigning a Name to a Attribute (line 232):
        # Getting the type of 'old_expander' (line 232)
        old_expander_37369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 33), 'old_expander')
        # Getting the type of 'os' (line 232)
        os_37370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'os')
        # Obtaining the member 'path' of a type (line 232)
        path_37371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), os_37370, 'path')
        # Setting the type of the member 'expanduser' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), path_37371, 'expanduser', old_expander_37369)
        
        
        # Call to assertEqual(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Call to len(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'all_files' (line 235)
        all_files_37375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 29), 'all_files', False)
        # Processing the call keyword arguments (line 235)
        kwargs_37376 = {}
        # Getting the type of 'len' (line 235)
        len_37374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'len', False)
        # Calling len(args, kwargs) (line 235)
        len_call_result_37377 = invoke(stypy.reporting.localization.Localization(__file__, 235, 25), len_37374, *[all_files_37375], **kwargs_37376)
        
        int_37378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 40), 'int')
        # Applying the binary operator '-' (line 235)
        result_sub_37379 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 25), '-', len_call_result_37377, int_37378)
        
        
        # Call to len(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'files' (line 235)
        files_37381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 47), 'files', False)
        # Processing the call keyword arguments (line 235)
        kwargs_37382 = {}
        # Getting the type of 'len' (line 235)
        len_37380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 43), 'len', False)
        # Calling len(args, kwargs) (line 235)
        len_call_result_37383 = invoke(stypy.reporting.localization.Localization(__file__, 235, 43), len_37380, *[files_37381], **kwargs_37382)
        
        # Processing the call keyword arguments (line 235)
        kwargs_37384 = {}
        # Getting the type of 'self' (line 235)
        self_37372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 235)
        assertEqual_37373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_37372, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 235)
        assertEqual_call_result_37385 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), assertEqual_37373, *[result_sub_37379, len_call_result_37383], **kwargs_37384)
        
        
        # ################# End of 'test_find_config_files_disable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_find_config_files_disable' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_37386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_find_config_files_disable'
        return stypy_return_type_37386


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 42, 0, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistributionTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistributionTestCase' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'DistributionTestCase', DistributionTestCase)
# Declaration of the 'MetadataTestCase' class
# Getting the type of 'support' (line 238)
support_37387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'support')
# Obtaining the member 'TempdirManager' of a type (line 238)
TempdirManager_37388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 23), support_37387, 'TempdirManager')
# Getting the type of 'support' (line 238)
support_37389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'support')
# Obtaining the member 'EnvironGuard' of a type (line 238)
EnvironGuard_37390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 47), support_37389, 'EnvironGuard')
# Getting the type of 'unittest' (line 239)
unittest_37391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'unittest')
# Obtaining the member 'TestCase' of a type (line 239)
TestCase_37392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 23), unittest_37391, 'TestCase')

class MetadataTestCase(TempdirManager_37388, EnvironGuard_37390, TestCase_37392, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.setUp')
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.setUp', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setUp(...): (line 242)
        # Processing the call keyword arguments (line 242)
        kwargs_37399 = {}
        
        # Call to super(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'MetadataTestCase' (line 242)
        MetadataTestCase_37394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'MetadataTestCase', False)
        # Getting the type of 'self' (line 242)
        self_37395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 32), 'self', False)
        # Processing the call keyword arguments (line 242)
        kwargs_37396 = {}
        # Getting the type of 'super' (line 242)
        super_37393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'super', False)
        # Calling super(args, kwargs) (line 242)
        super_call_result_37397 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), super_37393, *[MetadataTestCase_37394, self_37395], **kwargs_37396)
        
        # Obtaining the member 'setUp' of a type (line 242)
        setUp_37398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), super_call_result_37397, 'setUp')
        # Calling setUp(args, kwargs) (line 242)
        setUp_call_result_37400 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), setUp_37398, *[], **kwargs_37399)
        
        
        # Assigning a Tuple to a Attribute (line 243):
        
        # Obtaining an instance of the builtin type 'tuple' (line 243)
        tuple_37401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 243)
        # Adding element type (line 243)
        # Getting the type of 'sys' (line 243)
        sys_37402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'sys')
        # Obtaining the member 'argv' of a type (line 243)
        argv_37403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 20), sys_37402, 'argv')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 20), tuple_37401, argv_37403)
        # Adding element type (line 243)
        
        # Obtaining the type of the subscript
        slice_37404 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 243, 30), None, None, None)
        # Getting the type of 'sys' (line 243)
        sys_37405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 30), 'sys')
        # Obtaining the member 'argv' of a type (line 243)
        argv_37406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 30), sys_37405, 'argv')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___37407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 30), argv_37406, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_37408 = invoke(stypy.reporting.localization.Localization(__file__, 243, 30), getitem___37407, slice_37404)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 20), tuple_37401, subscript_call_result_37408)
        
        # Getting the type of 'self' (line 243)
        self_37409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'self')
        # Setting the type of the member 'argv' of a type (line 243)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), self_37409, 'argv', tuple_37401)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 241)
        stypy_return_type_37410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37410)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_37410


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.tearDown')
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.tearDown', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Subscript to a Attribute (line 246):
        
        # Obtaining the type of the subscript
        int_37411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 29), 'int')
        # Getting the type of 'self' (line 246)
        self_37412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'self')
        # Obtaining the member 'argv' of a type (line 246)
        argv_37413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), self_37412, 'argv')
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___37414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), argv_37413, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_37415 = invoke(stypy.reporting.localization.Localization(__file__, 246, 19), getitem___37414, int_37411)
        
        # Getting the type of 'sys' (line 246)
        sys_37416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 246)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), sys_37416, 'argv', subscript_call_result_37415)
        
        # Assigning a Subscript to a Subscript (line 247):
        
        # Obtaining the type of the subscript
        int_37417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 32), 'int')
        # Getting the type of 'self' (line 247)
        self_37418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'self')
        # Obtaining the member 'argv' of a type (line 247)
        argv_37419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 22), self_37418, 'argv')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___37420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 22), argv_37419, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_37421 = invoke(stypy.reporting.localization.Localization(__file__, 247, 22), getitem___37420, int_37417)
        
        # Getting the type of 'sys' (line 247)
        sys_37422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'sys')
        # Obtaining the member 'argv' of a type (line 247)
        argv_37423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), sys_37422, 'argv')
        slice_37424 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 247, 8), None, None, None)
        # Storing an element on a container (line 247)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 8), argv_37423, (slice_37424, subscript_call_result_37421))
        
        # Call to tearDown(...): (line 248)
        # Processing the call keyword arguments (line 248)
        kwargs_37431 = {}
        
        # Call to super(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'MetadataTestCase' (line 248)
        MetadataTestCase_37426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 14), 'MetadataTestCase', False)
        # Getting the type of 'self' (line 248)
        self_37427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 32), 'self', False)
        # Processing the call keyword arguments (line 248)
        kwargs_37428 = {}
        # Getting the type of 'super' (line 248)
        super_37425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'super', False)
        # Calling super(args, kwargs) (line 248)
        super_call_result_37429 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), super_37425, *[MetadataTestCase_37426, self_37427], **kwargs_37428)
        
        # Obtaining the member 'tearDown' of a type (line 248)
        tearDown_37430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), super_call_result_37429, 'tearDown')
        # Calling tearDown(args, kwargs) (line 248)
        tearDown_call_result_37432 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), tearDown_37430, *[], **kwargs_37431)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 245)
        stypy_return_type_37433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37433)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_37433


    @norecursion
    def test_classifier(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_classifier'
        module_type_store = module_type_store.open_function_context('test_classifier', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_classifier')
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_classifier.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_classifier', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_classifier', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_classifier(...)' code ##################

        
        # Assigning a Dict to a Name (line 251):
        
        # Obtaining an instance of the builtin type 'dict' (line 251)
        dict_37434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 251)
        # Adding element type (key, value) (line 251)
        str_37435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 17), 'str', 'name')
        str_37436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'str', 'Boa')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), dict_37434, (str_37435, str_37436))
        # Adding element type (key, value) (line 251)
        str_37437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 32), 'str', 'version')
        str_37438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 43), 'str', '3.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), dict_37434, (str_37437, str_37438))
        # Adding element type (key, value) (line 251)
        str_37439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 17), 'str', 'classifiers')
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_37440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        # Adding element type (line 252)
        str_37441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 33), 'str', 'Programming Language :: Python :: 3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 32), list_37440, str_37441)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 16), dict_37434, (str_37439, list_37440))
        
        # Assigning a type to the variable 'attrs' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'attrs', dict_37434)
        
        # Assigning a Call to a Name (line 253):
        
        # Call to Distribution(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'attrs' (line 253)
        attrs_37443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 28), 'attrs', False)
        # Processing the call keyword arguments (line 253)
        kwargs_37444 = {}
        # Getting the type of 'Distribution' (line 253)
        Distribution_37442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 253)
        Distribution_call_result_37445 = invoke(stypy.reporting.localization.Localization(__file__, 253, 15), Distribution_37442, *[attrs_37443], **kwargs_37444)
        
        # Assigning a type to the variable 'dist' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'dist', Distribution_call_result_37445)
        
        # Assigning a Call to a Name (line 254):
        
        # Call to format_metadata(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'dist' (line 254)
        dist_37448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 36), 'dist', False)
        # Processing the call keyword arguments (line 254)
        kwargs_37449 = {}
        # Getting the type of 'self' (line 254)
        self_37446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'self', False)
        # Obtaining the member 'format_metadata' of a type (line 254)
        format_metadata_37447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 15), self_37446, 'format_metadata')
        # Calling format_metadata(args, kwargs) (line 254)
        format_metadata_call_result_37450 = invoke(stypy.reporting.localization.Localization(__file__, 254, 15), format_metadata_37447, *[dist_37448], **kwargs_37449)
        
        # Assigning a type to the variable 'meta' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'meta', format_metadata_call_result_37450)
        
        # Call to assertIn(...): (line 255)
        # Processing the call arguments (line 255)
        str_37453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 22), 'str', 'Metadata-Version: 1.1')
        # Getting the type of 'meta' (line 255)
        meta_37454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 47), 'meta', False)
        # Processing the call keyword arguments (line 255)
        kwargs_37455 = {}
        # Getting the type of 'self' (line 255)
        self_37451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 255)
        assertIn_37452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_37451, 'assertIn')
        # Calling assertIn(args, kwargs) (line 255)
        assertIn_call_result_37456 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assertIn_37452, *[str_37453, meta_37454], **kwargs_37455)
        
        
        # ################# End of 'test_classifier(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_classifier' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_37457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37457)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_classifier'
        return stypy_return_type_37457


    @norecursion
    def test_download_url(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_download_url'
        module_type_store = module_type_store.open_function_context('test_download_url', 257, 4, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_download_url')
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_download_url.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_download_url', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_download_url', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_download_url(...)' code ##################

        
        # Assigning a Dict to a Name (line 258):
        
        # Obtaining an instance of the builtin type 'dict' (line 258)
        dict_37458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 258)
        # Adding element type (key, value) (line 258)
        str_37459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 17), 'str', 'name')
        str_37460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 25), 'str', 'Boa')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 16), dict_37458, (str_37459, str_37460))
        # Adding element type (key, value) (line 258)
        str_37461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 32), 'str', 'version')
        str_37462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 43), 'str', '3.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 16), dict_37458, (str_37461, str_37462))
        # Adding element type (key, value) (line 258)
        str_37463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 17), 'str', 'download_url')
        str_37464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 33), 'str', 'http://example.org/boa')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 16), dict_37458, (str_37463, str_37464))
        
        # Assigning a type to the variable 'attrs' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'attrs', dict_37458)
        
        # Assigning a Call to a Name (line 260):
        
        # Call to Distribution(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'attrs' (line 260)
        attrs_37466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 28), 'attrs', False)
        # Processing the call keyword arguments (line 260)
        kwargs_37467 = {}
        # Getting the type of 'Distribution' (line 260)
        Distribution_37465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 260)
        Distribution_call_result_37468 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), Distribution_37465, *[attrs_37466], **kwargs_37467)
        
        # Assigning a type to the variable 'dist' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'dist', Distribution_call_result_37468)
        
        # Assigning a Call to a Name (line 261):
        
        # Call to format_metadata(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'dist' (line 261)
        dist_37471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 36), 'dist', False)
        # Processing the call keyword arguments (line 261)
        kwargs_37472 = {}
        # Getting the type of 'self' (line 261)
        self_37469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'self', False)
        # Obtaining the member 'format_metadata' of a type (line 261)
        format_metadata_37470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 15), self_37469, 'format_metadata')
        # Calling format_metadata(args, kwargs) (line 261)
        format_metadata_call_result_37473 = invoke(stypy.reporting.localization.Localization(__file__, 261, 15), format_metadata_37470, *[dist_37471], **kwargs_37472)
        
        # Assigning a type to the variable 'meta' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'meta', format_metadata_call_result_37473)
        
        # Call to assertIn(...): (line 262)
        # Processing the call arguments (line 262)
        str_37476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 22), 'str', 'Metadata-Version: 1.1')
        # Getting the type of 'meta' (line 262)
        meta_37477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 47), 'meta', False)
        # Processing the call keyword arguments (line 262)
        kwargs_37478 = {}
        # Getting the type of 'self' (line 262)
        self_37474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 262)
        assertIn_37475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_37474, 'assertIn')
        # Calling assertIn(args, kwargs) (line 262)
        assertIn_call_result_37479 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), assertIn_37475, *[str_37476, meta_37477], **kwargs_37478)
        
        
        # ################# End of 'test_download_url(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_download_url' in the type store
        # Getting the type of 'stypy_return_type' (line 257)
        stypy_return_type_37480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37480)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_download_url'
        return stypy_return_type_37480


    @norecursion
    def test_long_description(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_long_description'
        module_type_store = module_type_store.open_function_context('test_long_description', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_long_description')
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_long_description.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_long_description', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_long_description', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_long_description(...)' code ##################

        
        # Assigning a Call to a Name (line 265):
        
        # Call to dedent(...): (line 265)
        # Processing the call arguments (line 265)
        str_37483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, (-1)), 'str', '        example::\n              We start here\n            and continue here\n          and end here.')
        # Processing the call keyword arguments (line 265)
        kwargs_37484 = {}
        # Getting the type of 'textwrap' (line 265)
        textwrap_37481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'textwrap', False)
        # Obtaining the member 'dedent' of a type (line 265)
        dedent_37482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 20), textwrap_37481, 'dedent')
        # Calling dedent(args, kwargs) (line 265)
        dedent_call_result_37485 = invoke(stypy.reporting.localization.Localization(__file__, 265, 20), dedent_37482, *[str_37483], **kwargs_37484)
        
        # Assigning a type to the variable 'long_desc' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'long_desc', dedent_call_result_37485)
        
        # Assigning a Dict to a Name (line 270):
        
        # Obtaining an instance of the builtin type 'dict' (line 270)
        dict_37486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 270)
        # Adding element type (key, value) (line 270)
        str_37487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 17), 'str', 'name')
        str_37488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 25), 'str', 'package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 16), dict_37486, (str_37487, str_37488))
        # Adding element type (key, value) (line 270)
        str_37489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 17), 'str', 'version')
        str_37490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 28), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 16), dict_37486, (str_37489, str_37490))
        # Adding element type (key, value) (line 270)
        str_37491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 17), 'str', 'long_description')
        # Getting the type of 'long_desc' (line 272)
        long_desc_37492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 37), 'long_desc')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 16), dict_37486, (str_37491, long_desc_37492))
        
        # Assigning a type to the variable 'attrs' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'attrs', dict_37486)
        
        # Assigning a Call to a Name (line 274):
        
        # Call to Distribution(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'attrs' (line 274)
        attrs_37494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 28), 'attrs', False)
        # Processing the call keyword arguments (line 274)
        kwargs_37495 = {}
        # Getting the type of 'Distribution' (line 274)
        Distribution_37493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 274)
        Distribution_call_result_37496 = invoke(stypy.reporting.localization.Localization(__file__, 274, 15), Distribution_37493, *[attrs_37494], **kwargs_37495)
        
        # Assigning a type to the variable 'dist' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'dist', Distribution_call_result_37496)
        
        # Assigning a Call to a Name (line 275):
        
        # Call to format_metadata(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'dist' (line 275)
        dist_37499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 36), 'dist', False)
        # Processing the call keyword arguments (line 275)
        kwargs_37500 = {}
        # Getting the type of 'self' (line 275)
        self_37497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'self', False)
        # Obtaining the member 'format_metadata' of a type (line 275)
        format_metadata_37498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 15), self_37497, 'format_metadata')
        # Calling format_metadata(args, kwargs) (line 275)
        format_metadata_call_result_37501 = invoke(stypy.reporting.localization.Localization(__file__, 275, 15), format_metadata_37498, *[dist_37499], **kwargs_37500)
        
        # Assigning a type to the variable 'meta' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'meta', format_metadata_call_result_37501)
        
        # Assigning a Call to a Name (line 276):
        
        # Call to replace(...): (line 276)
        # Processing the call arguments (line 276)
        str_37504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 28), 'str', '\n')
        int_37505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 35), 'int')
        str_37506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 39), 'str', ' ')
        # Applying the binary operator '*' (line 276)
        result_mul_37507 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 35), '*', int_37505, str_37506)
        
        # Applying the binary operator '+' (line 276)
        result_add_37508 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 28), '+', str_37504, result_mul_37507)
        
        str_37509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 44), 'str', '\n')
        # Processing the call keyword arguments (line 276)
        kwargs_37510 = {}
        # Getting the type of 'meta' (line 276)
        meta_37502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'meta', False)
        # Obtaining the member 'replace' of a type (line 276)
        replace_37503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 15), meta_37502, 'replace')
        # Calling replace(args, kwargs) (line 276)
        replace_call_result_37511 = invoke(stypy.reporting.localization.Localization(__file__, 276, 15), replace_37503, *[result_add_37508, str_37509], **kwargs_37510)
        
        # Assigning a type to the variable 'meta' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'meta', replace_call_result_37511)
        
        # Call to assertIn(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'long_desc' (line 277)
        long_desc_37514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 22), 'long_desc', False)
        # Getting the type of 'meta' (line 277)
        meta_37515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 33), 'meta', False)
        # Processing the call keyword arguments (line 277)
        kwargs_37516 = {}
        # Getting the type of 'self' (line 277)
        self_37512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 277)
        assertIn_37513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), self_37512, 'assertIn')
        # Calling assertIn(args, kwargs) (line 277)
        assertIn_call_result_37517 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), assertIn_37513, *[long_desc_37514, meta_37515], **kwargs_37516)
        
        
        # ################# End of 'test_long_description(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_long_description' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_37518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_long_description'
        return stypy_return_type_37518


    @norecursion
    def test_simple_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple_metadata'
        module_type_store = module_type_store.open_function_context('test_simple_metadata', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_simple_metadata')
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_simple_metadata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_simple_metadata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple_metadata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple_metadata(...)' code ##################

        
        # Assigning a Dict to a Name (line 280):
        
        # Obtaining an instance of the builtin type 'dict' (line 280)
        dict_37519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 280)
        # Adding element type (key, value) (line 280)
        str_37520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 17), 'str', 'name')
        str_37521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 25), 'str', 'package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 16), dict_37519, (str_37520, str_37521))
        # Adding element type (key, value) (line 280)
        str_37522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 17), 'str', 'version')
        str_37523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 28), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 16), dict_37519, (str_37522, str_37523))
        
        # Assigning a type to the variable 'attrs' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'attrs', dict_37519)
        
        # Assigning a Call to a Name (line 282):
        
        # Call to Distribution(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'attrs' (line 282)
        attrs_37525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'attrs', False)
        # Processing the call keyword arguments (line 282)
        kwargs_37526 = {}
        # Getting the type of 'Distribution' (line 282)
        Distribution_37524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 282)
        Distribution_call_result_37527 = invoke(stypy.reporting.localization.Localization(__file__, 282, 15), Distribution_37524, *[attrs_37525], **kwargs_37526)
        
        # Assigning a type to the variable 'dist' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'dist', Distribution_call_result_37527)
        
        # Assigning a Call to a Name (line 283):
        
        # Call to format_metadata(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'dist' (line 283)
        dist_37530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 36), 'dist', False)
        # Processing the call keyword arguments (line 283)
        kwargs_37531 = {}
        # Getting the type of 'self' (line 283)
        self_37528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'self', False)
        # Obtaining the member 'format_metadata' of a type (line 283)
        format_metadata_37529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 15), self_37528, 'format_metadata')
        # Calling format_metadata(args, kwargs) (line 283)
        format_metadata_call_result_37532 = invoke(stypy.reporting.localization.Localization(__file__, 283, 15), format_metadata_37529, *[dist_37530], **kwargs_37531)
        
        # Assigning a type to the variable 'meta' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'meta', format_metadata_call_result_37532)
        
        # Call to assertIn(...): (line 284)
        # Processing the call arguments (line 284)
        str_37535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 22), 'str', 'Metadata-Version: 1.0')
        # Getting the type of 'meta' (line 284)
        meta_37536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 47), 'meta', False)
        # Processing the call keyword arguments (line 284)
        kwargs_37537 = {}
        # Getting the type of 'self' (line 284)
        self_37533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 284)
        assertIn_37534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_37533, 'assertIn')
        # Calling assertIn(args, kwargs) (line 284)
        assertIn_call_result_37538 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), assertIn_37534, *[str_37535, meta_37536], **kwargs_37537)
        
        
        # Call to assertNotIn(...): (line 285)
        # Processing the call arguments (line 285)
        str_37541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 25), 'str', 'provides:')
        
        # Call to lower(...): (line 285)
        # Processing the call keyword arguments (line 285)
        kwargs_37544 = {}
        # Getting the type of 'meta' (line 285)
        meta_37542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 38), 'meta', False)
        # Obtaining the member 'lower' of a type (line 285)
        lower_37543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 38), meta_37542, 'lower')
        # Calling lower(args, kwargs) (line 285)
        lower_call_result_37545 = invoke(stypy.reporting.localization.Localization(__file__, 285, 38), lower_37543, *[], **kwargs_37544)
        
        # Processing the call keyword arguments (line 285)
        kwargs_37546 = {}
        # Getting the type of 'self' (line 285)
        self_37539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 285)
        assertNotIn_37540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_37539, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 285)
        assertNotIn_call_result_37547 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), assertNotIn_37540, *[str_37541, lower_call_result_37545], **kwargs_37546)
        
        
        # Call to assertNotIn(...): (line 286)
        # Processing the call arguments (line 286)
        str_37550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 25), 'str', 'requires:')
        
        # Call to lower(...): (line 286)
        # Processing the call keyword arguments (line 286)
        kwargs_37553 = {}
        # Getting the type of 'meta' (line 286)
        meta_37551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 38), 'meta', False)
        # Obtaining the member 'lower' of a type (line 286)
        lower_37552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 38), meta_37551, 'lower')
        # Calling lower(args, kwargs) (line 286)
        lower_call_result_37554 = invoke(stypy.reporting.localization.Localization(__file__, 286, 38), lower_37552, *[], **kwargs_37553)
        
        # Processing the call keyword arguments (line 286)
        kwargs_37555 = {}
        # Getting the type of 'self' (line 286)
        self_37548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 286)
        assertNotIn_37549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), self_37548, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 286)
        assertNotIn_call_result_37556 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), assertNotIn_37549, *[str_37550, lower_call_result_37554], **kwargs_37555)
        
        
        # Call to assertNotIn(...): (line 287)
        # Processing the call arguments (line 287)
        str_37559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 25), 'str', 'obsoletes:')
        
        # Call to lower(...): (line 287)
        # Processing the call keyword arguments (line 287)
        kwargs_37562 = {}
        # Getting the type of 'meta' (line 287)
        meta_37560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 39), 'meta', False)
        # Obtaining the member 'lower' of a type (line 287)
        lower_37561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 39), meta_37560, 'lower')
        # Calling lower(args, kwargs) (line 287)
        lower_call_result_37563 = invoke(stypy.reporting.localization.Localization(__file__, 287, 39), lower_37561, *[], **kwargs_37562)
        
        # Processing the call keyword arguments (line 287)
        kwargs_37564 = {}
        # Getting the type of 'self' (line 287)
        self_37557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 287)
        assertNotIn_37558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_37557, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 287)
        assertNotIn_call_result_37565 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), assertNotIn_37558, *[str_37559, lower_call_result_37563], **kwargs_37564)
        
        
        # ################# End of 'test_simple_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_37566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37566)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple_metadata'
        return stypy_return_type_37566


    @norecursion
    def test_provides(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_provides'
        module_type_store = module_type_store.open_function_context('test_provides', 289, 4, False)
        # Assigning a type to the variable 'self' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_provides')
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_provides.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_provides', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_provides', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_provides(...)' code ##################

        
        # Assigning a Dict to a Name (line 290):
        
        # Obtaining an instance of the builtin type 'dict' (line 290)
        dict_37567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 290)
        # Adding element type (key, value) (line 290)
        str_37568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 17), 'str', 'name')
        str_37569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 25), 'str', 'package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 16), dict_37567, (str_37568, str_37569))
        # Adding element type (key, value) (line 290)
        str_37570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 17), 'str', 'version')
        str_37571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 28), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 16), dict_37567, (str_37570, str_37571))
        # Adding element type (key, value) (line 290)
        str_37572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 17), 'str', 'provides')
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_37573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        str_37574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 30), 'str', 'package')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 29), list_37573, str_37574)
        # Adding element type (line 292)
        str_37575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 41), 'str', 'package.sub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 29), list_37573, str_37575)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 16), dict_37567, (str_37572, list_37573))
        
        # Assigning a type to the variable 'attrs' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'attrs', dict_37567)
        
        # Assigning a Call to a Name (line 293):
        
        # Call to Distribution(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'attrs' (line 293)
        attrs_37577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 28), 'attrs', False)
        # Processing the call keyword arguments (line 293)
        kwargs_37578 = {}
        # Getting the type of 'Distribution' (line 293)
        Distribution_37576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 293)
        Distribution_call_result_37579 = invoke(stypy.reporting.localization.Localization(__file__, 293, 15), Distribution_37576, *[attrs_37577], **kwargs_37578)
        
        # Assigning a type to the variable 'dist' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'dist', Distribution_call_result_37579)
        
        # Call to assertEqual(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to get_provides(...): (line 294)
        # Processing the call keyword arguments (line 294)
        kwargs_37585 = {}
        # Getting the type of 'dist' (line 294)
        dist_37582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 25), 'dist', False)
        # Obtaining the member 'metadata' of a type (line 294)
        metadata_37583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 25), dist_37582, 'metadata')
        # Obtaining the member 'get_provides' of a type (line 294)
        get_provides_37584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 25), metadata_37583, 'get_provides')
        # Calling get_provides(args, kwargs) (line 294)
        get_provides_call_result_37586 = invoke(stypy.reporting.localization.Localization(__file__, 294, 25), get_provides_37584, *[], **kwargs_37585)
        
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_37587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        str_37588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 26), 'str', 'package')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 25), list_37587, str_37588)
        # Adding element type (line 295)
        str_37589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 37), 'str', 'package.sub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 25), list_37587, str_37589)
        
        # Processing the call keyword arguments (line 294)
        kwargs_37590 = {}
        # Getting the type of 'self' (line 294)
        self_37580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 294)
        assertEqual_37581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_37580, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 294)
        assertEqual_call_result_37591 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), assertEqual_37581, *[get_provides_call_result_37586, list_37587], **kwargs_37590)
        
        
        # Call to assertEqual(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Call to get_provides(...): (line 296)
        # Processing the call keyword arguments (line 296)
        kwargs_37596 = {}
        # Getting the type of 'dist' (line 296)
        dist_37594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 25), 'dist', False)
        # Obtaining the member 'get_provides' of a type (line 296)
        get_provides_37595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 25), dist_37594, 'get_provides')
        # Calling get_provides(args, kwargs) (line 296)
        get_provides_call_result_37597 = invoke(stypy.reporting.localization.Localization(__file__, 296, 25), get_provides_37595, *[], **kwargs_37596)
        
        
        # Obtaining an instance of the builtin type 'list' (line 297)
        list_37598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 297)
        # Adding element type (line 297)
        str_37599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 26), 'str', 'package')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 25), list_37598, str_37599)
        # Adding element type (line 297)
        str_37600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 37), 'str', 'package.sub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 25), list_37598, str_37600)
        
        # Processing the call keyword arguments (line 296)
        kwargs_37601 = {}
        # Getting the type of 'self' (line 296)
        self_37592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 296)
        assertEqual_37593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_37592, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 296)
        assertEqual_call_result_37602 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), assertEqual_37593, *[get_provides_call_result_37597, list_37598], **kwargs_37601)
        
        
        # Assigning a Call to a Name (line 298):
        
        # Call to format_metadata(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'dist' (line 298)
        dist_37605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 36), 'dist', False)
        # Processing the call keyword arguments (line 298)
        kwargs_37606 = {}
        # Getting the type of 'self' (line 298)
        self_37603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'self', False)
        # Obtaining the member 'format_metadata' of a type (line 298)
        format_metadata_37604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 15), self_37603, 'format_metadata')
        # Calling format_metadata(args, kwargs) (line 298)
        format_metadata_call_result_37607 = invoke(stypy.reporting.localization.Localization(__file__, 298, 15), format_metadata_37604, *[dist_37605], **kwargs_37606)
        
        # Assigning a type to the variable 'meta' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'meta', format_metadata_call_result_37607)
        
        # Call to assertIn(...): (line 299)
        # Processing the call arguments (line 299)
        str_37610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 22), 'str', 'Metadata-Version: 1.1')
        # Getting the type of 'meta' (line 299)
        meta_37611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 47), 'meta', False)
        # Processing the call keyword arguments (line 299)
        kwargs_37612 = {}
        # Getting the type of 'self' (line 299)
        self_37608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 299)
        assertIn_37609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), self_37608, 'assertIn')
        # Calling assertIn(args, kwargs) (line 299)
        assertIn_call_result_37613 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), assertIn_37609, *[str_37610, meta_37611], **kwargs_37612)
        
        
        # Call to assertNotIn(...): (line 300)
        # Processing the call arguments (line 300)
        str_37616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 25), 'str', 'requires:')
        
        # Call to lower(...): (line 300)
        # Processing the call keyword arguments (line 300)
        kwargs_37619 = {}
        # Getting the type of 'meta' (line 300)
        meta_37617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 38), 'meta', False)
        # Obtaining the member 'lower' of a type (line 300)
        lower_37618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 38), meta_37617, 'lower')
        # Calling lower(args, kwargs) (line 300)
        lower_call_result_37620 = invoke(stypy.reporting.localization.Localization(__file__, 300, 38), lower_37618, *[], **kwargs_37619)
        
        # Processing the call keyword arguments (line 300)
        kwargs_37621 = {}
        # Getting the type of 'self' (line 300)
        self_37614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 300)
        assertNotIn_37615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), self_37614, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 300)
        assertNotIn_call_result_37622 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), assertNotIn_37615, *[str_37616, lower_call_result_37620], **kwargs_37621)
        
        
        # Call to assertNotIn(...): (line 301)
        # Processing the call arguments (line 301)
        str_37625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 25), 'str', 'obsoletes:')
        
        # Call to lower(...): (line 301)
        # Processing the call keyword arguments (line 301)
        kwargs_37628 = {}
        # Getting the type of 'meta' (line 301)
        meta_37626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 39), 'meta', False)
        # Obtaining the member 'lower' of a type (line 301)
        lower_37627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 39), meta_37626, 'lower')
        # Calling lower(args, kwargs) (line 301)
        lower_call_result_37629 = invoke(stypy.reporting.localization.Localization(__file__, 301, 39), lower_37627, *[], **kwargs_37628)
        
        # Processing the call keyword arguments (line 301)
        kwargs_37630 = {}
        # Getting the type of 'self' (line 301)
        self_37623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 301)
        assertNotIn_37624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), self_37623, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 301)
        assertNotIn_call_result_37631 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), assertNotIn_37624, *[str_37625, lower_call_result_37629], **kwargs_37630)
        
        
        # ################# End of 'test_provides(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_provides' in the type store
        # Getting the type of 'stypy_return_type' (line 289)
        stypy_return_type_37632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37632)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_provides'
        return stypy_return_type_37632


    @norecursion
    def test_provides_illegal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_provides_illegal'
        module_type_store = module_type_store.open_function_context('test_provides_illegal', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_provides_illegal')
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_provides_illegal.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_provides_illegal', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_provides_illegal', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_provides_illegal(...)' code ##################

        
        # Call to assertRaises(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'ValueError' (line 304)
        ValueError_37635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 26), 'ValueError', False)
        # Getting the type of 'Distribution' (line 304)
        Distribution_37636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 38), 'Distribution', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 305)
        dict_37637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 26), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 305)
        # Adding element type (key, value) (line 305)
        str_37638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 27), 'str', 'name')
        str_37639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 35), 'str', 'package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 26), dict_37637, (str_37638, str_37639))
        # Adding element type (key, value) (line 305)
        str_37640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 27), 'str', 'version')
        str_37641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 38), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 26), dict_37637, (str_37640, str_37641))
        # Adding element type (key, value) (line 305)
        str_37642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 27), 'str', 'provides')
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_37643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        str_37644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 40), 'str', 'my.pkg (splat)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 39), list_37643, str_37644)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 26), dict_37637, (str_37642, list_37643))
        
        # Processing the call keyword arguments (line 304)
        kwargs_37645 = {}
        # Getting the type of 'self' (line 304)
        self_37633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 304)
        assertRaises_37634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), self_37633, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 304)
        assertRaises_call_result_37646 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), assertRaises_37634, *[ValueError_37635, Distribution_37636, dict_37637], **kwargs_37645)
        
        
        # ################# End of 'test_provides_illegal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_provides_illegal' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_37647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37647)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_provides_illegal'
        return stypy_return_type_37647


    @norecursion
    def test_requires(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_requires'
        module_type_store = module_type_store.open_function_context('test_requires', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_requires')
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_requires.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_requires', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_requires', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_requires(...)' code ##################

        
        # Assigning a Dict to a Name (line 310):
        
        # Obtaining an instance of the builtin type 'dict' (line 310)
        dict_37648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 310)
        # Adding element type (key, value) (line 310)
        str_37649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 17), 'str', 'name')
        str_37650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 25), 'str', 'package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 16), dict_37648, (str_37649, str_37650))
        # Adding element type (key, value) (line 310)
        str_37651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 17), 'str', 'version')
        str_37652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 28), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 16), dict_37648, (str_37651, str_37652))
        # Adding element type (key, value) (line 310)
        str_37653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 17), 'str', 'requires')
        
        # Obtaining an instance of the builtin type 'list' (line 312)
        list_37654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 312)
        # Adding element type (line 312)
        str_37655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 30), 'str', 'other')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 29), list_37654, str_37655)
        # Adding element type (line 312)
        str_37656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 39), 'str', 'another (==1.0)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 29), list_37654, str_37656)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 16), dict_37648, (str_37653, list_37654))
        
        # Assigning a type to the variable 'attrs' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'attrs', dict_37648)
        
        # Assigning a Call to a Name (line 313):
        
        # Call to Distribution(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'attrs' (line 313)
        attrs_37658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 28), 'attrs', False)
        # Processing the call keyword arguments (line 313)
        kwargs_37659 = {}
        # Getting the type of 'Distribution' (line 313)
        Distribution_37657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 313)
        Distribution_call_result_37660 = invoke(stypy.reporting.localization.Localization(__file__, 313, 15), Distribution_37657, *[attrs_37658], **kwargs_37659)
        
        # Assigning a type to the variable 'dist' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'dist', Distribution_call_result_37660)
        
        # Call to assertEqual(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Call to get_requires(...): (line 314)
        # Processing the call keyword arguments (line 314)
        kwargs_37666 = {}
        # Getting the type of 'dist' (line 314)
        dist_37663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 25), 'dist', False)
        # Obtaining the member 'metadata' of a type (line 314)
        metadata_37664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 25), dist_37663, 'metadata')
        # Obtaining the member 'get_requires' of a type (line 314)
        get_requires_37665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 25), metadata_37664, 'get_requires')
        # Calling get_requires(args, kwargs) (line 314)
        get_requires_call_result_37667 = invoke(stypy.reporting.localization.Localization(__file__, 314, 25), get_requires_37665, *[], **kwargs_37666)
        
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_37668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        str_37669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 26), 'str', 'other')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 25), list_37668, str_37669)
        # Adding element type (line 315)
        str_37670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 35), 'str', 'another (==1.0)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 25), list_37668, str_37670)
        
        # Processing the call keyword arguments (line 314)
        kwargs_37671 = {}
        # Getting the type of 'self' (line 314)
        self_37661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 314)
        assertEqual_37662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), self_37661, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 314)
        assertEqual_call_result_37672 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), assertEqual_37662, *[get_requires_call_result_37667, list_37668], **kwargs_37671)
        
        
        # Call to assertEqual(...): (line 316)
        # Processing the call arguments (line 316)
        
        # Call to get_requires(...): (line 316)
        # Processing the call keyword arguments (line 316)
        kwargs_37677 = {}
        # Getting the type of 'dist' (line 316)
        dist_37675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 25), 'dist', False)
        # Obtaining the member 'get_requires' of a type (line 316)
        get_requires_37676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 25), dist_37675, 'get_requires')
        # Calling get_requires(args, kwargs) (line 316)
        get_requires_call_result_37678 = invoke(stypy.reporting.localization.Localization(__file__, 316, 25), get_requires_37676, *[], **kwargs_37677)
        
        
        # Obtaining an instance of the builtin type 'list' (line 317)
        list_37679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 317)
        # Adding element type (line 317)
        str_37680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 26), 'str', 'other')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 25), list_37679, str_37680)
        # Adding element type (line 317)
        str_37681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 35), 'str', 'another (==1.0)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 25), list_37679, str_37681)
        
        # Processing the call keyword arguments (line 316)
        kwargs_37682 = {}
        # Getting the type of 'self' (line 316)
        self_37673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 316)
        assertEqual_37674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), self_37673, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 316)
        assertEqual_call_result_37683 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), assertEqual_37674, *[get_requires_call_result_37678, list_37679], **kwargs_37682)
        
        
        # Assigning a Call to a Name (line 318):
        
        # Call to format_metadata(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'dist' (line 318)
        dist_37686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 36), 'dist', False)
        # Processing the call keyword arguments (line 318)
        kwargs_37687 = {}
        # Getting the type of 'self' (line 318)
        self_37684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 15), 'self', False)
        # Obtaining the member 'format_metadata' of a type (line 318)
        format_metadata_37685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 15), self_37684, 'format_metadata')
        # Calling format_metadata(args, kwargs) (line 318)
        format_metadata_call_result_37688 = invoke(stypy.reporting.localization.Localization(__file__, 318, 15), format_metadata_37685, *[dist_37686], **kwargs_37687)
        
        # Assigning a type to the variable 'meta' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'meta', format_metadata_call_result_37688)
        
        # Call to assertIn(...): (line 319)
        # Processing the call arguments (line 319)
        str_37691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 22), 'str', 'Metadata-Version: 1.1')
        # Getting the type of 'meta' (line 319)
        meta_37692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 47), 'meta', False)
        # Processing the call keyword arguments (line 319)
        kwargs_37693 = {}
        # Getting the type of 'self' (line 319)
        self_37689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 319)
        assertIn_37690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), self_37689, 'assertIn')
        # Calling assertIn(args, kwargs) (line 319)
        assertIn_call_result_37694 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), assertIn_37690, *[str_37691, meta_37692], **kwargs_37693)
        
        
        # Call to assertNotIn(...): (line 320)
        # Processing the call arguments (line 320)
        str_37697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 25), 'str', 'provides:')
        
        # Call to lower(...): (line 320)
        # Processing the call keyword arguments (line 320)
        kwargs_37700 = {}
        # Getting the type of 'meta' (line 320)
        meta_37698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 38), 'meta', False)
        # Obtaining the member 'lower' of a type (line 320)
        lower_37699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 38), meta_37698, 'lower')
        # Calling lower(args, kwargs) (line 320)
        lower_call_result_37701 = invoke(stypy.reporting.localization.Localization(__file__, 320, 38), lower_37699, *[], **kwargs_37700)
        
        # Processing the call keyword arguments (line 320)
        kwargs_37702 = {}
        # Getting the type of 'self' (line 320)
        self_37695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 320)
        assertNotIn_37696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), self_37695, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 320)
        assertNotIn_call_result_37703 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), assertNotIn_37696, *[str_37697, lower_call_result_37701], **kwargs_37702)
        
        
        # Call to assertIn(...): (line 321)
        # Processing the call arguments (line 321)
        str_37706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 22), 'str', 'Requires: other')
        # Getting the type of 'meta' (line 321)
        meta_37707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 41), 'meta', False)
        # Processing the call keyword arguments (line 321)
        kwargs_37708 = {}
        # Getting the type of 'self' (line 321)
        self_37704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 321)
        assertIn_37705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), self_37704, 'assertIn')
        # Calling assertIn(args, kwargs) (line 321)
        assertIn_call_result_37709 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), assertIn_37705, *[str_37706, meta_37707], **kwargs_37708)
        
        
        # Call to assertIn(...): (line 322)
        # Processing the call arguments (line 322)
        str_37712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 22), 'str', 'Requires: another (==1.0)')
        # Getting the type of 'meta' (line 322)
        meta_37713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 51), 'meta', False)
        # Processing the call keyword arguments (line 322)
        kwargs_37714 = {}
        # Getting the type of 'self' (line 322)
        self_37710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 322)
        assertIn_37711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), self_37710, 'assertIn')
        # Calling assertIn(args, kwargs) (line 322)
        assertIn_call_result_37715 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), assertIn_37711, *[str_37712, meta_37713], **kwargs_37714)
        
        
        # Call to assertNotIn(...): (line 323)
        # Processing the call arguments (line 323)
        str_37718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 25), 'str', 'obsoletes:')
        
        # Call to lower(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_37721 = {}
        # Getting the type of 'meta' (line 323)
        meta_37719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 39), 'meta', False)
        # Obtaining the member 'lower' of a type (line 323)
        lower_37720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 39), meta_37719, 'lower')
        # Calling lower(args, kwargs) (line 323)
        lower_call_result_37722 = invoke(stypy.reporting.localization.Localization(__file__, 323, 39), lower_37720, *[], **kwargs_37721)
        
        # Processing the call keyword arguments (line 323)
        kwargs_37723 = {}
        # Getting the type of 'self' (line 323)
        self_37716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 323)
        assertNotIn_37717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_37716, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 323)
        assertNotIn_call_result_37724 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), assertNotIn_37717, *[str_37718, lower_call_result_37722], **kwargs_37723)
        
        
        # ################# End of 'test_requires(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_requires' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_37725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37725)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_requires'
        return stypy_return_type_37725


    @norecursion
    def test_requires_illegal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_requires_illegal'
        module_type_store = module_type_store.open_function_context('test_requires_illegal', 325, 4, False)
        # Assigning a type to the variable 'self' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_requires_illegal')
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_requires_illegal.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_requires_illegal', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_requires_illegal', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_requires_illegal(...)' code ##################

        
        # Call to assertRaises(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'ValueError' (line 326)
        ValueError_37728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), 'ValueError', False)
        # Getting the type of 'Distribution' (line 326)
        Distribution_37729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 38), 'Distribution', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 327)
        dict_37730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 26), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 327)
        # Adding element type (key, value) (line 327)
        str_37731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 27), 'str', 'name')
        str_37732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 35), 'str', 'package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 26), dict_37730, (str_37731, str_37732))
        # Adding element type (key, value) (line 327)
        str_37733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 27), 'str', 'version')
        str_37734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 38), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 26), dict_37730, (str_37733, str_37734))
        # Adding element type (key, value) (line 327)
        str_37735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 27), 'str', 'requires')
        
        # Obtaining an instance of the builtin type 'list' (line 329)
        list_37736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 329)
        # Adding element type (line 329)
        str_37737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 40), 'str', 'my.pkg (splat)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 39), list_37736, str_37737)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 26), dict_37730, (str_37735, list_37736))
        
        # Processing the call keyword arguments (line 326)
        kwargs_37738 = {}
        # Getting the type of 'self' (line 326)
        self_37726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 326)
        assertRaises_37727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_37726, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 326)
        assertRaises_call_result_37739 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), assertRaises_37727, *[ValueError_37728, Distribution_37729, dict_37730], **kwargs_37738)
        
        
        # ################# End of 'test_requires_illegal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_requires_illegal' in the type store
        # Getting the type of 'stypy_return_type' (line 325)
        stypy_return_type_37740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37740)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_requires_illegal'
        return stypy_return_type_37740


    @norecursion
    def test_obsoletes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_obsoletes'
        module_type_store = module_type_store.open_function_context('test_obsoletes', 331, 4, False)
        # Assigning a type to the variable 'self' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_obsoletes')
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_obsoletes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_obsoletes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_obsoletes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_obsoletes(...)' code ##################

        
        # Assigning a Dict to a Name (line 332):
        
        # Obtaining an instance of the builtin type 'dict' (line 332)
        dict_37741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 332)
        # Adding element type (key, value) (line 332)
        str_37742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 17), 'str', 'name')
        str_37743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 25), 'str', 'package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 16), dict_37741, (str_37742, str_37743))
        # Adding element type (key, value) (line 332)
        str_37744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 17), 'str', 'version')
        str_37745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 28), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 16), dict_37741, (str_37744, str_37745))
        # Adding element type (key, value) (line 332)
        str_37746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 17), 'str', 'obsoletes')
        
        # Obtaining an instance of the builtin type 'list' (line 334)
        list_37747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 334)
        # Adding element type (line 334)
        str_37748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 31), 'str', 'other')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 30), list_37747, str_37748)
        # Adding element type (line 334)
        str_37749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 40), 'str', 'another (<1.0)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 30), list_37747, str_37749)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 16), dict_37741, (str_37746, list_37747))
        
        # Assigning a type to the variable 'attrs' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'attrs', dict_37741)
        
        # Assigning a Call to a Name (line 335):
        
        # Call to Distribution(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'attrs' (line 335)
        attrs_37751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 28), 'attrs', False)
        # Processing the call keyword arguments (line 335)
        kwargs_37752 = {}
        # Getting the type of 'Distribution' (line 335)
        Distribution_37750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 335)
        Distribution_call_result_37753 = invoke(stypy.reporting.localization.Localization(__file__, 335, 15), Distribution_37750, *[attrs_37751], **kwargs_37752)
        
        # Assigning a type to the variable 'dist' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'dist', Distribution_call_result_37753)
        
        # Call to assertEqual(...): (line 336)
        # Processing the call arguments (line 336)
        
        # Call to get_obsoletes(...): (line 336)
        # Processing the call keyword arguments (line 336)
        kwargs_37759 = {}
        # Getting the type of 'dist' (line 336)
        dist_37756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 25), 'dist', False)
        # Obtaining the member 'metadata' of a type (line 336)
        metadata_37757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 25), dist_37756, 'metadata')
        # Obtaining the member 'get_obsoletes' of a type (line 336)
        get_obsoletes_37758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 25), metadata_37757, 'get_obsoletes')
        # Calling get_obsoletes(args, kwargs) (line 336)
        get_obsoletes_call_result_37760 = invoke(stypy.reporting.localization.Localization(__file__, 336, 25), get_obsoletes_37758, *[], **kwargs_37759)
        
        
        # Obtaining an instance of the builtin type 'list' (line 337)
        list_37761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 337)
        # Adding element type (line 337)
        str_37762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 26), 'str', 'other')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 25), list_37761, str_37762)
        # Adding element type (line 337)
        str_37763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 35), 'str', 'another (<1.0)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 25), list_37761, str_37763)
        
        # Processing the call keyword arguments (line 336)
        kwargs_37764 = {}
        # Getting the type of 'self' (line 336)
        self_37754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 336)
        assertEqual_37755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), self_37754, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 336)
        assertEqual_call_result_37765 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), assertEqual_37755, *[get_obsoletes_call_result_37760, list_37761], **kwargs_37764)
        
        
        # Call to assertEqual(...): (line 338)
        # Processing the call arguments (line 338)
        
        # Call to get_obsoletes(...): (line 338)
        # Processing the call keyword arguments (line 338)
        kwargs_37770 = {}
        # Getting the type of 'dist' (line 338)
        dist_37768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 25), 'dist', False)
        # Obtaining the member 'get_obsoletes' of a type (line 338)
        get_obsoletes_37769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 25), dist_37768, 'get_obsoletes')
        # Calling get_obsoletes(args, kwargs) (line 338)
        get_obsoletes_call_result_37771 = invoke(stypy.reporting.localization.Localization(__file__, 338, 25), get_obsoletes_37769, *[], **kwargs_37770)
        
        
        # Obtaining an instance of the builtin type 'list' (line 339)
        list_37772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 339)
        # Adding element type (line 339)
        str_37773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 26), 'str', 'other')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 25), list_37772, str_37773)
        # Adding element type (line 339)
        str_37774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 35), 'str', 'another (<1.0)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 25), list_37772, str_37774)
        
        # Processing the call keyword arguments (line 338)
        kwargs_37775 = {}
        # Getting the type of 'self' (line 338)
        self_37766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 338)
        assertEqual_37767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), self_37766, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 338)
        assertEqual_call_result_37776 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), assertEqual_37767, *[get_obsoletes_call_result_37771, list_37772], **kwargs_37775)
        
        
        # Assigning a Call to a Name (line 340):
        
        # Call to format_metadata(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'dist' (line 340)
        dist_37779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 36), 'dist', False)
        # Processing the call keyword arguments (line 340)
        kwargs_37780 = {}
        # Getting the type of 'self' (line 340)
        self_37777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'self', False)
        # Obtaining the member 'format_metadata' of a type (line 340)
        format_metadata_37778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 15), self_37777, 'format_metadata')
        # Calling format_metadata(args, kwargs) (line 340)
        format_metadata_call_result_37781 = invoke(stypy.reporting.localization.Localization(__file__, 340, 15), format_metadata_37778, *[dist_37779], **kwargs_37780)
        
        # Assigning a type to the variable 'meta' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'meta', format_metadata_call_result_37781)
        
        # Call to assertIn(...): (line 341)
        # Processing the call arguments (line 341)
        str_37784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 22), 'str', 'Metadata-Version: 1.1')
        # Getting the type of 'meta' (line 341)
        meta_37785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 47), 'meta', False)
        # Processing the call keyword arguments (line 341)
        kwargs_37786 = {}
        # Getting the type of 'self' (line 341)
        self_37782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 341)
        assertIn_37783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_37782, 'assertIn')
        # Calling assertIn(args, kwargs) (line 341)
        assertIn_call_result_37787 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), assertIn_37783, *[str_37784, meta_37785], **kwargs_37786)
        
        
        # Call to assertNotIn(...): (line 342)
        # Processing the call arguments (line 342)
        str_37790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 25), 'str', 'provides:')
        
        # Call to lower(...): (line 342)
        # Processing the call keyword arguments (line 342)
        kwargs_37793 = {}
        # Getting the type of 'meta' (line 342)
        meta_37791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 38), 'meta', False)
        # Obtaining the member 'lower' of a type (line 342)
        lower_37792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 38), meta_37791, 'lower')
        # Calling lower(args, kwargs) (line 342)
        lower_call_result_37794 = invoke(stypy.reporting.localization.Localization(__file__, 342, 38), lower_37792, *[], **kwargs_37793)
        
        # Processing the call keyword arguments (line 342)
        kwargs_37795 = {}
        # Getting the type of 'self' (line 342)
        self_37788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 342)
        assertNotIn_37789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_37788, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 342)
        assertNotIn_call_result_37796 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), assertNotIn_37789, *[str_37790, lower_call_result_37794], **kwargs_37795)
        
        
        # Call to assertNotIn(...): (line 343)
        # Processing the call arguments (line 343)
        str_37799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 25), 'str', 'requires:')
        
        # Call to lower(...): (line 343)
        # Processing the call keyword arguments (line 343)
        kwargs_37802 = {}
        # Getting the type of 'meta' (line 343)
        meta_37800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 38), 'meta', False)
        # Obtaining the member 'lower' of a type (line 343)
        lower_37801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 38), meta_37800, 'lower')
        # Calling lower(args, kwargs) (line 343)
        lower_call_result_37803 = invoke(stypy.reporting.localization.Localization(__file__, 343, 38), lower_37801, *[], **kwargs_37802)
        
        # Processing the call keyword arguments (line 343)
        kwargs_37804 = {}
        # Getting the type of 'self' (line 343)
        self_37797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self', False)
        # Obtaining the member 'assertNotIn' of a type (line 343)
        assertNotIn_37798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_37797, 'assertNotIn')
        # Calling assertNotIn(args, kwargs) (line 343)
        assertNotIn_call_result_37805 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), assertNotIn_37798, *[str_37799, lower_call_result_37803], **kwargs_37804)
        
        
        # Call to assertIn(...): (line 344)
        # Processing the call arguments (line 344)
        str_37808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 22), 'str', 'Obsoletes: other')
        # Getting the type of 'meta' (line 344)
        meta_37809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 42), 'meta', False)
        # Processing the call keyword arguments (line 344)
        kwargs_37810 = {}
        # Getting the type of 'self' (line 344)
        self_37806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 344)
        assertIn_37807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), self_37806, 'assertIn')
        # Calling assertIn(args, kwargs) (line 344)
        assertIn_call_result_37811 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), assertIn_37807, *[str_37808, meta_37809], **kwargs_37810)
        
        
        # Call to assertIn(...): (line 345)
        # Processing the call arguments (line 345)
        str_37814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 22), 'str', 'Obsoletes: another (<1.0)')
        # Getting the type of 'meta' (line 345)
        meta_37815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 51), 'meta', False)
        # Processing the call keyword arguments (line 345)
        kwargs_37816 = {}
        # Getting the type of 'self' (line 345)
        self_37812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 345)
        assertIn_37813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), self_37812, 'assertIn')
        # Calling assertIn(args, kwargs) (line 345)
        assertIn_call_result_37817 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assertIn_37813, *[str_37814, meta_37815], **kwargs_37816)
        
        
        # ################# End of 'test_obsoletes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_obsoletes' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_37818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37818)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_obsoletes'
        return stypy_return_type_37818


    @norecursion
    def test_obsoletes_illegal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_obsoletes_illegal'
        module_type_store = module_type_store.open_function_context('test_obsoletes_illegal', 347, 4, False)
        # Assigning a type to the variable 'self' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_obsoletes_illegal')
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_obsoletes_illegal.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_obsoletes_illegal', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_obsoletes_illegal', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_obsoletes_illegal(...)' code ##################

        
        # Call to assertRaises(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'ValueError' (line 348)
        ValueError_37821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'ValueError', False)
        # Getting the type of 'Distribution' (line 348)
        Distribution_37822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 38), 'Distribution', False)
        
        # Obtaining an instance of the builtin type 'dict' (line 349)
        dict_37823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 26), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 349)
        # Adding element type (key, value) (line 349)
        str_37824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 27), 'str', 'name')
        str_37825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 35), 'str', 'package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 26), dict_37823, (str_37824, str_37825))
        # Adding element type (key, value) (line 349)
        str_37826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 27), 'str', 'version')
        str_37827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 38), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 26), dict_37823, (str_37826, str_37827))
        # Adding element type (key, value) (line 349)
        str_37828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 27), 'str', 'obsoletes')
        
        # Obtaining an instance of the builtin type 'list' (line 351)
        list_37829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 351)
        # Adding element type (line 351)
        str_37830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 41), 'str', 'my.pkg (splat)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 40), list_37829, str_37830)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 26), dict_37823, (str_37828, list_37829))
        
        # Processing the call keyword arguments (line 348)
        kwargs_37831 = {}
        # Getting the type of 'self' (line 348)
        self_37819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 348)
        assertRaises_37820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), self_37819, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 348)
        assertRaises_call_result_37832 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), assertRaises_37820, *[ValueError_37821, Distribution_37822, dict_37823], **kwargs_37831)
        
        
        # ################# End of 'test_obsoletes_illegal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_obsoletes_illegal' in the type store
        # Getting the type of 'stypy_return_type' (line 347)
        stypy_return_type_37833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37833)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_obsoletes_illegal'
        return stypy_return_type_37833


    @norecursion
    def format_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'format_metadata'
        module_type_store = module_type_store.open_function_context('format_metadata', 353, 4, False)
        # Assigning a type to the variable 'self' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.format_metadata')
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_param_names_list', ['dist'])
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.format_metadata.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.format_metadata', ['dist'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'format_metadata', localization, ['dist'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'format_metadata(...)' code ##################

        
        # Assigning a Call to a Name (line 354):
        
        # Call to StringIO(...): (line 354)
        # Processing the call keyword arguments (line 354)
        kwargs_37836 = {}
        # Getting the type of 'StringIO' (line 354)
        StringIO_37834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 14), 'StringIO', False)
        # Obtaining the member 'StringIO' of a type (line 354)
        StringIO_37835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 14), StringIO_37834, 'StringIO')
        # Calling StringIO(args, kwargs) (line 354)
        StringIO_call_result_37837 = invoke(stypy.reporting.localization.Localization(__file__, 354, 14), StringIO_37835, *[], **kwargs_37836)
        
        # Assigning a type to the variable 'sio' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'sio', StringIO_call_result_37837)
        
        # Call to write_pkg_file(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'sio' (line 355)
        sio_37841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 37), 'sio', False)
        # Processing the call keyword arguments (line 355)
        kwargs_37842 = {}
        # Getting the type of 'dist' (line 355)
        dist_37838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'dist', False)
        # Obtaining the member 'metadata' of a type (line 355)
        metadata_37839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), dist_37838, 'metadata')
        # Obtaining the member 'write_pkg_file' of a type (line 355)
        write_pkg_file_37840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), metadata_37839, 'write_pkg_file')
        # Calling write_pkg_file(args, kwargs) (line 355)
        write_pkg_file_call_result_37843 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), write_pkg_file_37840, *[sio_37841], **kwargs_37842)
        
        
        # Call to getvalue(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_37846 = {}
        # Getting the type of 'sio' (line 356)
        sio_37844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'sio', False)
        # Obtaining the member 'getvalue' of a type (line 356)
        getvalue_37845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 15), sio_37844, 'getvalue')
        # Calling getvalue(args, kwargs) (line 356)
        getvalue_call_result_37847 = invoke(stypy.reporting.localization.Localization(__file__, 356, 15), getvalue_37845, *[], **kwargs_37846)
        
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'stypy_return_type', getvalue_call_result_37847)
        
        # ################# End of 'format_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 353)
        stypy_return_type_37848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format_metadata'
        return stypy_return_type_37848


    @norecursion
    def test_custom_pydistutils(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_custom_pydistutils'
        module_type_store = module_type_store.open_function_context('test_custom_pydistutils', 358, 4, False)
        # Assigning a type to the variable 'self' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_custom_pydistutils')
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_custom_pydistutils.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_custom_pydistutils', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_custom_pydistutils', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_custom_pydistutils(...)' code ##################

        
        
        # Getting the type of 'os' (line 361)
        os_37849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 11), 'os')
        # Obtaining the member 'name' of a type (line 361)
        name_37850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 11), os_37849, 'name')
        str_37851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 22), 'str', 'posix')
        # Applying the binary operator '==' (line 361)
        result_eq_37852 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 11), '==', name_37850, str_37851)
        
        # Testing the type of an if condition (line 361)
        if_condition_37853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 8), result_eq_37852)
        # Assigning a type to the variable 'if_condition_37853' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'if_condition_37853', if_condition_37853)
        # SSA begins for if statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 362):
        str_37854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 28), 'str', '.pydistutils.cfg')
        # Assigning a type to the variable 'user_filename' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'user_filename', str_37854)
        # SSA branch for the else part of an if statement (line 361)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 364):
        str_37855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 28), 'str', 'pydistutils.cfg')
        # Assigning a type to the variable 'user_filename' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'user_filename', str_37855)
        # SSA join for if statement (line 361)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 366):
        
        # Call to mkdtemp(...): (line 366)
        # Processing the call keyword arguments (line 366)
        kwargs_37858 = {}
        # Getting the type of 'self' (line 366)
        self_37856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 19), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 366)
        mkdtemp_37857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 19), self_37856, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 366)
        mkdtemp_call_result_37859 = invoke(stypy.reporting.localization.Localization(__file__, 366, 19), mkdtemp_37857, *[], **kwargs_37858)
        
        # Assigning a type to the variable 'temp_dir' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'temp_dir', mkdtemp_call_result_37859)
        
        # Assigning a Call to a Name (line 367):
        
        # Call to join(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'temp_dir' (line 367)
        temp_dir_37863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 37), 'temp_dir', False)
        # Getting the type of 'user_filename' (line 367)
        user_filename_37864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 47), 'user_filename', False)
        # Processing the call keyword arguments (line 367)
        kwargs_37865 = {}
        # Getting the type of 'os' (line 367)
        os_37860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 367)
        path_37861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 24), os_37860, 'path')
        # Obtaining the member 'join' of a type (line 367)
        join_37862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 24), path_37861, 'join')
        # Calling join(args, kwargs) (line 367)
        join_call_result_37866 = invoke(stypy.reporting.localization.Localization(__file__, 367, 24), join_37862, *[temp_dir_37863, user_filename_37864], **kwargs_37865)
        
        # Assigning a type to the variable 'user_filename' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'user_filename', join_call_result_37866)
        
        # Assigning a Call to a Name (line 368):
        
        # Call to open(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'user_filename' (line 368)
        user_filename_37868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 17), 'user_filename', False)
        str_37869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 32), 'str', 'w')
        # Processing the call keyword arguments (line 368)
        kwargs_37870 = {}
        # Getting the type of 'open' (line 368)
        open_37867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'open', False)
        # Calling open(args, kwargs) (line 368)
        open_call_result_37871 = invoke(stypy.reporting.localization.Localization(__file__, 368, 12), open_37867, *[user_filename_37868, str_37869], **kwargs_37870)
        
        # Assigning a type to the variable 'f' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'f', open_call_result_37871)
        
        # Try-finally block (line 369)
        
        # Call to write(...): (line 370)
        # Processing the call arguments (line 370)
        str_37874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 20), 'str', '.')
        # Processing the call keyword arguments (line 370)
        kwargs_37875 = {}
        # Getting the type of 'f' (line 370)
        f_37872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 370)
        write_37873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), f_37872, 'write')
        # Calling write(args, kwargs) (line 370)
        write_call_result_37876 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), write_37873, *[str_37874], **kwargs_37875)
        
        
        # finally branch of the try-finally block (line 369)
        
        # Call to close(...): (line 372)
        # Processing the call keyword arguments (line 372)
        kwargs_37879 = {}
        # Getting the type of 'f' (line 372)
        f_37877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 372)
        close_37878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), f_37877, 'close')
        # Calling close(args, kwargs) (line 372)
        close_call_result_37880 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), close_37878, *[], **kwargs_37879)
        
        
        
        # Try-finally block (line 374)
        
        # Assigning a Call to a Name (line 375):
        
        # Call to Distribution(...): (line 375)
        # Processing the call keyword arguments (line 375)
        kwargs_37882 = {}
        # Getting the type of 'Distribution' (line 375)
        Distribution_37881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 19), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 375)
        Distribution_call_result_37883 = invoke(stypy.reporting.localization.Localization(__file__, 375, 19), Distribution_37881, *[], **kwargs_37882)
        
        # Assigning a type to the variable 'dist' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'dist', Distribution_call_result_37883)
        
        
        # Getting the type of 'sys' (line 378)
        sys_37884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 378)
        platform_37885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 15), sys_37884, 'platform')
        
        # Obtaining an instance of the builtin type 'tuple' (line 378)
        tuple_37886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 378)
        # Adding element type (line 378)
        str_37887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 32), 'str', 'linux')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 32), tuple_37886, str_37887)
        # Adding element type (line 378)
        str_37888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 41), 'str', 'darwin')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 32), tuple_37886, str_37888)
        
        # Applying the binary operator 'in' (line 378)
        result_contains_37889 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 15), 'in', platform_37885, tuple_37886)
        
        # Testing the type of an if condition (line 378)
        if_condition_37890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 12), result_contains_37889)
        # Assigning a type to the variable 'if_condition_37890' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'if_condition_37890', if_condition_37890)
        # SSA begins for if statement (line 378)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 379):
        # Getting the type of 'temp_dir' (line 379)
        temp_dir_37891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 37), 'temp_dir')
        # Getting the type of 'os' (line 379)
        os_37892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 16), 'os')
        # Obtaining the member 'environ' of a type (line 379)
        environ_37893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 16), os_37892, 'environ')
        str_37894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 27), 'str', 'HOME')
        # Storing an element on a container (line 379)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 16), environ_37893, (str_37894, temp_dir_37891))
        
        # Assigning a Call to a Name (line 380):
        
        # Call to find_config_files(...): (line 380)
        # Processing the call keyword arguments (line 380)
        kwargs_37897 = {}
        # Getting the type of 'dist' (line 380)
        dist_37895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'dist', False)
        # Obtaining the member 'find_config_files' of a type (line 380)
        find_config_files_37896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 24), dist_37895, 'find_config_files')
        # Calling find_config_files(args, kwargs) (line 380)
        find_config_files_call_result_37898 = invoke(stypy.reporting.localization.Localization(__file__, 380, 24), find_config_files_37896, *[], **kwargs_37897)
        
        # Assigning a type to the variable 'files' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'files', find_config_files_call_result_37898)
        
        # Call to assertIn(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'user_filename' (line 381)
        user_filename_37901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 30), 'user_filename', False)
        # Getting the type of 'files' (line 381)
        files_37902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 45), 'files', False)
        # Processing the call keyword arguments (line 381)
        kwargs_37903 = {}
        # Getting the type of 'self' (line 381)
        self_37899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 381)
        assertIn_37900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 16), self_37899, 'assertIn')
        # Calling assertIn(args, kwargs) (line 381)
        assertIn_call_result_37904 = invoke(stypy.reporting.localization.Localization(__file__, 381, 16), assertIn_37900, *[user_filename_37901, files_37902], **kwargs_37903)
        
        # SSA join for if statement (line 378)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'sys' (line 384)
        sys_37905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 384)
        platform_37906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 15), sys_37905, 'platform')
        str_37907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 31), 'str', 'win32')
        # Applying the binary operator '==' (line 384)
        result_eq_37908 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 15), '==', platform_37906, str_37907)
        
        # Testing the type of an if condition (line 384)
        if_condition_37909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 12), result_eq_37908)
        # Assigning a type to the variable 'if_condition_37909' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'if_condition_37909', if_condition_37909)
        # SSA begins for if statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 386):
        # Getting the type of 'temp_dir' (line 386)
        temp_dir_37910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 37), 'temp_dir')
        # Getting the type of 'os' (line 386)
        os_37911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'os')
        # Obtaining the member 'environ' of a type (line 386)
        environ_37912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 16), os_37911, 'environ')
        str_37913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 27), 'str', 'HOME')
        # Storing an element on a container (line 386)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 16), environ_37912, (str_37913, temp_dir_37910))
        
        # Assigning a Call to a Name (line 387):
        
        # Call to find_config_files(...): (line 387)
        # Processing the call keyword arguments (line 387)
        kwargs_37916 = {}
        # Getting the type of 'dist' (line 387)
        dist_37914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 24), 'dist', False)
        # Obtaining the member 'find_config_files' of a type (line 387)
        find_config_files_37915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 24), dist_37914, 'find_config_files')
        # Calling find_config_files(args, kwargs) (line 387)
        find_config_files_call_result_37917 = invoke(stypy.reporting.localization.Localization(__file__, 387, 24), find_config_files_37915, *[], **kwargs_37916)
        
        # Assigning a type to the variable 'files' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'files', find_config_files_call_result_37917)
        
        # Call to assertIn(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'user_filename' (line 388)
        user_filename_37920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 30), 'user_filename', False)
        # Getting the type of 'files' (line 388)
        files_37921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 45), 'files', False)
        str_37922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 29), 'str', '%r not found in %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 389)
        tuple_37923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 389)
        # Adding element type (line 389)
        # Getting the type of 'user_filename' (line 389)
        user_filename_37924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 53), 'user_filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 53), tuple_37923, user_filename_37924)
        # Adding element type (line 389)
        # Getting the type of 'files' (line 389)
        files_37925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 68), 'files', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 53), tuple_37923, files_37925)
        
        # Applying the binary operator '%' (line 389)
        result_mod_37926 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 29), '%', str_37922, tuple_37923)
        
        # Processing the call keyword arguments (line 388)
        kwargs_37927 = {}
        # Getting the type of 'self' (line 388)
        self_37918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 388)
        assertIn_37919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 16), self_37918, 'assertIn')
        # Calling assertIn(args, kwargs) (line 388)
        assertIn_call_result_37928 = invoke(stypy.reporting.localization.Localization(__file__, 388, 16), assertIn_37919, *[user_filename_37920, files_37921, result_mod_37926], **kwargs_37927)
        
        # SSA join for if statement (line 384)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 374)
        
        # Call to remove(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'user_filename' (line 391)
        user_filename_37931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 22), 'user_filename', False)
        # Processing the call keyword arguments (line 391)
        kwargs_37932 = {}
        # Getting the type of 'os' (line 391)
        os_37929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'os', False)
        # Obtaining the member 'remove' of a type (line 391)
        remove_37930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 12), os_37929, 'remove')
        # Calling remove(args, kwargs) (line 391)
        remove_call_result_37933 = invoke(stypy.reporting.localization.Localization(__file__, 391, 12), remove_37930, *[user_filename_37931], **kwargs_37932)
        
        
        
        # ################# End of 'test_custom_pydistutils(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_custom_pydistutils' in the type store
        # Getting the type of 'stypy_return_type' (line 358)
        stypy_return_type_37934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37934)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_custom_pydistutils'
        return stypy_return_type_37934


    @norecursion
    def test_fix_help_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fix_help_options'
        module_type_store = module_type_store.open_function_context('test_fix_help_options', 393, 4, False)
        # Assigning a type to the variable 'self' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_fix_help_options')
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_fix_help_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_fix_help_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fix_help_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fix_help_options(...)' code ##################

        
        # Assigning a List to a Name (line 394):
        
        # Obtaining an instance of the builtin type 'list' (line 394)
        list_37935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 394)
        # Adding element type (line 394)
        
        # Obtaining an instance of the builtin type 'tuple' (line 394)
        tuple_37936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 394)
        # Adding element type (line 394)
        str_37937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 24), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 24), tuple_37936, str_37937)
        # Adding element type (line 394)
        str_37938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 29), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 24), tuple_37936, str_37938)
        # Adding element type (line 394)
        str_37939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 34), 'str', 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 24), tuple_37936, str_37939)
        # Adding element type (line 394)
        str_37940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 39), 'str', 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 24), tuple_37936, str_37940)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 22), list_37935, tuple_37936)
        # Adding element type (line 394)
        
        # Obtaining an instance of the builtin type 'tuple' (line 394)
        tuple_37941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 394)
        # Adding element type (line 394)
        int_37942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 46), tuple_37941, int_37942)
        # Adding element type (line 394)
        int_37943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 46), tuple_37941, int_37943)
        # Adding element type (line 394)
        int_37944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 46), tuple_37941, int_37944)
        # Adding element type (line 394)
        int_37945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 46), tuple_37941, int_37945)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 22), list_37935, tuple_37941)
        
        # Assigning a type to the variable 'help_tuples' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'help_tuples', list_37935)
        
        # Assigning a Call to a Name (line 395):
        
        # Call to fix_help_options(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'help_tuples' (line 395)
        help_tuples_37947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 41), 'help_tuples', False)
        # Processing the call keyword arguments (line 395)
        kwargs_37948 = {}
        # Getting the type of 'fix_help_options' (line 395)
        fix_help_options_37946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 24), 'fix_help_options', False)
        # Calling fix_help_options(args, kwargs) (line 395)
        fix_help_options_call_result_37949 = invoke(stypy.reporting.localization.Localization(__file__, 395, 24), fix_help_options_37946, *[help_tuples_37947], **kwargs_37948)
        
        # Assigning a type to the variable 'fancy_options' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'fancy_options', fix_help_options_call_result_37949)
        
        # Call to assertEqual(...): (line 396)
        # Processing the call arguments (line 396)
        
        # Obtaining the type of the subscript
        int_37952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 39), 'int')
        # Getting the type of 'fancy_options' (line 396)
        fancy_options_37953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 25), 'fancy_options', False)
        # Obtaining the member '__getitem__' of a type (line 396)
        getitem___37954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 25), fancy_options_37953, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 396)
        subscript_call_result_37955 = invoke(stypy.reporting.localization.Localization(__file__, 396, 25), getitem___37954, int_37952)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 396)
        tuple_37956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 396)
        # Adding element type (line 396)
        str_37957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 44), 'str', 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 44), tuple_37956, str_37957)
        # Adding element type (line 396)
        str_37958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 49), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 44), tuple_37956, str_37958)
        # Adding element type (line 396)
        str_37959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 54), 'str', 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 44), tuple_37956, str_37959)
        
        # Processing the call keyword arguments (line 396)
        kwargs_37960 = {}
        # Getting the type of 'self' (line 396)
        self_37950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 396)
        assertEqual_37951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_37950, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 396)
        assertEqual_call_result_37961 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), assertEqual_37951, *[subscript_call_result_37955, tuple_37956], **kwargs_37960)
        
        
        # Call to assertEqual(...): (line 397)
        # Processing the call arguments (line 397)
        
        # Obtaining the type of the subscript
        int_37964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 39), 'int')
        # Getting the type of 'fancy_options' (line 397)
        fancy_options_37965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 25), 'fancy_options', False)
        # Obtaining the member '__getitem__' of a type (line 397)
        getitem___37966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 25), fancy_options_37965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 397)
        subscript_call_result_37967 = invoke(stypy.reporting.localization.Localization(__file__, 397, 25), getitem___37966, int_37964)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 397)
        tuple_37968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 397)
        # Adding element type (line 397)
        int_37969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 44), tuple_37968, int_37969)
        # Adding element type (line 397)
        int_37970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 44), tuple_37968, int_37970)
        # Adding element type (line 397)
        int_37971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 44), tuple_37968, int_37971)
        
        # Processing the call keyword arguments (line 397)
        kwargs_37972 = {}
        # Getting the type of 'self' (line 397)
        self_37962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 397)
        assertEqual_37963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_37962, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 397)
        assertEqual_call_result_37973 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), assertEqual_37963, *[subscript_call_result_37967, tuple_37968], **kwargs_37972)
        
        
        # ################# End of 'test_fix_help_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fix_help_options' in the type store
        # Getting the type of 'stypy_return_type' (line 393)
        stypy_return_type_37974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_37974)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fix_help_options'
        return stypy_return_type_37974


    @norecursion
    def test_show_help(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_show_help'
        module_type_store = module_type_store.open_function_context('test_show_help', 399, 4, False)
        # Assigning a type to the variable 'self' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_show_help')
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_show_help.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_show_help', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_show_help', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_show_help(...)' code ##################

        
        # Call to addCleanup(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'log' (line 401)
        log_37977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 24), 'log', False)
        # Obtaining the member 'set_threshold' of a type (line 401)
        set_threshold_37978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 24), log_37977, 'set_threshold')
        # Getting the type of 'log' (line 401)
        log_37979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 43), 'log', False)
        # Obtaining the member '_global_log' of a type (line 401)
        _global_log_37980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 43), log_37979, '_global_log')
        # Obtaining the member 'threshold' of a type (line 401)
        threshold_37981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 43), _global_log_37980, 'threshold')
        # Processing the call keyword arguments (line 401)
        kwargs_37982 = {}
        # Getting the type of 'self' (line 401)
        self_37975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 401)
        addCleanup_37976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), self_37975, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 401)
        addCleanup_call_result_37983 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), addCleanup_37976, *[set_threshold_37978, threshold_37981], **kwargs_37982)
        
        
        # Assigning a Call to a Name (line 402):
        
        # Call to Distribution(...): (line 402)
        # Processing the call keyword arguments (line 402)
        kwargs_37985 = {}
        # Getting the type of 'Distribution' (line 402)
        Distribution_37984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 402)
        Distribution_call_result_37986 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), Distribution_37984, *[], **kwargs_37985)
        
        # Assigning a type to the variable 'dist' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'dist', Distribution_call_result_37986)
        
        # Assigning a List to a Attribute (line 403):
        
        # Obtaining an instance of the builtin type 'list' (line 403)
        list_37987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 403)
        
        # Getting the type of 'sys' (line 403)
        sys_37988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'sys')
        # Setting the type of the member 'argv' of a type (line 403)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), sys_37988, 'argv', list_37987)
        
        # Assigning a Num to a Attribute (line 404):
        int_37989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 20), 'int')
        # Getting the type of 'dist' (line 404)
        dist_37990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'dist')
        # Setting the type of the member 'help' of a type (line 404)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), dist_37990, 'help', int_37989)
        
        # Assigning a Str to a Attribute (line 405):
        str_37991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 27), 'str', 'setup.py')
        # Getting the type of 'dist' (line 405)
        dist_37992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'dist')
        # Setting the type of the member 'script_name' of a type (line 405)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), dist_37992, 'script_name', str_37991)
        
        # Call to captured_stdout(...): (line 406)
        # Processing the call keyword arguments (line 406)
        kwargs_37994 = {}
        # Getting the type of 'captured_stdout' (line 406)
        captured_stdout_37993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 13), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 406)
        captured_stdout_call_result_37995 = invoke(stypy.reporting.localization.Localization(__file__, 406, 13), captured_stdout_37993, *[], **kwargs_37994)
        
        with_37996 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 406, 13), captured_stdout_call_result_37995, 'with parameter', '__enter__', '__exit__')

        if with_37996:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 406)
            enter___37997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 13), captured_stdout_call_result_37995, '__enter__')
            with_enter_37998 = invoke(stypy.reporting.localization.Localization(__file__, 406, 13), enter___37997)
            # Assigning a type to the variable 's' (line 406)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 13), 's', with_enter_37998)
            
            # Call to parse_command_line(...): (line 407)
            # Processing the call keyword arguments (line 407)
            kwargs_38001 = {}
            # Getting the type of 'dist' (line 407)
            dist_37999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'dist', False)
            # Obtaining the member 'parse_command_line' of a type (line 407)
            parse_command_line_38000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 12), dist_37999, 'parse_command_line')
            # Calling parse_command_line(args, kwargs) (line 407)
            parse_command_line_call_result_38002 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), parse_command_line_38000, *[], **kwargs_38001)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 406)
            exit___38003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 13), captured_stdout_call_result_37995, '__exit__')
            with_exit_38004 = invoke(stypy.reporting.localization.Localization(__file__, 406, 13), exit___38003, None, None, None)

        
        # Assigning a ListComp to a Name (line 409):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 409)
        # Processing the call arguments (line 409)
        str_38017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 54), 'str', '\n')
        # Processing the call keyword arguments (line 409)
        kwargs_38018 = {}
        
        # Call to getvalue(...): (line 409)
        # Processing the call keyword arguments (line 409)
        kwargs_38014 = {}
        # Getting the type of 's' (line 409)
        s_38012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 35), 's', False)
        # Obtaining the member 'getvalue' of a type (line 409)
        getvalue_38013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 35), s_38012, 'getvalue')
        # Calling getvalue(args, kwargs) (line 409)
        getvalue_call_result_38015 = invoke(stypy.reporting.localization.Localization(__file__, 409, 35), getvalue_38013, *[], **kwargs_38014)
        
        # Obtaining the member 'split' of a type (line 409)
        split_38016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 35), getvalue_call_result_38015, 'split')
        # Calling split(args, kwargs) (line 409)
        split_call_result_38019 = invoke(stypy.reporting.localization.Localization(__file__, 409, 35), split_38016, *[str_38017], **kwargs_38018)
        
        comprehension_38020 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 18), split_call_result_38019)
        # Assigning a type to the variable 'line' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 18), 'line', comprehension_38020)
        
        
        # Call to strip(...): (line 410)
        # Processing the call keyword arguments (line 410)
        kwargs_38008 = {}
        # Getting the type of 'line' (line 410)
        line_38006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 21), 'line', False)
        # Obtaining the member 'strip' of a type (line 410)
        strip_38007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 21), line_38006, 'strip')
        # Calling strip(args, kwargs) (line 410)
        strip_call_result_38009 = invoke(stypy.reporting.localization.Localization(__file__, 410, 21), strip_38007, *[], **kwargs_38008)
        
        str_38010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 37), 'str', '')
        # Applying the binary operator '!=' (line 410)
        result_ne_38011 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 21), '!=', strip_call_result_38009, str_38010)
        
        # Getting the type of 'line' (line 409)
        line_38005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 18), 'line')
        list_38021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 18), list_38021, line_38005)
        # Assigning a type to the variable 'output' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'output', list_38021)
        
        # Call to assertTrue(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'output' (line 411)
        output_38024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 24), 'output', False)
        # Processing the call keyword arguments (line 411)
        kwargs_38025 = {}
        # Getting the type of 'self' (line 411)
        self_38022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 411)
        assertTrue_38023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_38022, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 411)
        assertTrue_call_result_38026 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), assertTrue_38023, *[output_38024], **kwargs_38025)
        
        
        # ################# End of 'test_show_help(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_show_help' in the type store
        # Getting the type of 'stypy_return_type' (line 399)
        stypy_return_type_38027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_show_help'
        return stypy_return_type_38027


    @norecursion
    def test_read_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_read_metadata'
        module_type_store = module_type_store.open_function_context('test_read_metadata', 413, 4, False)
        # Assigning a type to the variable 'self' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_localization', localization)
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_function_name', 'MetadataTestCase.test_read_metadata')
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_param_names_list', [])
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MetadataTestCase.test_read_metadata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.test_read_metadata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_read_metadata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_read_metadata(...)' code ##################

        
        # Assigning a Dict to a Name (line 414):
        
        # Obtaining an instance of the builtin type 'dict' (line 414)
        dict_38028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 414)
        # Adding element type (key, value) (line 414)
        str_38029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 17), 'str', 'name')
        str_38030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 25), 'str', 'package')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 16), dict_38028, (str_38029, str_38030))
        # Adding element type (key, value) (line 414)
        str_38031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 17), 'str', 'version')
        str_38032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 28), 'str', '1.0')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 16), dict_38028, (str_38031, str_38032))
        # Adding element type (key, value) (line 414)
        str_38033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 17), 'str', 'long_description')
        str_38034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 37), 'str', 'desc')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 16), dict_38028, (str_38033, str_38034))
        # Adding element type (key, value) (line 414)
        str_38035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 17), 'str', 'description')
        str_38036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 32), 'str', 'xxx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 16), dict_38028, (str_38035, str_38036))
        # Adding element type (key, value) (line 414)
        str_38037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 17), 'str', 'download_url')
        str_38038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 33), 'str', 'http://example.com')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 16), dict_38028, (str_38037, str_38038))
        # Adding element type (key, value) (line 414)
        str_38039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 17), 'str', 'keywords')
        
        # Obtaining an instance of the builtin type 'list' (line 419)
        list_38040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 419)
        # Adding element type (line 419)
        str_38041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 30), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 29), list_38040, str_38041)
        # Adding element type (line 419)
        str_38042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 37), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 29), list_38040, str_38042)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 16), dict_38028, (str_38039, list_38040))
        # Adding element type (key, value) (line 414)
        str_38043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 17), 'str', 'requires')
        
        # Obtaining an instance of the builtin type 'list' (line 420)
        list_38044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 420)
        # Adding element type (line 420)
        str_38045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 30), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 29), list_38044, str_38045)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 16), dict_38028, (str_38043, list_38044))
        
        # Assigning a type to the variable 'attrs' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'attrs', dict_38028)
        
        # Assigning a Call to a Name (line 422):
        
        # Call to Distribution(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'attrs' (line 422)
        attrs_38047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 28), 'attrs', False)
        # Processing the call keyword arguments (line 422)
        kwargs_38048 = {}
        # Getting the type of 'Distribution' (line 422)
        Distribution_38046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 422)
        Distribution_call_result_38049 = invoke(stypy.reporting.localization.Localization(__file__, 422, 15), Distribution_38046, *[attrs_38047], **kwargs_38048)
        
        # Assigning a type to the variable 'dist' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'dist', Distribution_call_result_38049)
        
        # Assigning a Attribute to a Name (line 423):
        # Getting the type of 'dist' (line 423)
        dist_38050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 19), 'dist')
        # Obtaining the member 'metadata' of a type (line 423)
        metadata_38051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 19), dist_38050, 'metadata')
        # Assigning a type to the variable 'metadata' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'metadata', metadata_38051)
        
        # Assigning a Call to a Name (line 426):
        
        # Call to StringIO(...): (line 426)
        # Processing the call keyword arguments (line 426)
        kwargs_38054 = {}
        # Getting the type of 'StringIO' (line 426)
        StringIO_38052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 19), 'StringIO', False)
        # Obtaining the member 'StringIO' of a type (line 426)
        StringIO_38053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 19), StringIO_38052, 'StringIO')
        # Calling StringIO(args, kwargs) (line 426)
        StringIO_call_result_38055 = invoke(stypy.reporting.localization.Localization(__file__, 426, 19), StringIO_38053, *[], **kwargs_38054)
        
        # Assigning a type to the variable 'PKG_INFO' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'PKG_INFO', StringIO_call_result_38055)
        
        # Call to write_pkg_file(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'PKG_INFO' (line 427)
        PKG_INFO_38058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 32), 'PKG_INFO', False)
        # Processing the call keyword arguments (line 427)
        kwargs_38059 = {}
        # Getting the type of 'metadata' (line 427)
        metadata_38056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'metadata', False)
        # Obtaining the member 'write_pkg_file' of a type (line 427)
        write_pkg_file_38057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), metadata_38056, 'write_pkg_file')
        # Calling write_pkg_file(args, kwargs) (line 427)
        write_pkg_file_call_result_38060 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), write_pkg_file_38057, *[PKG_INFO_38058], **kwargs_38059)
        
        
        # Call to seek(...): (line 428)
        # Processing the call arguments (line 428)
        int_38063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 22), 'int')
        # Processing the call keyword arguments (line 428)
        kwargs_38064 = {}
        # Getting the type of 'PKG_INFO' (line 428)
        PKG_INFO_38061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'PKG_INFO', False)
        # Obtaining the member 'seek' of a type (line 428)
        seek_38062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), PKG_INFO_38061, 'seek')
        # Calling seek(args, kwargs) (line 428)
        seek_call_result_38065 = invoke(stypy.reporting.localization.Localization(__file__, 428, 8), seek_38062, *[int_38063], **kwargs_38064)
        
        
        # Call to read_pkg_file(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'PKG_INFO' (line 429)
        PKG_INFO_38068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 31), 'PKG_INFO', False)
        # Processing the call keyword arguments (line 429)
        kwargs_38069 = {}
        # Getting the type of 'metadata' (line 429)
        metadata_38066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'metadata', False)
        # Obtaining the member 'read_pkg_file' of a type (line 429)
        read_pkg_file_38067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), metadata_38066, 'read_pkg_file')
        # Calling read_pkg_file(args, kwargs) (line 429)
        read_pkg_file_call_result_38070 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), read_pkg_file_38067, *[PKG_INFO_38068], **kwargs_38069)
        
        
        # Call to assertEqual(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'metadata' (line 431)
        metadata_38073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 25), 'metadata', False)
        # Obtaining the member 'name' of a type (line 431)
        name_38074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 25), metadata_38073, 'name')
        str_38075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 40), 'str', 'package')
        # Processing the call keyword arguments (line 431)
        kwargs_38076 = {}
        # Getting the type of 'self' (line 431)
        self_38071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 431)
        assertEqual_38072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), self_38071, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 431)
        assertEqual_call_result_38077 = invoke(stypy.reporting.localization.Localization(__file__, 431, 8), assertEqual_38072, *[name_38074, str_38075], **kwargs_38076)
        
        
        # Call to assertEqual(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'metadata' (line 432)
        metadata_38080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 25), 'metadata', False)
        # Obtaining the member 'version' of a type (line 432)
        version_38081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 25), metadata_38080, 'version')
        str_38082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 43), 'str', '1.0')
        # Processing the call keyword arguments (line 432)
        kwargs_38083 = {}
        # Getting the type of 'self' (line 432)
        self_38078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 432)
        assertEqual_38079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), self_38078, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 432)
        assertEqual_call_result_38084 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), assertEqual_38079, *[version_38081, str_38082], **kwargs_38083)
        
        
        # Call to assertEqual(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'metadata' (line 433)
        metadata_38087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 25), 'metadata', False)
        # Obtaining the member 'description' of a type (line 433)
        description_38088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 25), metadata_38087, 'description')
        str_38089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 47), 'str', 'xxx')
        # Processing the call keyword arguments (line 433)
        kwargs_38090 = {}
        # Getting the type of 'self' (line 433)
        self_38085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 433)
        assertEqual_38086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_38085, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 433)
        assertEqual_call_result_38091 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), assertEqual_38086, *[description_38088, str_38089], **kwargs_38090)
        
        
        # Call to assertEqual(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'metadata' (line 434)
        metadata_38094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 25), 'metadata', False)
        # Obtaining the member 'download_url' of a type (line 434)
        download_url_38095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 25), metadata_38094, 'download_url')
        str_38096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 48), 'str', 'http://example.com')
        # Processing the call keyword arguments (line 434)
        kwargs_38097 = {}
        # Getting the type of 'self' (line 434)
        self_38092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 434)
        assertEqual_38093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), self_38092, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 434)
        assertEqual_call_result_38098 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), assertEqual_38093, *[download_url_38095, str_38096], **kwargs_38097)
        
        
        # Call to assertEqual(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'metadata' (line 435)
        metadata_38101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 25), 'metadata', False)
        # Obtaining the member 'keywords' of a type (line 435)
        keywords_38102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 25), metadata_38101, 'keywords')
        
        # Obtaining an instance of the builtin type 'list' (line 435)
        list_38103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 435)
        # Adding element type (line 435)
        str_38104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 45), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 44), list_38103, str_38104)
        # Adding element type (line 435)
        str_38105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 52), 'str', 'two')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 44), list_38103, str_38105)
        
        # Processing the call keyword arguments (line 435)
        kwargs_38106 = {}
        # Getting the type of 'self' (line 435)
        self_38099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 435)
        assertEqual_38100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 8), self_38099, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 435)
        assertEqual_call_result_38107 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), assertEqual_38100, *[keywords_38102, list_38103], **kwargs_38106)
        
        
        # Call to assertEqual(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'metadata' (line 436)
        metadata_38110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 25), 'metadata', False)
        # Obtaining the member 'platforms' of a type (line 436)
        platforms_38111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 25), metadata_38110, 'platforms')
        
        # Obtaining an instance of the builtin type 'list' (line 436)
        list_38112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 436)
        # Adding element type (line 436)
        str_38113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 46), 'str', 'UNKNOWN')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 45), list_38112, str_38113)
        
        # Processing the call keyword arguments (line 436)
        kwargs_38114 = {}
        # Getting the type of 'self' (line 436)
        self_38108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 436)
        assertEqual_38109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), self_38108, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 436)
        assertEqual_call_result_38115 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), assertEqual_38109, *[platforms_38111, list_38112], **kwargs_38114)
        
        
        # Call to assertEqual(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'metadata' (line 437)
        metadata_38118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 25), 'metadata', False)
        # Obtaining the member 'obsoletes' of a type (line 437)
        obsoletes_38119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 25), metadata_38118, 'obsoletes')
        # Getting the type of 'None' (line 437)
        None_38120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 45), 'None', False)
        # Processing the call keyword arguments (line 437)
        kwargs_38121 = {}
        # Getting the type of 'self' (line 437)
        self_38116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 437)
        assertEqual_38117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), self_38116, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 437)
        assertEqual_call_result_38122 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), assertEqual_38117, *[obsoletes_38119, None_38120], **kwargs_38121)
        
        
        # Call to assertEqual(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'metadata' (line 438)
        metadata_38125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 25), 'metadata', False)
        # Obtaining the member 'requires' of a type (line 438)
        requires_38126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 25), metadata_38125, 'requires')
        
        # Obtaining an instance of the builtin type 'list' (line 438)
        list_38127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 438)
        # Adding element type (line 438)
        str_38128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 45), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 44), list_38127, str_38128)
        
        # Processing the call keyword arguments (line 438)
        kwargs_38129 = {}
        # Getting the type of 'self' (line 438)
        self_38123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 438)
        assertEqual_38124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), self_38123, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 438)
        assertEqual_call_result_38130 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), assertEqual_38124, *[requires_38126, list_38127], **kwargs_38129)
        
        
        # ################# End of 'test_read_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_read_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 413)
        stypy_return_type_38131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38131)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_read_metadata'
        return stypy_return_type_38131


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 238, 0, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MetadataTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MetadataTestCase' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'MetadataTestCase', MetadataTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 441, 0, False)
    
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

    
    # Assigning a Call to a Name (line 442):
    
    # Call to TestSuite(...): (line 442)
    # Processing the call keyword arguments (line 442)
    kwargs_38134 = {}
    # Getting the type of 'unittest' (line 442)
    unittest_38132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'unittest', False)
    # Obtaining the member 'TestSuite' of a type (line 442)
    TestSuite_38133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), unittest_38132, 'TestSuite')
    # Calling TestSuite(args, kwargs) (line 442)
    TestSuite_call_result_38135 = invoke(stypy.reporting.localization.Localization(__file__, 442, 12), TestSuite_38133, *[], **kwargs_38134)
    
    # Assigning a type to the variable 'suite' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'suite', TestSuite_call_result_38135)
    
    # Call to addTest(...): (line 443)
    # Processing the call arguments (line 443)
    
    # Call to makeSuite(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'DistributionTestCase' (line 443)
    DistributionTestCase_38140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 37), 'DistributionTestCase', False)
    # Processing the call keyword arguments (line 443)
    kwargs_38141 = {}
    # Getting the type of 'unittest' (line 443)
    unittest_38138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 18), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 443)
    makeSuite_38139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 18), unittest_38138, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 443)
    makeSuite_call_result_38142 = invoke(stypy.reporting.localization.Localization(__file__, 443, 18), makeSuite_38139, *[DistributionTestCase_38140], **kwargs_38141)
    
    # Processing the call keyword arguments (line 443)
    kwargs_38143 = {}
    # Getting the type of 'suite' (line 443)
    suite_38136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'suite', False)
    # Obtaining the member 'addTest' of a type (line 443)
    addTest_38137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 4), suite_38136, 'addTest')
    # Calling addTest(args, kwargs) (line 443)
    addTest_call_result_38144 = invoke(stypy.reporting.localization.Localization(__file__, 443, 4), addTest_38137, *[makeSuite_call_result_38142], **kwargs_38143)
    
    
    # Call to addTest(...): (line 444)
    # Processing the call arguments (line 444)
    
    # Call to makeSuite(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'MetadataTestCase' (line 444)
    MetadataTestCase_38149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 37), 'MetadataTestCase', False)
    # Processing the call keyword arguments (line 444)
    kwargs_38150 = {}
    # Getting the type of 'unittest' (line 444)
    unittest_38147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 18), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 444)
    makeSuite_38148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 18), unittest_38147, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 444)
    makeSuite_call_result_38151 = invoke(stypy.reporting.localization.Localization(__file__, 444, 18), makeSuite_38148, *[MetadataTestCase_38149], **kwargs_38150)
    
    # Processing the call keyword arguments (line 444)
    kwargs_38152 = {}
    # Getting the type of 'suite' (line 444)
    suite_38145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'suite', False)
    # Obtaining the member 'addTest' of a type (line 444)
    addTest_38146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), suite_38145, 'addTest')
    # Calling addTest(args, kwargs) (line 444)
    addTest_call_result_38153 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), addTest_38146, *[makeSuite_call_result_38151], **kwargs_38152)
    
    # Getting the type of 'suite' (line 445)
    suite_38154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), 'suite')
    # Assigning a type to the variable 'stypy_return_type' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type', suite_38154)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 441)
    stypy_return_type_38155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_38155)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_38155

# Assigning a type to the variable 'test_suite' (line 441)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 448)
    # Processing the call arguments (line 448)
    
    # Call to test_suite(...): (line 448)
    # Processing the call keyword arguments (line 448)
    kwargs_38158 = {}
    # Getting the type of 'test_suite' (line 448)
    test_suite_38157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 448)
    test_suite_call_result_38159 = invoke(stypy.reporting.localization.Localization(__file__, 448, 17), test_suite_38157, *[], **kwargs_38158)
    
    # Processing the call keyword arguments (line 448)
    kwargs_38160 = {}
    # Getting the type of 'run_unittest' (line 448)
    run_unittest_38156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 448)
    run_unittest_call_result_38161 = invoke(stypy.reporting.localization.Localization(__file__, 448, 4), run_unittest_38156, *[test_suite_call_result_38159], **kwargs_38160)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
