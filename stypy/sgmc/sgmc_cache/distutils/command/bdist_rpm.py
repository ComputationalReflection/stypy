
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.bdist_rpm
2: 
3: Implements the Distutils 'bdist_rpm' command (create RPM source and binary
4: distributions).'''
5: 
6: __revision__ = "$Id$"
7: 
8: import sys
9: import os
10: import string
11: 
12: from distutils.core import Command
13: from distutils.debug import DEBUG
14: from distutils.file_util import write_file
15: from distutils.sysconfig import get_python_version
16: from distutils.errors import (DistutilsOptionError, DistutilsPlatformError,
17:                               DistutilsFileError, DistutilsExecError)
18: from distutils import log
19: 
20: class bdist_rpm (Command):
21: 
22:     description = "create an RPM distribution"
23: 
24:     user_options = [
25:         ('bdist-base=', None,
26:          "base directory for creating built distributions"),
27:         ('rpm-base=', None,
28:          "base directory for creating RPMs (defaults to \"rpm\" under "
29:          "--bdist-base; must be specified for RPM 2)"),
30:         ('dist-dir=', 'd',
31:          "directory to put final RPM files in "
32:          "(and .spec files if --spec-only)"),
33:         ('python=', None,
34:          "path to Python interpreter to hard-code in the .spec file "
35:          "(default: \"python\")"),
36:         ('fix-python', None,
37:          "hard-code the exact path to the current Python interpreter in "
38:          "the .spec file"),
39:         ('spec-only', None,
40:          "only regenerate spec file"),
41:         ('source-only', None,
42:          "only generate source RPM"),
43:         ('binary-only', None,
44:          "only generate binary RPM"),
45:         ('use-bzip2', None,
46:          "use bzip2 instead of gzip to create source distribution"),
47: 
48:         # More meta-data: too RPM-specific to put in the setup script,
49:         # but needs to go in the .spec file -- so we make these options
50:         # to "bdist_rpm".  The idea is that packagers would put this
51:         # info in setup.cfg, although they are of course free to
52:         # supply it on the command line.
53:         ('distribution-name=', None,
54:          "name of the (Linux) distribution to which this "
55:          "RPM applies (*not* the name of the module distribution!)"),
56:         ('group=', None,
57:          "package classification [default: \"Development/Libraries\"]"),
58:         ('release=', None,
59:          "RPM release number"),
60:         ('serial=', None,
61:          "RPM serial number"),
62:         ('vendor=', None,
63:          "RPM \"vendor\" (eg. \"Joe Blow <joe@example.com>\") "
64:          "[default: maintainer or author from setup script]"),
65:         ('packager=', None,
66:          "RPM packager (eg. \"Jane Doe <jane@example.net>\")"
67:          "[default: vendor]"),
68:         ('doc-files=', None,
69:          "list of documentation files (space or comma-separated)"),
70:         ('changelog=', None,
71:          "RPM changelog"),
72:         ('icon=', None,
73:          "name of icon file"),
74:         ('provides=', None,
75:          "capabilities provided by this package"),
76:         ('requires=', None,
77:          "capabilities required by this package"),
78:         ('conflicts=', None,
79:          "capabilities which conflict with this package"),
80:         ('build-requires=', None,
81:          "capabilities required to build this package"),
82:         ('obsoletes=', None,
83:          "capabilities made obsolete by this package"),
84:         ('no-autoreq', None,
85:          "do not automatically calculate dependencies"),
86: 
87:         # Actions to take when building RPM
88:         ('keep-temp', 'k',
89:          "don't clean up RPM build directory"),
90:         ('no-keep-temp', None,
91:          "clean up RPM build directory [default]"),
92:         ('use-rpm-opt-flags', None,
93:          "compile with RPM_OPT_FLAGS when building from source RPM"),
94:         ('no-rpm-opt-flags', None,
95:          "do not pass any RPM CFLAGS to compiler"),
96:         ('rpm3-mode', None,
97:          "RPM 3 compatibility mode (default)"),
98:         ('rpm2-mode', None,
99:          "RPM 2 compatibility mode"),
100: 
101:         # Add the hooks necessary for specifying custom scripts
102:         ('prep-script=', None,
103:          "Specify a script for the PREP phase of RPM building"),
104:         ('build-script=', None,
105:          "Specify a script for the BUILD phase of RPM building"),
106: 
107:         ('pre-install=', None,
108:          "Specify a script for the pre-INSTALL phase of RPM building"),
109:         ('install-script=', None,
110:          "Specify a script for the INSTALL phase of RPM building"),
111:         ('post-install=', None,
112:          "Specify a script for the post-INSTALL phase of RPM building"),
113: 
114:         ('pre-uninstall=', None,
115:          "Specify a script for the pre-UNINSTALL phase of RPM building"),
116:         ('post-uninstall=', None,
117:          "Specify a script for the post-UNINSTALL phase of RPM building"),
118: 
119:         ('clean-script=', None,
120:          "Specify a script for the CLEAN phase of RPM building"),
121: 
122:         ('verify-script=', None,
123:          "Specify a script for the VERIFY phase of the RPM build"),
124: 
125:         # Allow a packager to explicitly force an architecture
126:         ('force-arch=', None,
127:          "Force an architecture onto the RPM build process"),
128: 
129:         ('quiet', 'q',
130:          "Run the INSTALL phase of RPM building in quiet mode"),
131:         ]
132: 
133:     boolean_options = ['keep-temp', 'use-rpm-opt-flags', 'rpm3-mode',
134:                        'no-autoreq', 'quiet']
135: 
136:     negative_opt = {'no-keep-temp': 'keep-temp',
137:                     'no-rpm-opt-flags': 'use-rpm-opt-flags',
138:                     'rpm2-mode': 'rpm3-mode'}
139: 
140: 
141:     def initialize_options (self):
142:         self.bdist_base = None
143:         self.rpm_base = None
144:         self.dist_dir = None
145:         self.python = None
146:         self.fix_python = None
147:         self.spec_only = None
148:         self.binary_only = None
149:         self.source_only = None
150:         self.use_bzip2 = None
151: 
152:         self.distribution_name = None
153:         self.group = None
154:         self.release = None
155:         self.serial = None
156:         self.vendor = None
157:         self.packager = None
158:         self.doc_files = None
159:         self.changelog = None
160:         self.icon = None
161: 
162:         self.prep_script = None
163:         self.build_script = None
164:         self.install_script = None
165:         self.clean_script = None
166:         self.verify_script = None
167:         self.pre_install = None
168:         self.post_install = None
169:         self.pre_uninstall = None
170:         self.post_uninstall = None
171:         self.prep = None
172:         self.provides = None
173:         self.requires = None
174:         self.conflicts = None
175:         self.build_requires = None
176:         self.obsoletes = None
177: 
178:         self.keep_temp = 0
179:         self.use_rpm_opt_flags = 1
180:         self.rpm3_mode = 1
181:         self.no_autoreq = 0
182: 
183:         self.force_arch = None
184:         self.quiet = 0
185: 
186:     # initialize_options()
187: 
188: 
189:     def finalize_options (self):
190:         self.set_undefined_options('bdist', ('bdist_base', 'bdist_base'))
191:         if self.rpm_base is None:
192:             if not self.rpm3_mode:
193:                 raise DistutilsOptionError, \
194:                       "you must specify --rpm-base in RPM 2 mode"
195:             self.rpm_base = os.path.join(self.bdist_base, "rpm")
196: 
197:         if self.python is None:
198:             if self.fix_python:
199:                 self.python = sys.executable
200:             else:
201:                 self.python = "python"
202:         elif self.fix_python:
203:             raise DistutilsOptionError, \
204:                   "--python and --fix-python are mutually exclusive options"
205: 
206:         if os.name != 'posix':
207:             raise DistutilsPlatformError, \
208:                   ("don't know how to create RPM "
209:                    "distributions on platform %s" % os.name)
210:         if self.binary_only and self.source_only:
211:             raise DistutilsOptionError, \
212:                   "cannot supply both '--source-only' and '--binary-only'"
213: 
214:         # don't pass CFLAGS to pure python distributions
215:         if not self.distribution.has_ext_modules():
216:             self.use_rpm_opt_flags = 0
217: 
218:         self.set_undefined_options('bdist', ('dist_dir', 'dist_dir'))
219:         self.finalize_package_data()
220: 
221:     # finalize_options()
222: 
223:     def finalize_package_data (self):
224:         self.ensure_string('group', "Development/Libraries")
225:         self.ensure_string('vendor',
226:                            "%s <%s>" % (self.distribution.get_contact(),
227:                                         self.distribution.get_contact_email()))
228:         self.ensure_string('packager')
229:         self.ensure_string_list('doc_files')
230:         if isinstance(self.doc_files, list):
231:             for readme in ('README', 'README.txt'):
232:                 if os.path.exists(readme) and readme not in self.doc_files:
233:                     self.doc_files.append(readme)
234: 
235:         self.ensure_string('release', "1")
236:         self.ensure_string('serial')   # should it be an int?
237: 
238:         self.ensure_string('distribution_name')
239: 
240:         self.ensure_string('changelog')
241:           # Format changelog correctly
242:         self.changelog = self._format_changelog(self.changelog)
243: 
244:         self.ensure_filename('icon')
245: 
246:         self.ensure_filename('prep_script')
247:         self.ensure_filename('build_script')
248:         self.ensure_filename('install_script')
249:         self.ensure_filename('clean_script')
250:         self.ensure_filename('verify_script')
251:         self.ensure_filename('pre_install')
252:         self.ensure_filename('post_install')
253:         self.ensure_filename('pre_uninstall')
254:         self.ensure_filename('post_uninstall')
255: 
256:         # XXX don't forget we punted on summaries and descriptions -- they
257:         # should be handled here eventually!
258: 
259:         # Now *this* is some meta-data that belongs in the setup script...
260:         self.ensure_string_list('provides')
261:         self.ensure_string_list('requires')
262:         self.ensure_string_list('conflicts')
263:         self.ensure_string_list('build_requires')
264:         self.ensure_string_list('obsoletes')
265: 
266:         self.ensure_string('force_arch')
267:     # finalize_package_data ()
268: 
269: 
270:     def run (self):
271: 
272:         if DEBUG:
273:             print "before _get_package_data():"
274:             print "vendor =", self.vendor
275:             print "packager =", self.packager
276:             print "doc_files =", self.doc_files
277:             print "changelog =", self.changelog
278: 
279:         # make directories
280:         if self.spec_only:
281:             spec_dir = self.dist_dir
282:             self.mkpath(spec_dir)
283:         else:
284:             rpm_dir = {}
285:             for d in ('SOURCES', 'SPECS', 'BUILD', 'RPMS', 'SRPMS'):
286:                 rpm_dir[d] = os.path.join(self.rpm_base, d)
287:                 self.mkpath(rpm_dir[d])
288:             spec_dir = rpm_dir['SPECS']
289: 
290:         # Spec file goes into 'dist_dir' if '--spec-only specified',
291:         # build/rpm.<plat> otherwise.
292:         spec_path = os.path.join(spec_dir,
293:                                  "%s.spec" % self.distribution.get_name())
294:         self.execute(write_file,
295:                      (spec_path,
296:                       self._make_spec_file()),
297:                      "writing '%s'" % spec_path)
298: 
299:         if self.spec_only: # stop if requested
300:             return
301: 
302:         # Make a source distribution and copy to SOURCES directory with
303:         # optional icon.
304:         saved_dist_files = self.distribution.dist_files[:]
305:         sdist = self.reinitialize_command('sdist')
306:         if self.use_bzip2:
307:             sdist.formats = ['bztar']
308:         else:
309:             sdist.formats = ['gztar']
310:         self.run_command('sdist')
311:         self.distribution.dist_files = saved_dist_files
312: 
313:         source = sdist.get_archive_files()[0]
314:         source_dir = rpm_dir['SOURCES']
315:         self.copy_file(source, source_dir)
316: 
317:         if self.icon:
318:             if os.path.exists(self.icon):
319:                 self.copy_file(self.icon, source_dir)
320:             else:
321:                 raise DistutilsFileError, \
322:                       "icon file '%s' does not exist" % self.icon
323: 
324: 
325:         # build package
326:         log.info("building RPMs")
327:         rpm_cmd = ['rpm']
328:         if os.path.exists('/usr/bin/rpmbuild') or \
329:            os.path.exists('/bin/rpmbuild'):
330:             rpm_cmd = ['rpmbuild']
331: 
332:         if self.source_only: # what kind of RPMs?
333:             rpm_cmd.append('-bs')
334:         elif self.binary_only:
335:             rpm_cmd.append('-bb')
336:         else:
337:             rpm_cmd.append('-ba')
338:         if self.rpm3_mode:
339:             rpm_cmd.extend(['--define',
340:                              '_topdir %s' % os.path.abspath(self.rpm_base)])
341:         if not self.keep_temp:
342:             rpm_cmd.append('--clean')
343: 
344:         if self.quiet:
345:             rpm_cmd.append('--quiet')
346: 
347:         rpm_cmd.append(spec_path)
348:         # Determine the binary rpm names that should be built out of this spec
349:         # file
350:         # Note that some of these may not be really built (if the file
351:         # list is empty)
352:         nvr_string = "%{name}-%{version}-%{release}"
353:         src_rpm = nvr_string + ".src.rpm"
354:         non_src_rpm = "%{arch}/" + nvr_string + ".%{arch}.rpm"
355:         q_cmd = r"rpm -q --qf '%s %s\n' --specfile '%s'" % (
356:             src_rpm, non_src_rpm, spec_path)
357: 
358:         out = os.popen(q_cmd)
359:         try:
360:             binary_rpms = []
361:             source_rpm = None
362:             while 1:
363:                 line = out.readline()
364:                 if not line:
365:                     break
366:                 l = string.split(string.strip(line))
367:                 assert(len(l) == 2)
368:                 binary_rpms.append(l[1])
369:                 # The source rpm is named after the first entry in the spec file
370:                 if source_rpm is None:
371:                     source_rpm = l[0]
372: 
373:             status = out.close()
374:             if status:
375:                 raise DistutilsExecError("Failed to execute: %s" % repr(q_cmd))
376: 
377:         finally:
378:             out.close()
379: 
380:         self.spawn(rpm_cmd)
381: 
382:         if not self.dry_run:
383:             if self.distribution.has_ext_modules():
384:                 pyversion = get_python_version()
385:             else:
386:                 pyversion = 'any'
387: 
388:             if not self.binary_only:
389:                 srpm = os.path.join(rpm_dir['SRPMS'], source_rpm)
390:                 assert(os.path.exists(srpm))
391:                 self.move_file(srpm, self.dist_dir)
392:                 filename = os.path.join(self.dist_dir, source_rpm)
393:                 self.distribution.dist_files.append(
394:                     ('bdist_rpm', pyversion, filename))
395: 
396:             if not self.source_only:
397:                 for rpm in binary_rpms:
398:                     rpm = os.path.join(rpm_dir['RPMS'], rpm)
399:                     if os.path.exists(rpm):
400:                         self.move_file(rpm, self.dist_dir)
401:                         filename = os.path.join(self.dist_dir,
402:                                                 os.path.basename(rpm))
403:                         self.distribution.dist_files.append(
404:                             ('bdist_rpm', pyversion, filename))
405:     # run()
406: 
407:     def _dist_path(self, path):
408:         return os.path.join(self.dist_dir, os.path.basename(path))
409: 
410:     def _make_spec_file(self):
411:         '''Generate the text of an RPM spec file and return it as a
412:         list of strings (one per line).
413:         '''
414:         # definitions and headers
415:         spec_file = [
416:             '%define name ' + self.distribution.get_name(),
417:             '%define version ' + self.distribution.get_version().replace('-','_'),
418:             '%define unmangled_version ' + self.distribution.get_version(),
419:             '%define release ' + self.release.replace('-','_'),
420:             '',
421:             'Summary: ' + self.distribution.get_description(),
422:             ]
423: 
424:         # put locale summaries into spec file
425:         # XXX not supported for now (hard to put a dictionary
426:         # in a config file -- arg!)
427:         #for locale in self.summaries.keys():
428:         #    spec_file.append('Summary(%s): %s' % (locale,
429:         #                                          self.summaries[locale]))
430: 
431:         spec_file.extend([
432:             'Name: %{name}',
433:             'Version: %{version}',
434:             'Release: %{release}',])
435: 
436:         # XXX yuck! this filename is available from the "sdist" command,
437:         # but only after it has run: and we create the spec file before
438:         # running "sdist", in case of --spec-only.
439:         if self.use_bzip2:
440:             spec_file.append('Source0: %{name}-%{unmangled_version}.tar.bz2')
441:         else:
442:             spec_file.append('Source0: %{name}-%{unmangled_version}.tar.gz')
443: 
444:         spec_file.extend([
445:             'License: ' + self.distribution.get_license(),
446:             'Group: ' + self.group,
447:             'BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot',
448:             'Prefix: %{_prefix}', ])
449: 
450:         if not self.force_arch:
451:             # noarch if no extension modules
452:             if not self.distribution.has_ext_modules():
453:                 spec_file.append('BuildArch: noarch')
454:         else:
455:             spec_file.append( 'BuildArch: %s' % self.force_arch )
456: 
457:         for field in ('Vendor',
458:                       'Packager',
459:                       'Provides',
460:                       'Requires',
461:                       'Conflicts',
462:                       'Obsoletes',
463:                       ):
464:             val = getattr(self, string.lower(field))
465:             if isinstance(val, list):
466:                 spec_file.append('%s: %s' % (field, string.join(val)))
467:             elif val is not None:
468:                 spec_file.append('%s: %s' % (field, val))
469: 
470: 
471:         if self.distribution.get_url() != 'UNKNOWN':
472:             spec_file.append('Url: ' + self.distribution.get_url())
473: 
474:         if self.distribution_name:
475:             spec_file.append('Distribution: ' + self.distribution_name)
476: 
477:         if self.build_requires:
478:             spec_file.append('BuildRequires: ' +
479:                              string.join(self.build_requires))
480: 
481:         if self.icon:
482:             spec_file.append('Icon: ' + os.path.basename(self.icon))
483: 
484:         if self.no_autoreq:
485:             spec_file.append('AutoReq: 0')
486: 
487:         spec_file.extend([
488:             '',
489:             '%description',
490:             self.distribution.get_long_description()
491:             ])
492: 
493:         # put locale descriptions into spec file
494:         # XXX again, suppressed because config file syntax doesn't
495:         # easily support this ;-(
496:         #for locale in self.descriptions.keys():
497:         #    spec_file.extend([
498:         #        '',
499:         #        '%description -l ' + locale,
500:         #        self.descriptions[locale],
501:         #        ])
502: 
503:         # rpm scripts
504:         # figure out default build script
505:         def_setup_call = "%s %s" % (self.python,os.path.basename(sys.argv[0]))
506:         def_build = "%s build" % def_setup_call
507:         if self.use_rpm_opt_flags:
508:             def_build = 'env CFLAGS="$RPM_OPT_FLAGS" ' + def_build
509: 
510:         # insert contents of files
511: 
512:         # XXX this is kind of misleading: user-supplied options are files
513:         # that we open and interpolate into the spec file, but the defaults
514:         # are just text that we drop in as-is.  Hmmm.
515: 
516:         install_cmd = ('%s install -O1 --root=$RPM_BUILD_ROOT '
517:                        '--record=INSTALLED_FILES') % def_setup_call
518: 
519:         script_options = [
520:             ('prep', 'prep_script', "%setup -n %{name}-%{unmangled_version}"),
521:             ('build', 'build_script', def_build),
522:             ('install', 'install_script', install_cmd),
523:             ('clean', 'clean_script', "rm -rf $RPM_BUILD_ROOT"),
524:             ('verifyscript', 'verify_script', None),
525:             ('pre', 'pre_install', None),
526:             ('post', 'post_install', None),
527:             ('preun', 'pre_uninstall', None),
528:             ('postun', 'post_uninstall', None),
529:         ]
530: 
531:         for (rpm_opt, attr, default) in script_options:
532:             # Insert contents of file referred to, if no file is referred to
533:             # use 'default' as contents of script
534:             val = getattr(self, attr)
535:             if val or default:
536:                 spec_file.extend([
537:                     '',
538:                     '%' + rpm_opt,])
539:                 if val:
540:                     spec_file.extend(string.split(open(val, 'r').read(), '\n'))
541:                 else:
542:                     spec_file.append(default)
543: 
544: 
545:         # files section
546:         spec_file.extend([
547:             '',
548:             '%files -f INSTALLED_FILES',
549:             '%defattr(-,root,root)',
550:             ])
551: 
552:         if self.doc_files:
553:             spec_file.append('%doc ' + string.join(self.doc_files))
554: 
555:         if self.changelog:
556:             spec_file.extend([
557:                 '',
558:                 '%changelog',])
559:             spec_file.extend(self.changelog)
560: 
561:         return spec_file
562: 
563:     # _make_spec_file ()
564: 
565:     def _format_changelog(self, changelog):
566:         '''Format the changelog correctly and convert it to a list of strings
567:         '''
568:         if not changelog:
569:             return changelog
570:         new_changelog = []
571:         for line in string.split(string.strip(changelog), '\n'):
572:             line = string.strip(line)
573:             if line[0] == '*':
574:                 new_changelog.extend(['', line])
575:             elif line[0] == '-':
576:                 new_changelog.append(line)
577:             else:
578:                 new_changelog.append('  ' + line)
579: 
580:         # strip trailing newline inserted by first changelog entry
581:         if not new_changelog[0]:
582:             del new_changelog[0]
583: 
584:         return new_changelog
585: 
586:     # _format_changelog()
587: 
588: # class bdist_rpm
589: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_14895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.bdist_rpm\n\nImplements the Distutils 'bdist_rpm' command (create RPM source and binary\ndistributions).")

# Assigning a Str to a Name (line 6):
str_14896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_14896)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import string' statement (line 10)
import string

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'string', string, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.core import Command' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_14897 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.core')

if (type(import_14897) is not StypyTypeError):

    if (import_14897 != 'pyd_module'):
        __import__(import_14897)
        sys_modules_14898 = sys.modules[import_14897]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.core', sys_modules_14898.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_14898, sys_modules_14898.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.core', import_14897)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.debug import DEBUG' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_14899 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.debug')

if (type(import_14899) is not StypyTypeError):

    if (import_14899 != 'pyd_module'):
        __import__(import_14899)
        sys_modules_14900 = sys.modules[import_14899]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.debug', sys_modules_14900.module_type_store, module_type_store, ['DEBUG'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_14900, sys_modules_14900.module_type_store, module_type_store)
    else:
        from distutils.debug import DEBUG

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.debug', None, module_type_store, ['DEBUG'], [DEBUG])

else:
    # Assigning a type to the variable 'distutils.debug' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.debug', import_14899)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.file_util import write_file' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_14901 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util')

if (type(import_14901) is not StypyTypeError):

    if (import_14901 != 'pyd_module'):
        __import__(import_14901)
        sys_modules_14902 = sys.modules[import_14901]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util', sys_modules_14902.module_type_store, module_type_store, ['write_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_14902, sys_modules_14902.module_type_store, module_type_store)
    else:
        from distutils.file_util import write_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util', None, module_type_store, ['write_file'], [write_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.file_util', import_14901)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.sysconfig import get_python_version' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_14903 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.sysconfig')

if (type(import_14903) is not StypyTypeError):

    if (import_14903 != 'pyd_module'):
        __import__(import_14903)
        sys_modules_14904 = sys.modules[import_14903]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.sysconfig', sys_modules_14904.module_type_store, module_type_store, ['get_python_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_14904, sys_modules_14904.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import get_python_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.sysconfig', None, module_type_store, ['get_python_version'], [get_python_version])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.sysconfig', import_14903)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.errors import DistutilsOptionError, DistutilsPlatformError, DistutilsFileError, DistutilsExecError' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_14905 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors')

if (type(import_14905) is not StypyTypeError):

    if (import_14905 != 'pyd_module'):
        __import__(import_14905)
        sys_modules_14906 = sys.modules[import_14905]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', sys_modules_14906.module_type_store, module_type_store, ['DistutilsOptionError', 'DistutilsPlatformError', 'DistutilsFileError', 'DistutilsExecError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_14906, sys_modules_14906.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError, DistutilsPlatformError, DistutilsFileError, DistutilsExecError

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError', 'DistutilsPlatformError', 'DistutilsFileError', 'DistutilsExecError'], [DistutilsOptionError, DistutilsPlatformError, DistutilsFileError, DistutilsExecError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', import_14905)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils import log' statement (line 18)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'bdist_rpm' class
# Getting the type of 'Command' (line 20)
Command_14907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'Command')

class bdist_rpm(Command_14907, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_function_name', 'bdist_rpm.initialize_options')
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_rpm.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 142):
        # Getting the type of 'None' (line 142)
        None_14908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'None')
        # Getting the type of 'self' (line 142)
        self_14909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self')
        # Setting the type of the member 'bdist_base' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_14909, 'bdist_base', None_14908)
        
        # Assigning a Name to a Attribute (line 143):
        # Getting the type of 'None' (line 143)
        None_14910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'None')
        # Getting the type of 'self' (line 143)
        self_14911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Setting the type of the member 'rpm_base' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_14911, 'rpm_base', None_14910)
        
        # Assigning a Name to a Attribute (line 144):
        # Getting the type of 'None' (line 144)
        None_14912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'None')
        # Getting the type of 'self' (line 144)
        self_14913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self')
        # Setting the type of the member 'dist_dir' of a type (line 144)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_14913, 'dist_dir', None_14912)
        
        # Assigning a Name to a Attribute (line 145):
        # Getting the type of 'None' (line 145)
        None_14914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 22), 'None')
        # Getting the type of 'self' (line 145)
        self_14915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self')
        # Setting the type of the member 'python' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_14915, 'python', None_14914)
        
        # Assigning a Name to a Attribute (line 146):
        # Getting the type of 'None' (line 146)
        None_14916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'None')
        # Getting the type of 'self' (line 146)
        self_14917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'fix_python' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_14917, 'fix_python', None_14916)
        
        # Assigning a Name to a Attribute (line 147):
        # Getting the type of 'None' (line 147)
        None_14918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'None')
        # Getting the type of 'self' (line 147)
        self_14919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member 'spec_only' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_14919, 'spec_only', None_14918)
        
        # Assigning a Name to a Attribute (line 148):
        # Getting the type of 'None' (line 148)
        None_14920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'None')
        # Getting the type of 'self' (line 148)
        self_14921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self')
        # Setting the type of the member 'binary_only' of a type (line 148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_14921, 'binary_only', None_14920)
        
        # Assigning a Name to a Attribute (line 149):
        # Getting the type of 'None' (line 149)
        None_14922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'None')
        # Getting the type of 'self' (line 149)
        self_14923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Setting the type of the member 'source_only' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_14923, 'source_only', None_14922)
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 'None' (line 150)
        None_14924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 25), 'None')
        # Getting the type of 'self' (line 150)
        self_14925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'use_bzip2' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_14925, 'use_bzip2', None_14924)
        
        # Assigning a Name to a Attribute (line 152):
        # Getting the type of 'None' (line 152)
        None_14926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 33), 'None')
        # Getting the type of 'self' (line 152)
        self_14927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'self')
        # Setting the type of the member 'distribution_name' of a type (line 152)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), self_14927, 'distribution_name', None_14926)
        
        # Assigning a Name to a Attribute (line 153):
        # Getting the type of 'None' (line 153)
        None_14928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'None')
        # Getting the type of 'self' (line 153)
        self_14929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'self')
        # Setting the type of the member 'group' of a type (line 153)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), self_14929, 'group', None_14928)
        
        # Assigning a Name to a Attribute (line 154):
        # Getting the type of 'None' (line 154)
        None_14930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'None')
        # Getting the type of 'self' (line 154)
        self_14931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
        # Setting the type of the member 'release' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_14931, 'release', None_14930)
        
        # Assigning a Name to a Attribute (line 155):
        # Getting the type of 'None' (line 155)
        None_14932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'None')
        # Getting the type of 'self' (line 155)
        self_14933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self')
        # Setting the type of the member 'serial' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_14933, 'serial', None_14932)
        
        # Assigning a Name to a Attribute (line 156):
        # Getting the type of 'None' (line 156)
        None_14934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'None')
        # Getting the type of 'self' (line 156)
        self_14935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'self')
        # Setting the type of the member 'vendor' of a type (line 156)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), self_14935, 'vendor', None_14934)
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'None' (line 157)
        None_14936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'None')
        # Getting the type of 'self' (line 157)
        self_14937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member 'packager' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_14937, 'packager', None_14936)
        
        # Assigning a Name to a Attribute (line 158):
        # Getting the type of 'None' (line 158)
        None_14938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 25), 'None')
        # Getting the type of 'self' (line 158)
        self_14939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self')
        # Setting the type of the member 'doc_files' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_14939, 'doc_files', None_14938)
        
        # Assigning a Name to a Attribute (line 159):
        # Getting the type of 'None' (line 159)
        None_14940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'None')
        # Getting the type of 'self' (line 159)
        self_14941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self')
        # Setting the type of the member 'changelog' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_14941, 'changelog', None_14940)
        
        # Assigning a Name to a Attribute (line 160):
        # Getting the type of 'None' (line 160)
        None_14942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'None')
        # Getting the type of 'self' (line 160)
        self_14943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self')
        # Setting the type of the member 'icon' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_14943, 'icon', None_14942)
        
        # Assigning a Name to a Attribute (line 162):
        # Getting the type of 'None' (line 162)
        None_14944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'None')
        # Getting the type of 'self' (line 162)
        self_14945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self')
        # Setting the type of the member 'prep_script' of a type (line 162)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_14945, 'prep_script', None_14944)
        
        # Assigning a Name to a Attribute (line 163):
        # Getting the type of 'None' (line 163)
        None_14946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'None')
        # Getting the type of 'self' (line 163)
        self_14947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self')
        # Setting the type of the member 'build_script' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_14947, 'build_script', None_14946)
        
        # Assigning a Name to a Attribute (line 164):
        # Getting the type of 'None' (line 164)
        None_14948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 30), 'None')
        # Getting the type of 'self' (line 164)
        self_14949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self')
        # Setting the type of the member 'install_script' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_14949, 'install_script', None_14948)
        
        # Assigning a Name to a Attribute (line 165):
        # Getting the type of 'None' (line 165)
        None_14950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'None')
        # Getting the type of 'self' (line 165)
        self_14951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self')
        # Setting the type of the member 'clean_script' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_14951, 'clean_script', None_14950)
        
        # Assigning a Name to a Attribute (line 166):
        # Getting the type of 'None' (line 166)
        None_14952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), 'None')
        # Getting the type of 'self' (line 166)
        self_14953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self')
        # Setting the type of the member 'verify_script' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_14953, 'verify_script', None_14952)
        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'None' (line 167)
        None_14954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'None')
        # Getting the type of 'self' (line 167)
        self_14955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member 'pre_install' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_14955, 'pre_install', None_14954)
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of 'None' (line 168)
        None_14956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'None')
        # Getting the type of 'self' (line 168)
        self_14957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self')
        # Setting the type of the member 'post_install' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_14957, 'post_install', None_14956)
        
        # Assigning a Name to a Attribute (line 169):
        # Getting the type of 'None' (line 169)
        None_14958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'None')
        # Getting the type of 'self' (line 169)
        self_14959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self')
        # Setting the type of the member 'pre_uninstall' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_14959, 'pre_uninstall', None_14958)
        
        # Assigning a Name to a Attribute (line 170):
        # Getting the type of 'None' (line 170)
        None_14960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'None')
        # Getting the type of 'self' (line 170)
        self_14961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member 'post_uninstall' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_14961, 'post_uninstall', None_14960)
        
        # Assigning a Name to a Attribute (line 171):
        # Getting the type of 'None' (line 171)
        None_14962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'None')
        # Getting the type of 'self' (line 171)
        self_14963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self')
        # Setting the type of the member 'prep' of a type (line 171)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_14963, 'prep', None_14962)
        
        # Assigning a Name to a Attribute (line 172):
        # Getting the type of 'None' (line 172)
        None_14964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 24), 'None')
        # Getting the type of 'self' (line 172)
        self_14965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self')
        # Setting the type of the member 'provides' of a type (line 172)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_14965, 'provides', None_14964)
        
        # Assigning a Name to a Attribute (line 173):
        # Getting the type of 'None' (line 173)
        None_14966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'None')
        # Getting the type of 'self' (line 173)
        self_14967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self')
        # Setting the type of the member 'requires' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_14967, 'requires', None_14966)
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'None' (line 174)
        None_14968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'None')
        # Getting the type of 'self' (line 174)
        self_14969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'conflicts' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_14969, 'conflicts', None_14968)
        
        # Assigning a Name to a Attribute (line 175):
        # Getting the type of 'None' (line 175)
        None_14970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 30), 'None')
        # Getting the type of 'self' (line 175)
        self_14971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member 'build_requires' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_14971, 'build_requires', None_14970)
        
        # Assigning a Name to a Attribute (line 176):
        # Getting the type of 'None' (line 176)
        None_14972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'None')
        # Getting the type of 'self' (line 176)
        self_14973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self')
        # Setting the type of the member 'obsoletes' of a type (line 176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_14973, 'obsoletes', None_14972)
        
        # Assigning a Num to a Attribute (line 178):
        int_14974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 25), 'int')
        # Getting the type of 'self' (line 178)
        self_14975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self')
        # Setting the type of the member 'keep_temp' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_14975, 'keep_temp', int_14974)
        
        # Assigning a Num to a Attribute (line 179):
        int_14976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 33), 'int')
        # Getting the type of 'self' (line 179)
        self_14977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self')
        # Setting the type of the member 'use_rpm_opt_flags' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_14977, 'use_rpm_opt_flags', int_14976)
        
        # Assigning a Num to a Attribute (line 180):
        int_14978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 25), 'int')
        # Getting the type of 'self' (line 180)
        self_14979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self')
        # Setting the type of the member 'rpm3_mode' of a type (line 180)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_14979, 'rpm3_mode', int_14978)
        
        # Assigning a Num to a Attribute (line 181):
        int_14980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 26), 'int')
        # Getting the type of 'self' (line 181)
        self_14981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self')
        # Setting the type of the member 'no_autoreq' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_14981, 'no_autoreq', int_14980)
        
        # Assigning a Name to a Attribute (line 183):
        # Getting the type of 'None' (line 183)
        None_14982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'None')
        # Getting the type of 'self' (line 183)
        self_14983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'self')
        # Setting the type of the member 'force_arch' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), self_14983, 'force_arch', None_14982)
        
        # Assigning a Num to a Attribute (line 184):
        int_14984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 21), 'int')
        # Getting the type of 'self' (line 184)
        self_14985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self')
        # Setting the type of the member 'quiet' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_14985, 'quiet', int_14984)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_14986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14986)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_14986


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_function_name', 'bdist_rpm.finalize_options')
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_rpm.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm.finalize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finalize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finalize_options(...)' code ##################

        
        # Call to set_undefined_options(...): (line 190)
        # Processing the call arguments (line 190)
        str_14989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 35), 'str', 'bdist')
        
        # Obtaining an instance of the builtin type 'tuple' (line 190)
        tuple_14990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 190)
        # Adding element type (line 190)
        str_14991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 45), 'str', 'bdist_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 45), tuple_14990, str_14991)
        # Adding element type (line 190)
        str_14992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 59), 'str', 'bdist_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 45), tuple_14990, str_14992)
        
        # Processing the call keyword arguments (line 190)
        kwargs_14993 = {}
        # Getting the type of 'self' (line 190)
        self_14987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 190)
        set_undefined_options_14988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), self_14987, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 190)
        set_undefined_options_call_result_14994 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), set_undefined_options_14988, *[str_14989, tuple_14990], **kwargs_14993)
        
        
        # Type idiom detected: calculating its left and rigth part (line 191)
        # Getting the type of 'self' (line 191)
        self_14995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'self')
        # Obtaining the member 'rpm_base' of a type (line 191)
        rpm_base_14996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 11), self_14995, 'rpm_base')
        # Getting the type of 'None' (line 191)
        None_14997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 28), 'None')
        
        (may_be_14998, more_types_in_union_14999) = may_be_none(rpm_base_14996, None_14997)

        if may_be_14998:

            if more_types_in_union_14999:
                # Runtime conditional SSA (line 191)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'self' (line 192)
            self_15000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'self')
            # Obtaining the member 'rpm3_mode' of a type (line 192)
            rpm3_mode_15001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 19), self_15000, 'rpm3_mode')
            # Applying the 'not' unary operator (line 192)
            result_not__15002 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 15), 'not', rpm3_mode_15001)
            
            # Testing the type of an if condition (line 192)
            if_condition_15003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 12), result_not__15002)
            # Assigning a type to the variable 'if_condition_15003' (line 192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'if_condition_15003', if_condition_15003)
            # SSA begins for if statement (line 192)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'DistutilsOptionError' (line 193)
            DistutilsOptionError_15004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 22), 'DistutilsOptionError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 193, 16), DistutilsOptionError_15004, 'raise parameter', BaseException)
            # SSA join for if statement (line 192)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Attribute (line 195):
            
            # Call to join(...): (line 195)
            # Processing the call arguments (line 195)
            # Getting the type of 'self' (line 195)
            self_15008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 41), 'self', False)
            # Obtaining the member 'bdist_base' of a type (line 195)
            bdist_base_15009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 41), self_15008, 'bdist_base')
            str_15010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 58), 'str', 'rpm')
            # Processing the call keyword arguments (line 195)
            kwargs_15011 = {}
            # Getting the type of 'os' (line 195)
            os_15005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'os', False)
            # Obtaining the member 'path' of a type (line 195)
            path_15006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 28), os_15005, 'path')
            # Obtaining the member 'join' of a type (line 195)
            join_15007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 28), path_15006, 'join')
            # Calling join(args, kwargs) (line 195)
            join_call_result_15012 = invoke(stypy.reporting.localization.Localization(__file__, 195, 28), join_15007, *[bdist_base_15009, str_15010], **kwargs_15011)
            
            # Getting the type of 'self' (line 195)
            self_15013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self')
            # Setting the type of the member 'rpm_base' of a type (line 195)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_15013, 'rpm_base', join_call_result_15012)

            if more_types_in_union_14999:
                # SSA join for if statement (line 191)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 197)
        # Getting the type of 'self' (line 197)
        self_15014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'self')
        # Obtaining the member 'python' of a type (line 197)
        python_15015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 11), self_15014, 'python')
        # Getting the type of 'None' (line 197)
        None_15016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'None')
        
        (may_be_15017, more_types_in_union_15018) = may_be_none(python_15015, None_15016)

        if may_be_15017:

            if more_types_in_union_15018:
                # Runtime conditional SSA (line 197)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'self' (line 198)
            self_15019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'self')
            # Obtaining the member 'fix_python' of a type (line 198)
            fix_python_15020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 15), self_15019, 'fix_python')
            # Testing the type of an if condition (line 198)
            if_condition_15021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 12), fix_python_15020)
            # Assigning a type to the variable 'if_condition_15021' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'if_condition_15021', if_condition_15021)
            # SSA begins for if statement (line 198)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 199):
            # Getting the type of 'sys' (line 199)
            sys_15022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'sys')
            # Obtaining the member 'executable' of a type (line 199)
            executable_15023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 30), sys_15022, 'executable')
            # Getting the type of 'self' (line 199)
            self_15024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'self')
            # Setting the type of the member 'python' of a type (line 199)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), self_15024, 'python', executable_15023)
            # SSA branch for the else part of an if statement (line 198)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Attribute (line 201):
            str_15025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 30), 'str', 'python')
            # Getting the type of 'self' (line 201)
            self_15026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'self')
            # Setting the type of the member 'python' of a type (line 201)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), self_15026, 'python', str_15025)
            # SSA join for if statement (line 198)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_15018:
                # Runtime conditional SSA for else branch (line 197)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_15017) or more_types_in_union_15018):
            
            # Getting the type of 'self' (line 202)
            self_15027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'self')
            # Obtaining the member 'fix_python' of a type (line 202)
            fix_python_15028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 13), self_15027, 'fix_python')
            # Testing the type of an if condition (line 202)
            if_condition_15029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 13), fix_python_15028)
            # Assigning a type to the variable 'if_condition_15029' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'if_condition_15029', if_condition_15029)
            # SSA begins for if statement (line 202)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'DistutilsOptionError' (line 203)
            DistutilsOptionError_15030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'DistutilsOptionError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 203, 12), DistutilsOptionError_15030, 'raise parameter', BaseException)
            # SSA join for if statement (line 202)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_15017 and more_types_in_union_15018):
                # SSA join for if statement (line 197)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'os' (line 206)
        os_15031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'os')
        # Obtaining the member 'name' of a type (line 206)
        name_15032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 11), os_15031, 'name')
        str_15033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 22), 'str', 'posix')
        # Applying the binary operator '!=' (line 206)
        result_ne_15034 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 11), '!=', name_15032, str_15033)
        
        # Testing the type of an if condition (line 206)
        if_condition_15035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 8), result_ne_15034)
        # Assigning a type to the variable 'if_condition_15035' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'if_condition_15035', if_condition_15035)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsPlatformError' (line 207)
        DistutilsPlatformError_15036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 18), 'DistutilsPlatformError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 207, 12), DistutilsPlatformError_15036, 'raise parameter', BaseException)
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 210)
        self_15037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'self')
        # Obtaining the member 'binary_only' of a type (line 210)
        binary_only_15038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 11), self_15037, 'binary_only')
        # Getting the type of 'self' (line 210)
        self_15039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 32), 'self')
        # Obtaining the member 'source_only' of a type (line 210)
        source_only_15040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 32), self_15039, 'source_only')
        # Applying the binary operator 'and' (line 210)
        result_and_keyword_15041 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 11), 'and', binary_only_15038, source_only_15040)
        
        # Testing the type of an if condition (line 210)
        if_condition_15042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), result_and_keyword_15041)
        # Assigning a type to the variable 'if_condition_15042' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_15042', if_condition_15042)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsOptionError' (line 211)
        DistutilsOptionError_15043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 211, 12), DistutilsOptionError_15043, 'raise parameter', BaseException)
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to has_ext_modules(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_15047 = {}
        # Getting the type of 'self' (line 215)
        self_15044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 215)
        distribution_15045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 15), self_15044, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 215)
        has_ext_modules_15046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 15), distribution_15045, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 215)
        has_ext_modules_call_result_15048 = invoke(stypy.reporting.localization.Localization(__file__, 215, 15), has_ext_modules_15046, *[], **kwargs_15047)
        
        # Applying the 'not' unary operator (line 215)
        result_not__15049 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), 'not', has_ext_modules_call_result_15048)
        
        # Testing the type of an if condition (line 215)
        if_condition_15050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_not__15049)
        # Assigning a type to the variable 'if_condition_15050' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_15050', if_condition_15050)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 216):
        int_15051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 37), 'int')
        # Getting the type of 'self' (line 216)
        self_15052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'self')
        # Setting the type of the member 'use_rpm_opt_flags' of a type (line 216)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), self_15052, 'use_rpm_opt_flags', int_15051)
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_undefined_options(...): (line 218)
        # Processing the call arguments (line 218)
        str_15055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 35), 'str', 'bdist')
        
        # Obtaining an instance of the builtin type 'tuple' (line 218)
        tuple_15056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 218)
        # Adding element type (line 218)
        str_15057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 45), 'str', 'dist_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 45), tuple_15056, str_15057)
        # Adding element type (line 218)
        str_15058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 57), 'str', 'dist_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 45), tuple_15056, str_15058)
        
        # Processing the call keyword arguments (line 218)
        kwargs_15059 = {}
        # Getting the type of 'self' (line 218)
        self_15053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 218)
        set_undefined_options_15054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), self_15053, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 218)
        set_undefined_options_call_result_15060 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), set_undefined_options_15054, *[str_15055, tuple_15056], **kwargs_15059)
        
        
        # Call to finalize_package_data(...): (line 219)
        # Processing the call keyword arguments (line 219)
        kwargs_15063 = {}
        # Getting the type of 'self' (line 219)
        self_15061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self', False)
        # Obtaining the member 'finalize_package_data' of a type (line 219)
        finalize_package_data_15062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_15061, 'finalize_package_data')
        # Calling finalize_package_data(args, kwargs) (line 219)
        finalize_package_data_call_result_15064 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), finalize_package_data_15062, *[], **kwargs_15063)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_15065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15065)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_15065


    @norecursion
    def finalize_package_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_package_data'
        module_type_store = module_type_store.open_function_context('finalize_package_data', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_localization', localization)
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_function_name', 'bdist_rpm.finalize_package_data')
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_rpm.finalize_package_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm.finalize_package_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finalize_package_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finalize_package_data(...)' code ##################

        
        # Call to ensure_string(...): (line 224)
        # Processing the call arguments (line 224)
        str_15068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 27), 'str', 'group')
        str_15069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 36), 'str', 'Development/Libraries')
        # Processing the call keyword arguments (line 224)
        kwargs_15070 = {}
        # Getting the type of 'self' (line 224)
        self_15066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self', False)
        # Obtaining the member 'ensure_string' of a type (line 224)
        ensure_string_15067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_15066, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 224)
        ensure_string_call_result_15071 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), ensure_string_15067, *[str_15068, str_15069], **kwargs_15070)
        
        
        # Call to ensure_string(...): (line 225)
        # Processing the call arguments (line 225)
        str_15074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 27), 'str', 'vendor')
        str_15075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 27), 'str', '%s <%s>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_15076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        
        # Call to get_contact(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_15080 = {}
        # Getting the type of 'self' (line 226)
        self_15077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'self', False)
        # Obtaining the member 'distribution' of a type (line 226)
        distribution_15078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 40), self_15077, 'distribution')
        # Obtaining the member 'get_contact' of a type (line 226)
        get_contact_15079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 40), distribution_15078, 'get_contact')
        # Calling get_contact(args, kwargs) (line 226)
        get_contact_call_result_15081 = invoke(stypy.reporting.localization.Localization(__file__, 226, 40), get_contact_15079, *[], **kwargs_15080)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 40), tuple_15076, get_contact_call_result_15081)
        # Adding element type (line 226)
        
        # Call to get_contact_email(...): (line 227)
        # Processing the call keyword arguments (line 227)
        kwargs_15085 = {}
        # Getting the type of 'self' (line 227)
        self_15082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 40), 'self', False)
        # Obtaining the member 'distribution' of a type (line 227)
        distribution_15083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 40), self_15082, 'distribution')
        # Obtaining the member 'get_contact_email' of a type (line 227)
        get_contact_email_15084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 40), distribution_15083, 'get_contact_email')
        # Calling get_contact_email(args, kwargs) (line 227)
        get_contact_email_call_result_15086 = invoke(stypy.reporting.localization.Localization(__file__, 227, 40), get_contact_email_15084, *[], **kwargs_15085)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 40), tuple_15076, get_contact_email_call_result_15086)
        
        # Applying the binary operator '%' (line 226)
        result_mod_15087 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 27), '%', str_15075, tuple_15076)
        
        # Processing the call keyword arguments (line 225)
        kwargs_15088 = {}
        # Getting the type of 'self' (line 225)
        self_15072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'self', False)
        # Obtaining the member 'ensure_string' of a type (line 225)
        ensure_string_15073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), self_15072, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 225)
        ensure_string_call_result_15089 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), ensure_string_15073, *[str_15074, result_mod_15087], **kwargs_15088)
        
        
        # Call to ensure_string(...): (line 228)
        # Processing the call arguments (line 228)
        str_15092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 27), 'str', 'packager')
        # Processing the call keyword arguments (line 228)
        kwargs_15093 = {}
        # Getting the type of 'self' (line 228)
        self_15090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'self', False)
        # Obtaining the member 'ensure_string' of a type (line 228)
        ensure_string_15091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), self_15090, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 228)
        ensure_string_call_result_15094 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), ensure_string_15091, *[str_15092], **kwargs_15093)
        
        
        # Call to ensure_string_list(...): (line 229)
        # Processing the call arguments (line 229)
        str_15097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 32), 'str', 'doc_files')
        # Processing the call keyword arguments (line 229)
        kwargs_15098 = {}
        # Getting the type of 'self' (line 229)
        self_15095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 229)
        ensure_string_list_15096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), self_15095, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 229)
        ensure_string_list_call_result_15099 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), ensure_string_list_15096, *[str_15097], **kwargs_15098)
        
        
        # Type idiom detected: calculating its left and rigth part (line 230)
        # Getting the type of 'list' (line 230)
        list_15100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 38), 'list')
        # Getting the type of 'self' (line 230)
        self_15101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'self')
        # Obtaining the member 'doc_files' of a type (line 230)
        doc_files_15102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 22), self_15101, 'doc_files')
        
        (may_be_15103, more_types_in_union_15104) = may_be_subtype(list_15100, doc_files_15102)

        if may_be_15103:

            if more_types_in_union_15104:
                # Runtime conditional SSA (line 230)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 230)
            self_15105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self')
            # Obtaining the member 'doc_files' of a type (line 230)
            doc_files_15106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_15105, 'doc_files')
            # Setting the type of the member 'doc_files' of a type (line 230)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_15105, 'doc_files', remove_not_subtype_from_union(doc_files_15102, list))
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 231)
            tuple_15107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 231)
            # Adding element type (line 231)
            str_15108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 27), 'str', 'README')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 27), tuple_15107, str_15108)
            # Adding element type (line 231)
            str_15109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 37), 'str', 'README.txt')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 27), tuple_15107, str_15109)
            
            # Testing the type of a for loop iterable (line 231)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 231, 12), tuple_15107)
            # Getting the type of the for loop variable (line 231)
            for_loop_var_15110 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 231, 12), tuple_15107)
            # Assigning a type to the variable 'readme' (line 231)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'readme', for_loop_var_15110)
            # SSA begins for a for statement (line 231)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Evaluating a boolean operation
            
            # Call to exists(...): (line 232)
            # Processing the call arguments (line 232)
            # Getting the type of 'readme' (line 232)
            readme_15114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'readme', False)
            # Processing the call keyword arguments (line 232)
            kwargs_15115 = {}
            # Getting the type of 'os' (line 232)
            os_15111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'os', False)
            # Obtaining the member 'path' of a type (line 232)
            path_15112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 19), os_15111, 'path')
            # Obtaining the member 'exists' of a type (line 232)
            exists_15113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 19), path_15112, 'exists')
            # Calling exists(args, kwargs) (line 232)
            exists_call_result_15116 = invoke(stypy.reporting.localization.Localization(__file__, 232, 19), exists_15113, *[readme_15114], **kwargs_15115)
            
            
            # Getting the type of 'readme' (line 232)
            readme_15117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 46), 'readme')
            # Getting the type of 'self' (line 232)
            self_15118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 60), 'self')
            # Obtaining the member 'doc_files' of a type (line 232)
            doc_files_15119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 60), self_15118, 'doc_files')
            # Applying the binary operator 'notin' (line 232)
            result_contains_15120 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 46), 'notin', readme_15117, doc_files_15119)
            
            # Applying the binary operator 'and' (line 232)
            result_and_keyword_15121 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 19), 'and', exists_call_result_15116, result_contains_15120)
            
            # Testing the type of an if condition (line 232)
            if_condition_15122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 16), result_and_keyword_15121)
            # Assigning a type to the variable 'if_condition_15122' (line 232)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'if_condition_15122', if_condition_15122)
            # SSA begins for if statement (line 232)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 233)
            # Processing the call arguments (line 233)
            # Getting the type of 'readme' (line 233)
            readme_15126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 42), 'readme', False)
            # Processing the call keyword arguments (line 233)
            kwargs_15127 = {}
            # Getting the type of 'self' (line 233)
            self_15123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'self', False)
            # Obtaining the member 'doc_files' of a type (line 233)
            doc_files_15124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 20), self_15123, 'doc_files')
            # Obtaining the member 'append' of a type (line 233)
            append_15125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 20), doc_files_15124, 'append')
            # Calling append(args, kwargs) (line 233)
            append_call_result_15128 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), append_15125, *[readme_15126], **kwargs_15127)
            
            # SSA join for if statement (line 232)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_15104:
                # SSA join for if statement (line 230)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to ensure_string(...): (line 235)
        # Processing the call arguments (line 235)
        str_15131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 27), 'str', 'release')
        str_15132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 38), 'str', '1')
        # Processing the call keyword arguments (line 235)
        kwargs_15133 = {}
        # Getting the type of 'self' (line 235)
        self_15129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self', False)
        # Obtaining the member 'ensure_string' of a type (line 235)
        ensure_string_15130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_15129, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 235)
        ensure_string_call_result_15134 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), ensure_string_15130, *[str_15131, str_15132], **kwargs_15133)
        
        
        # Call to ensure_string(...): (line 236)
        # Processing the call arguments (line 236)
        str_15137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 27), 'str', 'serial')
        # Processing the call keyword arguments (line 236)
        kwargs_15138 = {}
        # Getting the type of 'self' (line 236)
        self_15135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self', False)
        # Obtaining the member 'ensure_string' of a type (line 236)
        ensure_string_15136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_15135, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 236)
        ensure_string_call_result_15139 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), ensure_string_15136, *[str_15137], **kwargs_15138)
        
        
        # Call to ensure_string(...): (line 238)
        # Processing the call arguments (line 238)
        str_15142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 27), 'str', 'distribution_name')
        # Processing the call keyword arguments (line 238)
        kwargs_15143 = {}
        # Getting the type of 'self' (line 238)
        self_15140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self', False)
        # Obtaining the member 'ensure_string' of a type (line 238)
        ensure_string_15141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_15140, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 238)
        ensure_string_call_result_15144 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), ensure_string_15141, *[str_15142], **kwargs_15143)
        
        
        # Call to ensure_string(...): (line 240)
        # Processing the call arguments (line 240)
        str_15147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 27), 'str', 'changelog')
        # Processing the call keyword arguments (line 240)
        kwargs_15148 = {}
        # Getting the type of 'self' (line 240)
        self_15145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'self', False)
        # Obtaining the member 'ensure_string' of a type (line 240)
        ensure_string_15146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), self_15145, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 240)
        ensure_string_call_result_15149 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), ensure_string_15146, *[str_15147], **kwargs_15148)
        
        
        # Assigning a Call to a Attribute (line 242):
        
        # Call to _format_changelog(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'self' (line 242)
        self_15152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 48), 'self', False)
        # Obtaining the member 'changelog' of a type (line 242)
        changelog_15153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 48), self_15152, 'changelog')
        # Processing the call keyword arguments (line 242)
        kwargs_15154 = {}
        # Getting the type of 'self' (line 242)
        self_15150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 25), 'self', False)
        # Obtaining the member '_format_changelog' of a type (line 242)
        _format_changelog_15151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 25), self_15150, '_format_changelog')
        # Calling _format_changelog(args, kwargs) (line 242)
        _format_changelog_call_result_15155 = invoke(stypy.reporting.localization.Localization(__file__, 242, 25), _format_changelog_15151, *[changelog_15153], **kwargs_15154)
        
        # Getting the type of 'self' (line 242)
        self_15156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self')
        # Setting the type of the member 'changelog' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_15156, 'changelog', _format_changelog_call_result_15155)
        
        # Call to ensure_filename(...): (line 244)
        # Processing the call arguments (line 244)
        str_15159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 29), 'str', 'icon')
        # Processing the call keyword arguments (line 244)
        kwargs_15160 = {}
        # Getting the type of 'self' (line 244)
        self_15157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 244)
        ensure_filename_15158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_15157, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 244)
        ensure_filename_call_result_15161 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), ensure_filename_15158, *[str_15159], **kwargs_15160)
        
        
        # Call to ensure_filename(...): (line 246)
        # Processing the call arguments (line 246)
        str_15164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 29), 'str', 'prep_script')
        # Processing the call keyword arguments (line 246)
        kwargs_15165 = {}
        # Getting the type of 'self' (line 246)
        self_15162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 246)
        ensure_filename_15163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), self_15162, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 246)
        ensure_filename_call_result_15166 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), ensure_filename_15163, *[str_15164], **kwargs_15165)
        
        
        # Call to ensure_filename(...): (line 247)
        # Processing the call arguments (line 247)
        str_15169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 29), 'str', 'build_script')
        # Processing the call keyword arguments (line 247)
        kwargs_15170 = {}
        # Getting the type of 'self' (line 247)
        self_15167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 247)
        ensure_filename_15168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_15167, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 247)
        ensure_filename_call_result_15171 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), ensure_filename_15168, *[str_15169], **kwargs_15170)
        
        
        # Call to ensure_filename(...): (line 248)
        # Processing the call arguments (line 248)
        str_15174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 29), 'str', 'install_script')
        # Processing the call keyword arguments (line 248)
        kwargs_15175 = {}
        # Getting the type of 'self' (line 248)
        self_15172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 248)
        ensure_filename_15173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_15172, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 248)
        ensure_filename_call_result_15176 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), ensure_filename_15173, *[str_15174], **kwargs_15175)
        
        
        # Call to ensure_filename(...): (line 249)
        # Processing the call arguments (line 249)
        str_15179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'str', 'clean_script')
        # Processing the call keyword arguments (line 249)
        kwargs_15180 = {}
        # Getting the type of 'self' (line 249)
        self_15177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 249)
        ensure_filename_15178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_15177, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 249)
        ensure_filename_call_result_15181 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), ensure_filename_15178, *[str_15179], **kwargs_15180)
        
        
        # Call to ensure_filename(...): (line 250)
        # Processing the call arguments (line 250)
        str_15184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 29), 'str', 'verify_script')
        # Processing the call keyword arguments (line 250)
        kwargs_15185 = {}
        # Getting the type of 'self' (line 250)
        self_15182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 250)
        ensure_filename_15183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_15182, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 250)
        ensure_filename_call_result_15186 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), ensure_filename_15183, *[str_15184], **kwargs_15185)
        
        
        # Call to ensure_filename(...): (line 251)
        # Processing the call arguments (line 251)
        str_15189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 29), 'str', 'pre_install')
        # Processing the call keyword arguments (line 251)
        kwargs_15190 = {}
        # Getting the type of 'self' (line 251)
        self_15187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 251)
        ensure_filename_15188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_15187, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 251)
        ensure_filename_call_result_15191 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), ensure_filename_15188, *[str_15189], **kwargs_15190)
        
        
        # Call to ensure_filename(...): (line 252)
        # Processing the call arguments (line 252)
        str_15194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 29), 'str', 'post_install')
        # Processing the call keyword arguments (line 252)
        kwargs_15195 = {}
        # Getting the type of 'self' (line 252)
        self_15192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 252)
        ensure_filename_15193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), self_15192, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 252)
        ensure_filename_call_result_15196 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), ensure_filename_15193, *[str_15194], **kwargs_15195)
        
        
        # Call to ensure_filename(...): (line 253)
        # Processing the call arguments (line 253)
        str_15199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 29), 'str', 'pre_uninstall')
        # Processing the call keyword arguments (line 253)
        kwargs_15200 = {}
        # Getting the type of 'self' (line 253)
        self_15197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 253)
        ensure_filename_15198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_15197, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 253)
        ensure_filename_call_result_15201 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), ensure_filename_15198, *[str_15199], **kwargs_15200)
        
        
        # Call to ensure_filename(...): (line 254)
        # Processing the call arguments (line 254)
        str_15204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'str', 'post_uninstall')
        # Processing the call keyword arguments (line 254)
        kwargs_15205 = {}
        # Getting the type of 'self' (line 254)
        self_15202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'self', False)
        # Obtaining the member 'ensure_filename' of a type (line 254)
        ensure_filename_15203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), self_15202, 'ensure_filename')
        # Calling ensure_filename(args, kwargs) (line 254)
        ensure_filename_call_result_15206 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), ensure_filename_15203, *[str_15204], **kwargs_15205)
        
        
        # Call to ensure_string_list(...): (line 260)
        # Processing the call arguments (line 260)
        str_15209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'str', 'provides')
        # Processing the call keyword arguments (line 260)
        kwargs_15210 = {}
        # Getting the type of 'self' (line 260)
        self_15207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 260)
        ensure_string_list_15208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), self_15207, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 260)
        ensure_string_list_call_result_15211 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), ensure_string_list_15208, *[str_15209], **kwargs_15210)
        
        
        # Call to ensure_string_list(...): (line 261)
        # Processing the call arguments (line 261)
        str_15214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 32), 'str', 'requires')
        # Processing the call keyword arguments (line 261)
        kwargs_15215 = {}
        # Getting the type of 'self' (line 261)
        self_15212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 261)
        ensure_string_list_15213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_15212, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 261)
        ensure_string_list_call_result_15216 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), ensure_string_list_15213, *[str_15214], **kwargs_15215)
        
        
        # Call to ensure_string_list(...): (line 262)
        # Processing the call arguments (line 262)
        str_15219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 32), 'str', 'conflicts')
        # Processing the call keyword arguments (line 262)
        kwargs_15220 = {}
        # Getting the type of 'self' (line 262)
        self_15217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 262)
        ensure_string_list_15218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_15217, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 262)
        ensure_string_list_call_result_15221 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), ensure_string_list_15218, *[str_15219], **kwargs_15220)
        
        
        # Call to ensure_string_list(...): (line 263)
        # Processing the call arguments (line 263)
        str_15224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 32), 'str', 'build_requires')
        # Processing the call keyword arguments (line 263)
        kwargs_15225 = {}
        # Getting the type of 'self' (line 263)
        self_15222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 263)
        ensure_string_list_15223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), self_15222, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 263)
        ensure_string_list_call_result_15226 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), ensure_string_list_15223, *[str_15224], **kwargs_15225)
        
        
        # Call to ensure_string_list(...): (line 264)
        # Processing the call arguments (line 264)
        str_15229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 32), 'str', 'obsoletes')
        # Processing the call keyword arguments (line 264)
        kwargs_15230 = {}
        # Getting the type of 'self' (line 264)
        self_15227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 264)
        ensure_string_list_15228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_15227, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 264)
        ensure_string_list_call_result_15231 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), ensure_string_list_15228, *[str_15229], **kwargs_15230)
        
        
        # Call to ensure_string(...): (line 266)
        # Processing the call arguments (line 266)
        str_15234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 27), 'str', 'force_arch')
        # Processing the call keyword arguments (line 266)
        kwargs_15235 = {}
        # Getting the type of 'self' (line 266)
        self_15232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self', False)
        # Obtaining the member 'ensure_string' of a type (line 266)
        ensure_string_15233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_15232, 'ensure_string')
        # Calling ensure_string(args, kwargs) (line 266)
        ensure_string_call_result_15236 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), ensure_string_15233, *[str_15234], **kwargs_15235)
        
        
        # ################# End of 'finalize_package_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_package_data' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_15237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_package_data'
        return stypy_return_type_15237


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_rpm.run.__dict__.__setitem__('stypy_localization', localization)
        bdist_rpm.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_rpm.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_rpm.run.__dict__.__setitem__('stypy_function_name', 'bdist_rpm.run')
        bdist_rpm.run.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_rpm.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_rpm.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_rpm.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_rpm.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_rpm.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_rpm.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm.run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Getting the type of 'DEBUG' (line 272)
        DEBUG_15238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'DEBUG')
        # Testing the type of an if condition (line 272)
        if_condition_15239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 8), DEBUG_15238)
        # Assigning a type to the variable 'if_condition_15239' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'if_condition_15239', if_condition_15239)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_15240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 18), 'str', 'before _get_package_data():')
        str_15241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 18), 'str', 'vendor =')
        # Getting the type of 'self' (line 274)
        self_15242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 30), 'self')
        # Obtaining the member 'vendor' of a type (line 274)
        vendor_15243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 30), self_15242, 'vendor')
        str_15244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 18), 'str', 'packager =')
        # Getting the type of 'self' (line 275)
        self_15245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 32), 'self')
        # Obtaining the member 'packager' of a type (line 275)
        packager_15246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 32), self_15245, 'packager')
        str_15247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 18), 'str', 'doc_files =')
        # Getting the type of 'self' (line 276)
        self_15248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 33), 'self')
        # Obtaining the member 'doc_files' of a type (line 276)
        doc_files_15249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 33), self_15248, 'doc_files')
        str_15250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 18), 'str', 'changelog =')
        # Getting the type of 'self' (line 277)
        self_15251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 33), 'self')
        # Obtaining the member 'changelog' of a type (line 277)
        changelog_15252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 33), self_15251, 'changelog')
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 280)
        self_15253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'self')
        # Obtaining the member 'spec_only' of a type (line 280)
        spec_only_15254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), self_15253, 'spec_only')
        # Testing the type of an if condition (line 280)
        if_condition_15255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 8), spec_only_15254)
        # Assigning a type to the variable 'if_condition_15255' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'if_condition_15255', if_condition_15255)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 281):
        # Getting the type of 'self' (line 281)
        self_15256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'self')
        # Obtaining the member 'dist_dir' of a type (line 281)
        dist_dir_15257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 23), self_15256, 'dist_dir')
        # Assigning a type to the variable 'spec_dir' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'spec_dir', dist_dir_15257)
        
        # Call to mkpath(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'spec_dir' (line 282)
        spec_dir_15260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'spec_dir', False)
        # Processing the call keyword arguments (line 282)
        kwargs_15261 = {}
        # Getting the type of 'self' (line 282)
        self_15258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 282)
        mkpath_15259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), self_15258, 'mkpath')
        # Calling mkpath(args, kwargs) (line 282)
        mkpath_call_result_15262 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), mkpath_15259, *[spec_dir_15260], **kwargs_15261)
        
        # SSA branch for the else part of an if statement (line 280)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Dict to a Name (line 284):
        
        # Obtaining an instance of the builtin type 'dict' (line 284)
        dict_15263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 284)
        
        # Assigning a type to the variable 'rpm_dir' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'rpm_dir', dict_15263)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 285)
        tuple_15264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 285)
        # Adding element type (line 285)
        str_15265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 22), 'str', 'SOURCES')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 22), tuple_15264, str_15265)
        # Adding element type (line 285)
        str_15266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 33), 'str', 'SPECS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 22), tuple_15264, str_15266)
        # Adding element type (line 285)
        str_15267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 42), 'str', 'BUILD')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 22), tuple_15264, str_15267)
        # Adding element type (line 285)
        str_15268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 51), 'str', 'RPMS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 22), tuple_15264, str_15268)
        # Adding element type (line 285)
        str_15269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 59), 'str', 'SRPMS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 22), tuple_15264, str_15269)
        
        # Testing the type of a for loop iterable (line 285)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 285, 12), tuple_15264)
        # Getting the type of the for loop variable (line 285)
        for_loop_var_15270 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 285, 12), tuple_15264)
        # Assigning a type to the variable 'd' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'd', for_loop_var_15270)
        # SSA begins for a for statement (line 285)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 286):
        
        # Call to join(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'self' (line 286)
        self_15274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 42), 'self', False)
        # Obtaining the member 'rpm_base' of a type (line 286)
        rpm_base_15275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 42), self_15274, 'rpm_base')
        # Getting the type of 'd' (line 286)
        d_15276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 57), 'd', False)
        # Processing the call keyword arguments (line 286)
        kwargs_15277 = {}
        # Getting the type of 'os' (line 286)
        os_15271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 286)
        path_15272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 29), os_15271, 'path')
        # Obtaining the member 'join' of a type (line 286)
        join_15273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 29), path_15272, 'join')
        # Calling join(args, kwargs) (line 286)
        join_call_result_15278 = invoke(stypy.reporting.localization.Localization(__file__, 286, 29), join_15273, *[rpm_base_15275, d_15276], **kwargs_15277)
        
        # Getting the type of 'rpm_dir' (line 286)
        rpm_dir_15279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'rpm_dir')
        # Getting the type of 'd' (line 286)
        d_15280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 24), 'd')
        # Storing an element on a container (line 286)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 16), rpm_dir_15279, (d_15280, join_call_result_15278))
        
        # Call to mkpath(...): (line 287)
        # Processing the call arguments (line 287)
        
        # Obtaining the type of the subscript
        # Getting the type of 'd' (line 287)
        d_15283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 'd', False)
        # Getting the type of 'rpm_dir' (line 287)
        rpm_dir_15284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 28), 'rpm_dir', False)
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___15285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 28), rpm_dir_15284, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_15286 = invoke(stypy.reporting.localization.Localization(__file__, 287, 28), getitem___15285, d_15283)
        
        # Processing the call keyword arguments (line 287)
        kwargs_15287 = {}
        # Getting the type of 'self' (line 287)
        self_15281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 287)
        mkpath_15282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 16), self_15281, 'mkpath')
        # Calling mkpath(args, kwargs) (line 287)
        mkpath_call_result_15288 = invoke(stypy.reporting.localization.Localization(__file__, 287, 16), mkpath_15282, *[subscript_call_result_15286], **kwargs_15287)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 288):
        
        # Obtaining the type of the subscript
        str_15289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 31), 'str', 'SPECS')
        # Getting the type of 'rpm_dir' (line 288)
        rpm_dir_15290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'rpm_dir')
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___15291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 23), rpm_dir_15290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 288)
        subscript_call_result_15292 = invoke(stypy.reporting.localization.Localization(__file__, 288, 23), getitem___15291, str_15289)
        
        # Assigning a type to the variable 'spec_dir' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'spec_dir', subscript_call_result_15292)
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 292):
        
        # Call to join(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'spec_dir' (line 292)
        spec_dir_15296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 33), 'spec_dir', False)
        str_15297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 33), 'str', '%s.spec')
        
        # Call to get_name(...): (line 293)
        # Processing the call keyword arguments (line 293)
        kwargs_15301 = {}
        # Getting the type of 'self' (line 293)
        self_15298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 45), 'self', False)
        # Obtaining the member 'distribution' of a type (line 293)
        distribution_15299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 45), self_15298, 'distribution')
        # Obtaining the member 'get_name' of a type (line 293)
        get_name_15300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 45), distribution_15299, 'get_name')
        # Calling get_name(args, kwargs) (line 293)
        get_name_call_result_15302 = invoke(stypy.reporting.localization.Localization(__file__, 293, 45), get_name_15300, *[], **kwargs_15301)
        
        # Applying the binary operator '%' (line 293)
        result_mod_15303 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 33), '%', str_15297, get_name_call_result_15302)
        
        # Processing the call keyword arguments (line 292)
        kwargs_15304 = {}
        # Getting the type of 'os' (line 292)
        os_15293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 292)
        path_15294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 20), os_15293, 'path')
        # Obtaining the member 'join' of a type (line 292)
        join_15295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 20), path_15294, 'join')
        # Calling join(args, kwargs) (line 292)
        join_call_result_15305 = invoke(stypy.reporting.localization.Localization(__file__, 292, 20), join_15295, *[spec_dir_15296, result_mod_15303], **kwargs_15304)
        
        # Assigning a type to the variable 'spec_path' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'spec_path', join_call_result_15305)
        
        # Call to execute(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'write_file' (line 294)
        write_file_15308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'write_file', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_15309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        # Getting the type of 'spec_path' (line 295)
        spec_path_15310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 22), 'spec_path', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 22), tuple_15309, spec_path_15310)
        # Adding element type (line 295)
        
        # Call to _make_spec_file(...): (line 296)
        # Processing the call keyword arguments (line 296)
        kwargs_15313 = {}
        # Getting the type of 'self' (line 296)
        self_15311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'self', False)
        # Obtaining the member '_make_spec_file' of a type (line 296)
        _make_spec_file_15312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 22), self_15311, '_make_spec_file')
        # Calling _make_spec_file(args, kwargs) (line 296)
        _make_spec_file_call_result_15314 = invoke(stypy.reporting.localization.Localization(__file__, 296, 22), _make_spec_file_15312, *[], **kwargs_15313)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 22), tuple_15309, _make_spec_file_call_result_15314)
        
        str_15315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 21), 'str', "writing '%s'")
        # Getting the type of 'spec_path' (line 297)
        spec_path_15316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 38), 'spec_path', False)
        # Applying the binary operator '%' (line 297)
        result_mod_15317 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 21), '%', str_15315, spec_path_15316)
        
        # Processing the call keyword arguments (line 294)
        kwargs_15318 = {}
        # Getting the type of 'self' (line 294)
        self_15306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self', False)
        # Obtaining the member 'execute' of a type (line 294)
        execute_15307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_15306, 'execute')
        # Calling execute(args, kwargs) (line 294)
        execute_call_result_15319 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), execute_15307, *[write_file_15308, tuple_15309, result_mod_15317], **kwargs_15318)
        
        
        # Getting the type of 'self' (line 299)
        self_15320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'self')
        # Obtaining the member 'spec_only' of a type (line 299)
        spec_only_15321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), self_15320, 'spec_only')
        # Testing the type of an if condition (line 299)
        if_condition_15322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 8), spec_only_15321)
        # Assigning a type to the variable 'if_condition_15322' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'if_condition_15322', if_condition_15322)
        # SSA begins for if statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 304):
        
        # Obtaining the type of the subscript
        slice_15323 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 304, 27), None, None, None)
        # Getting the type of 'self' (line 304)
        self_15324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'self')
        # Obtaining the member 'distribution' of a type (line 304)
        distribution_15325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 27), self_15324, 'distribution')
        # Obtaining the member 'dist_files' of a type (line 304)
        dist_files_15326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 27), distribution_15325, 'dist_files')
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___15327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 27), dist_files_15326, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_15328 = invoke(stypy.reporting.localization.Localization(__file__, 304, 27), getitem___15327, slice_15323)
        
        # Assigning a type to the variable 'saved_dist_files' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'saved_dist_files', subscript_call_result_15328)
        
        # Assigning a Call to a Name (line 305):
        
        # Call to reinitialize_command(...): (line 305)
        # Processing the call arguments (line 305)
        str_15331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 42), 'str', 'sdist')
        # Processing the call keyword arguments (line 305)
        kwargs_15332 = {}
        # Getting the type of 'self' (line 305)
        self_15329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'self', False)
        # Obtaining the member 'reinitialize_command' of a type (line 305)
        reinitialize_command_15330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 16), self_15329, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 305)
        reinitialize_command_call_result_15333 = invoke(stypy.reporting.localization.Localization(__file__, 305, 16), reinitialize_command_15330, *[str_15331], **kwargs_15332)
        
        # Assigning a type to the variable 'sdist' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'sdist', reinitialize_command_call_result_15333)
        
        # Getting the type of 'self' (line 306)
        self_15334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'self')
        # Obtaining the member 'use_bzip2' of a type (line 306)
        use_bzip2_15335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 11), self_15334, 'use_bzip2')
        # Testing the type of an if condition (line 306)
        if_condition_15336 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 8), use_bzip2_15335)
        # Assigning a type to the variable 'if_condition_15336' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'if_condition_15336', if_condition_15336)
        # SSA begins for if statement (line 306)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 307):
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_15337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        str_15338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 29), 'str', 'bztar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 28), list_15337, str_15338)
        
        # Getting the type of 'sdist' (line 307)
        sdist_15339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'sdist')
        # Setting the type of the member 'formats' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), sdist_15339, 'formats', list_15337)
        # SSA branch for the else part of an if statement (line 306)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 309):
        
        # Obtaining an instance of the builtin type 'list' (line 309)
        list_15340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 309)
        # Adding element type (line 309)
        str_15341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 29), 'str', 'gztar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 28), list_15340, str_15341)
        
        # Getting the type of 'sdist' (line 309)
        sdist_15342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'sdist')
        # Setting the type of the member 'formats' of a type (line 309)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), sdist_15342, 'formats', list_15340)
        # SSA join for if statement (line 306)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to run_command(...): (line 310)
        # Processing the call arguments (line 310)
        str_15345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 25), 'str', 'sdist')
        # Processing the call keyword arguments (line 310)
        kwargs_15346 = {}
        # Getting the type of 'self' (line 310)
        self_15343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self', False)
        # Obtaining the member 'run_command' of a type (line 310)
        run_command_15344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_15343, 'run_command')
        # Calling run_command(args, kwargs) (line 310)
        run_command_call_result_15347 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), run_command_15344, *[str_15345], **kwargs_15346)
        
        
        # Assigning a Name to a Attribute (line 311):
        # Getting the type of 'saved_dist_files' (line 311)
        saved_dist_files_15348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 39), 'saved_dist_files')
        # Getting the type of 'self' (line 311)
        self_15349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'self')
        # Obtaining the member 'distribution' of a type (line 311)
        distribution_15350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), self_15349, 'distribution')
        # Setting the type of the member 'dist_files' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), distribution_15350, 'dist_files', saved_dist_files_15348)
        
        # Assigning a Subscript to a Name (line 313):
        
        # Obtaining the type of the subscript
        int_15351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 43), 'int')
        
        # Call to get_archive_files(...): (line 313)
        # Processing the call keyword arguments (line 313)
        kwargs_15354 = {}
        # Getting the type of 'sdist' (line 313)
        sdist_15352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 17), 'sdist', False)
        # Obtaining the member 'get_archive_files' of a type (line 313)
        get_archive_files_15353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 17), sdist_15352, 'get_archive_files')
        # Calling get_archive_files(args, kwargs) (line 313)
        get_archive_files_call_result_15355 = invoke(stypy.reporting.localization.Localization(__file__, 313, 17), get_archive_files_15353, *[], **kwargs_15354)
        
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___15356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 17), get_archive_files_call_result_15355, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_15357 = invoke(stypy.reporting.localization.Localization(__file__, 313, 17), getitem___15356, int_15351)
        
        # Assigning a type to the variable 'source' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'source', subscript_call_result_15357)
        
        # Assigning a Subscript to a Name (line 314):
        
        # Obtaining the type of the subscript
        str_15358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 29), 'str', 'SOURCES')
        # Getting the type of 'rpm_dir' (line 314)
        rpm_dir_15359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 21), 'rpm_dir')
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___15360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 21), rpm_dir_15359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_15361 = invoke(stypy.reporting.localization.Localization(__file__, 314, 21), getitem___15360, str_15358)
        
        # Assigning a type to the variable 'source_dir' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'source_dir', subscript_call_result_15361)
        
        # Call to copy_file(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'source' (line 315)
        source_15364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'source', False)
        # Getting the type of 'source_dir' (line 315)
        source_dir_15365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 31), 'source_dir', False)
        # Processing the call keyword arguments (line 315)
        kwargs_15366 = {}
        # Getting the type of 'self' (line 315)
        self_15362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 315)
        copy_file_15363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_15362, 'copy_file')
        # Calling copy_file(args, kwargs) (line 315)
        copy_file_call_result_15367 = invoke(stypy.reporting.localization.Localization(__file__, 315, 8), copy_file_15363, *[source_15364, source_dir_15365], **kwargs_15366)
        
        
        # Getting the type of 'self' (line 317)
        self_15368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'self')
        # Obtaining the member 'icon' of a type (line 317)
        icon_15369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 11), self_15368, 'icon')
        # Testing the type of an if condition (line 317)
        if_condition_15370 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 8), icon_15369)
        # Assigning a type to the variable 'if_condition_15370' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'if_condition_15370', if_condition_15370)
        # SSA begins for if statement (line 317)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to exists(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'self' (line 318)
        self_15374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 30), 'self', False)
        # Obtaining the member 'icon' of a type (line 318)
        icon_15375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 30), self_15374, 'icon')
        # Processing the call keyword arguments (line 318)
        kwargs_15376 = {}
        # Getting the type of 'os' (line 318)
        os_15371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 318)
        path_15372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 15), os_15371, 'path')
        # Obtaining the member 'exists' of a type (line 318)
        exists_15373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 15), path_15372, 'exists')
        # Calling exists(args, kwargs) (line 318)
        exists_call_result_15377 = invoke(stypy.reporting.localization.Localization(__file__, 318, 15), exists_15373, *[icon_15375], **kwargs_15376)
        
        # Testing the type of an if condition (line 318)
        if_condition_15378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 12), exists_call_result_15377)
        # Assigning a type to the variable 'if_condition_15378' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'if_condition_15378', if_condition_15378)
        # SSA begins for if statement (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy_file(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'self' (line 319)
        self_15381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 31), 'self', False)
        # Obtaining the member 'icon' of a type (line 319)
        icon_15382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 31), self_15381, 'icon')
        # Getting the type of 'source_dir' (line 319)
        source_dir_15383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 42), 'source_dir', False)
        # Processing the call keyword arguments (line 319)
        kwargs_15384 = {}
        # Getting the type of 'self' (line 319)
        self_15379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 16), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 319)
        copy_file_15380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 16), self_15379, 'copy_file')
        # Calling copy_file(args, kwargs) (line 319)
        copy_file_call_result_15385 = invoke(stypy.reporting.localization.Localization(__file__, 319, 16), copy_file_15380, *[icon_15382, source_dir_15383], **kwargs_15384)
        
        # SSA branch for the else part of an if statement (line 318)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'DistutilsFileError' (line 321)
        DistutilsFileError_15386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 22), 'DistutilsFileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 321, 16), DistutilsFileError_15386, 'raise parameter', BaseException)
        # SSA join for if statement (line 318)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 317)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 326)
        # Processing the call arguments (line 326)
        str_15389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 17), 'str', 'building RPMs')
        # Processing the call keyword arguments (line 326)
        kwargs_15390 = {}
        # Getting the type of 'log' (line 326)
        log_15387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 326)
        info_15388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), log_15387, 'info')
        # Calling info(args, kwargs) (line 326)
        info_call_result_15391 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), info_15388, *[str_15389], **kwargs_15390)
        
        
        # Assigning a List to a Name (line 327):
        
        # Obtaining an instance of the builtin type 'list' (line 327)
        list_15392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 327)
        # Adding element type (line 327)
        str_15393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 19), 'str', 'rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 18), list_15392, str_15393)
        
        # Assigning a type to the variable 'rpm_cmd' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'rpm_cmd', list_15392)
        
        
        # Evaluating a boolean operation
        
        # Call to exists(...): (line 328)
        # Processing the call arguments (line 328)
        str_15397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 26), 'str', '/usr/bin/rpmbuild')
        # Processing the call keyword arguments (line 328)
        kwargs_15398 = {}
        # Getting the type of 'os' (line 328)
        os_15394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 328)
        path_15395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 11), os_15394, 'path')
        # Obtaining the member 'exists' of a type (line 328)
        exists_15396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 11), path_15395, 'exists')
        # Calling exists(args, kwargs) (line 328)
        exists_call_result_15399 = invoke(stypy.reporting.localization.Localization(__file__, 328, 11), exists_15396, *[str_15397], **kwargs_15398)
        
        
        # Call to exists(...): (line 329)
        # Processing the call arguments (line 329)
        str_15403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 26), 'str', '/bin/rpmbuild')
        # Processing the call keyword arguments (line 329)
        kwargs_15404 = {}
        # Getting the type of 'os' (line 329)
        os_15400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 329)
        path_15401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 11), os_15400, 'path')
        # Obtaining the member 'exists' of a type (line 329)
        exists_15402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 11), path_15401, 'exists')
        # Calling exists(args, kwargs) (line 329)
        exists_call_result_15405 = invoke(stypy.reporting.localization.Localization(__file__, 329, 11), exists_15402, *[str_15403], **kwargs_15404)
        
        # Applying the binary operator 'or' (line 328)
        result_or_keyword_15406 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 11), 'or', exists_call_result_15399, exists_call_result_15405)
        
        # Testing the type of an if condition (line 328)
        if_condition_15407 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 8), result_or_keyword_15406)
        # Assigning a type to the variable 'if_condition_15407' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'if_condition_15407', if_condition_15407)
        # SSA begins for if statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 330):
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_15408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        str_15409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 23), 'str', 'rpmbuild')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 22), list_15408, str_15409)
        
        # Assigning a type to the variable 'rpm_cmd' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'rpm_cmd', list_15408)
        # SSA join for if statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 332)
        self_15410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 11), 'self')
        # Obtaining the member 'source_only' of a type (line 332)
        source_only_15411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 11), self_15410, 'source_only')
        # Testing the type of an if condition (line 332)
        if_condition_15412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 8), source_only_15411)
        # Assigning a type to the variable 'if_condition_15412' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'if_condition_15412', if_condition_15412)
        # SSA begins for if statement (line 332)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 333)
        # Processing the call arguments (line 333)
        str_15415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 27), 'str', '-bs')
        # Processing the call keyword arguments (line 333)
        kwargs_15416 = {}
        # Getting the type of 'rpm_cmd' (line 333)
        rpm_cmd_15413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'rpm_cmd', False)
        # Obtaining the member 'append' of a type (line 333)
        append_15414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), rpm_cmd_15413, 'append')
        # Calling append(args, kwargs) (line 333)
        append_call_result_15417 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), append_15414, *[str_15415], **kwargs_15416)
        
        # SSA branch for the else part of an if statement (line 332)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 334)
        self_15418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 13), 'self')
        # Obtaining the member 'binary_only' of a type (line 334)
        binary_only_15419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 13), self_15418, 'binary_only')
        # Testing the type of an if condition (line 334)
        if_condition_15420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 13), binary_only_15419)
        # Assigning a type to the variable 'if_condition_15420' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 13), 'if_condition_15420', if_condition_15420)
        # SSA begins for if statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 335)
        # Processing the call arguments (line 335)
        str_15423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 27), 'str', '-bb')
        # Processing the call keyword arguments (line 335)
        kwargs_15424 = {}
        # Getting the type of 'rpm_cmd' (line 335)
        rpm_cmd_15421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'rpm_cmd', False)
        # Obtaining the member 'append' of a type (line 335)
        append_15422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), rpm_cmd_15421, 'append')
        # Calling append(args, kwargs) (line 335)
        append_call_result_15425 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), append_15422, *[str_15423], **kwargs_15424)
        
        # SSA branch for the else part of an if statement (line 334)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 337)
        # Processing the call arguments (line 337)
        str_15428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 27), 'str', '-ba')
        # Processing the call keyword arguments (line 337)
        kwargs_15429 = {}
        # Getting the type of 'rpm_cmd' (line 337)
        rpm_cmd_15426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'rpm_cmd', False)
        # Obtaining the member 'append' of a type (line 337)
        append_15427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), rpm_cmd_15426, 'append')
        # Calling append(args, kwargs) (line 337)
        append_call_result_15430 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), append_15427, *[str_15428], **kwargs_15429)
        
        # SSA join for if statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 332)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 338)
        self_15431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'self')
        # Obtaining the member 'rpm3_mode' of a type (line 338)
        rpm3_mode_15432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 11), self_15431, 'rpm3_mode')
        # Testing the type of an if condition (line 338)
        if_condition_15433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 8), rpm3_mode_15432)
        # Assigning a type to the variable 'if_condition_15433' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'if_condition_15433', if_condition_15433)
        # SSA begins for if statement (line 338)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 339)
        # Processing the call arguments (line 339)
        
        # Obtaining an instance of the builtin type 'list' (line 339)
        list_15436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 339)
        # Adding element type (line 339)
        str_15437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 28), 'str', '--define')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 27), list_15436, str_15437)
        # Adding element type (line 339)
        str_15438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 29), 'str', '_topdir %s')
        
        # Call to abspath(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'self' (line 340)
        self_15442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 60), 'self', False)
        # Obtaining the member 'rpm_base' of a type (line 340)
        rpm_base_15443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 60), self_15442, 'rpm_base')
        # Processing the call keyword arguments (line 340)
        kwargs_15444 = {}
        # Getting the type of 'os' (line 340)
        os_15439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 340)
        path_15440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 44), os_15439, 'path')
        # Obtaining the member 'abspath' of a type (line 340)
        abspath_15441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 44), path_15440, 'abspath')
        # Calling abspath(args, kwargs) (line 340)
        abspath_call_result_15445 = invoke(stypy.reporting.localization.Localization(__file__, 340, 44), abspath_15441, *[rpm_base_15443], **kwargs_15444)
        
        # Applying the binary operator '%' (line 340)
        result_mod_15446 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 29), '%', str_15438, abspath_call_result_15445)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 27), list_15436, result_mod_15446)
        
        # Processing the call keyword arguments (line 339)
        kwargs_15447 = {}
        # Getting the type of 'rpm_cmd' (line 339)
        rpm_cmd_15434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'rpm_cmd', False)
        # Obtaining the member 'extend' of a type (line 339)
        extend_15435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), rpm_cmd_15434, 'extend')
        # Calling extend(args, kwargs) (line 339)
        extend_call_result_15448 = invoke(stypy.reporting.localization.Localization(__file__, 339, 12), extend_15435, *[list_15436], **kwargs_15447)
        
        # SSA join for if statement (line 338)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 341)
        self_15449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'self')
        # Obtaining the member 'keep_temp' of a type (line 341)
        keep_temp_15450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), self_15449, 'keep_temp')
        # Applying the 'not' unary operator (line 341)
        result_not__15451 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 11), 'not', keep_temp_15450)
        
        # Testing the type of an if condition (line 341)
        if_condition_15452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 8), result_not__15451)
        # Assigning a type to the variable 'if_condition_15452' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'if_condition_15452', if_condition_15452)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 342)
        # Processing the call arguments (line 342)
        str_15455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 27), 'str', '--clean')
        # Processing the call keyword arguments (line 342)
        kwargs_15456 = {}
        # Getting the type of 'rpm_cmd' (line 342)
        rpm_cmd_15453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'rpm_cmd', False)
        # Obtaining the member 'append' of a type (line 342)
        append_15454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), rpm_cmd_15453, 'append')
        # Calling append(args, kwargs) (line 342)
        append_call_result_15457 = invoke(stypy.reporting.localization.Localization(__file__, 342, 12), append_15454, *[str_15455], **kwargs_15456)
        
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 344)
        self_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 11), 'self')
        # Obtaining the member 'quiet' of a type (line 344)
        quiet_15459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 11), self_15458, 'quiet')
        # Testing the type of an if condition (line 344)
        if_condition_15460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 344, 8), quiet_15459)
        # Assigning a type to the variable 'if_condition_15460' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'if_condition_15460', if_condition_15460)
        # SSA begins for if statement (line 344)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 345)
        # Processing the call arguments (line 345)
        str_15463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 27), 'str', '--quiet')
        # Processing the call keyword arguments (line 345)
        kwargs_15464 = {}
        # Getting the type of 'rpm_cmd' (line 345)
        rpm_cmd_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'rpm_cmd', False)
        # Obtaining the member 'append' of a type (line 345)
        append_15462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), rpm_cmd_15461, 'append')
        # Calling append(args, kwargs) (line 345)
        append_call_result_15465 = invoke(stypy.reporting.localization.Localization(__file__, 345, 12), append_15462, *[str_15463], **kwargs_15464)
        
        # SSA join for if statement (line 344)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'spec_path' (line 347)
        spec_path_15468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 23), 'spec_path', False)
        # Processing the call keyword arguments (line 347)
        kwargs_15469 = {}
        # Getting the type of 'rpm_cmd' (line 347)
        rpm_cmd_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'rpm_cmd', False)
        # Obtaining the member 'append' of a type (line 347)
        append_15467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), rpm_cmd_15466, 'append')
        # Calling append(args, kwargs) (line 347)
        append_call_result_15470 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), append_15467, *[spec_path_15468], **kwargs_15469)
        
        
        # Assigning a Str to a Name (line 352):
        str_15471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 21), 'str', '%{name}-%{version}-%{release}')
        # Assigning a type to the variable 'nvr_string' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'nvr_string', str_15471)
        
        # Assigning a BinOp to a Name (line 353):
        # Getting the type of 'nvr_string' (line 353)
        nvr_string_15472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 18), 'nvr_string')
        str_15473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 31), 'str', '.src.rpm')
        # Applying the binary operator '+' (line 353)
        result_add_15474 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 18), '+', nvr_string_15472, str_15473)
        
        # Assigning a type to the variable 'src_rpm' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'src_rpm', result_add_15474)
        
        # Assigning a BinOp to a Name (line 354):
        str_15475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 22), 'str', '%{arch}/')
        # Getting the type of 'nvr_string' (line 354)
        nvr_string_15476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 35), 'nvr_string')
        # Applying the binary operator '+' (line 354)
        result_add_15477 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 22), '+', str_15475, nvr_string_15476)
        
        str_15478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 48), 'str', '.%{arch}.rpm')
        # Applying the binary operator '+' (line 354)
        result_add_15479 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 46), '+', result_add_15477, str_15478)
        
        # Assigning a type to the variable 'non_src_rpm' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'non_src_rpm', result_add_15479)
        
        # Assigning a BinOp to a Name (line 355):
        str_15480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 16), 'str', "rpm -q --qf '%s %s\\n' --specfile '%s'")
        
        # Obtaining an instance of the builtin type 'tuple' (line 356)
        tuple_15481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 356)
        # Adding element type (line 356)
        # Getting the type of 'src_rpm' (line 356)
        src_rpm_15482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'src_rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 12), tuple_15481, src_rpm_15482)
        # Adding element type (line 356)
        # Getting the type of 'non_src_rpm' (line 356)
        non_src_rpm_15483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 21), 'non_src_rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 12), tuple_15481, non_src_rpm_15483)
        # Adding element type (line 356)
        # Getting the type of 'spec_path' (line 356)
        spec_path_15484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 34), 'spec_path')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 12), tuple_15481, spec_path_15484)
        
        # Applying the binary operator '%' (line 355)
        result_mod_15485 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 16), '%', str_15480, tuple_15481)
        
        # Assigning a type to the variable 'q_cmd' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'q_cmd', result_mod_15485)
        
        # Assigning a Call to a Name (line 358):
        
        # Call to popen(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'q_cmd' (line 358)
        q_cmd_15488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 23), 'q_cmd', False)
        # Processing the call keyword arguments (line 358)
        kwargs_15489 = {}
        # Getting the type of 'os' (line 358)
        os_15486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 14), 'os', False)
        # Obtaining the member 'popen' of a type (line 358)
        popen_15487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 14), os_15486, 'popen')
        # Calling popen(args, kwargs) (line 358)
        popen_call_result_15490 = invoke(stypy.reporting.localization.Localization(__file__, 358, 14), popen_15487, *[q_cmd_15488], **kwargs_15489)
        
        # Assigning a type to the variable 'out' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'out', popen_call_result_15490)
        
        # Try-finally block (line 359)
        
        # Assigning a List to a Name (line 360):
        
        # Obtaining an instance of the builtin type 'list' (line 360)
        list_15491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 360)
        
        # Assigning a type to the variable 'binary_rpms' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'binary_rpms', list_15491)
        
        # Assigning a Name to a Name (line 361):
        # Getting the type of 'None' (line 361)
        None_15492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 25), 'None')
        # Assigning a type to the variable 'source_rpm' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'source_rpm', None_15492)
        
        int_15493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 18), 'int')
        # Testing the type of an if condition (line 362)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 12), int_15493)
        # SSA begins for while statement (line 362)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 363):
        
        # Call to readline(...): (line 363)
        # Processing the call keyword arguments (line 363)
        kwargs_15496 = {}
        # Getting the type of 'out' (line 363)
        out_15494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 23), 'out', False)
        # Obtaining the member 'readline' of a type (line 363)
        readline_15495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 23), out_15494, 'readline')
        # Calling readline(args, kwargs) (line 363)
        readline_call_result_15497 = invoke(stypy.reporting.localization.Localization(__file__, 363, 23), readline_15495, *[], **kwargs_15496)
        
        # Assigning a type to the variable 'line' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'line', readline_call_result_15497)
        
        
        # Getting the type of 'line' (line 364)
        line_15498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'line')
        # Applying the 'not' unary operator (line 364)
        result_not__15499 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 19), 'not', line_15498)
        
        # Testing the type of an if condition (line 364)
        if_condition_15500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 16), result_not__15499)
        # Assigning a type to the variable 'if_condition_15500' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'if_condition_15500', if_condition_15500)
        # SSA begins for if statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 366):
        
        # Call to split(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Call to strip(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'line' (line 366)
        line_15505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 46), 'line', False)
        # Processing the call keyword arguments (line 366)
        kwargs_15506 = {}
        # Getting the type of 'string' (line 366)
        string_15503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 33), 'string', False)
        # Obtaining the member 'strip' of a type (line 366)
        strip_15504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 33), string_15503, 'strip')
        # Calling strip(args, kwargs) (line 366)
        strip_call_result_15507 = invoke(stypy.reporting.localization.Localization(__file__, 366, 33), strip_15504, *[line_15505], **kwargs_15506)
        
        # Processing the call keyword arguments (line 366)
        kwargs_15508 = {}
        # Getting the type of 'string' (line 366)
        string_15501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 20), 'string', False)
        # Obtaining the member 'split' of a type (line 366)
        split_15502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 20), string_15501, 'split')
        # Calling split(args, kwargs) (line 366)
        split_call_result_15509 = invoke(stypy.reporting.localization.Localization(__file__, 366, 20), split_15502, *[strip_call_result_15507], **kwargs_15508)
        
        # Assigning a type to the variable 'l' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'l', split_call_result_15509)
        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'l' (line 367)
        l_15511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 27), 'l', False)
        # Processing the call keyword arguments (line 367)
        kwargs_15512 = {}
        # Getting the type of 'len' (line 367)
        len_15510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 23), 'len', False)
        # Calling len(args, kwargs) (line 367)
        len_call_result_15513 = invoke(stypy.reporting.localization.Localization(__file__, 367, 23), len_15510, *[l_15511], **kwargs_15512)
        
        int_15514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 33), 'int')
        # Applying the binary operator '==' (line 367)
        result_eq_15515 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 23), '==', len_call_result_15513, int_15514)
        
        
        # Call to append(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Obtaining the type of the subscript
        int_15518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 37), 'int')
        # Getting the type of 'l' (line 368)
        l_15519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 35), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___15520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 35), l_15519, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_15521 = invoke(stypy.reporting.localization.Localization(__file__, 368, 35), getitem___15520, int_15518)
        
        # Processing the call keyword arguments (line 368)
        kwargs_15522 = {}
        # Getting the type of 'binary_rpms' (line 368)
        binary_rpms_15516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'binary_rpms', False)
        # Obtaining the member 'append' of a type (line 368)
        append_15517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 16), binary_rpms_15516, 'append')
        # Calling append(args, kwargs) (line 368)
        append_call_result_15523 = invoke(stypy.reporting.localization.Localization(__file__, 368, 16), append_15517, *[subscript_call_result_15521], **kwargs_15522)
        
        
        # Type idiom detected: calculating its left and rigth part (line 370)
        # Getting the type of 'source_rpm' (line 370)
        source_rpm_15524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 19), 'source_rpm')
        # Getting the type of 'None' (line 370)
        None_15525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 33), 'None')
        
        (may_be_15526, more_types_in_union_15527) = may_be_none(source_rpm_15524, None_15525)

        if may_be_15526:

            if more_types_in_union_15527:
                # Runtime conditional SSA (line 370)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 371):
            
            # Obtaining the type of the subscript
            int_15528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 35), 'int')
            # Getting the type of 'l' (line 371)
            l_15529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 33), 'l')
            # Obtaining the member '__getitem__' of a type (line 371)
            getitem___15530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 33), l_15529, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 371)
            subscript_call_result_15531 = invoke(stypy.reporting.localization.Localization(__file__, 371, 33), getitem___15530, int_15528)
            
            # Assigning a type to the variable 'source_rpm' (line 371)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 20), 'source_rpm', subscript_call_result_15531)

            if more_types_in_union_15527:
                # SSA join for if statement (line 370)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for while statement (line 362)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 373):
        
        # Call to close(...): (line 373)
        # Processing the call keyword arguments (line 373)
        kwargs_15534 = {}
        # Getting the type of 'out' (line 373)
        out_15532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 21), 'out', False)
        # Obtaining the member 'close' of a type (line 373)
        close_15533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 21), out_15532, 'close')
        # Calling close(args, kwargs) (line 373)
        close_call_result_15535 = invoke(stypy.reporting.localization.Localization(__file__, 373, 21), close_15533, *[], **kwargs_15534)
        
        # Assigning a type to the variable 'status' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'status', close_call_result_15535)
        
        # Getting the type of 'status' (line 374)
        status_15536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'status')
        # Testing the type of an if condition (line 374)
        if_condition_15537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 12), status_15536)
        # Assigning a type to the variable 'if_condition_15537' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'if_condition_15537', if_condition_15537)
        # SSA begins for if statement (line 374)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsExecError(...): (line 375)
        # Processing the call arguments (line 375)
        str_15539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 41), 'str', 'Failed to execute: %s')
        
        # Call to repr(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'q_cmd' (line 375)
        q_cmd_15541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 72), 'q_cmd', False)
        # Processing the call keyword arguments (line 375)
        kwargs_15542 = {}
        # Getting the type of 'repr' (line 375)
        repr_15540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 67), 'repr', False)
        # Calling repr(args, kwargs) (line 375)
        repr_call_result_15543 = invoke(stypy.reporting.localization.Localization(__file__, 375, 67), repr_15540, *[q_cmd_15541], **kwargs_15542)
        
        # Applying the binary operator '%' (line 375)
        result_mod_15544 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 41), '%', str_15539, repr_call_result_15543)
        
        # Processing the call keyword arguments (line 375)
        kwargs_15545 = {}
        # Getting the type of 'DistutilsExecError' (line 375)
        DistutilsExecError_15538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 22), 'DistutilsExecError', False)
        # Calling DistutilsExecError(args, kwargs) (line 375)
        DistutilsExecError_call_result_15546 = invoke(stypy.reporting.localization.Localization(__file__, 375, 22), DistutilsExecError_15538, *[result_mod_15544], **kwargs_15545)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 375, 16), DistutilsExecError_call_result_15546, 'raise parameter', BaseException)
        # SSA join for if statement (line 374)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 359)
        
        # Call to close(...): (line 378)
        # Processing the call keyword arguments (line 378)
        kwargs_15549 = {}
        # Getting the type of 'out' (line 378)
        out_15547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'out', False)
        # Obtaining the member 'close' of a type (line 378)
        close_15548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 12), out_15547, 'close')
        # Calling close(args, kwargs) (line 378)
        close_call_result_15550 = invoke(stypy.reporting.localization.Localization(__file__, 378, 12), close_15548, *[], **kwargs_15549)
        
        
        
        # Call to spawn(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'rpm_cmd' (line 380)
        rpm_cmd_15553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'rpm_cmd', False)
        # Processing the call keyword arguments (line 380)
        kwargs_15554 = {}
        # Getting the type of 'self' (line 380)
        self_15551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'self', False)
        # Obtaining the member 'spawn' of a type (line 380)
        spawn_15552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), self_15551, 'spawn')
        # Calling spawn(args, kwargs) (line 380)
        spawn_call_result_15555 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), spawn_15552, *[rpm_cmd_15553], **kwargs_15554)
        
        
        
        # Getting the type of 'self' (line 382)
        self_15556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 15), 'self')
        # Obtaining the member 'dry_run' of a type (line 382)
        dry_run_15557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 15), self_15556, 'dry_run')
        # Applying the 'not' unary operator (line 382)
        result_not__15558 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 11), 'not', dry_run_15557)
        
        # Testing the type of an if condition (line 382)
        if_condition_15559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 8), result_not__15558)
        # Assigning a type to the variable 'if_condition_15559' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'if_condition_15559', if_condition_15559)
        # SSA begins for if statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to has_ext_modules(...): (line 383)
        # Processing the call keyword arguments (line 383)
        kwargs_15563 = {}
        # Getting the type of 'self' (line 383)
        self_15560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 383)
        distribution_15561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 15), self_15560, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 383)
        has_ext_modules_15562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 15), distribution_15561, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 383)
        has_ext_modules_call_result_15564 = invoke(stypy.reporting.localization.Localization(__file__, 383, 15), has_ext_modules_15562, *[], **kwargs_15563)
        
        # Testing the type of an if condition (line 383)
        if_condition_15565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 12), has_ext_modules_call_result_15564)
        # Assigning a type to the variable 'if_condition_15565' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'if_condition_15565', if_condition_15565)
        # SSA begins for if statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 384):
        
        # Call to get_python_version(...): (line 384)
        # Processing the call keyword arguments (line 384)
        kwargs_15567 = {}
        # Getting the type of 'get_python_version' (line 384)
        get_python_version_15566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), 'get_python_version', False)
        # Calling get_python_version(args, kwargs) (line 384)
        get_python_version_call_result_15568 = invoke(stypy.reporting.localization.Localization(__file__, 384, 28), get_python_version_15566, *[], **kwargs_15567)
        
        # Assigning a type to the variable 'pyversion' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'pyversion', get_python_version_call_result_15568)
        # SSA branch for the else part of an if statement (line 383)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 386):
        str_15569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 28), 'str', 'any')
        # Assigning a type to the variable 'pyversion' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'pyversion', str_15569)
        # SSA join for if statement (line 383)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 388)
        self_15570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 19), 'self')
        # Obtaining the member 'binary_only' of a type (line 388)
        binary_only_15571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 19), self_15570, 'binary_only')
        # Applying the 'not' unary operator (line 388)
        result_not__15572 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 15), 'not', binary_only_15571)
        
        # Testing the type of an if condition (line 388)
        if_condition_15573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 12), result_not__15572)
        # Assigning a type to the variable 'if_condition_15573' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'if_condition_15573', if_condition_15573)
        # SSA begins for if statement (line 388)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 389):
        
        # Call to join(...): (line 389)
        # Processing the call arguments (line 389)
        
        # Obtaining the type of the subscript
        str_15577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 44), 'str', 'SRPMS')
        # Getting the type of 'rpm_dir' (line 389)
        rpm_dir_15578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 36), 'rpm_dir', False)
        # Obtaining the member '__getitem__' of a type (line 389)
        getitem___15579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 36), rpm_dir_15578, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 389)
        subscript_call_result_15580 = invoke(stypy.reporting.localization.Localization(__file__, 389, 36), getitem___15579, str_15577)
        
        # Getting the type of 'source_rpm' (line 389)
        source_rpm_15581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 54), 'source_rpm', False)
        # Processing the call keyword arguments (line 389)
        kwargs_15582 = {}
        # Getting the type of 'os' (line 389)
        os_15574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 389)
        path_15575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 23), os_15574, 'path')
        # Obtaining the member 'join' of a type (line 389)
        join_15576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 23), path_15575, 'join')
        # Calling join(args, kwargs) (line 389)
        join_call_result_15583 = invoke(stypy.reporting.localization.Localization(__file__, 389, 23), join_15576, *[subscript_call_result_15580, source_rpm_15581], **kwargs_15582)
        
        # Assigning a type to the variable 'srpm' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'srpm', join_call_result_15583)
        # Evaluating assert statement condition
        
        # Call to exists(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'srpm' (line 390)
        srpm_15587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 38), 'srpm', False)
        # Processing the call keyword arguments (line 390)
        kwargs_15588 = {}
        # Getting the type of 'os' (line 390)
        os_15584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 390)
        path_15585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 23), os_15584, 'path')
        # Obtaining the member 'exists' of a type (line 390)
        exists_15586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 23), path_15585, 'exists')
        # Calling exists(args, kwargs) (line 390)
        exists_call_result_15589 = invoke(stypy.reporting.localization.Localization(__file__, 390, 23), exists_15586, *[srpm_15587], **kwargs_15588)
        
        
        # Call to move_file(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'srpm' (line 391)
        srpm_15592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 31), 'srpm', False)
        # Getting the type of 'self' (line 391)
        self_15593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 37), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 391)
        dist_dir_15594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 37), self_15593, 'dist_dir')
        # Processing the call keyword arguments (line 391)
        kwargs_15595 = {}
        # Getting the type of 'self' (line 391)
        self_15590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'self', False)
        # Obtaining the member 'move_file' of a type (line 391)
        move_file_15591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 16), self_15590, 'move_file')
        # Calling move_file(args, kwargs) (line 391)
        move_file_call_result_15596 = invoke(stypy.reporting.localization.Localization(__file__, 391, 16), move_file_15591, *[srpm_15592, dist_dir_15594], **kwargs_15595)
        
        
        # Assigning a Call to a Name (line 392):
        
        # Call to join(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'self' (line 392)
        self_15600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 40), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 392)
        dist_dir_15601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 40), self_15600, 'dist_dir')
        # Getting the type of 'source_rpm' (line 392)
        source_rpm_15602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 55), 'source_rpm', False)
        # Processing the call keyword arguments (line 392)
        kwargs_15603 = {}
        # Getting the type of 'os' (line 392)
        os_15597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 392)
        path_15598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 27), os_15597, 'path')
        # Obtaining the member 'join' of a type (line 392)
        join_15599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 27), path_15598, 'join')
        # Calling join(args, kwargs) (line 392)
        join_call_result_15604 = invoke(stypy.reporting.localization.Localization(__file__, 392, 27), join_15599, *[dist_dir_15601, source_rpm_15602], **kwargs_15603)
        
        # Assigning a type to the variable 'filename' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'filename', join_call_result_15604)
        
        # Call to append(...): (line 393)
        # Processing the call arguments (line 393)
        
        # Obtaining an instance of the builtin type 'tuple' (line 394)
        tuple_15609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 394)
        # Adding element type (line 394)
        str_15610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 21), 'str', 'bdist_rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 21), tuple_15609, str_15610)
        # Adding element type (line 394)
        # Getting the type of 'pyversion' (line 394)
        pyversion_15611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 34), 'pyversion', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 21), tuple_15609, pyversion_15611)
        # Adding element type (line 394)
        # Getting the type of 'filename' (line 394)
        filename_15612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 45), 'filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 21), tuple_15609, filename_15612)
        
        # Processing the call keyword arguments (line 393)
        kwargs_15613 = {}
        # Getting the type of 'self' (line 393)
        self_15605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'self', False)
        # Obtaining the member 'distribution' of a type (line 393)
        distribution_15606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 16), self_15605, 'distribution')
        # Obtaining the member 'dist_files' of a type (line 393)
        dist_files_15607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 16), distribution_15606, 'dist_files')
        # Obtaining the member 'append' of a type (line 393)
        append_15608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 16), dist_files_15607, 'append')
        # Calling append(args, kwargs) (line 393)
        append_call_result_15614 = invoke(stypy.reporting.localization.Localization(__file__, 393, 16), append_15608, *[tuple_15609], **kwargs_15613)
        
        # SSA join for if statement (line 388)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 396)
        self_15615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'self')
        # Obtaining the member 'source_only' of a type (line 396)
        source_only_15616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 19), self_15615, 'source_only')
        # Applying the 'not' unary operator (line 396)
        result_not__15617 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 15), 'not', source_only_15616)
        
        # Testing the type of an if condition (line 396)
        if_condition_15618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 12), result_not__15617)
        # Assigning a type to the variable 'if_condition_15618' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'if_condition_15618', if_condition_15618)
        # SSA begins for if statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'binary_rpms' (line 397)
        binary_rpms_15619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 27), 'binary_rpms')
        # Testing the type of a for loop iterable (line 397)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 397, 16), binary_rpms_15619)
        # Getting the type of the for loop variable (line 397)
        for_loop_var_15620 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 397, 16), binary_rpms_15619)
        # Assigning a type to the variable 'rpm' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'rpm', for_loop_var_15620)
        # SSA begins for a for statement (line 397)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 398):
        
        # Call to join(...): (line 398)
        # Processing the call arguments (line 398)
        
        # Obtaining the type of the subscript
        str_15624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 47), 'str', 'RPMS')
        # Getting the type of 'rpm_dir' (line 398)
        rpm_dir_15625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 39), 'rpm_dir', False)
        # Obtaining the member '__getitem__' of a type (line 398)
        getitem___15626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 39), rpm_dir_15625, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 398)
        subscript_call_result_15627 = invoke(stypy.reporting.localization.Localization(__file__, 398, 39), getitem___15626, str_15624)
        
        # Getting the type of 'rpm' (line 398)
        rpm_15628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 56), 'rpm', False)
        # Processing the call keyword arguments (line 398)
        kwargs_15629 = {}
        # Getting the type of 'os' (line 398)
        os_15621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 398)
        path_15622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 26), os_15621, 'path')
        # Obtaining the member 'join' of a type (line 398)
        join_15623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 26), path_15622, 'join')
        # Calling join(args, kwargs) (line 398)
        join_call_result_15630 = invoke(stypy.reporting.localization.Localization(__file__, 398, 26), join_15623, *[subscript_call_result_15627, rpm_15628], **kwargs_15629)
        
        # Assigning a type to the variable 'rpm' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 20), 'rpm', join_call_result_15630)
        
        
        # Call to exists(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'rpm' (line 399)
        rpm_15634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 38), 'rpm', False)
        # Processing the call keyword arguments (line 399)
        kwargs_15635 = {}
        # Getting the type of 'os' (line 399)
        os_15631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 399)
        path_15632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 23), os_15631, 'path')
        # Obtaining the member 'exists' of a type (line 399)
        exists_15633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 23), path_15632, 'exists')
        # Calling exists(args, kwargs) (line 399)
        exists_call_result_15636 = invoke(stypy.reporting.localization.Localization(__file__, 399, 23), exists_15633, *[rpm_15634], **kwargs_15635)
        
        # Testing the type of an if condition (line 399)
        if_condition_15637 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 20), exists_call_result_15636)
        # Assigning a type to the variable 'if_condition_15637' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 20), 'if_condition_15637', if_condition_15637)
        # SSA begins for if statement (line 399)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to move_file(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'rpm' (line 400)
        rpm_15640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 39), 'rpm', False)
        # Getting the type of 'self' (line 400)
        self_15641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 44), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 400)
        dist_dir_15642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 44), self_15641, 'dist_dir')
        # Processing the call keyword arguments (line 400)
        kwargs_15643 = {}
        # Getting the type of 'self' (line 400)
        self_15638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 24), 'self', False)
        # Obtaining the member 'move_file' of a type (line 400)
        move_file_15639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 24), self_15638, 'move_file')
        # Calling move_file(args, kwargs) (line 400)
        move_file_call_result_15644 = invoke(stypy.reporting.localization.Localization(__file__, 400, 24), move_file_15639, *[rpm_15640, dist_dir_15642], **kwargs_15643)
        
        
        # Assigning a Call to a Name (line 401):
        
        # Call to join(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'self' (line 401)
        self_15648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 48), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 401)
        dist_dir_15649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 48), self_15648, 'dist_dir')
        
        # Call to basename(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'rpm' (line 402)
        rpm_15653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 65), 'rpm', False)
        # Processing the call keyword arguments (line 402)
        kwargs_15654 = {}
        # Getting the type of 'os' (line 402)
        os_15650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 48), 'os', False)
        # Obtaining the member 'path' of a type (line 402)
        path_15651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 48), os_15650, 'path')
        # Obtaining the member 'basename' of a type (line 402)
        basename_15652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 48), path_15651, 'basename')
        # Calling basename(args, kwargs) (line 402)
        basename_call_result_15655 = invoke(stypy.reporting.localization.Localization(__file__, 402, 48), basename_15652, *[rpm_15653], **kwargs_15654)
        
        # Processing the call keyword arguments (line 401)
        kwargs_15656 = {}
        # Getting the type of 'os' (line 401)
        os_15645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 35), 'os', False)
        # Obtaining the member 'path' of a type (line 401)
        path_15646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 35), os_15645, 'path')
        # Obtaining the member 'join' of a type (line 401)
        join_15647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 35), path_15646, 'join')
        # Calling join(args, kwargs) (line 401)
        join_call_result_15657 = invoke(stypy.reporting.localization.Localization(__file__, 401, 35), join_15647, *[dist_dir_15649, basename_call_result_15655], **kwargs_15656)
        
        # Assigning a type to the variable 'filename' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 24), 'filename', join_call_result_15657)
        
        # Call to append(...): (line 403)
        # Processing the call arguments (line 403)
        
        # Obtaining an instance of the builtin type 'tuple' (line 404)
        tuple_15662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 404)
        # Adding element type (line 404)
        str_15663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 29), 'str', 'bdist_rpm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 29), tuple_15662, str_15663)
        # Adding element type (line 404)
        # Getting the type of 'pyversion' (line 404)
        pyversion_15664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 42), 'pyversion', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 29), tuple_15662, pyversion_15664)
        # Adding element type (line 404)
        # Getting the type of 'filename' (line 404)
        filename_15665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 53), 'filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 29), tuple_15662, filename_15665)
        
        # Processing the call keyword arguments (line 403)
        kwargs_15666 = {}
        # Getting the type of 'self' (line 403)
        self_15658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 24), 'self', False)
        # Obtaining the member 'distribution' of a type (line 403)
        distribution_15659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 24), self_15658, 'distribution')
        # Obtaining the member 'dist_files' of a type (line 403)
        dist_files_15660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 24), distribution_15659, 'dist_files')
        # Obtaining the member 'append' of a type (line 403)
        append_15661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 24), dist_files_15660, 'append')
        # Calling append(args, kwargs) (line 403)
        append_call_result_15667 = invoke(stypy.reporting.localization.Localization(__file__, 403, 24), append_15661, *[tuple_15662], **kwargs_15666)
        
        # SSA join for if statement (line 399)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 382)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_15668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15668)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_15668


    @norecursion
    def _dist_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dist_path'
        module_type_store = module_type_store.open_function_context('_dist_path', 407, 4, False)
        # Assigning a type to the variable 'self' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_localization', localization)
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_function_name', 'bdist_rpm._dist_path')
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_param_names_list', ['path'])
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_rpm._dist_path.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm._dist_path', ['path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dist_path', localization, ['path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dist_path(...)' code ##################

        
        # Call to join(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'self' (line 408)
        self_15672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 408)
        dist_dir_15673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 28), self_15672, 'dist_dir')
        
        # Call to basename(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'path' (line 408)
        path_15677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 60), 'path', False)
        # Processing the call keyword arguments (line 408)
        kwargs_15678 = {}
        # Getting the type of 'os' (line 408)
        os_15674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 43), 'os', False)
        # Obtaining the member 'path' of a type (line 408)
        path_15675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 43), os_15674, 'path')
        # Obtaining the member 'basename' of a type (line 408)
        basename_15676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 43), path_15675, 'basename')
        # Calling basename(args, kwargs) (line 408)
        basename_call_result_15679 = invoke(stypy.reporting.localization.Localization(__file__, 408, 43), basename_15676, *[path_15677], **kwargs_15678)
        
        # Processing the call keyword arguments (line 408)
        kwargs_15680 = {}
        # Getting the type of 'os' (line 408)
        os_15669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 408)
        path_15670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 15), os_15669, 'path')
        # Obtaining the member 'join' of a type (line 408)
        join_15671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 15), path_15670, 'join')
        # Calling join(args, kwargs) (line 408)
        join_call_result_15681 = invoke(stypy.reporting.localization.Localization(__file__, 408, 15), join_15671, *[dist_dir_15673, basename_call_result_15679], **kwargs_15680)
        
        # Assigning a type to the variable 'stypy_return_type' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'stypy_return_type', join_call_result_15681)
        
        # ################# End of '_dist_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dist_path' in the type store
        # Getting the type of 'stypy_return_type' (line 407)
        stypy_return_type_15682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dist_path'
        return stypy_return_type_15682


    @norecursion
    def _make_spec_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_make_spec_file'
        module_type_store = module_type_store.open_function_context('_make_spec_file', 410, 4, False)
        # Assigning a type to the variable 'self' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_localization', localization)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_function_name', 'bdist_rpm._make_spec_file')
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_rpm._make_spec_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm._make_spec_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_make_spec_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_make_spec_file(...)' code ##################

        str_15683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, (-1)), 'str', 'Generate the text of an RPM spec file and return it as a\n        list of strings (one per line).\n        ')
        
        # Assigning a List to a Name (line 415):
        
        # Obtaining an instance of the builtin type 'list' (line 415)
        list_15684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 415)
        # Adding element type (line 415)
        str_15685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 12), 'str', '%define name ')
        
        # Call to get_name(...): (line 416)
        # Processing the call keyword arguments (line 416)
        kwargs_15689 = {}
        # Getting the type of 'self' (line 416)
        self_15686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 30), 'self', False)
        # Obtaining the member 'distribution' of a type (line 416)
        distribution_15687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 30), self_15686, 'distribution')
        # Obtaining the member 'get_name' of a type (line 416)
        get_name_15688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 30), distribution_15687, 'get_name')
        # Calling get_name(args, kwargs) (line 416)
        get_name_call_result_15690 = invoke(stypy.reporting.localization.Localization(__file__, 416, 30), get_name_15688, *[], **kwargs_15689)
        
        # Applying the binary operator '+' (line 416)
        result_add_15691 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 12), '+', str_15685, get_name_call_result_15690)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 20), list_15684, result_add_15691)
        # Adding element type (line 415)
        str_15692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 12), 'str', '%define version ')
        
        # Call to replace(...): (line 417)
        # Processing the call arguments (line 417)
        str_15699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 73), 'str', '-')
        str_15700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 77), 'str', '_')
        # Processing the call keyword arguments (line 417)
        kwargs_15701 = {}
        
        # Call to get_version(...): (line 417)
        # Processing the call keyword arguments (line 417)
        kwargs_15696 = {}
        # Getting the type of 'self' (line 417)
        self_15693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 33), 'self', False)
        # Obtaining the member 'distribution' of a type (line 417)
        distribution_15694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 33), self_15693, 'distribution')
        # Obtaining the member 'get_version' of a type (line 417)
        get_version_15695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 33), distribution_15694, 'get_version')
        # Calling get_version(args, kwargs) (line 417)
        get_version_call_result_15697 = invoke(stypy.reporting.localization.Localization(__file__, 417, 33), get_version_15695, *[], **kwargs_15696)
        
        # Obtaining the member 'replace' of a type (line 417)
        replace_15698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 33), get_version_call_result_15697, 'replace')
        # Calling replace(args, kwargs) (line 417)
        replace_call_result_15702 = invoke(stypy.reporting.localization.Localization(__file__, 417, 33), replace_15698, *[str_15699, str_15700], **kwargs_15701)
        
        # Applying the binary operator '+' (line 417)
        result_add_15703 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 12), '+', str_15692, replace_call_result_15702)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 20), list_15684, result_add_15703)
        # Adding element type (line 415)
        str_15704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 12), 'str', '%define unmangled_version ')
        
        # Call to get_version(...): (line 418)
        # Processing the call keyword arguments (line 418)
        kwargs_15708 = {}
        # Getting the type of 'self' (line 418)
        self_15705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 43), 'self', False)
        # Obtaining the member 'distribution' of a type (line 418)
        distribution_15706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 43), self_15705, 'distribution')
        # Obtaining the member 'get_version' of a type (line 418)
        get_version_15707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 43), distribution_15706, 'get_version')
        # Calling get_version(args, kwargs) (line 418)
        get_version_call_result_15709 = invoke(stypy.reporting.localization.Localization(__file__, 418, 43), get_version_15707, *[], **kwargs_15708)
        
        # Applying the binary operator '+' (line 418)
        result_add_15710 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 12), '+', str_15704, get_version_call_result_15709)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 20), list_15684, result_add_15710)
        # Adding element type (line 415)
        str_15711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 12), 'str', '%define release ')
        
        # Call to replace(...): (line 419)
        # Processing the call arguments (line 419)
        str_15715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 54), 'str', '-')
        str_15716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 58), 'str', '_')
        # Processing the call keyword arguments (line 419)
        kwargs_15717 = {}
        # Getting the type of 'self' (line 419)
        self_15712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 33), 'self', False)
        # Obtaining the member 'release' of a type (line 419)
        release_15713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 33), self_15712, 'release')
        # Obtaining the member 'replace' of a type (line 419)
        replace_15714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 33), release_15713, 'replace')
        # Calling replace(args, kwargs) (line 419)
        replace_call_result_15718 = invoke(stypy.reporting.localization.Localization(__file__, 419, 33), replace_15714, *[str_15715, str_15716], **kwargs_15717)
        
        # Applying the binary operator '+' (line 419)
        result_add_15719 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 12), '+', str_15711, replace_call_result_15718)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 20), list_15684, result_add_15719)
        # Adding element type (line 415)
        str_15720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 12), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 20), list_15684, str_15720)
        # Adding element type (line 415)
        str_15721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 12), 'str', 'Summary: ')
        
        # Call to get_description(...): (line 421)
        # Processing the call keyword arguments (line 421)
        kwargs_15725 = {}
        # Getting the type of 'self' (line 421)
        self_15722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 26), 'self', False)
        # Obtaining the member 'distribution' of a type (line 421)
        distribution_15723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 26), self_15722, 'distribution')
        # Obtaining the member 'get_description' of a type (line 421)
        get_description_15724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 26), distribution_15723, 'get_description')
        # Calling get_description(args, kwargs) (line 421)
        get_description_call_result_15726 = invoke(stypy.reporting.localization.Localization(__file__, 421, 26), get_description_15724, *[], **kwargs_15725)
        
        # Applying the binary operator '+' (line 421)
        result_add_15727 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 12), '+', str_15721, get_description_call_result_15726)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 20), list_15684, result_add_15727)
        
        # Assigning a type to the variable 'spec_file' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'spec_file', list_15684)
        
        # Call to extend(...): (line 431)
        # Processing the call arguments (line 431)
        
        # Obtaining an instance of the builtin type 'list' (line 431)
        list_15730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 431)
        # Adding element type (line 431)
        str_15731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 12), 'str', 'Name: %{name}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 25), list_15730, str_15731)
        # Adding element type (line 431)
        str_15732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 12), 'str', 'Version: %{version}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 25), list_15730, str_15732)
        # Adding element type (line 431)
        str_15733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 12), 'str', 'Release: %{release}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 25), list_15730, str_15733)
        
        # Processing the call keyword arguments (line 431)
        kwargs_15734 = {}
        # Getting the type of 'spec_file' (line 431)
        spec_file_15728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'spec_file', False)
        # Obtaining the member 'extend' of a type (line 431)
        extend_15729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), spec_file_15728, 'extend')
        # Calling extend(args, kwargs) (line 431)
        extend_call_result_15735 = invoke(stypy.reporting.localization.Localization(__file__, 431, 8), extend_15729, *[list_15730], **kwargs_15734)
        
        
        # Getting the type of 'self' (line 439)
        self_15736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 11), 'self')
        # Obtaining the member 'use_bzip2' of a type (line 439)
        use_bzip2_15737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 11), self_15736, 'use_bzip2')
        # Testing the type of an if condition (line 439)
        if_condition_15738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 8), use_bzip2_15737)
        # Assigning a type to the variable 'if_condition_15738' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'if_condition_15738', if_condition_15738)
        # SSA begins for if statement (line 439)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 440)
        # Processing the call arguments (line 440)
        str_15741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 29), 'str', 'Source0: %{name}-%{unmangled_version}.tar.bz2')
        # Processing the call keyword arguments (line 440)
        kwargs_15742 = {}
        # Getting the type of 'spec_file' (line 440)
        spec_file_15739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 440)
        append_15740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 12), spec_file_15739, 'append')
        # Calling append(args, kwargs) (line 440)
        append_call_result_15743 = invoke(stypy.reporting.localization.Localization(__file__, 440, 12), append_15740, *[str_15741], **kwargs_15742)
        
        # SSA branch for the else part of an if statement (line 439)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 442)
        # Processing the call arguments (line 442)
        str_15746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 29), 'str', 'Source0: %{name}-%{unmangled_version}.tar.gz')
        # Processing the call keyword arguments (line 442)
        kwargs_15747 = {}
        # Getting the type of 'spec_file' (line 442)
        spec_file_15744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 442)
        append_15745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), spec_file_15744, 'append')
        # Calling append(args, kwargs) (line 442)
        append_call_result_15748 = invoke(stypy.reporting.localization.Localization(__file__, 442, 12), append_15745, *[str_15746], **kwargs_15747)
        
        # SSA join for if statement (line 439)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 444)
        # Processing the call arguments (line 444)
        
        # Obtaining an instance of the builtin type 'list' (line 444)
        list_15751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 444)
        # Adding element type (line 444)
        str_15752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 12), 'str', 'License: ')
        
        # Call to get_license(...): (line 445)
        # Processing the call keyword arguments (line 445)
        kwargs_15756 = {}
        # Getting the type of 'self' (line 445)
        self_15753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 26), 'self', False)
        # Obtaining the member 'distribution' of a type (line 445)
        distribution_15754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 26), self_15753, 'distribution')
        # Obtaining the member 'get_license' of a type (line 445)
        get_license_15755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 26), distribution_15754, 'get_license')
        # Calling get_license(args, kwargs) (line 445)
        get_license_call_result_15757 = invoke(stypy.reporting.localization.Localization(__file__, 445, 26), get_license_15755, *[], **kwargs_15756)
        
        # Applying the binary operator '+' (line 445)
        result_add_15758 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 12), '+', str_15752, get_license_call_result_15757)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 25), list_15751, result_add_15758)
        # Adding element type (line 444)
        str_15759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 12), 'str', 'Group: ')
        # Getting the type of 'self' (line 446)
        self_15760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 24), 'self', False)
        # Obtaining the member 'group' of a type (line 446)
        group_15761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 24), self_15760, 'group')
        # Applying the binary operator '+' (line 446)
        result_add_15762 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 12), '+', str_15759, group_15761)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 25), list_15751, result_add_15762)
        # Adding element type (line 444)
        str_15763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 12), 'str', 'BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 25), list_15751, str_15763)
        # Adding element type (line 444)
        str_15764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 12), 'str', 'Prefix: %{_prefix}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 25), list_15751, str_15764)
        
        # Processing the call keyword arguments (line 444)
        kwargs_15765 = {}
        # Getting the type of 'spec_file' (line 444)
        spec_file_15749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'spec_file', False)
        # Obtaining the member 'extend' of a type (line 444)
        extend_15750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), spec_file_15749, 'extend')
        # Calling extend(args, kwargs) (line 444)
        extend_call_result_15766 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), extend_15750, *[list_15751], **kwargs_15765)
        
        
        
        # Getting the type of 'self' (line 450)
        self_15767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 15), 'self')
        # Obtaining the member 'force_arch' of a type (line 450)
        force_arch_15768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 15), self_15767, 'force_arch')
        # Applying the 'not' unary operator (line 450)
        result_not__15769 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 11), 'not', force_arch_15768)
        
        # Testing the type of an if condition (line 450)
        if_condition_15770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 8), result_not__15769)
        # Assigning a type to the variable 'if_condition_15770' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'if_condition_15770', if_condition_15770)
        # SSA begins for if statement (line 450)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to has_ext_modules(...): (line 452)
        # Processing the call keyword arguments (line 452)
        kwargs_15774 = {}
        # Getting the type of 'self' (line 452)
        self_15771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 19), 'self', False)
        # Obtaining the member 'distribution' of a type (line 452)
        distribution_15772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 19), self_15771, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 452)
        has_ext_modules_15773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 19), distribution_15772, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 452)
        has_ext_modules_call_result_15775 = invoke(stypy.reporting.localization.Localization(__file__, 452, 19), has_ext_modules_15773, *[], **kwargs_15774)
        
        # Applying the 'not' unary operator (line 452)
        result_not__15776 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 15), 'not', has_ext_modules_call_result_15775)
        
        # Testing the type of an if condition (line 452)
        if_condition_15777 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 452, 12), result_not__15776)
        # Assigning a type to the variable 'if_condition_15777' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'if_condition_15777', if_condition_15777)
        # SSA begins for if statement (line 452)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 453)
        # Processing the call arguments (line 453)
        str_15780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 33), 'str', 'BuildArch: noarch')
        # Processing the call keyword arguments (line 453)
        kwargs_15781 = {}
        # Getting the type of 'spec_file' (line 453)
        spec_file_15778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 16), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 453)
        append_15779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 16), spec_file_15778, 'append')
        # Calling append(args, kwargs) (line 453)
        append_call_result_15782 = invoke(stypy.reporting.localization.Localization(__file__, 453, 16), append_15779, *[str_15780], **kwargs_15781)
        
        # SSA join for if statement (line 452)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 450)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 455)
        # Processing the call arguments (line 455)
        str_15785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 30), 'str', 'BuildArch: %s')
        # Getting the type of 'self' (line 455)
        self_15786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 48), 'self', False)
        # Obtaining the member 'force_arch' of a type (line 455)
        force_arch_15787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 48), self_15786, 'force_arch')
        # Applying the binary operator '%' (line 455)
        result_mod_15788 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 30), '%', str_15785, force_arch_15787)
        
        # Processing the call keyword arguments (line 455)
        kwargs_15789 = {}
        # Getting the type of 'spec_file' (line 455)
        spec_file_15783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 455)
        append_15784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 12), spec_file_15783, 'append')
        # Calling append(args, kwargs) (line 455)
        append_call_result_15790 = invoke(stypy.reporting.localization.Localization(__file__, 455, 12), append_15784, *[result_mod_15788], **kwargs_15789)
        
        # SSA join for if statement (line 450)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 457)
        tuple_15791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 457)
        # Adding element type (line 457)
        str_15792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 22), 'str', 'Vendor')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 22), tuple_15791, str_15792)
        # Adding element type (line 457)
        str_15793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 22), 'str', 'Packager')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 22), tuple_15791, str_15793)
        # Adding element type (line 457)
        str_15794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 22), 'str', 'Provides')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 22), tuple_15791, str_15794)
        # Adding element type (line 457)
        str_15795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 22), 'str', 'Requires')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 22), tuple_15791, str_15795)
        # Adding element type (line 457)
        str_15796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 22), 'str', 'Conflicts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 22), tuple_15791, str_15796)
        # Adding element type (line 457)
        str_15797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 22), 'str', 'Obsoletes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 22), tuple_15791, str_15797)
        
        # Testing the type of a for loop iterable (line 457)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 457, 8), tuple_15791)
        # Getting the type of the for loop variable (line 457)
        for_loop_var_15798 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 457, 8), tuple_15791)
        # Assigning a type to the variable 'field' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'field', for_loop_var_15798)
        # SSA begins for a for statement (line 457)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 464):
        
        # Call to getattr(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'self' (line 464)
        self_15800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 26), 'self', False)
        
        # Call to lower(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'field' (line 464)
        field_15803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 45), 'field', False)
        # Processing the call keyword arguments (line 464)
        kwargs_15804 = {}
        # Getting the type of 'string' (line 464)
        string_15801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 32), 'string', False)
        # Obtaining the member 'lower' of a type (line 464)
        lower_15802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 32), string_15801, 'lower')
        # Calling lower(args, kwargs) (line 464)
        lower_call_result_15805 = invoke(stypy.reporting.localization.Localization(__file__, 464, 32), lower_15802, *[field_15803], **kwargs_15804)
        
        # Processing the call keyword arguments (line 464)
        kwargs_15806 = {}
        # Getting the type of 'getattr' (line 464)
        getattr_15799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 464)
        getattr_call_result_15807 = invoke(stypy.reporting.localization.Localization(__file__, 464, 18), getattr_15799, *[self_15800, lower_call_result_15805], **kwargs_15806)
        
        # Assigning a type to the variable 'val' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'val', getattr_call_result_15807)
        
        # Type idiom detected: calculating its left and rigth part (line 465)
        # Getting the type of 'list' (line 465)
        list_15808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 31), 'list')
        # Getting the type of 'val' (line 465)
        val_15809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 26), 'val')
        
        (may_be_15810, more_types_in_union_15811) = may_be_subtype(list_15808, val_15809)

        if may_be_15810:

            if more_types_in_union_15811:
                # Runtime conditional SSA (line 465)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'val' (line 465)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'val', remove_not_subtype_from_union(val_15809, list))
            
            # Call to append(...): (line 466)
            # Processing the call arguments (line 466)
            str_15814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 33), 'str', '%s: %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 466)
            tuple_15815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 45), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 466)
            # Adding element type (line 466)
            # Getting the type of 'field' (line 466)
            field_15816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 45), 'field', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 45), tuple_15815, field_15816)
            # Adding element type (line 466)
            
            # Call to join(...): (line 466)
            # Processing the call arguments (line 466)
            # Getting the type of 'val' (line 466)
            val_15819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 64), 'val', False)
            # Processing the call keyword arguments (line 466)
            kwargs_15820 = {}
            # Getting the type of 'string' (line 466)
            string_15817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 52), 'string', False)
            # Obtaining the member 'join' of a type (line 466)
            join_15818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 52), string_15817, 'join')
            # Calling join(args, kwargs) (line 466)
            join_call_result_15821 = invoke(stypy.reporting.localization.Localization(__file__, 466, 52), join_15818, *[val_15819], **kwargs_15820)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 45), tuple_15815, join_call_result_15821)
            
            # Applying the binary operator '%' (line 466)
            result_mod_15822 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 33), '%', str_15814, tuple_15815)
            
            # Processing the call keyword arguments (line 466)
            kwargs_15823 = {}
            # Getting the type of 'spec_file' (line 466)
            spec_file_15812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'spec_file', False)
            # Obtaining the member 'append' of a type (line 466)
            append_15813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 16), spec_file_15812, 'append')
            # Calling append(args, kwargs) (line 466)
            append_call_result_15824 = invoke(stypy.reporting.localization.Localization(__file__, 466, 16), append_15813, *[result_mod_15822], **kwargs_15823)
            

            if more_types_in_union_15811:
                # Runtime conditional SSA for else branch (line 465)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_15810) or more_types_in_union_15811):
            # Assigning a type to the variable 'val' (line 465)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'val', remove_subtype_from_union(val_15809, list))
            
            # Type idiom detected: calculating its left and rigth part (line 467)
            # Getting the type of 'val' (line 467)
            val_15825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 17), 'val')
            # Getting the type of 'None' (line 467)
            None_15826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 28), 'None')
            
            (may_be_15827, more_types_in_union_15828) = may_not_be_none(val_15825, None_15826)

            if may_be_15827:

                if more_types_in_union_15828:
                    # Runtime conditional SSA (line 467)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to append(...): (line 468)
                # Processing the call arguments (line 468)
                str_15831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 33), 'str', '%s: %s')
                
                # Obtaining an instance of the builtin type 'tuple' (line 468)
                tuple_15832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 45), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 468)
                # Adding element type (line 468)
                # Getting the type of 'field' (line 468)
                field_15833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 45), 'field', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 45), tuple_15832, field_15833)
                # Adding element type (line 468)
                # Getting the type of 'val' (line 468)
                val_15834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 52), 'val', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 45), tuple_15832, val_15834)
                
                # Applying the binary operator '%' (line 468)
                result_mod_15835 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 33), '%', str_15831, tuple_15832)
                
                # Processing the call keyword arguments (line 468)
                kwargs_15836 = {}
                # Getting the type of 'spec_file' (line 468)
                spec_file_15829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'spec_file', False)
                # Obtaining the member 'append' of a type (line 468)
                append_15830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 16), spec_file_15829, 'append')
                # Calling append(args, kwargs) (line 468)
                append_call_result_15837 = invoke(stypy.reporting.localization.Localization(__file__, 468, 16), append_15830, *[result_mod_15835], **kwargs_15836)
                

                if more_types_in_union_15828:
                    # SSA join for if statement (line 467)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_15810 and more_types_in_union_15811):
                # SSA join for if statement (line 465)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to get_url(...): (line 471)
        # Processing the call keyword arguments (line 471)
        kwargs_15841 = {}
        # Getting the type of 'self' (line 471)
        self_15838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 471)
        distribution_15839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 11), self_15838, 'distribution')
        # Obtaining the member 'get_url' of a type (line 471)
        get_url_15840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 11), distribution_15839, 'get_url')
        # Calling get_url(args, kwargs) (line 471)
        get_url_call_result_15842 = invoke(stypy.reporting.localization.Localization(__file__, 471, 11), get_url_15840, *[], **kwargs_15841)
        
        str_15843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 42), 'str', 'UNKNOWN')
        # Applying the binary operator '!=' (line 471)
        result_ne_15844 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 11), '!=', get_url_call_result_15842, str_15843)
        
        # Testing the type of an if condition (line 471)
        if_condition_15845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 471, 8), result_ne_15844)
        # Assigning a type to the variable 'if_condition_15845' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'if_condition_15845', if_condition_15845)
        # SSA begins for if statement (line 471)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 472)
        # Processing the call arguments (line 472)
        str_15848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 29), 'str', 'Url: ')
        
        # Call to get_url(...): (line 472)
        # Processing the call keyword arguments (line 472)
        kwargs_15852 = {}
        # Getting the type of 'self' (line 472)
        self_15849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 39), 'self', False)
        # Obtaining the member 'distribution' of a type (line 472)
        distribution_15850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 39), self_15849, 'distribution')
        # Obtaining the member 'get_url' of a type (line 472)
        get_url_15851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 39), distribution_15850, 'get_url')
        # Calling get_url(args, kwargs) (line 472)
        get_url_call_result_15853 = invoke(stypy.reporting.localization.Localization(__file__, 472, 39), get_url_15851, *[], **kwargs_15852)
        
        # Applying the binary operator '+' (line 472)
        result_add_15854 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 29), '+', str_15848, get_url_call_result_15853)
        
        # Processing the call keyword arguments (line 472)
        kwargs_15855 = {}
        # Getting the type of 'spec_file' (line 472)
        spec_file_15846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 472)
        append_15847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), spec_file_15846, 'append')
        # Calling append(args, kwargs) (line 472)
        append_call_result_15856 = invoke(stypy.reporting.localization.Localization(__file__, 472, 12), append_15847, *[result_add_15854], **kwargs_15855)
        
        # SSA join for if statement (line 471)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 474)
        self_15857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'self')
        # Obtaining the member 'distribution_name' of a type (line 474)
        distribution_name_15858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 11), self_15857, 'distribution_name')
        # Testing the type of an if condition (line 474)
        if_condition_15859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 8), distribution_name_15858)
        # Assigning a type to the variable 'if_condition_15859' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'if_condition_15859', if_condition_15859)
        # SSA begins for if statement (line 474)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 475)
        # Processing the call arguments (line 475)
        str_15862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 29), 'str', 'Distribution: ')
        # Getting the type of 'self' (line 475)
        self_15863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 48), 'self', False)
        # Obtaining the member 'distribution_name' of a type (line 475)
        distribution_name_15864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 48), self_15863, 'distribution_name')
        # Applying the binary operator '+' (line 475)
        result_add_15865 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 29), '+', str_15862, distribution_name_15864)
        
        # Processing the call keyword arguments (line 475)
        kwargs_15866 = {}
        # Getting the type of 'spec_file' (line 475)
        spec_file_15860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 475)
        append_15861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), spec_file_15860, 'append')
        # Calling append(args, kwargs) (line 475)
        append_call_result_15867 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), append_15861, *[result_add_15865], **kwargs_15866)
        
        # SSA join for if statement (line 474)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 477)
        self_15868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 11), 'self')
        # Obtaining the member 'build_requires' of a type (line 477)
        build_requires_15869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 11), self_15868, 'build_requires')
        # Testing the type of an if condition (line 477)
        if_condition_15870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 8), build_requires_15869)
        # Assigning a type to the variable 'if_condition_15870' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'if_condition_15870', if_condition_15870)
        # SSA begins for if statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 478)
        # Processing the call arguments (line 478)
        str_15873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 29), 'str', 'BuildRequires: ')
        
        # Call to join(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'self' (line 479)
        self_15876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 41), 'self', False)
        # Obtaining the member 'build_requires' of a type (line 479)
        build_requires_15877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 41), self_15876, 'build_requires')
        # Processing the call keyword arguments (line 479)
        kwargs_15878 = {}
        # Getting the type of 'string' (line 479)
        string_15874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 29), 'string', False)
        # Obtaining the member 'join' of a type (line 479)
        join_15875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 29), string_15874, 'join')
        # Calling join(args, kwargs) (line 479)
        join_call_result_15879 = invoke(stypy.reporting.localization.Localization(__file__, 479, 29), join_15875, *[build_requires_15877], **kwargs_15878)
        
        # Applying the binary operator '+' (line 478)
        result_add_15880 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 29), '+', str_15873, join_call_result_15879)
        
        # Processing the call keyword arguments (line 478)
        kwargs_15881 = {}
        # Getting the type of 'spec_file' (line 478)
        spec_file_15871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 478)
        append_15872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), spec_file_15871, 'append')
        # Calling append(args, kwargs) (line 478)
        append_call_result_15882 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), append_15872, *[result_add_15880], **kwargs_15881)
        
        # SSA join for if statement (line 477)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 481)
        self_15883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 11), 'self')
        # Obtaining the member 'icon' of a type (line 481)
        icon_15884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 11), self_15883, 'icon')
        # Testing the type of an if condition (line 481)
        if_condition_15885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 8), icon_15884)
        # Assigning a type to the variable 'if_condition_15885' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'if_condition_15885', if_condition_15885)
        # SSA begins for if statement (line 481)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 482)
        # Processing the call arguments (line 482)
        str_15888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 29), 'str', 'Icon: ')
        
        # Call to basename(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'self' (line 482)
        self_15892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 57), 'self', False)
        # Obtaining the member 'icon' of a type (line 482)
        icon_15893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 57), self_15892, 'icon')
        # Processing the call keyword arguments (line 482)
        kwargs_15894 = {}
        # Getting the type of 'os' (line 482)
        os_15889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 40), 'os', False)
        # Obtaining the member 'path' of a type (line 482)
        path_15890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 40), os_15889, 'path')
        # Obtaining the member 'basename' of a type (line 482)
        basename_15891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 40), path_15890, 'basename')
        # Calling basename(args, kwargs) (line 482)
        basename_call_result_15895 = invoke(stypy.reporting.localization.Localization(__file__, 482, 40), basename_15891, *[icon_15893], **kwargs_15894)
        
        # Applying the binary operator '+' (line 482)
        result_add_15896 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 29), '+', str_15888, basename_call_result_15895)
        
        # Processing the call keyword arguments (line 482)
        kwargs_15897 = {}
        # Getting the type of 'spec_file' (line 482)
        spec_file_15886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 482)
        append_15887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 12), spec_file_15886, 'append')
        # Calling append(args, kwargs) (line 482)
        append_call_result_15898 = invoke(stypy.reporting.localization.Localization(__file__, 482, 12), append_15887, *[result_add_15896], **kwargs_15897)
        
        # SSA join for if statement (line 481)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 484)
        self_15899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), 'self')
        # Obtaining the member 'no_autoreq' of a type (line 484)
        no_autoreq_15900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 11), self_15899, 'no_autoreq')
        # Testing the type of an if condition (line 484)
        if_condition_15901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 484, 8), no_autoreq_15900)
        # Assigning a type to the variable 'if_condition_15901' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'if_condition_15901', if_condition_15901)
        # SSA begins for if statement (line 484)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 485)
        # Processing the call arguments (line 485)
        str_15904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 29), 'str', 'AutoReq: 0')
        # Processing the call keyword arguments (line 485)
        kwargs_15905 = {}
        # Getting the type of 'spec_file' (line 485)
        spec_file_15902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 485)
        append_15903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 12), spec_file_15902, 'append')
        # Calling append(args, kwargs) (line 485)
        append_call_result_15906 = invoke(stypy.reporting.localization.Localization(__file__, 485, 12), append_15903, *[str_15904], **kwargs_15905)
        
        # SSA join for if statement (line 484)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 487)
        # Processing the call arguments (line 487)
        
        # Obtaining an instance of the builtin type 'list' (line 487)
        list_15909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 487)
        # Adding element type (line 487)
        str_15910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 12), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 25), list_15909, str_15910)
        # Adding element type (line 487)
        str_15911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 12), 'str', '%description')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 25), list_15909, str_15911)
        # Adding element type (line 487)
        
        # Call to get_long_description(...): (line 490)
        # Processing the call keyword arguments (line 490)
        kwargs_15915 = {}
        # Getting the type of 'self' (line 490)
        self_15912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'self', False)
        # Obtaining the member 'distribution' of a type (line 490)
        distribution_15913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), self_15912, 'distribution')
        # Obtaining the member 'get_long_description' of a type (line 490)
        get_long_description_15914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), distribution_15913, 'get_long_description')
        # Calling get_long_description(args, kwargs) (line 490)
        get_long_description_call_result_15916 = invoke(stypy.reporting.localization.Localization(__file__, 490, 12), get_long_description_15914, *[], **kwargs_15915)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 25), list_15909, get_long_description_call_result_15916)
        
        # Processing the call keyword arguments (line 487)
        kwargs_15917 = {}
        # Getting the type of 'spec_file' (line 487)
        spec_file_15907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'spec_file', False)
        # Obtaining the member 'extend' of a type (line 487)
        extend_15908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), spec_file_15907, 'extend')
        # Calling extend(args, kwargs) (line 487)
        extend_call_result_15918 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), extend_15908, *[list_15909], **kwargs_15917)
        
        
        # Assigning a BinOp to a Name (line 505):
        str_15919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 25), 'str', '%s %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 505)
        tuple_15920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 505)
        # Adding element type (line 505)
        # Getting the type of 'self' (line 505)
        self_15921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 36), 'self')
        # Obtaining the member 'python' of a type (line 505)
        python_15922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 36), self_15921, 'python')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 36), tuple_15920, python_15922)
        # Adding element type (line 505)
        
        # Call to basename(...): (line 505)
        # Processing the call arguments (line 505)
        
        # Obtaining the type of the subscript
        int_15926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 74), 'int')
        # Getting the type of 'sys' (line 505)
        sys_15927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 65), 'sys', False)
        # Obtaining the member 'argv' of a type (line 505)
        argv_15928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 65), sys_15927, 'argv')
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___15929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 65), argv_15928, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_15930 = invoke(stypy.reporting.localization.Localization(__file__, 505, 65), getitem___15929, int_15926)
        
        # Processing the call keyword arguments (line 505)
        kwargs_15931 = {}
        # Getting the type of 'os' (line 505)
        os_15923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 48), 'os', False)
        # Obtaining the member 'path' of a type (line 505)
        path_15924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 48), os_15923, 'path')
        # Obtaining the member 'basename' of a type (line 505)
        basename_15925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 48), path_15924, 'basename')
        # Calling basename(args, kwargs) (line 505)
        basename_call_result_15932 = invoke(stypy.reporting.localization.Localization(__file__, 505, 48), basename_15925, *[subscript_call_result_15930], **kwargs_15931)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 36), tuple_15920, basename_call_result_15932)
        
        # Applying the binary operator '%' (line 505)
        result_mod_15933 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 25), '%', str_15919, tuple_15920)
        
        # Assigning a type to the variable 'def_setup_call' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'def_setup_call', result_mod_15933)
        
        # Assigning a BinOp to a Name (line 506):
        str_15934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 20), 'str', '%s build')
        # Getting the type of 'def_setup_call' (line 506)
        def_setup_call_15935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 33), 'def_setup_call')
        # Applying the binary operator '%' (line 506)
        result_mod_15936 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 20), '%', str_15934, def_setup_call_15935)
        
        # Assigning a type to the variable 'def_build' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'def_build', result_mod_15936)
        
        # Getting the type of 'self' (line 507)
        self_15937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), 'self')
        # Obtaining the member 'use_rpm_opt_flags' of a type (line 507)
        use_rpm_opt_flags_15938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 11), self_15937, 'use_rpm_opt_flags')
        # Testing the type of an if condition (line 507)
        if_condition_15939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 8), use_rpm_opt_flags_15938)
        # Assigning a type to the variable 'if_condition_15939' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'if_condition_15939', if_condition_15939)
        # SSA begins for if statement (line 507)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 508):
        str_15940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 24), 'str', 'env CFLAGS="$RPM_OPT_FLAGS" ')
        # Getting the type of 'def_build' (line 508)
        def_build_15941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 57), 'def_build')
        # Applying the binary operator '+' (line 508)
        result_add_15942 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 24), '+', str_15940, def_build_15941)
        
        # Assigning a type to the variable 'def_build' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'def_build', result_add_15942)
        # SSA join for if statement (line 507)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 516):
        str_15943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 23), 'str', '%s install -O1 --root=$RPM_BUILD_ROOT --record=INSTALLED_FILES')
        # Getting the type of 'def_setup_call' (line 517)
        def_setup_call_15944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 53), 'def_setup_call')
        # Applying the binary operator '%' (line 516)
        result_mod_15945 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 22), '%', str_15943, def_setup_call_15944)
        
        # Assigning a type to the variable 'install_cmd' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'install_cmd', result_mod_15945)
        
        # Assigning a List to a Name (line 519):
        
        # Obtaining an instance of the builtin type 'list' (line 519)
        list_15946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 519)
        # Adding element type (line 519)
        
        # Obtaining an instance of the builtin type 'tuple' (line 520)
        tuple_15947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 520)
        # Adding element type (line 520)
        str_15948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 13), 'str', 'prep')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 13), tuple_15947, str_15948)
        # Adding element type (line 520)
        str_15949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 21), 'str', 'prep_script')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 13), tuple_15947, str_15949)
        # Adding element type (line 520)
        str_15950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 36), 'str', '%setup -n %{name}-%{unmangled_version}')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 520, 13), tuple_15947, str_15950)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 25), list_15946, tuple_15947)
        # Adding element type (line 519)
        
        # Obtaining an instance of the builtin type 'tuple' (line 521)
        tuple_15951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 521)
        # Adding element type (line 521)
        str_15952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 13), 'str', 'build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 13), tuple_15951, str_15952)
        # Adding element type (line 521)
        str_15953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 22), 'str', 'build_script')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 13), tuple_15951, str_15953)
        # Adding element type (line 521)
        # Getting the type of 'def_build' (line 521)
        def_build_15954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 38), 'def_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 13), tuple_15951, def_build_15954)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 25), list_15946, tuple_15951)
        # Adding element type (line 519)
        
        # Obtaining an instance of the builtin type 'tuple' (line 522)
        tuple_15955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 522)
        # Adding element type (line 522)
        str_15956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 13), 'str', 'install')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 13), tuple_15955, str_15956)
        # Adding element type (line 522)
        str_15957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 24), 'str', 'install_script')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 13), tuple_15955, str_15957)
        # Adding element type (line 522)
        # Getting the type of 'install_cmd' (line 522)
        install_cmd_15958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 42), 'install_cmd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 522, 13), tuple_15955, install_cmd_15958)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 25), list_15946, tuple_15955)
        # Adding element type (line 519)
        
        # Obtaining an instance of the builtin type 'tuple' (line 523)
        tuple_15959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 523)
        # Adding element type (line 523)
        str_15960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 13), 'str', 'clean')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 13), tuple_15959, str_15960)
        # Adding element type (line 523)
        str_15961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 22), 'str', 'clean_script')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 13), tuple_15959, str_15961)
        # Adding element type (line 523)
        str_15962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 38), 'str', 'rm -rf $RPM_BUILD_ROOT')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 13), tuple_15959, str_15962)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 25), list_15946, tuple_15959)
        # Adding element type (line 519)
        
        # Obtaining an instance of the builtin type 'tuple' (line 524)
        tuple_15963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 524)
        # Adding element type (line 524)
        str_15964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 13), 'str', 'verifyscript')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_15963, str_15964)
        # Adding element type (line 524)
        str_15965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 29), 'str', 'verify_script')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_15963, str_15965)
        # Adding element type (line 524)
        # Getting the type of 'None' (line 524)
        None_15966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 46), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 13), tuple_15963, None_15966)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 25), list_15946, tuple_15963)
        # Adding element type (line 519)
        
        # Obtaining an instance of the builtin type 'tuple' (line 525)
        tuple_15967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 525)
        # Adding element type (line 525)
        str_15968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 13), 'str', 'pre')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 13), tuple_15967, str_15968)
        # Adding element type (line 525)
        str_15969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 20), 'str', 'pre_install')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 13), tuple_15967, str_15969)
        # Adding element type (line 525)
        # Getting the type of 'None' (line 525)
        None_15970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 35), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 13), tuple_15967, None_15970)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 25), list_15946, tuple_15967)
        # Adding element type (line 519)
        
        # Obtaining an instance of the builtin type 'tuple' (line 526)
        tuple_15971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 526)
        # Adding element type (line 526)
        str_15972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 13), 'str', 'post')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 13), tuple_15971, str_15972)
        # Adding element type (line 526)
        str_15973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 21), 'str', 'post_install')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 13), tuple_15971, str_15973)
        # Adding element type (line 526)
        # Getting the type of 'None' (line 526)
        None_15974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 37), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 13), tuple_15971, None_15974)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 25), list_15946, tuple_15971)
        # Adding element type (line 519)
        
        # Obtaining an instance of the builtin type 'tuple' (line 527)
        tuple_15975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 527)
        # Adding element type (line 527)
        str_15976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 13), 'str', 'preun')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 13), tuple_15975, str_15976)
        # Adding element type (line 527)
        str_15977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 22), 'str', 'pre_uninstall')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 13), tuple_15975, str_15977)
        # Adding element type (line 527)
        # Getting the type of 'None' (line 527)
        None_15978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 39), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 13), tuple_15975, None_15978)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 25), list_15946, tuple_15975)
        # Adding element type (line 519)
        
        # Obtaining an instance of the builtin type 'tuple' (line 528)
        tuple_15979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 528)
        # Adding element type (line 528)
        str_15980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 13), 'str', 'postun')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 13), tuple_15979, str_15980)
        # Adding element type (line 528)
        str_15981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 23), 'str', 'post_uninstall')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 13), tuple_15979, str_15981)
        # Adding element type (line 528)
        # Getting the type of 'None' (line 528)
        None_15982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 41), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 13), tuple_15979, None_15982)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 25), list_15946, tuple_15979)
        
        # Assigning a type to the variable 'script_options' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'script_options', list_15946)
        
        # Getting the type of 'script_options' (line 531)
        script_options_15983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 40), 'script_options')
        # Testing the type of a for loop iterable (line 531)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 531, 8), script_options_15983)
        # Getting the type of the for loop variable (line 531)
        for_loop_var_15984 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 531, 8), script_options_15983)
        # Assigning a type to the variable 'rpm_opt' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'rpm_opt', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 8), for_loop_var_15984))
        # Assigning a type to the variable 'attr' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'attr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 8), for_loop_var_15984))
        # Assigning a type to the variable 'default' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'default', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 8), for_loop_var_15984))
        # SSA begins for a for statement (line 531)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 534):
        
        # Call to getattr(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'self' (line 534)
        self_15986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 26), 'self', False)
        # Getting the type of 'attr' (line 534)
        attr_15987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 32), 'attr', False)
        # Processing the call keyword arguments (line 534)
        kwargs_15988 = {}
        # Getting the type of 'getattr' (line 534)
        getattr_15985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 534)
        getattr_call_result_15989 = invoke(stypy.reporting.localization.Localization(__file__, 534, 18), getattr_15985, *[self_15986, attr_15987], **kwargs_15988)
        
        # Assigning a type to the variable 'val' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'val', getattr_call_result_15989)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'val' (line 535)
        val_15990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'val')
        # Getting the type of 'default' (line 535)
        default_15991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 22), 'default')
        # Applying the binary operator 'or' (line 535)
        result_or_keyword_15992 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 15), 'or', val_15990, default_15991)
        
        # Testing the type of an if condition (line 535)
        if_condition_15993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 12), result_or_keyword_15992)
        # Assigning a type to the variable 'if_condition_15993' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'if_condition_15993', if_condition_15993)
        # SSA begins for if statement (line 535)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 536)
        # Processing the call arguments (line 536)
        
        # Obtaining an instance of the builtin type 'list' (line 536)
        list_15996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 536)
        # Adding element type (line 536)
        str_15997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 20), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 33), list_15996, str_15997)
        # Adding element type (line 536)
        str_15998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 20), 'str', '%')
        # Getting the type of 'rpm_opt' (line 538)
        rpm_opt_15999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 26), 'rpm_opt', False)
        # Applying the binary operator '+' (line 538)
        result_add_16000 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 20), '+', str_15998, rpm_opt_15999)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 33), list_15996, result_add_16000)
        
        # Processing the call keyword arguments (line 536)
        kwargs_16001 = {}
        # Getting the type of 'spec_file' (line 536)
        spec_file_15994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'spec_file', False)
        # Obtaining the member 'extend' of a type (line 536)
        extend_15995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 16), spec_file_15994, 'extend')
        # Calling extend(args, kwargs) (line 536)
        extend_call_result_16002 = invoke(stypy.reporting.localization.Localization(__file__, 536, 16), extend_15995, *[list_15996], **kwargs_16001)
        
        
        # Getting the type of 'val' (line 539)
        val_16003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 19), 'val')
        # Testing the type of an if condition (line 539)
        if_condition_16004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 16), val_16003)
        # Assigning a type to the variable 'if_condition_16004' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 16), 'if_condition_16004', if_condition_16004)
        # SSA begins for if statement (line 539)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 540)
        # Processing the call arguments (line 540)
        
        # Call to split(...): (line 540)
        # Processing the call arguments (line 540)
        
        # Call to read(...): (line 540)
        # Processing the call keyword arguments (line 540)
        kwargs_16015 = {}
        
        # Call to open(...): (line 540)
        # Processing the call arguments (line 540)
        # Getting the type of 'val' (line 540)
        val_16010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 55), 'val', False)
        str_16011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 60), 'str', 'r')
        # Processing the call keyword arguments (line 540)
        kwargs_16012 = {}
        # Getting the type of 'open' (line 540)
        open_16009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 50), 'open', False)
        # Calling open(args, kwargs) (line 540)
        open_call_result_16013 = invoke(stypy.reporting.localization.Localization(__file__, 540, 50), open_16009, *[val_16010, str_16011], **kwargs_16012)
        
        # Obtaining the member 'read' of a type (line 540)
        read_16014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 50), open_call_result_16013, 'read')
        # Calling read(args, kwargs) (line 540)
        read_call_result_16016 = invoke(stypy.reporting.localization.Localization(__file__, 540, 50), read_16014, *[], **kwargs_16015)
        
        str_16017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 73), 'str', '\n')
        # Processing the call keyword arguments (line 540)
        kwargs_16018 = {}
        # Getting the type of 'string' (line 540)
        string_16007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 37), 'string', False)
        # Obtaining the member 'split' of a type (line 540)
        split_16008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 37), string_16007, 'split')
        # Calling split(args, kwargs) (line 540)
        split_call_result_16019 = invoke(stypy.reporting.localization.Localization(__file__, 540, 37), split_16008, *[read_call_result_16016, str_16017], **kwargs_16018)
        
        # Processing the call keyword arguments (line 540)
        kwargs_16020 = {}
        # Getting the type of 'spec_file' (line 540)
        spec_file_16005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 20), 'spec_file', False)
        # Obtaining the member 'extend' of a type (line 540)
        extend_16006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 20), spec_file_16005, 'extend')
        # Calling extend(args, kwargs) (line 540)
        extend_call_result_16021 = invoke(stypy.reporting.localization.Localization(__file__, 540, 20), extend_16006, *[split_call_result_16019], **kwargs_16020)
        
        # SSA branch for the else part of an if statement (line 539)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'default' (line 542)
        default_16024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 37), 'default', False)
        # Processing the call keyword arguments (line 542)
        kwargs_16025 = {}
        # Getting the type of 'spec_file' (line 542)
        spec_file_16022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 20), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 542)
        append_16023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 20), spec_file_16022, 'append')
        # Calling append(args, kwargs) (line 542)
        append_call_result_16026 = invoke(stypy.reporting.localization.Localization(__file__, 542, 20), append_16023, *[default_16024], **kwargs_16025)
        
        # SSA join for if statement (line 539)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 535)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 546)
        # Processing the call arguments (line 546)
        
        # Obtaining an instance of the builtin type 'list' (line 546)
        list_16029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 546)
        # Adding element type (line 546)
        str_16030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 12), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 25), list_16029, str_16030)
        # Adding element type (line 546)
        str_16031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 12), 'str', '%files -f INSTALLED_FILES')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 25), list_16029, str_16031)
        # Adding element type (line 546)
        str_16032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 12), 'str', '%defattr(-,root,root)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 25), list_16029, str_16032)
        
        # Processing the call keyword arguments (line 546)
        kwargs_16033 = {}
        # Getting the type of 'spec_file' (line 546)
        spec_file_16027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'spec_file', False)
        # Obtaining the member 'extend' of a type (line 546)
        extend_16028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 8), spec_file_16027, 'extend')
        # Calling extend(args, kwargs) (line 546)
        extend_call_result_16034 = invoke(stypy.reporting.localization.Localization(__file__, 546, 8), extend_16028, *[list_16029], **kwargs_16033)
        
        
        # Getting the type of 'self' (line 552)
        self_16035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 11), 'self')
        # Obtaining the member 'doc_files' of a type (line 552)
        doc_files_16036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 11), self_16035, 'doc_files')
        # Testing the type of an if condition (line 552)
        if_condition_16037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 552, 8), doc_files_16036)
        # Assigning a type to the variable 'if_condition_16037' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'if_condition_16037', if_condition_16037)
        # SSA begins for if statement (line 552)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 553)
        # Processing the call arguments (line 553)
        str_16040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 29), 'str', '%doc ')
        
        # Call to join(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'self' (line 553)
        self_16043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 51), 'self', False)
        # Obtaining the member 'doc_files' of a type (line 553)
        doc_files_16044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 51), self_16043, 'doc_files')
        # Processing the call keyword arguments (line 553)
        kwargs_16045 = {}
        # Getting the type of 'string' (line 553)
        string_16041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 39), 'string', False)
        # Obtaining the member 'join' of a type (line 553)
        join_16042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 39), string_16041, 'join')
        # Calling join(args, kwargs) (line 553)
        join_call_result_16046 = invoke(stypy.reporting.localization.Localization(__file__, 553, 39), join_16042, *[doc_files_16044], **kwargs_16045)
        
        # Applying the binary operator '+' (line 553)
        result_add_16047 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 29), '+', str_16040, join_call_result_16046)
        
        # Processing the call keyword arguments (line 553)
        kwargs_16048 = {}
        # Getting the type of 'spec_file' (line 553)
        spec_file_16038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'spec_file', False)
        # Obtaining the member 'append' of a type (line 553)
        append_16039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 12), spec_file_16038, 'append')
        # Calling append(args, kwargs) (line 553)
        append_call_result_16049 = invoke(stypy.reporting.localization.Localization(__file__, 553, 12), append_16039, *[result_add_16047], **kwargs_16048)
        
        # SSA join for if statement (line 552)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 555)
        self_16050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 11), 'self')
        # Obtaining the member 'changelog' of a type (line 555)
        changelog_16051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 11), self_16050, 'changelog')
        # Testing the type of an if condition (line 555)
        if_condition_16052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 8), changelog_16051)
        # Assigning a type to the variable 'if_condition_16052' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'if_condition_16052', if_condition_16052)
        # SSA begins for if statement (line 555)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 556)
        # Processing the call arguments (line 556)
        
        # Obtaining an instance of the builtin type 'list' (line 556)
        list_16055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 556)
        # Adding element type (line 556)
        str_16056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 16), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 29), list_16055, str_16056)
        # Adding element type (line 556)
        str_16057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 16), 'str', '%changelog')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 29), list_16055, str_16057)
        
        # Processing the call keyword arguments (line 556)
        kwargs_16058 = {}
        # Getting the type of 'spec_file' (line 556)
        spec_file_16053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'spec_file', False)
        # Obtaining the member 'extend' of a type (line 556)
        extend_16054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 12), spec_file_16053, 'extend')
        # Calling extend(args, kwargs) (line 556)
        extend_call_result_16059 = invoke(stypy.reporting.localization.Localization(__file__, 556, 12), extend_16054, *[list_16055], **kwargs_16058)
        
        
        # Call to extend(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'self' (line 559)
        self_16062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 29), 'self', False)
        # Obtaining the member 'changelog' of a type (line 559)
        changelog_16063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 29), self_16062, 'changelog')
        # Processing the call keyword arguments (line 559)
        kwargs_16064 = {}
        # Getting the type of 'spec_file' (line 559)
        spec_file_16060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'spec_file', False)
        # Obtaining the member 'extend' of a type (line 559)
        extend_16061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 12), spec_file_16060, 'extend')
        # Calling extend(args, kwargs) (line 559)
        extend_call_result_16065 = invoke(stypy.reporting.localization.Localization(__file__, 559, 12), extend_16061, *[changelog_16063], **kwargs_16064)
        
        # SSA join for if statement (line 555)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'spec_file' (line 561)
        spec_file_16066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 15), 'spec_file')
        # Assigning a type to the variable 'stypy_return_type' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'stypy_return_type', spec_file_16066)
        
        # ################# End of '_make_spec_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_make_spec_file' in the type store
        # Getting the type of 'stypy_return_type' (line 410)
        stypy_return_type_16067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16067)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_make_spec_file'
        return stypy_return_type_16067


    @norecursion
    def _format_changelog(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_format_changelog'
        module_type_store = module_type_store.open_function_context('_format_changelog', 565, 4, False)
        # Assigning a type to the variable 'self' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_localization', localization)
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_function_name', 'bdist_rpm._format_changelog')
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_param_names_list', ['changelog'])
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_rpm._format_changelog.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm._format_changelog', ['changelog'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_format_changelog', localization, ['changelog'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_format_changelog(...)' code ##################

        str_16068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, (-1)), 'str', 'Format the changelog correctly and convert it to a list of strings\n        ')
        
        
        # Getting the type of 'changelog' (line 568)
        changelog_16069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'changelog')
        # Applying the 'not' unary operator (line 568)
        result_not__16070 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 11), 'not', changelog_16069)
        
        # Testing the type of an if condition (line 568)
        if_condition_16071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 8), result_not__16070)
        # Assigning a type to the variable 'if_condition_16071' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'if_condition_16071', if_condition_16071)
        # SSA begins for if statement (line 568)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'changelog' (line 569)
        changelog_16072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 19), 'changelog')
        # Assigning a type to the variable 'stypy_return_type' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'stypy_return_type', changelog_16072)
        # SSA join for if statement (line 568)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 570):
        
        # Obtaining an instance of the builtin type 'list' (line 570)
        list_16073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 570)
        
        # Assigning a type to the variable 'new_changelog' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'new_changelog', list_16073)
        
        
        # Call to split(...): (line 571)
        # Processing the call arguments (line 571)
        
        # Call to strip(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'changelog' (line 571)
        changelog_16078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 46), 'changelog', False)
        # Processing the call keyword arguments (line 571)
        kwargs_16079 = {}
        # Getting the type of 'string' (line 571)
        string_16076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 33), 'string', False)
        # Obtaining the member 'strip' of a type (line 571)
        strip_16077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 33), string_16076, 'strip')
        # Calling strip(args, kwargs) (line 571)
        strip_call_result_16080 = invoke(stypy.reporting.localization.Localization(__file__, 571, 33), strip_16077, *[changelog_16078], **kwargs_16079)
        
        str_16081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 58), 'str', '\n')
        # Processing the call keyword arguments (line 571)
        kwargs_16082 = {}
        # Getting the type of 'string' (line 571)
        string_16074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 20), 'string', False)
        # Obtaining the member 'split' of a type (line 571)
        split_16075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 20), string_16074, 'split')
        # Calling split(args, kwargs) (line 571)
        split_call_result_16083 = invoke(stypy.reporting.localization.Localization(__file__, 571, 20), split_16075, *[strip_call_result_16080, str_16081], **kwargs_16082)
        
        # Testing the type of a for loop iterable (line 571)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 571, 8), split_call_result_16083)
        # Getting the type of the for loop variable (line 571)
        for_loop_var_16084 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 571, 8), split_call_result_16083)
        # Assigning a type to the variable 'line' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'line', for_loop_var_16084)
        # SSA begins for a for statement (line 571)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 572):
        
        # Call to strip(...): (line 572)
        # Processing the call arguments (line 572)
        # Getting the type of 'line' (line 572)
        line_16087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 32), 'line', False)
        # Processing the call keyword arguments (line 572)
        kwargs_16088 = {}
        # Getting the type of 'string' (line 572)
        string_16085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 19), 'string', False)
        # Obtaining the member 'strip' of a type (line 572)
        strip_16086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 19), string_16085, 'strip')
        # Calling strip(args, kwargs) (line 572)
        strip_call_result_16089 = invoke(stypy.reporting.localization.Localization(__file__, 572, 19), strip_16086, *[line_16087], **kwargs_16088)
        
        # Assigning a type to the variable 'line' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'line', strip_call_result_16089)
        
        
        
        # Obtaining the type of the subscript
        int_16090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 20), 'int')
        # Getting the type of 'line' (line 573)
        line_16091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 15), 'line')
        # Obtaining the member '__getitem__' of a type (line 573)
        getitem___16092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 15), line_16091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 573)
        subscript_call_result_16093 = invoke(stypy.reporting.localization.Localization(__file__, 573, 15), getitem___16092, int_16090)
        
        str_16094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 26), 'str', '*')
        # Applying the binary operator '==' (line 573)
        result_eq_16095 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 15), '==', subscript_call_result_16093, str_16094)
        
        # Testing the type of an if condition (line 573)
        if_condition_16096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 12), result_eq_16095)
        # Assigning a type to the variable 'if_condition_16096' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'if_condition_16096', if_condition_16096)
        # SSA begins for if statement (line 573)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 574)
        # Processing the call arguments (line 574)
        
        # Obtaining an instance of the builtin type 'list' (line 574)
        list_16099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 574)
        # Adding element type (line 574)
        str_16100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 38), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 37), list_16099, str_16100)
        # Adding element type (line 574)
        # Getting the type of 'line' (line 574)
        line_16101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'line', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 37), list_16099, line_16101)
        
        # Processing the call keyword arguments (line 574)
        kwargs_16102 = {}
        # Getting the type of 'new_changelog' (line 574)
        new_changelog_16097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 16), 'new_changelog', False)
        # Obtaining the member 'extend' of a type (line 574)
        extend_16098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 16), new_changelog_16097, 'extend')
        # Calling extend(args, kwargs) (line 574)
        extend_call_result_16103 = invoke(stypy.reporting.localization.Localization(__file__, 574, 16), extend_16098, *[list_16099], **kwargs_16102)
        
        # SSA branch for the else part of an if statement (line 573)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_16104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 22), 'int')
        # Getting the type of 'line' (line 575)
        line_16105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 17), 'line')
        # Obtaining the member '__getitem__' of a type (line 575)
        getitem___16106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 17), line_16105, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 575)
        subscript_call_result_16107 = invoke(stypy.reporting.localization.Localization(__file__, 575, 17), getitem___16106, int_16104)
        
        str_16108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 28), 'str', '-')
        # Applying the binary operator '==' (line 575)
        result_eq_16109 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 17), '==', subscript_call_result_16107, str_16108)
        
        # Testing the type of an if condition (line 575)
        if_condition_16110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 17), result_eq_16109)
        # Assigning a type to the variable 'if_condition_16110' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 17), 'if_condition_16110', if_condition_16110)
        # SSA begins for if statement (line 575)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 576)
        # Processing the call arguments (line 576)
        # Getting the type of 'line' (line 576)
        line_16113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 37), 'line', False)
        # Processing the call keyword arguments (line 576)
        kwargs_16114 = {}
        # Getting the type of 'new_changelog' (line 576)
        new_changelog_16111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'new_changelog', False)
        # Obtaining the member 'append' of a type (line 576)
        append_16112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 16), new_changelog_16111, 'append')
        # Calling append(args, kwargs) (line 576)
        append_call_result_16115 = invoke(stypy.reporting.localization.Localization(__file__, 576, 16), append_16112, *[line_16113], **kwargs_16114)
        
        # SSA branch for the else part of an if statement (line 575)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 578)
        # Processing the call arguments (line 578)
        str_16118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 37), 'str', '  ')
        # Getting the type of 'line' (line 578)
        line_16119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 44), 'line', False)
        # Applying the binary operator '+' (line 578)
        result_add_16120 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 37), '+', str_16118, line_16119)
        
        # Processing the call keyword arguments (line 578)
        kwargs_16121 = {}
        # Getting the type of 'new_changelog' (line 578)
        new_changelog_16116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 16), 'new_changelog', False)
        # Obtaining the member 'append' of a type (line 578)
        append_16117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 16), new_changelog_16116, 'append')
        # Calling append(args, kwargs) (line 578)
        append_call_result_16122 = invoke(stypy.reporting.localization.Localization(__file__, 578, 16), append_16117, *[result_add_16120], **kwargs_16121)
        
        # SSA join for if statement (line 575)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 573)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_16123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 29), 'int')
        # Getting the type of 'new_changelog' (line 581)
        new_changelog_16124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'new_changelog')
        # Obtaining the member '__getitem__' of a type (line 581)
        getitem___16125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 15), new_changelog_16124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 581)
        subscript_call_result_16126 = invoke(stypy.reporting.localization.Localization(__file__, 581, 15), getitem___16125, int_16123)
        
        # Applying the 'not' unary operator (line 581)
        result_not__16127 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 11), 'not', subscript_call_result_16126)
        
        # Testing the type of an if condition (line 581)
        if_condition_16128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 581, 8), result_not__16127)
        # Assigning a type to the variable 'if_condition_16128' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'if_condition_16128', if_condition_16128)
        # SSA begins for if statement (line 581)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Deleting a member
        # Getting the type of 'new_changelog' (line 582)
        new_changelog_16129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'new_changelog')
        
        # Obtaining the type of the subscript
        int_16130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 30), 'int')
        # Getting the type of 'new_changelog' (line 582)
        new_changelog_16131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'new_changelog')
        # Obtaining the member '__getitem__' of a type (line 582)
        getitem___16132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 16), new_changelog_16131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 582)
        subscript_call_result_16133 = invoke(stypy.reporting.localization.Localization(__file__, 582, 16), getitem___16132, int_16130)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 12), new_changelog_16129, subscript_call_result_16133)
        # SSA join for if statement (line 581)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_changelog' (line 584)
        new_changelog_16134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 15), 'new_changelog')
        # Assigning a type to the variable 'stypy_return_type' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'stypy_return_type', new_changelog_16134)
        
        # ################# End of '_format_changelog(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_format_changelog' in the type store
        # Getting the type of 'stypy_return_type' (line 565)
        stypy_return_type_16135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_format_changelog'
        return stypy_return_type_16135


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 0, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_rpm.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'bdist_rpm' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'bdist_rpm', bdist_rpm)

# Assigning a Str to a Name (line 22):
str_16136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'str', 'create an RPM distribution')
# Getting the type of 'bdist_rpm'
bdist_rpm_16137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_rpm')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_rpm_16137, 'description', str_16136)

# Assigning a List to a Name (line 24):

# Obtaining an instance of the builtin type 'list' (line 24)
list_16138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 25)
tuple_16139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 25)
# Adding element type (line 25)
str_16140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'str', 'bdist-base=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), tuple_16139, str_16140)
# Adding element type (line 25)
# Getting the type of 'None' (line 25)
None_16141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), tuple_16139, None_16141)
# Adding element type (line 25)
str_16142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'str', 'base directory for creating built distributions')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), tuple_16139, str_16142)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16139)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_16143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)
str_16144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'str', 'rpm-base=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_16143, str_16144)
# Adding element type (line 27)
# Getting the type of 'None' (line 27)
None_16145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_16143, None_16145)
# Adding element type (line 27)
str_16146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'str', 'base directory for creating RPMs (defaults to "rpm" under --bdist-base; must be specified for RPM 2)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_16143, str_16146)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16143)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_16147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
str_16148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'str', 'dist-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_16147, str_16148)
# Adding element type (line 30)
str_16149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_16147, str_16149)
# Adding element type (line 30)
str_16150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'str', 'directory to put final RPM files in (and .spec files if --spec-only)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_16147, str_16150)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16147)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_16151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
str_16152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'str', 'python=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_16151, str_16152)
# Adding element type (line 33)
# Getting the type of 'None' (line 33)
None_16153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_16151, None_16153)
# Adding element type (line 33)
str_16154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'str', 'path to Python interpreter to hard-code in the .spec file (default: "python")')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_16151, str_16154)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16151)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_16155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)
str_16156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'str', 'fix-python')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 9), tuple_16155, str_16156)
# Adding element type (line 36)
# Getting the type of 'None' (line 36)
None_16157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 9), tuple_16155, None_16157)
# Adding element type (line 36)
str_16158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'str', 'hard-code the exact path to the current Python interpreter in the .spec file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 9), tuple_16155, str_16158)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16155)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 39)
tuple_16159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 39)
# Adding element type (line 39)
str_16160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'str', 'spec-only')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_16159, str_16160)
# Adding element type (line 39)
# Getting the type of 'None' (line 39)
None_16161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_16159, None_16161)
# Adding element type (line 39)
str_16162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'str', 'only regenerate spec file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_16159, str_16162)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16159)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_16163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
str_16164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'str', 'source-only')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_16163, str_16164)
# Adding element type (line 41)
# Getting the type of 'None' (line 41)
None_16165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_16163, None_16165)
# Adding element type (line 41)
str_16166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'str', 'only generate source RPM')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_16163, str_16166)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16163)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_16167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
str_16168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'str', 'binary-only')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_16167, str_16168)
# Adding element type (line 43)
# Getting the type of 'None' (line 43)
None_16169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_16167, None_16169)
# Adding element type (line 43)
str_16170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'str', 'only generate binary RPM')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_16167, str_16170)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16167)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 45)
tuple_16171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 45)
# Adding element type (line 45)
str_16172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'str', 'use-bzip2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 9), tuple_16171, str_16172)
# Adding element type (line 45)
# Getting the type of 'None' (line 45)
None_16173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 9), tuple_16171, None_16173)
# Adding element type (line 45)
str_16174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'str', 'use bzip2 instead of gzip to create source distribution')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 9), tuple_16171, str_16174)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16171)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 53)
tuple_16175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 53)
# Adding element type (line 53)
str_16176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 9), 'str', 'distribution-name=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 9), tuple_16175, str_16176)
# Adding element type (line 53)
# Getting the type of 'None' (line 53)
None_16177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 9), tuple_16175, None_16177)
# Adding element type (line 53)
str_16178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'str', 'name of the (Linux) distribution to which this RPM applies (*not* the name of the module distribution!)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 9), tuple_16175, str_16178)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16175)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 56)
tuple_16179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 56)
# Adding element type (line 56)
str_16180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'str', 'group=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_16179, str_16180)
# Adding element type (line 56)
# Getting the type of 'None' (line 56)
None_16181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_16179, None_16181)
# Adding element type (line 56)
str_16182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'str', 'package classification [default: "Development/Libraries"]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_16179, str_16182)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16179)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 58)
tuple_16183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 58)
# Adding element type (line 58)
str_16184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'str', 'release=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_16183, str_16184)
# Adding element type (line 58)
# Getting the type of 'None' (line 58)
None_16185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_16183, None_16185)
# Adding element type (line 58)
str_16186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 9), 'str', 'RPM release number')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 9), tuple_16183, str_16186)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16183)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 60)
tuple_16187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 60)
# Adding element type (line 60)
str_16188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 9), 'str', 'serial=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 9), tuple_16187, str_16188)
# Adding element type (line 60)
# Getting the type of 'None' (line 60)
None_16189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 9), tuple_16187, None_16189)
# Adding element type (line 60)
str_16190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 9), 'str', 'RPM serial number')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 9), tuple_16187, str_16190)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16187)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 62)
tuple_16191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 62)
# Adding element type (line 62)
str_16192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 9), 'str', 'vendor=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 9), tuple_16191, str_16192)
# Adding element type (line 62)
# Getting the type of 'None' (line 62)
None_16193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 9), tuple_16191, None_16193)
# Adding element type (line 62)
str_16194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 9), 'str', 'RPM "vendor" (eg. "Joe Blow <joe@example.com>") [default: maintainer or author from setup script]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 9), tuple_16191, str_16194)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16191)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 65)
tuple_16195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 65)
# Adding element type (line 65)
str_16196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'str', 'packager=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_16195, str_16196)
# Adding element type (line 65)
# Getting the type of 'None' (line 65)
None_16197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_16195, None_16197)
# Adding element type (line 65)
str_16198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 9), 'str', 'RPM packager (eg. "Jane Doe <jane@example.net>")[default: vendor]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_16195, str_16198)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16195)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_16199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)
# Adding element type (line 68)
str_16200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 9), 'str', 'doc-files=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_16199, str_16200)
# Adding element type (line 68)
# Getting the type of 'None' (line 68)
None_16201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_16199, None_16201)
# Adding element type (line 68)
str_16202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 9), 'str', 'list of documentation files (space or comma-separated)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_16199, str_16202)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16199)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 70)
tuple_16203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 70)
# Adding element type (line 70)
str_16204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 9), 'str', 'changelog=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), tuple_16203, str_16204)
# Adding element type (line 70)
# Getting the type of 'None' (line 70)
None_16205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), tuple_16203, None_16205)
# Adding element type (line 70)
str_16206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 9), 'str', 'RPM changelog')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), tuple_16203, str_16206)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16203)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 72)
tuple_16207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 72)
# Adding element type (line 72)
str_16208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 9), 'str', 'icon=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), tuple_16207, str_16208)
# Adding element type (line 72)
# Getting the type of 'None' (line 72)
None_16209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), tuple_16207, None_16209)
# Adding element type (line 72)
str_16210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 9), 'str', 'name of icon file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 9), tuple_16207, str_16210)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16207)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 74)
tuple_16211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 74)
# Adding element type (line 74)
str_16212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 9), 'str', 'provides=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 9), tuple_16211, str_16212)
# Adding element type (line 74)
# Getting the type of 'None' (line 74)
None_16213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 9), tuple_16211, None_16213)
# Adding element type (line 74)
str_16214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 9), 'str', 'capabilities provided by this package')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 9), tuple_16211, str_16214)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16211)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_16215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
str_16216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'str', 'requires=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_16215, str_16216)
# Adding element type (line 76)
# Getting the type of 'None' (line 76)
None_16217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_16215, None_16217)
# Adding element type (line 76)
str_16218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 9), 'str', 'capabilities required by this package')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_16215, str_16218)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16215)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_16219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)
str_16220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 9), 'str', 'conflicts=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 9), tuple_16219, str_16220)
# Adding element type (line 78)
# Getting the type of 'None' (line 78)
None_16221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 9), tuple_16219, None_16221)
# Adding element type (line 78)
str_16222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 9), 'str', 'capabilities which conflict with this package')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 9), tuple_16219, str_16222)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16219)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 80)
tuple_16223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 80)
# Adding element type (line 80)
str_16224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 9), 'str', 'build-requires=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 9), tuple_16223, str_16224)
# Adding element type (line 80)
# Getting the type of 'None' (line 80)
None_16225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 9), tuple_16223, None_16225)
# Adding element type (line 80)
str_16226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 9), 'str', 'capabilities required to build this package')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 9), tuple_16223, str_16226)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16223)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_16227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)
str_16228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 9), 'str', 'obsoletes=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 9), tuple_16227, str_16228)
# Adding element type (line 82)
# Getting the type of 'None' (line 82)
None_16229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 9), tuple_16227, None_16229)
# Adding element type (line 82)
str_16230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 9), 'str', 'capabilities made obsolete by this package')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 9), tuple_16227, str_16230)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16227)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 84)
tuple_16231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 84)
# Adding element type (line 84)
str_16232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 9), 'str', 'no-autoreq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 9), tuple_16231, str_16232)
# Adding element type (line 84)
# Getting the type of 'None' (line 84)
None_16233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 9), tuple_16231, None_16233)
# Adding element type (line 84)
str_16234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 9), 'str', 'do not automatically calculate dependencies')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 9), tuple_16231, str_16234)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16231)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 88)
tuple_16235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 88)
# Adding element type (line 88)
str_16236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 9), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 9), tuple_16235, str_16236)
# Adding element type (line 88)
str_16237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 22), 'str', 'k')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 9), tuple_16235, str_16237)
# Adding element type (line 88)
str_16238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 9), 'str', "don't clean up RPM build directory")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 9), tuple_16235, str_16238)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16235)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 90)
tuple_16239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 90)
# Adding element type (line 90)
str_16240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 9), 'str', 'no-keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_16239, str_16240)
# Adding element type (line 90)
# Getting the type of 'None' (line 90)
None_16241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_16239, None_16241)
# Adding element type (line 90)
str_16242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 9), 'str', 'clean up RPM build directory [default]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_16239, str_16242)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16239)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 92)
tuple_16243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 92)
# Adding element type (line 92)
str_16244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 9), 'str', 'use-rpm-opt-flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 9), tuple_16243, str_16244)
# Adding element type (line 92)
# Getting the type of 'None' (line 92)
None_16245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 30), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 9), tuple_16243, None_16245)
# Adding element type (line 92)
str_16246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 9), 'str', 'compile with RPM_OPT_FLAGS when building from source RPM')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 9), tuple_16243, str_16246)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16243)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 94)
tuple_16247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 94)
# Adding element type (line 94)
str_16248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 9), 'str', 'no-rpm-opt-flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 9), tuple_16247, str_16248)
# Adding element type (line 94)
# Getting the type of 'None' (line 94)
None_16249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 29), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 9), tuple_16247, None_16249)
# Adding element type (line 94)
str_16250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 9), 'str', 'do not pass any RPM CFLAGS to compiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 9), tuple_16247, str_16250)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16247)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 96)
tuple_16251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 96)
# Adding element type (line 96)
str_16252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 9), 'str', 'rpm3-mode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 9), tuple_16251, str_16252)
# Adding element type (line 96)
# Getting the type of 'None' (line 96)
None_16253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 9), tuple_16251, None_16253)
# Adding element type (line 96)
str_16254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 9), 'str', 'RPM 3 compatibility mode (default)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 9), tuple_16251, str_16254)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16251)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 98)
tuple_16255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 98)
# Adding element type (line 98)
str_16256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 9), 'str', 'rpm2-mode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 9), tuple_16255, str_16256)
# Adding element type (line 98)
# Getting the type of 'None' (line 98)
None_16257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 9), tuple_16255, None_16257)
# Adding element type (line 98)
str_16258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 9), 'str', 'RPM 2 compatibility mode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 9), tuple_16255, str_16258)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16255)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 102)
tuple_16259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 102)
# Adding element type (line 102)
str_16260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 9), 'str', 'prep-script=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 9), tuple_16259, str_16260)
# Adding element type (line 102)
# Getting the type of 'None' (line 102)
None_16261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 9), tuple_16259, None_16261)
# Adding element type (line 102)
str_16262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 9), 'str', 'Specify a script for the PREP phase of RPM building')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 9), tuple_16259, str_16262)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16259)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 104)
tuple_16263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 104)
# Adding element type (line 104)
str_16264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'str', 'build-script=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_16263, str_16264)
# Adding element type (line 104)
# Getting the type of 'None' (line 104)
None_16265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_16263, None_16265)
# Adding element type (line 104)
str_16266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 9), 'str', 'Specify a script for the BUILD phase of RPM building')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_16263, str_16266)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16263)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 107)
tuple_16267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 107)
# Adding element type (line 107)
str_16268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 9), 'str', 'pre-install=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 9), tuple_16267, str_16268)
# Adding element type (line 107)
# Getting the type of 'None' (line 107)
None_16269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 9), tuple_16267, None_16269)
# Adding element type (line 107)
str_16270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 9), 'str', 'Specify a script for the pre-INSTALL phase of RPM building')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 9), tuple_16267, str_16270)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16267)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 109)
tuple_16271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 109)
# Adding element type (line 109)
str_16272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 9), 'str', 'install-script=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 9), tuple_16271, str_16272)
# Adding element type (line 109)
# Getting the type of 'None' (line 109)
None_16273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 9), tuple_16271, None_16273)
# Adding element type (line 109)
str_16274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 9), 'str', 'Specify a script for the INSTALL phase of RPM building')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 9), tuple_16271, str_16274)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16271)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 111)
tuple_16275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 111)
# Adding element type (line 111)
str_16276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 9), 'str', 'post-install=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), tuple_16275, str_16276)
# Adding element type (line 111)
# Getting the type of 'None' (line 111)
None_16277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), tuple_16275, None_16277)
# Adding element type (line 111)
str_16278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 9), 'str', 'Specify a script for the post-INSTALL phase of RPM building')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 9), tuple_16275, str_16278)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16275)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 114)
tuple_16279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 114)
# Adding element type (line 114)
str_16280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 9), 'str', 'pre-uninstall=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 9), tuple_16279, str_16280)
# Adding element type (line 114)
# Getting the type of 'None' (line 114)
None_16281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 9), tuple_16279, None_16281)
# Adding element type (line 114)
str_16282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 9), 'str', 'Specify a script for the pre-UNINSTALL phase of RPM building')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 9), tuple_16279, str_16282)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16279)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 116)
tuple_16283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 116)
# Adding element type (line 116)
str_16284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 9), 'str', 'post-uninstall=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 9), tuple_16283, str_16284)
# Adding element type (line 116)
# Getting the type of 'None' (line 116)
None_16285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 9), tuple_16283, None_16285)
# Adding element type (line 116)
str_16286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 9), 'str', 'Specify a script for the post-UNINSTALL phase of RPM building')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 9), tuple_16283, str_16286)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16283)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 119)
tuple_16287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 119)
# Adding element type (line 119)
str_16288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 9), 'str', 'clean-script=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), tuple_16287, str_16288)
# Adding element type (line 119)
# Getting the type of 'None' (line 119)
None_16289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), tuple_16287, None_16289)
# Adding element type (line 119)
str_16290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 9), 'str', 'Specify a script for the CLEAN phase of RPM building')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), tuple_16287, str_16290)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16287)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 122)
tuple_16291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 122)
# Adding element type (line 122)
str_16292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 9), 'str', 'verify-script=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 9), tuple_16291, str_16292)
# Adding element type (line 122)
# Getting the type of 'None' (line 122)
None_16293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 9), tuple_16291, None_16293)
# Adding element type (line 122)
str_16294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 9), 'str', 'Specify a script for the VERIFY phase of the RPM build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 9), tuple_16291, str_16294)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16291)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 126)
tuple_16295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 126)
# Adding element type (line 126)
str_16296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 9), 'str', 'force-arch=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 9), tuple_16295, str_16296)
# Adding element type (line 126)
# Getting the type of 'None' (line 126)
None_16297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 9), tuple_16295, None_16297)
# Adding element type (line 126)
str_16298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 9), 'str', 'Force an architecture onto the RPM build process')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 9), tuple_16295, str_16298)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16295)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 129)
tuple_16299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 129)
# Adding element type (line 129)
str_16300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 9), 'str', 'quiet')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 9), tuple_16299, str_16300)
# Adding element type (line 129)
str_16301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 18), 'str', 'q')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 9), tuple_16299, str_16301)
# Adding element type (line 129)
str_16302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 9), 'str', 'Run the INSTALL phase of RPM building in quiet mode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 9), tuple_16299, str_16302)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_16138, tuple_16299)

# Getting the type of 'bdist_rpm'
bdist_rpm_16303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_rpm')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_rpm_16303, 'user_options', list_16138)

# Assigning a List to a Name (line 133):

# Obtaining an instance of the builtin type 'list' (line 133)
list_16304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 133)
# Adding element type (line 133)
str_16305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 23), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_16304, str_16305)
# Adding element type (line 133)
str_16306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 36), 'str', 'use-rpm-opt-flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_16304, str_16306)
# Adding element type (line 133)
str_16307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 57), 'str', 'rpm3-mode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_16304, str_16307)
# Adding element type (line 133)
str_16308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 23), 'str', 'no-autoreq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_16304, str_16308)
# Adding element type (line 133)
str_16309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 37), 'str', 'quiet')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 22), list_16304, str_16309)

# Getting the type of 'bdist_rpm'
bdist_rpm_16310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_rpm')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_rpm_16310, 'boolean_options', list_16304)

# Assigning a Dict to a Name (line 136):

# Obtaining an instance of the builtin type 'dict' (line 136)
dict_16311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 136)
# Adding element type (key, value) (line 136)
str_16312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'str', 'no-keep-temp')
str_16313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 36), 'str', 'keep-temp')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), dict_16311, (str_16312, str_16313))
# Adding element type (key, value) (line 136)
str_16314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 20), 'str', 'no-rpm-opt-flags')
str_16315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 40), 'str', 'use-rpm-opt-flags')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), dict_16311, (str_16314, str_16315))
# Adding element type (key, value) (line 136)
str_16316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'str', 'rpm2-mode')
str_16317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 33), 'str', 'rpm3-mode')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), dict_16311, (str_16316, str_16317))

# Getting the type of 'bdist_rpm'
bdist_rpm_16318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_rpm')
# Setting the type of the member 'negative_opt' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_rpm_16318, 'negative_opt', dict_16311)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
