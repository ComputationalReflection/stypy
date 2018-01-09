
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.sdist
2: 
3: Implements the Distutils 'sdist' command (create a source distribution).'''
4: 
5: __revision__ = "$Id$"
6: 
7: import os
8: import string
9: import sys
10: from glob import glob
11: from warnings import warn
12: 
13: from distutils.core import Command
14: from distutils import dir_util, dep_util, file_util, archive_util
15: from distutils.text_file import TextFile
16: from distutils.errors import (DistutilsPlatformError, DistutilsOptionError,
17:                               DistutilsTemplateError)
18: from distutils.filelist import FileList
19: from distutils import log
20: from distutils.util import convert_path
21: 
22: def show_formats():
23:     '''Print all possible values for the 'formats' option (used by
24:     the "--help-formats" command-line option).
25:     '''
26:     from distutils.fancy_getopt import FancyGetopt
27:     from distutils.archive_util import ARCHIVE_FORMATS
28:     formats = []
29:     for format in ARCHIVE_FORMATS.keys():
30:         formats.append(("formats=" + format, None,
31:                         ARCHIVE_FORMATS[format][2]))
32:     formats.sort()
33:     FancyGetopt(formats).print_help(
34:         "List of available source distribution formats:")
35: 
36: class sdist(Command):
37: 
38:     description = "create a source distribution (tarball, zip file, etc.)"
39: 
40:     def checking_metadata(self):
41:         '''Callable used for the check sub-command.
42: 
43:         Placed here so user_options can view it'''
44:         return self.metadata_check
45: 
46:     user_options = [
47:         ('template=', 't',
48:          "name of manifest template file [default: MANIFEST.in]"),
49:         ('manifest=', 'm',
50:          "name of manifest file [default: MANIFEST]"),
51:         ('use-defaults', None,
52:          "include the default file set in the manifest "
53:          "[default; disable with --no-defaults]"),
54:         ('no-defaults', None,
55:          "don't include the default file set"),
56:         ('prune', None,
57:          "specifically exclude files/directories that should not be "
58:          "distributed (build tree, RCS/CVS dirs, etc.) "
59:          "[default; disable with --no-prune]"),
60:         ('no-prune', None,
61:          "don't automatically exclude anything"),
62:         ('manifest-only', 'o',
63:          "just regenerate the manifest and then stop "
64:          "(implies --force-manifest)"),
65:         ('force-manifest', 'f',
66:          "forcibly regenerate the manifest and carry on as usual. "
67:          "Deprecated: now the manifest is always regenerated."),
68:         ('formats=', None,
69:          "formats for source distribution (comma-separated list)"),
70:         ('keep-temp', 'k',
71:          "keep the distribution tree around after creating " +
72:          "archive file(s)"),
73:         ('dist-dir=', 'd',
74:          "directory to put the source distribution archive(s) in "
75:          "[default: dist]"),
76:         ('metadata-check', None,
77:          "Ensure that all required elements of meta-data "
78:          "are supplied. Warn if any missing. [default]"),
79:         ('owner=', 'u',
80:          "Owner name used when creating a tar file [default: current user]"),
81:         ('group=', 'g',
82:          "Group name used when creating a tar file [default: current group]"),
83:         ]
84: 
85:     boolean_options = ['use-defaults', 'prune',
86:                        'manifest-only', 'force-manifest',
87:                        'keep-temp', 'metadata-check']
88: 
89:     help_options = [
90:         ('help-formats', None,
91:          "list available distribution formats", show_formats),
92:         ]
93: 
94:     negative_opt = {'no-defaults': 'use-defaults',
95:                     'no-prune': 'prune' }
96: 
97:     default_format = {'posix': 'gztar',
98:                       'nt': 'zip' }
99: 
100:     sub_commands = [('check', checking_metadata)]
101: 
102:     def initialize_options(self):
103:         # 'template' and 'manifest' are, respectively, the names of
104:         # the manifest template and manifest file.
105:         self.template = None
106:         self.manifest = None
107: 
108:         # 'use_defaults': if true, we will include the default file set
109:         # in the manifest
110:         self.use_defaults = 1
111:         self.prune = 1
112: 
113:         self.manifest_only = 0
114:         self.force_manifest = 0
115: 
116:         self.formats = None
117:         self.keep_temp = 0
118:         self.dist_dir = None
119: 
120:         self.archive_files = None
121:         self.metadata_check = 1
122:         self.owner = None
123:         self.group = None
124: 
125:     def finalize_options(self):
126:         if self.manifest is None:
127:             self.manifest = "MANIFEST"
128:         if self.template is None:
129:             self.template = "MANIFEST.in"
130: 
131:         self.ensure_string_list('formats')
132:         if self.formats is None:
133:             try:
134:                 self.formats = [self.default_format[os.name]]
135:             except KeyError:
136:                 raise DistutilsPlatformError, \
137:                       "don't know how to create source distributions " + \
138:                       "on platform %s" % os.name
139: 
140:         bad_format = archive_util.check_archive_formats(self.formats)
141:         if bad_format:
142:             raise DistutilsOptionError, \
143:                   "unknown archive format '%s'" % bad_format
144: 
145:         if self.dist_dir is None:
146:             self.dist_dir = "dist"
147: 
148:     def run(self):
149:         # 'filelist' contains the list of files that will make up the
150:         # manifest
151:         self.filelist = FileList()
152: 
153:         # Run sub commands
154:         for cmd_name in self.get_sub_commands():
155:             self.run_command(cmd_name)
156: 
157:         # Do whatever it takes to get the list of files to process
158:         # (process the manifest template, read an existing manifest,
159:         # whatever).  File list is accumulated in 'self.filelist'.
160:         self.get_file_list()
161: 
162:         # If user just wanted us to regenerate the manifest, stop now.
163:         if self.manifest_only:
164:             return
165: 
166:         # Otherwise, go ahead and create the source distribution tarball,
167:         # or zipfile, or whatever.
168:         self.make_distribution()
169: 
170:     def check_metadata(self):
171:         '''Deprecated API.'''
172:         warn("distutils.command.sdist.check_metadata is deprecated, \
173:               use the check command instead", PendingDeprecationWarning)
174:         check = self.distribution.get_command_obj('check')
175:         check.ensure_finalized()
176:         check.run()
177: 
178:     def get_file_list(self):
179:         '''Figure out the list of files to include in the source
180:         distribution, and put it in 'self.filelist'.  This might involve
181:         reading the manifest template (and writing the manifest), or just
182:         reading the manifest, or just using the default file set -- it all
183:         depends on the user's options.
184:         '''
185:         # new behavior when using a template:
186:         # the file list is recalculated every time because
187:         # even if MANIFEST.in or setup.py are not changed
188:         # the user might have added some files in the tree that
189:         # need to be included.
190:         #
191:         #  This makes --force the default and only behavior with templates.
192:         template_exists = os.path.isfile(self.template)
193:         if not template_exists and self._manifest_is_not_generated():
194:             self.read_manifest()
195:             self.filelist.sort()
196:             self.filelist.remove_duplicates()
197:             return
198: 
199:         if not template_exists:
200:             self.warn(("manifest template '%s' does not exist " +
201:                         "(using default file list)") %
202:                         self.template)
203:         self.filelist.findall()
204: 
205:         if self.use_defaults:
206:             self.add_defaults()
207: 
208:         if template_exists:
209:             self.read_template()
210: 
211:         if self.prune:
212:             self.prune_file_list()
213: 
214:         self.filelist.sort()
215:         self.filelist.remove_duplicates()
216:         self.write_manifest()
217: 
218:     def add_defaults(self):
219:         '''Add all the default files to self.filelist:
220:           - README or README.txt
221:           - setup.py
222:           - test/test*.py
223:           - all pure Python modules mentioned in setup script
224:           - all files pointed by package_data (build_py)
225:           - all files defined in data_files.
226:           - all files defined as scripts.
227:           - all C sources listed as part of extensions or C libraries
228:             in the setup script (doesn't catch C headers!)
229:         Warns if (README or README.txt) or setup.py are missing; everything
230:         else is optional.
231:         '''
232: 
233:         standards = [('README', 'README.txt'), self.distribution.script_name]
234:         for fn in standards:
235:             if isinstance(fn, tuple):
236:                 alts = fn
237:                 got_it = 0
238:                 for fn in alts:
239:                     if os.path.exists(fn):
240:                         got_it = 1
241:                         self.filelist.append(fn)
242:                         break
243: 
244:                 if not got_it:
245:                     self.warn("standard file not found: should have one of " +
246:                               string.join(alts, ', '))
247:             else:
248:                 if os.path.exists(fn):
249:                     self.filelist.append(fn)
250:                 else:
251:                     self.warn("standard file '%s' not found" % fn)
252: 
253:         optional = ['test/test*.py', 'setup.cfg']
254:         for pattern in optional:
255:             files = filter(os.path.isfile, glob(pattern))
256:             if files:
257:                 self.filelist.extend(files)
258: 
259:         # build_py is used to get:
260:         #  - python modules
261:         #  - files defined in package_data
262:         build_py = self.get_finalized_command('build_py')
263: 
264:         # getting python files
265:         if self.distribution.has_pure_modules():
266:             self.filelist.extend(build_py.get_source_files())
267: 
268:         # getting package_data files
269:         # (computed in build_py.data_files by build_py.finalize_options)
270:         for pkg, src_dir, build_dir, filenames in build_py.data_files:
271:             for filename in filenames:
272:                 self.filelist.append(os.path.join(src_dir, filename))
273: 
274:         # getting distribution.data_files
275:         if self.distribution.has_data_files():
276:             for item in self.distribution.data_files:
277:                 if isinstance(item, str): # plain file
278:                     item = convert_path(item)
279:                     if os.path.isfile(item):
280:                         self.filelist.append(item)
281:                 else:    # a (dirname, filenames) tuple
282:                     dirname, filenames = item
283:                     for f in filenames:
284:                         f = convert_path(f)
285:                         if os.path.isfile(f):
286:                             self.filelist.append(f)
287: 
288:         if self.distribution.has_ext_modules():
289:             build_ext = self.get_finalized_command('build_ext')
290:             self.filelist.extend(build_ext.get_source_files())
291: 
292:         if self.distribution.has_c_libraries():
293:             build_clib = self.get_finalized_command('build_clib')
294:             self.filelist.extend(build_clib.get_source_files())
295: 
296:         if self.distribution.has_scripts():
297:             build_scripts = self.get_finalized_command('build_scripts')
298:             self.filelist.extend(build_scripts.get_source_files())
299: 
300:     def read_template(self):
301:         '''Read and parse manifest template file named by self.template.
302: 
303:         (usually "MANIFEST.in") The parsing and processing is done by
304:         'self.filelist', which updates itself accordingly.
305:         '''
306:         log.info("reading manifest template '%s'", self.template)
307:         template = TextFile(self.template,
308:                             strip_comments=1,
309:                             skip_blanks=1,
310:                             join_lines=1,
311:                             lstrip_ws=1,
312:                             rstrip_ws=1,
313:                             collapse_join=1)
314: 
315:         try:
316:             while 1:
317:                 line = template.readline()
318:                 if line is None:            # end of file
319:                     break
320: 
321:                 try:
322:                     self.filelist.process_template_line(line)
323:                 # the call above can raise a DistutilsTemplateError for
324:                 # malformed lines, or a ValueError from the lower-level
325:                 # convert_path function
326:                 except (DistutilsTemplateError, ValueError) as msg:
327:                     self.warn("%s, line %d: %s" % (template.filename,
328:                                                    template.current_line,
329:                                                    msg))
330:         finally:
331:             template.close()
332: 
333:     def prune_file_list(self):
334:         '''Prune off branches that might slip into the file list as created
335:         by 'read_template()', but really don't belong there:
336:           * the build tree (typically "build")
337:           * the release tree itself (only an issue if we ran "sdist"
338:             previously with --keep-temp, or it aborted)
339:           * any RCS, CVS, .svn, .hg, .git, .bzr, _darcs directories
340:         '''
341:         build = self.get_finalized_command('build')
342:         base_dir = self.distribution.get_fullname()
343: 
344:         self.filelist.exclude_pattern(None, prefix=build.build_base)
345:         self.filelist.exclude_pattern(None, prefix=base_dir)
346: 
347:         # pruning out vcs directories
348:         # both separators are used under win32
349:         if sys.platform == 'win32':
350:             seps = r'/|\\'
351:         else:
352:             seps = '/'
353: 
354:         vcs_dirs = ['RCS', 'CVS', r'\.svn', r'\.hg', r'\.git', r'\.bzr',
355:                     '_darcs']
356:         vcs_ptrn = r'(^|%s)(%s)(%s).*' % (seps, '|'.join(vcs_dirs), seps)
357:         self.filelist.exclude_pattern(vcs_ptrn, is_regex=1)
358: 
359:     def write_manifest(self):
360:         '''Write the file list in 'self.filelist' (presumably as filled in
361:         by 'add_defaults()' and 'read_template()') to the manifest file
362:         named by 'self.manifest'.
363:         '''
364:         if self._manifest_is_not_generated():
365:             log.info("not writing to manually maintained "
366:                      "manifest file '%s'" % self.manifest)
367:             return
368: 
369:         content = self.filelist.files[:]
370:         content.insert(0, '# file GENERATED by distutils, do NOT edit')
371:         self.execute(file_util.write_file, (self.manifest, content),
372:                      "writing manifest file '%s'" % self.manifest)
373: 
374:     def _manifest_is_not_generated(self):
375:         # check for special comment used in 2.7.1 and higher
376:         if not os.path.isfile(self.manifest):
377:             return False
378: 
379:         fp = open(self.manifest, 'rU')
380:         try:
381:             first_line = fp.readline()
382:         finally:
383:             fp.close()
384:         return first_line != '# file GENERATED by distutils, do NOT edit\n'
385: 
386:     def read_manifest(self):
387:         '''Read the manifest file (named by 'self.manifest') and use it to
388:         fill in 'self.filelist', the list of files to include in the source
389:         distribution.
390:         '''
391:         log.info("reading manifest file '%s'", self.manifest)
392:         manifest = open(self.manifest)
393:         for line in manifest:
394:             # ignore comments and blank lines
395:             line = line.strip()
396:             if line.startswith('#') or not line:
397:                 continue
398:             self.filelist.append(line)
399:         manifest.close()
400: 
401:     def make_release_tree(self, base_dir, files):
402:         '''Create the directory tree that will become the source
403:         distribution archive.  All directories implied by the filenames in
404:         'files' are created under 'base_dir', and then we hard link or copy
405:         (if hard linking is unavailable) those files into place.
406:         Essentially, this duplicates the developer's source tree, but in a
407:         directory named after the distribution, containing only the files
408:         to be distributed.
409:         '''
410:         # Create all the directories under 'base_dir' necessary to
411:         # put 'files' there; the 'mkpath()' is just so we don't die
412:         # if the manifest happens to be empty.
413:         self.mkpath(base_dir)
414:         dir_util.create_tree(base_dir, files, dry_run=self.dry_run)
415: 
416:         # And walk over the list of files, either making a hard link (if
417:         # os.link exists) to each one that doesn't already exist in its
418:         # corresponding location under 'base_dir', or copying each file
419:         # that's out-of-date in 'base_dir'.  (Usually, all files will be
420:         # out-of-date, because by default we blow away 'base_dir' when
421:         # we're done making the distribution archives.)
422: 
423:         if hasattr(os, 'link'):        # can make hard links on this system
424:             link = 'hard'
425:             msg = "making hard links in %s..." % base_dir
426:         else:                           # nope, have to copy
427:             link = None
428:             msg = "copying files to %s..." % base_dir
429: 
430:         if not files:
431:             log.warn("no files to distribute -- empty manifest?")
432:         else:
433:             log.info(msg)
434:         for file in files:
435:             if not os.path.isfile(file):
436:                 log.warn("'%s' not a regular file -- skipping" % file)
437:             else:
438:                 dest = os.path.join(base_dir, file)
439:                 self.copy_file(file, dest, link=link)
440: 
441:         self.distribution.metadata.write_pkg_info(base_dir)
442: 
443:     def make_distribution(self):
444:         '''Create the source distribution(s).  First, we create the release
445:         tree with 'make_release_tree()'; then, we create all required
446:         archive files (according to 'self.formats') from the release tree.
447:         Finally, we clean up by blowing away the release tree (unless
448:         'self.keep_temp' is true).  The list of archive files created is
449:         stored so it can be retrieved later by 'get_archive_files()'.
450:         '''
451:         # Don't warn about missing meta-data here -- should be (and is!)
452:         # done elsewhere.
453:         base_dir = self.distribution.get_fullname()
454:         base_name = os.path.join(self.dist_dir, base_dir)
455: 
456:         self.make_release_tree(base_dir, self.filelist.files)
457:         archive_files = []              # remember names of files we create
458:         # tar archive must be created last to avoid overwrite and remove
459:         if 'tar' in self.formats:
460:             self.formats.append(self.formats.pop(self.formats.index('tar')))
461: 
462:         for fmt in self.formats:
463:             file = self.make_archive(base_name, fmt, base_dir=base_dir,
464:                                      owner=self.owner, group=self.group)
465:             archive_files.append(file)
466:             self.distribution.dist_files.append(('sdist', '', file))
467: 
468:         self.archive_files = archive_files
469: 
470:         if not self.keep_temp:
471:             dir_util.remove_tree(base_dir, dry_run=self.dry_run)
472: 
473:     def get_archive_files(self):
474:         '''Return the list of archive files created when the command
475:         was run, or None if the command hasn't run yet.
476:         '''
477:         return self.archive_files
478: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_25648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', "distutils.command.sdist\n\nImplements the Distutils 'sdist' command (create a source distribution).")

# Assigning a Str to a Name (line 5):

# Assigning a Str to a Name (line 5):
str_25649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__revision__', str_25649)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import os' statement (line 7)
import os

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import string' statement (line 8)
import string

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'string', string, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from glob import glob' statement (line 10)
try:
    from glob import glob

except:
    glob = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'glob', None, module_type_store, ['glob'], [glob])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from warnings import warn' statement (line 11)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.core import Command' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_25650 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core')

if (type(import_25650) is not StypyTypeError):

    if (import_25650 != 'pyd_module'):
        __import__(import_25650)
        sys_modules_25651 = sys.modules[import_25650]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core', sys_modules_25651.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_25651, sys_modules_25651.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core', import_25650)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils import dir_util, dep_util, file_util, archive_util' statement (line 14)
try:
    from distutils import dir_util, dep_util, file_util, archive_util

except:
    dir_util = UndefinedType
    dep_util = UndefinedType
    file_util = UndefinedType
    archive_util = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils', None, module_type_store, ['dir_util', 'dep_util', 'file_util', 'archive_util'], [dir_util, dep_util, file_util, archive_util])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.text_file import TextFile' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_25652 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.text_file')

if (type(import_25652) is not StypyTypeError):

    if (import_25652 != 'pyd_module'):
        __import__(import_25652)
        sys_modules_25653 = sys.modules[import_25652]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.text_file', sys_modules_25653.module_type_store, module_type_store, ['TextFile'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_25653, sys_modules_25653.module_type_store, module_type_store)
    else:
        from distutils.text_file import TextFile

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.text_file', None, module_type_store, ['TextFile'], [TextFile])

else:
    # Assigning a type to the variable 'distutils.text_file' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.text_file', import_25652)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.errors import DistutilsPlatformError, DistutilsOptionError, DistutilsTemplateError' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_25654 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors')

if (type(import_25654) is not StypyTypeError):

    if (import_25654 != 'pyd_module'):
        __import__(import_25654)
        sys_modules_25655 = sys.modules[import_25654]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', sys_modules_25655.module_type_store, module_type_store, ['DistutilsPlatformError', 'DistutilsOptionError', 'DistutilsTemplateError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_25655, sys_modules_25655.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsPlatformError, DistutilsOptionError, DistutilsTemplateError

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', None, module_type_store, ['DistutilsPlatformError', 'DistutilsOptionError', 'DistutilsTemplateError'], [DistutilsPlatformError, DistutilsOptionError, DistutilsTemplateError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', import_25654)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.filelist import FileList' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_25656 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.filelist')

if (type(import_25656) is not StypyTypeError):

    if (import_25656 != 'pyd_module'):
        __import__(import_25656)
        sys_modules_25657 = sys.modules[import_25656]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.filelist', sys_modules_25657.module_type_store, module_type_store, ['FileList'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_25657, sys_modules_25657.module_type_store, module_type_store)
    else:
        from distutils.filelist import FileList

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.filelist', None, module_type_store, ['FileList'], [FileList])

else:
    # Assigning a type to the variable 'distutils.filelist' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.filelist', import_25656)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils import log' statement (line 19)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils.util import convert_path' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_25658 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.util')

if (type(import_25658) is not StypyTypeError):

    if (import_25658 != 'pyd_module'):
        __import__(import_25658)
        sys_modules_25659 = sys.modules[import_25658]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.util', sys_modules_25659.module_type_store, module_type_store, ['convert_path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_25659, sys_modules_25659.module_type_store, module_type_store)
    else:
        from distutils.util import convert_path

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.util', None, module_type_store, ['convert_path'], [convert_path])

else:
    # Assigning a type to the variable 'distutils.util' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.util', import_25658)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')


@norecursion
def show_formats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'show_formats'
    module_type_store = module_type_store.open_function_context('show_formats', 22, 0, False)
    
    # Passed parameters checking function
    show_formats.stypy_localization = localization
    show_formats.stypy_type_of_self = None
    show_formats.stypy_type_store = module_type_store
    show_formats.stypy_function_name = 'show_formats'
    show_formats.stypy_param_names_list = []
    show_formats.stypy_varargs_param_name = None
    show_formats.stypy_kwargs_param_name = None
    show_formats.stypy_call_defaults = defaults
    show_formats.stypy_call_varargs = varargs
    show_formats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'show_formats', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'show_formats', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'show_formats(...)' code ##################

    str_25660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', 'Print all possible values for the \'formats\' option (used by\n    the "--help-formats" command-line option).\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 4))
    
    # 'from distutils.fancy_getopt import FancyGetopt' statement (line 26)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
    import_25661 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 4), 'distutils.fancy_getopt')

    if (type(import_25661) is not StypyTypeError):

        if (import_25661 != 'pyd_module'):
            __import__(import_25661)
            sys_modules_25662 = sys.modules[import_25661]
            import_from_module(stypy.reporting.localization.Localization(__file__, 26, 4), 'distutils.fancy_getopt', sys_modules_25662.module_type_store, module_type_store, ['FancyGetopt'])
            nest_module(stypy.reporting.localization.Localization(__file__, 26, 4), __file__, sys_modules_25662, sys_modules_25662.module_type_store, module_type_store)
        else:
            from distutils.fancy_getopt import FancyGetopt

            import_from_module(stypy.reporting.localization.Localization(__file__, 26, 4), 'distutils.fancy_getopt', None, module_type_store, ['FancyGetopt'], [FancyGetopt])

    else:
        # Assigning a type to the variable 'distutils.fancy_getopt' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'distutils.fancy_getopt', import_25661)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 4))
    
    # 'from distutils.archive_util import ARCHIVE_FORMATS' statement (line 27)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
    import_25663 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 4), 'distutils.archive_util')

    if (type(import_25663) is not StypyTypeError):

        if (import_25663 != 'pyd_module'):
            __import__(import_25663)
            sys_modules_25664 = sys.modules[import_25663]
            import_from_module(stypy.reporting.localization.Localization(__file__, 27, 4), 'distutils.archive_util', sys_modules_25664.module_type_store, module_type_store, ['ARCHIVE_FORMATS'])
            nest_module(stypy.reporting.localization.Localization(__file__, 27, 4), __file__, sys_modules_25664, sys_modules_25664.module_type_store, module_type_store)
        else:
            from distutils.archive_util import ARCHIVE_FORMATS

            import_from_module(stypy.reporting.localization.Localization(__file__, 27, 4), 'distutils.archive_util', None, module_type_store, ['ARCHIVE_FORMATS'], [ARCHIVE_FORMATS])

    else:
        # Assigning a type to the variable 'distutils.archive_util' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'distutils.archive_util', import_25663)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
    
    
    # Assigning a List to a Name (line 28):
    
    # Assigning a List to a Name (line 28):
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_25665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    
    # Assigning a type to the variable 'formats' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'formats', list_25665)
    
    
    # Call to keys(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_25668 = {}
    # Getting the type of 'ARCHIVE_FORMATS' (line 29)
    ARCHIVE_FORMATS_25666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'ARCHIVE_FORMATS', False)
    # Obtaining the member 'keys' of a type (line 29)
    keys_25667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 18), ARCHIVE_FORMATS_25666, 'keys')
    # Calling keys(args, kwargs) (line 29)
    keys_call_result_25669 = invoke(stypy.reporting.localization.Localization(__file__, 29, 18), keys_25667, *[], **kwargs_25668)
    
    # Testing the type of a for loop iterable (line 29)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 29, 4), keys_call_result_25669)
    # Getting the type of the for loop variable (line 29)
    for_loop_var_25670 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 29, 4), keys_call_result_25669)
    # Assigning a type to the variable 'format' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'format', for_loop_var_25670)
    # SSA begins for a for statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Obtaining an instance of the builtin type 'tuple' (line 30)
    tuple_25673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 30)
    # Adding element type (line 30)
    str_25674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'str', 'formats=')
    # Getting the type of 'format' (line 30)
    format_25675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 37), 'format', False)
    # Applying the binary operator '+' (line 30)
    result_add_25676 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 24), '+', str_25674, format_25675)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), tuple_25673, result_add_25676)
    # Adding element type (line 30)
    # Getting the type of 'None' (line 30)
    None_25677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 45), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), tuple_25673, None_25677)
    # Adding element type (line 30)
    
    # Obtaining the type of the subscript
    int_25678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 48), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'format' (line 31)
    format_25679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 40), 'format', False)
    # Getting the type of 'ARCHIVE_FORMATS' (line 31)
    ARCHIVE_FORMATS_25680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'ARCHIVE_FORMATS', False)
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___25681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 24), ARCHIVE_FORMATS_25680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_25682 = invoke(stypy.reporting.localization.Localization(__file__, 31, 24), getitem___25681, format_25679)
    
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___25683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 24), subscript_call_result_25682, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_25684 = invoke(stypy.reporting.localization.Localization(__file__, 31, 24), getitem___25683, int_25678)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), tuple_25673, subscript_call_result_25684)
    
    # Processing the call keyword arguments (line 30)
    kwargs_25685 = {}
    # Getting the type of 'formats' (line 30)
    formats_25671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'formats', False)
    # Obtaining the member 'append' of a type (line 30)
    append_25672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), formats_25671, 'append')
    # Calling append(args, kwargs) (line 30)
    append_call_result_25686 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), append_25672, *[tuple_25673], **kwargs_25685)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sort(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_25689 = {}
    # Getting the type of 'formats' (line 32)
    formats_25687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'formats', False)
    # Obtaining the member 'sort' of a type (line 32)
    sort_25688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 4), formats_25687, 'sort')
    # Calling sort(args, kwargs) (line 32)
    sort_call_result_25690 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), sort_25688, *[], **kwargs_25689)
    
    
    # Call to print_help(...): (line 33)
    # Processing the call arguments (line 33)
    str_25696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'str', 'List of available source distribution formats:')
    # Processing the call keyword arguments (line 33)
    kwargs_25697 = {}
    
    # Call to FancyGetopt(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'formats' (line 33)
    formats_25692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'formats', False)
    # Processing the call keyword arguments (line 33)
    kwargs_25693 = {}
    # Getting the type of 'FancyGetopt' (line 33)
    FancyGetopt_25691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'FancyGetopt', False)
    # Calling FancyGetopt(args, kwargs) (line 33)
    FancyGetopt_call_result_25694 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), FancyGetopt_25691, *[formats_25692], **kwargs_25693)
    
    # Obtaining the member 'print_help' of a type (line 33)
    print_help_25695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), FancyGetopt_call_result_25694, 'print_help')
    # Calling print_help(args, kwargs) (line 33)
    print_help_call_result_25698 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), print_help_25695, *[str_25696], **kwargs_25697)
    
    
    # ################# End of 'show_formats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'show_formats' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_25699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25699)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'show_formats'
    return stypy_return_type_25699

# Assigning a type to the variable 'show_formats' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'show_formats', show_formats)
# Declaration of the 'sdist' class
# Getting the type of 'Command' (line 36)
Command_25700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'Command')

class sdist(Command_25700, ):
    
    # Assigning a Str to a Name (line 38):

    @norecursion
    def checking_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'checking_metadata'
        module_type_store = module_type_store.open_function_context('checking_metadata', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.checking_metadata.__dict__.__setitem__('stypy_localization', localization)
        sdist.checking_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.checking_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.checking_metadata.__dict__.__setitem__('stypy_function_name', 'sdist.checking_metadata')
        sdist.checking_metadata.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.checking_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.checking_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.checking_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.checking_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.checking_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.checking_metadata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.checking_metadata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'checking_metadata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'checking_metadata(...)' code ##################

        str_25701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', 'Callable used for the check sub-command.\n\n        Placed here so user_options can view it')
        # Getting the type of 'self' (line 44)
        self_25702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'self')
        # Obtaining the member 'metadata_check' of a type (line 44)
        metadata_check_25703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), self_25702, 'metadata_check')
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', metadata_check_25703)
        
        # ################# End of 'checking_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'checking_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_25704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25704)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'checking_metadata'
        return stypy_return_type_25704

    
    # Assigning a List to a Name (line 46):
    
    # Assigning a List to a Name (line 85):
    
    # Assigning a List to a Name (line 89):
    
    # Assigning a Dict to a Name (line 94):
    
    # Assigning a Dict to a Name (line 97):
    
    # Assigning a List to a Name (line 100):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 102, 4, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        sdist.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.initialize_options.__dict__.__setitem__('stypy_function_name', 'sdist.initialize_options')
        sdist.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 105):
        
        # Assigning a Name to a Attribute (line 105):
        # Getting the type of 'None' (line 105)
        None_25705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'None')
        # Getting the type of 'self' (line 105)
        self_25706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self')
        # Setting the type of the member 'template' of a type (line 105)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_25706, 'template', None_25705)
        
        # Assigning a Name to a Attribute (line 106):
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of 'None' (line 106)
        None_25707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'None')
        # Getting the type of 'self' (line 106)
        self_25708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self')
        # Setting the type of the member 'manifest' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_25708, 'manifest', None_25707)
        
        # Assigning a Num to a Attribute (line 110):
        
        # Assigning a Num to a Attribute (line 110):
        int_25709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 28), 'int')
        # Getting the type of 'self' (line 110)
        self_25710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member 'use_defaults' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_25710, 'use_defaults', int_25709)
        
        # Assigning a Num to a Attribute (line 111):
        
        # Assigning a Num to a Attribute (line 111):
        int_25711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'int')
        # Getting the type of 'self' (line 111)
        self_25712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member 'prune' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_25712, 'prune', int_25711)
        
        # Assigning a Num to a Attribute (line 113):
        
        # Assigning a Num to a Attribute (line 113):
        int_25713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 29), 'int')
        # Getting the type of 'self' (line 113)
        self_25714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'manifest_only' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_25714, 'manifest_only', int_25713)
        
        # Assigning a Num to a Attribute (line 114):
        
        # Assigning a Num to a Attribute (line 114):
        int_25715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'int')
        # Getting the type of 'self' (line 114)
        self_25716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member 'force_manifest' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_25716, 'force_manifest', int_25715)
        
        # Assigning a Name to a Attribute (line 116):
        
        # Assigning a Name to a Attribute (line 116):
        # Getting the type of 'None' (line 116)
        None_25717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'None')
        # Getting the type of 'self' (line 116)
        self_25718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self')
        # Setting the type of the member 'formats' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_25718, 'formats', None_25717)
        
        # Assigning a Num to a Attribute (line 117):
        
        # Assigning a Num to a Attribute (line 117):
        int_25719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 25), 'int')
        # Getting the type of 'self' (line 117)
        self_25720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self')
        # Setting the type of the member 'keep_temp' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_25720, 'keep_temp', int_25719)
        
        # Assigning a Name to a Attribute (line 118):
        
        # Assigning a Name to a Attribute (line 118):
        # Getting the type of 'None' (line 118)
        None_25721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'None')
        # Getting the type of 'self' (line 118)
        self_25722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self')
        # Setting the type of the member 'dist_dir' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_25722, 'dist_dir', None_25721)
        
        # Assigning a Name to a Attribute (line 120):
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'None' (line 120)
        None_25723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'None')
        # Getting the type of 'self' (line 120)
        self_25724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member 'archive_files' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_25724, 'archive_files', None_25723)
        
        # Assigning a Num to a Attribute (line 121):
        
        # Assigning a Num to a Attribute (line 121):
        int_25725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 30), 'int')
        # Getting the type of 'self' (line 121)
        self_25726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member 'metadata_check' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_25726, 'metadata_check', int_25725)
        
        # Assigning a Name to a Attribute (line 122):
        
        # Assigning a Name to a Attribute (line 122):
        # Getting the type of 'None' (line 122)
        None_25727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'None')
        # Getting the type of 'self' (line 122)
        self_25728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member 'owner' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_25728, 'owner', None_25727)
        
        # Assigning a Name to a Attribute (line 123):
        
        # Assigning a Name to a Attribute (line 123):
        # Getting the type of 'None' (line 123)
        None_25729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 21), 'None')
        # Getting the type of 'self' (line 123)
        self_25730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self')
        # Setting the type of the member 'group' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_25730, 'group', None_25729)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_25731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25731)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_25731


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        sdist.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.finalize_options.__dict__.__setitem__('stypy_function_name', 'sdist.finalize_options')
        sdist.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 126)
        # Getting the type of 'self' (line 126)
        self_25732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'self')
        # Obtaining the member 'manifest' of a type (line 126)
        manifest_25733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 11), self_25732, 'manifest')
        # Getting the type of 'None' (line 126)
        None_25734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 28), 'None')
        
        (may_be_25735, more_types_in_union_25736) = may_be_none(manifest_25733, None_25734)

        if may_be_25735:

            if more_types_in_union_25736:
                # Runtime conditional SSA (line 126)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Attribute (line 127):
            
            # Assigning a Str to a Attribute (line 127):
            str_25737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 28), 'str', 'MANIFEST')
            # Getting the type of 'self' (line 127)
            self_25738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self')
            # Setting the type of the member 'manifest' of a type (line 127)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), self_25738, 'manifest', str_25737)

            if more_types_in_union_25736:
                # SSA join for if statement (line 126)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 128)
        # Getting the type of 'self' (line 128)
        self_25739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'self')
        # Obtaining the member 'template' of a type (line 128)
        template_25740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 11), self_25739, 'template')
        # Getting the type of 'None' (line 128)
        None_25741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'None')
        
        (may_be_25742, more_types_in_union_25743) = may_be_none(template_25740, None_25741)

        if may_be_25742:

            if more_types_in_union_25743:
                # Runtime conditional SSA (line 128)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Attribute (line 129):
            
            # Assigning a Str to a Attribute (line 129):
            str_25744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'str', 'MANIFEST.in')
            # Getting the type of 'self' (line 129)
            self_25745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'self')
            # Setting the type of the member 'template' of a type (line 129)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), self_25745, 'template', str_25744)

            if more_types_in_union_25743:
                # SSA join for if statement (line 128)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to ensure_string_list(...): (line 131)
        # Processing the call arguments (line 131)
        str_25748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 32), 'str', 'formats')
        # Processing the call keyword arguments (line 131)
        kwargs_25749 = {}
        # Getting the type of 'self' (line 131)
        self_25746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 131)
        ensure_string_list_25747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_25746, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 131)
        ensure_string_list_call_result_25750 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), ensure_string_list_25747, *[str_25748], **kwargs_25749)
        
        
        # Type idiom detected: calculating its left and rigth part (line 132)
        # Getting the type of 'self' (line 132)
        self_25751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'self')
        # Obtaining the member 'formats' of a type (line 132)
        formats_25752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 11), self_25751, 'formats')
        # Getting the type of 'None' (line 132)
        None_25753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'None')
        
        (may_be_25754, more_types_in_union_25755) = may_be_none(formats_25752, None_25753)

        if may_be_25754:

            if more_types_in_union_25755:
                # Runtime conditional SSA (line 132)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 133)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a List to a Attribute (line 134):
            
            # Assigning a List to a Attribute (line 134):
            
            # Obtaining an instance of the builtin type 'list' (line 134)
            list_25756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 134)
            # Adding element type (line 134)
            
            # Obtaining the type of the subscript
            # Getting the type of 'os' (line 134)
            os_25757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 52), 'os')
            # Obtaining the member 'name' of a type (line 134)
            name_25758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 52), os_25757, 'name')
            # Getting the type of 'self' (line 134)
            self_25759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 32), 'self')
            # Obtaining the member 'default_format' of a type (line 134)
            default_format_25760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 32), self_25759, 'default_format')
            # Obtaining the member '__getitem__' of a type (line 134)
            getitem___25761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 32), default_format_25760, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 134)
            subscript_call_result_25762 = invoke(stypy.reporting.localization.Localization(__file__, 134, 32), getitem___25761, name_25758)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 31), list_25756, subscript_call_result_25762)
            
            # Getting the type of 'self' (line 134)
            self_25763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'self')
            # Setting the type of the member 'formats' of a type (line 134)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), self_25763, 'formats', list_25756)
            # SSA branch for the except part of a try statement (line 133)
            # SSA branch for the except 'KeyError' branch of a try statement (line 133)
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'DistutilsPlatformError' (line 136)
            DistutilsPlatformError_25764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 22), 'DistutilsPlatformError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 136, 16), DistutilsPlatformError_25764, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 133)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_25755:
                # SSA join for if statement (line 132)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to check_archive_formats(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'self' (line 140)
        self_25767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 56), 'self', False)
        # Obtaining the member 'formats' of a type (line 140)
        formats_25768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 56), self_25767, 'formats')
        # Processing the call keyword arguments (line 140)
        kwargs_25769 = {}
        # Getting the type of 'archive_util' (line 140)
        archive_util_25765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'archive_util', False)
        # Obtaining the member 'check_archive_formats' of a type (line 140)
        check_archive_formats_25766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), archive_util_25765, 'check_archive_formats')
        # Calling check_archive_formats(args, kwargs) (line 140)
        check_archive_formats_call_result_25770 = invoke(stypy.reporting.localization.Localization(__file__, 140, 21), check_archive_formats_25766, *[formats_25768], **kwargs_25769)
        
        # Assigning a type to the variable 'bad_format' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'bad_format', check_archive_formats_call_result_25770)
        
        # Getting the type of 'bad_format' (line 141)
        bad_format_25771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'bad_format')
        # Testing the type of an if condition (line 141)
        if_condition_25772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), bad_format_25771)
        # Assigning a type to the variable 'if_condition_25772' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_25772', if_condition_25772)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsOptionError' (line 142)
        DistutilsOptionError_25773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 142, 12), DistutilsOptionError_25773, 'raise parameter', BaseException)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 145)
        # Getting the type of 'self' (line 145)
        self_25774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'self')
        # Obtaining the member 'dist_dir' of a type (line 145)
        dist_dir_25775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 11), self_25774, 'dist_dir')
        # Getting the type of 'None' (line 145)
        None_25776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'None')
        
        (may_be_25777, more_types_in_union_25778) = may_be_none(dist_dir_25775, None_25776)

        if may_be_25777:

            if more_types_in_union_25778:
                # Runtime conditional SSA (line 145)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Attribute (line 146):
            
            # Assigning a Str to a Attribute (line 146):
            str_25779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 28), 'str', 'dist')
            # Getting the type of 'self' (line 146)
            self_25780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'self')
            # Setting the type of the member 'dist_dir' of a type (line 146)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), self_25780, 'dist_dir', str_25779)

            if more_types_in_union_25778:
                # SSA join for if statement (line 145)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_25781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_25781


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.run.__dict__.__setitem__('stypy_localization', localization)
        sdist.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.run.__dict__.__setitem__('stypy_function_name', 'sdist.run')
        sdist.run.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 151):
        
        # Assigning a Call to a Attribute (line 151):
        
        # Call to FileList(...): (line 151)
        # Processing the call keyword arguments (line 151)
        kwargs_25783 = {}
        # Getting the type of 'FileList' (line 151)
        FileList_25782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'FileList', False)
        # Calling FileList(args, kwargs) (line 151)
        FileList_call_result_25784 = invoke(stypy.reporting.localization.Localization(__file__, 151, 24), FileList_25782, *[], **kwargs_25783)
        
        # Getting the type of 'self' (line 151)
        self_25785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'filelist' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_25785, 'filelist', FileList_call_result_25784)
        
        
        # Call to get_sub_commands(...): (line 154)
        # Processing the call keyword arguments (line 154)
        kwargs_25788 = {}
        # Getting the type of 'self' (line 154)
        self_25786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'self', False)
        # Obtaining the member 'get_sub_commands' of a type (line 154)
        get_sub_commands_25787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 24), self_25786, 'get_sub_commands')
        # Calling get_sub_commands(args, kwargs) (line 154)
        get_sub_commands_call_result_25789 = invoke(stypy.reporting.localization.Localization(__file__, 154, 24), get_sub_commands_25787, *[], **kwargs_25788)
        
        # Testing the type of a for loop iterable (line 154)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 154, 8), get_sub_commands_call_result_25789)
        # Getting the type of the for loop variable (line 154)
        for_loop_var_25790 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 154, 8), get_sub_commands_call_result_25789)
        # Assigning a type to the variable 'cmd_name' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'cmd_name', for_loop_var_25790)
        # SSA begins for a for statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to run_command(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'cmd_name' (line 155)
        cmd_name_25793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'cmd_name', False)
        # Processing the call keyword arguments (line 155)
        kwargs_25794 = {}
        # Getting the type of 'self' (line 155)
        self_25791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 155)
        run_command_25792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_25791, 'run_command')
        # Calling run_command(args, kwargs) (line 155)
        run_command_call_result_25795 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), run_command_25792, *[cmd_name_25793], **kwargs_25794)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to get_file_list(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_25798 = {}
        # Getting the type of 'self' (line 160)
        self_25796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self', False)
        # Obtaining the member 'get_file_list' of a type (line 160)
        get_file_list_25797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_25796, 'get_file_list')
        # Calling get_file_list(args, kwargs) (line 160)
        get_file_list_call_result_25799 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), get_file_list_25797, *[], **kwargs_25798)
        
        
        # Getting the type of 'self' (line 163)
        self_25800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'self')
        # Obtaining the member 'manifest_only' of a type (line 163)
        manifest_only_25801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 11), self_25800, 'manifest_only')
        # Testing the type of an if condition (line 163)
        if_condition_25802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), manifest_only_25801)
        # Assigning a type to the variable 'if_condition_25802' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_25802', if_condition_25802)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to make_distribution(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_25805 = {}
        # Getting the type of 'self' (line 168)
        self_25803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self', False)
        # Obtaining the member 'make_distribution' of a type (line 168)
        make_distribution_25804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_25803, 'make_distribution')
        # Calling make_distribution(args, kwargs) (line 168)
        make_distribution_call_result_25806 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), make_distribution_25804, *[], **kwargs_25805)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_25807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25807)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_25807


    @norecursion
    def check_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_metadata'
        module_type_store = module_type_store.open_function_context('check_metadata', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.check_metadata.__dict__.__setitem__('stypy_localization', localization)
        sdist.check_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.check_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.check_metadata.__dict__.__setitem__('stypy_function_name', 'sdist.check_metadata')
        sdist.check_metadata.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.check_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.check_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.check_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.check_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.check_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.check_metadata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.check_metadata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_metadata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_metadata(...)' code ##################

        str_25808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'str', 'Deprecated API.')
        
        # Call to warn(...): (line 172)
        # Processing the call arguments (line 172)
        str_25810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, (-1)), 'str', 'distutils.command.sdist.check_metadata is deprecated,               use the check command instead')
        # Getting the type of 'PendingDeprecationWarning' (line 173)
        PendingDeprecationWarning_25811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 46), 'PendingDeprecationWarning', False)
        # Processing the call keyword arguments (line 172)
        kwargs_25812 = {}
        # Getting the type of 'warn' (line 172)
        warn_25809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'warn', False)
        # Calling warn(args, kwargs) (line 172)
        warn_call_result_25813 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), warn_25809, *[str_25810, PendingDeprecationWarning_25811], **kwargs_25812)
        
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to get_command_obj(...): (line 174)
        # Processing the call arguments (line 174)
        str_25817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 50), 'str', 'check')
        # Processing the call keyword arguments (line 174)
        kwargs_25818 = {}
        # Getting the type of 'self' (line 174)
        self_25814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'self', False)
        # Obtaining the member 'distribution' of a type (line 174)
        distribution_25815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), self_25814, 'distribution')
        # Obtaining the member 'get_command_obj' of a type (line 174)
        get_command_obj_25816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), distribution_25815, 'get_command_obj')
        # Calling get_command_obj(args, kwargs) (line 174)
        get_command_obj_call_result_25819 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), get_command_obj_25816, *[str_25817], **kwargs_25818)
        
        # Assigning a type to the variable 'check' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'check', get_command_obj_call_result_25819)
        
        # Call to ensure_finalized(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_25822 = {}
        # Getting the type of 'check' (line 175)
        check_25820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'check', False)
        # Obtaining the member 'ensure_finalized' of a type (line 175)
        ensure_finalized_25821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), check_25820, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 175)
        ensure_finalized_call_result_25823 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), ensure_finalized_25821, *[], **kwargs_25822)
        
        
        # Call to run(...): (line 176)
        # Processing the call keyword arguments (line 176)
        kwargs_25826 = {}
        # Getting the type of 'check' (line 176)
        check_25824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'check', False)
        # Obtaining the member 'run' of a type (line 176)
        run_25825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), check_25824, 'run')
        # Calling run(args, kwargs) (line 176)
        run_call_result_25827 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), run_25825, *[], **kwargs_25826)
        
        
        # ################# End of 'check_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_25828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25828)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_metadata'
        return stypy_return_type_25828


    @norecursion
    def get_file_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_file_list'
        module_type_store = module_type_store.open_function_context('get_file_list', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.get_file_list.__dict__.__setitem__('stypy_localization', localization)
        sdist.get_file_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.get_file_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.get_file_list.__dict__.__setitem__('stypy_function_name', 'sdist.get_file_list')
        sdist.get_file_list.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.get_file_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.get_file_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.get_file_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.get_file_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.get_file_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.get_file_list.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.get_file_list', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_file_list', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_file_list(...)' code ##################

        str_25829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, (-1)), 'str', "Figure out the list of files to include in the source\n        distribution, and put it in 'self.filelist'.  This might involve\n        reading the manifest template (and writing the manifest), or just\n        reading the manifest, or just using the default file set -- it all\n        depends on the user's options.\n        ")
        
        # Assigning a Call to a Name (line 192):
        
        # Assigning a Call to a Name (line 192):
        
        # Call to isfile(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'self' (line 192)
        self_25833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 41), 'self', False)
        # Obtaining the member 'template' of a type (line 192)
        template_25834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 41), self_25833, 'template')
        # Processing the call keyword arguments (line 192)
        kwargs_25835 = {}
        # Getting the type of 'os' (line 192)
        os_25830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 192)
        path_25831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 26), os_25830, 'path')
        # Obtaining the member 'isfile' of a type (line 192)
        isfile_25832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 26), path_25831, 'isfile')
        # Calling isfile(args, kwargs) (line 192)
        isfile_call_result_25836 = invoke(stypy.reporting.localization.Localization(__file__, 192, 26), isfile_25832, *[template_25834], **kwargs_25835)
        
        # Assigning a type to the variable 'template_exists' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'template_exists', isfile_call_result_25836)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'template_exists' (line 193)
        template_exists_25837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'template_exists')
        # Applying the 'not' unary operator (line 193)
        result_not__25838 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), 'not', template_exists_25837)
        
        
        # Call to _manifest_is_not_generated(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_25841 = {}
        # Getting the type of 'self' (line 193)
        self_25839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 35), 'self', False)
        # Obtaining the member '_manifest_is_not_generated' of a type (line 193)
        _manifest_is_not_generated_25840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 35), self_25839, '_manifest_is_not_generated')
        # Calling _manifest_is_not_generated(args, kwargs) (line 193)
        _manifest_is_not_generated_call_result_25842 = invoke(stypy.reporting.localization.Localization(__file__, 193, 35), _manifest_is_not_generated_25840, *[], **kwargs_25841)
        
        # Applying the binary operator 'and' (line 193)
        result_and_keyword_25843 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), 'and', result_not__25838, _manifest_is_not_generated_call_result_25842)
        
        # Testing the type of an if condition (line 193)
        if_condition_25844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), result_and_keyword_25843)
        # Assigning a type to the variable 'if_condition_25844' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_25844', if_condition_25844)
        # SSA begins for if statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to read_manifest(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_25847 = {}
        # Getting the type of 'self' (line 194)
        self_25845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'self', False)
        # Obtaining the member 'read_manifest' of a type (line 194)
        read_manifest_25846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), self_25845, 'read_manifest')
        # Calling read_manifest(args, kwargs) (line 194)
        read_manifest_call_result_25848 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), read_manifest_25846, *[], **kwargs_25847)
        
        
        # Call to sort(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_25852 = {}
        # Getting the type of 'self' (line 195)
        self_25849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self', False)
        # Obtaining the member 'filelist' of a type (line 195)
        filelist_25850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_25849, 'filelist')
        # Obtaining the member 'sort' of a type (line 195)
        sort_25851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), filelist_25850, 'sort')
        # Calling sort(args, kwargs) (line 195)
        sort_call_result_25853 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), sort_25851, *[], **kwargs_25852)
        
        
        # Call to remove_duplicates(...): (line 196)
        # Processing the call keyword arguments (line 196)
        kwargs_25857 = {}
        # Getting the type of 'self' (line 196)
        self_25854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'self', False)
        # Obtaining the member 'filelist' of a type (line 196)
        filelist_25855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), self_25854, 'filelist')
        # Obtaining the member 'remove_duplicates' of a type (line 196)
        remove_duplicates_25856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), filelist_25855, 'remove_duplicates')
        # Calling remove_duplicates(args, kwargs) (line 196)
        remove_duplicates_call_result_25858 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), remove_duplicates_25856, *[], **kwargs_25857)
        
        # Assigning a type to the variable 'stypy_return_type' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 193)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'template_exists' (line 199)
        template_exists_25859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'template_exists')
        # Applying the 'not' unary operator (line 199)
        result_not__25860 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 11), 'not', template_exists_25859)
        
        # Testing the type of an if condition (line 199)
        if_condition_25861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), result_not__25860)
        # Assigning a type to the variable 'if_condition_25861' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_25861', if_condition_25861)
        # SSA begins for if statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 200)
        # Processing the call arguments (line 200)
        str_25864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 23), 'str', "manifest template '%s' does not exist ")
        str_25865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 24), 'str', '(using default file list)')
        # Applying the binary operator '+' (line 200)
        result_add_25866 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 23), '+', str_25864, str_25865)
        
        # Getting the type of 'self' (line 202)
        self_25867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'self', False)
        # Obtaining the member 'template' of a type (line 202)
        template_25868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 24), self_25867, 'template')
        # Applying the binary operator '%' (line 200)
        result_mod_25869 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 22), '%', result_add_25866, template_25868)
        
        # Processing the call keyword arguments (line 200)
        kwargs_25870 = {}
        # Getting the type of 'self' (line 200)
        self_25862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 200)
        warn_25863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), self_25862, 'warn')
        # Calling warn(args, kwargs) (line 200)
        warn_call_result_25871 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), warn_25863, *[result_mod_25869], **kwargs_25870)
        
        # SSA join for if statement (line 199)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to findall(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_25875 = {}
        # Getting the type of 'self' (line 203)
        self_25872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self', False)
        # Obtaining the member 'filelist' of a type (line 203)
        filelist_25873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_25872, 'filelist')
        # Obtaining the member 'findall' of a type (line 203)
        findall_25874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), filelist_25873, 'findall')
        # Calling findall(args, kwargs) (line 203)
        findall_call_result_25876 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), findall_25874, *[], **kwargs_25875)
        
        
        # Getting the type of 'self' (line 205)
        self_25877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'self')
        # Obtaining the member 'use_defaults' of a type (line 205)
        use_defaults_25878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 11), self_25877, 'use_defaults')
        # Testing the type of an if condition (line 205)
        if_condition_25879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), use_defaults_25878)
        # Assigning a type to the variable 'if_condition_25879' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_25879', if_condition_25879)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add_defaults(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_25882 = {}
        # Getting the type of 'self' (line 206)
        self_25880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'self', False)
        # Obtaining the member 'add_defaults' of a type (line 206)
        add_defaults_25881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), self_25880, 'add_defaults')
        # Calling add_defaults(args, kwargs) (line 206)
        add_defaults_call_result_25883 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), add_defaults_25881, *[], **kwargs_25882)
        
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'template_exists' (line 208)
        template_exists_25884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'template_exists')
        # Testing the type of an if condition (line 208)
        if_condition_25885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), template_exists_25884)
        # Assigning a type to the variable 'if_condition_25885' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_25885', if_condition_25885)
        # SSA begins for if statement (line 208)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to read_template(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_25888 = {}
        # Getting the type of 'self' (line 209)
        self_25886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'self', False)
        # Obtaining the member 'read_template' of a type (line 209)
        read_template_25887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), self_25886, 'read_template')
        # Calling read_template(args, kwargs) (line 209)
        read_template_call_result_25889 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), read_template_25887, *[], **kwargs_25888)
        
        # SSA join for if statement (line 208)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 211)
        self_25890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'self')
        # Obtaining the member 'prune' of a type (line 211)
        prune_25891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), self_25890, 'prune')
        # Testing the type of an if condition (line 211)
        if_condition_25892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), prune_25891)
        # Assigning a type to the variable 'if_condition_25892' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_25892', if_condition_25892)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to prune_file_list(...): (line 212)
        # Processing the call keyword arguments (line 212)
        kwargs_25895 = {}
        # Getting the type of 'self' (line 212)
        self_25893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'self', False)
        # Obtaining the member 'prune_file_list' of a type (line 212)
        prune_file_list_25894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), self_25893, 'prune_file_list')
        # Calling prune_file_list(args, kwargs) (line 212)
        prune_file_list_call_result_25896 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), prune_file_list_25894, *[], **kwargs_25895)
        
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to sort(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_25900 = {}
        # Getting the type of 'self' (line 214)
        self_25897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self', False)
        # Obtaining the member 'filelist' of a type (line 214)
        filelist_25898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_25897, 'filelist')
        # Obtaining the member 'sort' of a type (line 214)
        sort_25899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), filelist_25898, 'sort')
        # Calling sort(args, kwargs) (line 214)
        sort_call_result_25901 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), sort_25899, *[], **kwargs_25900)
        
        
        # Call to remove_duplicates(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_25905 = {}
        # Getting the type of 'self' (line 215)
        self_25902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'self', False)
        # Obtaining the member 'filelist' of a type (line 215)
        filelist_25903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), self_25902, 'filelist')
        # Obtaining the member 'remove_duplicates' of a type (line 215)
        remove_duplicates_25904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), filelist_25903, 'remove_duplicates')
        # Calling remove_duplicates(args, kwargs) (line 215)
        remove_duplicates_call_result_25906 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), remove_duplicates_25904, *[], **kwargs_25905)
        
        
        # Call to write_manifest(...): (line 216)
        # Processing the call keyword arguments (line 216)
        kwargs_25909 = {}
        # Getting the type of 'self' (line 216)
        self_25907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'self', False)
        # Obtaining the member 'write_manifest' of a type (line 216)
        write_manifest_25908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), self_25907, 'write_manifest')
        # Calling write_manifest(args, kwargs) (line 216)
        write_manifest_call_result_25910 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), write_manifest_25908, *[], **kwargs_25909)
        
        
        # ################# End of 'get_file_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_file_list' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_25911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_file_list'
        return stypy_return_type_25911


    @norecursion
    def add_defaults(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_defaults'
        module_type_store = module_type_store.open_function_context('add_defaults', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.add_defaults.__dict__.__setitem__('stypy_localization', localization)
        sdist.add_defaults.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.add_defaults.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.add_defaults.__dict__.__setitem__('stypy_function_name', 'sdist.add_defaults')
        sdist.add_defaults.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.add_defaults.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.add_defaults.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.add_defaults.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.add_defaults.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.add_defaults.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.add_defaults.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.add_defaults', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_defaults', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_defaults(...)' code ##################

        str_25912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, (-1)), 'str', "Add all the default files to self.filelist:\n          - README or README.txt\n          - setup.py\n          - test/test*.py\n          - all pure Python modules mentioned in setup script\n          - all files pointed by package_data (build_py)\n          - all files defined in data_files.\n          - all files defined as scripts.\n          - all C sources listed as part of extensions or C libraries\n            in the setup script (doesn't catch C headers!)\n        Warns if (README or README.txt) or setup.py are missing; everything\n        else is optional.\n        ")
        
        # Assigning a List to a Name (line 233):
        
        # Assigning a List to a Name (line 233):
        
        # Obtaining an instance of the builtin type 'list' (line 233)
        list_25913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 233)
        # Adding element type (line 233)
        
        # Obtaining an instance of the builtin type 'tuple' (line 233)
        tuple_25914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 233)
        # Adding element type (line 233)
        str_25915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 22), 'str', 'README')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 22), tuple_25914, str_25915)
        # Adding element type (line 233)
        str_25916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 32), 'str', 'README.txt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 22), tuple_25914, str_25916)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 20), list_25913, tuple_25914)
        # Adding element type (line 233)
        # Getting the type of 'self' (line 233)
        self_25917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 47), 'self')
        # Obtaining the member 'distribution' of a type (line 233)
        distribution_25918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 47), self_25917, 'distribution')
        # Obtaining the member 'script_name' of a type (line 233)
        script_name_25919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 47), distribution_25918, 'script_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 20), list_25913, script_name_25919)
        
        # Assigning a type to the variable 'standards' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'standards', list_25913)
        
        # Getting the type of 'standards' (line 234)
        standards_25920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 18), 'standards')
        # Testing the type of a for loop iterable (line 234)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 234, 8), standards_25920)
        # Getting the type of the for loop variable (line 234)
        for_loop_var_25921 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 234, 8), standards_25920)
        # Assigning a type to the variable 'fn' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'fn', for_loop_var_25921)
        # SSA begins for a for statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 235)
        # Getting the type of 'tuple' (line 235)
        tuple_25922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), 'tuple')
        # Getting the type of 'fn' (line 235)
        fn_25923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 26), 'fn')
        
        (may_be_25924, more_types_in_union_25925) = may_be_subtype(tuple_25922, fn_25923)

        if may_be_25924:

            if more_types_in_union_25925:
                # Runtime conditional SSA (line 235)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'fn' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'fn', remove_not_subtype_from_union(fn_25923, tuple))
            
            # Assigning a Name to a Name (line 236):
            
            # Assigning a Name to a Name (line 236):
            # Getting the type of 'fn' (line 236)
            fn_25926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'fn')
            # Assigning a type to the variable 'alts' (line 236)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'alts', fn_25926)
            
            # Assigning a Num to a Name (line 237):
            
            # Assigning a Num to a Name (line 237):
            int_25927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 25), 'int')
            # Assigning a type to the variable 'got_it' (line 237)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'got_it', int_25927)
            
            # Getting the type of 'alts' (line 238)
            alts_25928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 26), 'alts')
            # Testing the type of a for loop iterable (line 238)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 238, 16), alts_25928)
            # Getting the type of the for loop variable (line 238)
            for_loop_var_25929 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 238, 16), alts_25928)
            # Assigning a type to the variable 'fn' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'fn', for_loop_var_25929)
            # SSA begins for a for statement (line 238)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to exists(...): (line 239)
            # Processing the call arguments (line 239)
            # Getting the type of 'fn' (line 239)
            fn_25933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 38), 'fn', False)
            # Processing the call keyword arguments (line 239)
            kwargs_25934 = {}
            # Getting the type of 'os' (line 239)
            os_25930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 239)
            path_25931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 23), os_25930, 'path')
            # Obtaining the member 'exists' of a type (line 239)
            exists_25932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 23), path_25931, 'exists')
            # Calling exists(args, kwargs) (line 239)
            exists_call_result_25935 = invoke(stypy.reporting.localization.Localization(__file__, 239, 23), exists_25932, *[fn_25933], **kwargs_25934)
            
            # Testing the type of an if condition (line 239)
            if_condition_25936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 20), exists_call_result_25935)
            # Assigning a type to the variable 'if_condition_25936' (line 239)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 20), 'if_condition_25936', if_condition_25936)
            # SSA begins for if statement (line 239)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 240):
            
            # Assigning a Num to a Name (line 240):
            int_25937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 33), 'int')
            # Assigning a type to the variable 'got_it' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 'got_it', int_25937)
            
            # Call to append(...): (line 241)
            # Processing the call arguments (line 241)
            # Getting the type of 'fn' (line 241)
            fn_25941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 45), 'fn', False)
            # Processing the call keyword arguments (line 241)
            kwargs_25942 = {}
            # Getting the type of 'self' (line 241)
            self_25938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'self', False)
            # Obtaining the member 'filelist' of a type (line 241)
            filelist_25939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 24), self_25938, 'filelist')
            # Obtaining the member 'append' of a type (line 241)
            append_25940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 24), filelist_25939, 'append')
            # Calling append(args, kwargs) (line 241)
            append_call_result_25943 = invoke(stypy.reporting.localization.Localization(__file__, 241, 24), append_25940, *[fn_25941], **kwargs_25942)
            
            # SSA join for if statement (line 239)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'got_it' (line 244)
            got_it_25944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 23), 'got_it')
            # Applying the 'not' unary operator (line 244)
            result_not__25945 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 19), 'not', got_it_25944)
            
            # Testing the type of an if condition (line 244)
            if_condition_25946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 16), result_not__25945)
            # Assigning a type to the variable 'if_condition_25946' (line 244)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'if_condition_25946', if_condition_25946)
            # SSA begins for if statement (line 244)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to warn(...): (line 245)
            # Processing the call arguments (line 245)
            str_25949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 30), 'str', 'standard file not found: should have one of ')
            
            # Call to join(...): (line 246)
            # Processing the call arguments (line 246)
            # Getting the type of 'alts' (line 246)
            alts_25952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), 'alts', False)
            str_25953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 48), 'str', ', ')
            # Processing the call keyword arguments (line 246)
            kwargs_25954 = {}
            # Getting the type of 'string' (line 246)
            string_25950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 30), 'string', False)
            # Obtaining the member 'join' of a type (line 246)
            join_25951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 30), string_25950, 'join')
            # Calling join(args, kwargs) (line 246)
            join_call_result_25955 = invoke(stypy.reporting.localization.Localization(__file__, 246, 30), join_25951, *[alts_25952, str_25953], **kwargs_25954)
            
            # Applying the binary operator '+' (line 245)
            result_add_25956 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 30), '+', str_25949, join_call_result_25955)
            
            # Processing the call keyword arguments (line 245)
            kwargs_25957 = {}
            # Getting the type of 'self' (line 245)
            self_25947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'self', False)
            # Obtaining the member 'warn' of a type (line 245)
            warn_25948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 20), self_25947, 'warn')
            # Calling warn(args, kwargs) (line 245)
            warn_call_result_25958 = invoke(stypy.reporting.localization.Localization(__file__, 245, 20), warn_25948, *[result_add_25956], **kwargs_25957)
            
            # SSA join for if statement (line 244)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_25925:
                # Runtime conditional SSA for else branch (line 235)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_25924) or more_types_in_union_25925):
            # Assigning a type to the variable 'fn' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'fn', remove_subtype_from_union(fn_25923, tuple))
            
            
            # Call to exists(...): (line 248)
            # Processing the call arguments (line 248)
            # Getting the type of 'fn' (line 248)
            fn_25962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 34), 'fn', False)
            # Processing the call keyword arguments (line 248)
            kwargs_25963 = {}
            # Getting the type of 'os' (line 248)
            os_25959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'os', False)
            # Obtaining the member 'path' of a type (line 248)
            path_25960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 19), os_25959, 'path')
            # Obtaining the member 'exists' of a type (line 248)
            exists_25961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 19), path_25960, 'exists')
            # Calling exists(args, kwargs) (line 248)
            exists_call_result_25964 = invoke(stypy.reporting.localization.Localization(__file__, 248, 19), exists_25961, *[fn_25962], **kwargs_25963)
            
            # Testing the type of an if condition (line 248)
            if_condition_25965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 16), exists_call_result_25964)
            # Assigning a type to the variable 'if_condition_25965' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'if_condition_25965', if_condition_25965)
            # SSA begins for if statement (line 248)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 249)
            # Processing the call arguments (line 249)
            # Getting the type of 'fn' (line 249)
            fn_25969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 41), 'fn', False)
            # Processing the call keyword arguments (line 249)
            kwargs_25970 = {}
            # Getting the type of 'self' (line 249)
            self_25966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'self', False)
            # Obtaining the member 'filelist' of a type (line 249)
            filelist_25967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 20), self_25966, 'filelist')
            # Obtaining the member 'append' of a type (line 249)
            append_25968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 20), filelist_25967, 'append')
            # Calling append(args, kwargs) (line 249)
            append_call_result_25971 = invoke(stypy.reporting.localization.Localization(__file__, 249, 20), append_25968, *[fn_25969], **kwargs_25970)
            
            # SSA branch for the else part of an if statement (line 248)
            module_type_store.open_ssa_branch('else')
            
            # Call to warn(...): (line 251)
            # Processing the call arguments (line 251)
            str_25974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 30), 'str', "standard file '%s' not found")
            # Getting the type of 'fn' (line 251)
            fn_25975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 63), 'fn', False)
            # Applying the binary operator '%' (line 251)
            result_mod_25976 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 30), '%', str_25974, fn_25975)
            
            # Processing the call keyword arguments (line 251)
            kwargs_25977 = {}
            # Getting the type of 'self' (line 251)
            self_25972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'self', False)
            # Obtaining the member 'warn' of a type (line 251)
            warn_25973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 20), self_25972, 'warn')
            # Calling warn(args, kwargs) (line 251)
            warn_call_result_25978 = invoke(stypy.reporting.localization.Localization(__file__, 251, 20), warn_25973, *[result_mod_25976], **kwargs_25977)
            
            # SSA join for if statement (line 248)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_25924 and more_types_in_union_25925):
                # SSA join for if statement (line 235)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 253):
        
        # Assigning a List to a Name (line 253):
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_25979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        str_25980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 20), 'str', 'test/test*.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 19), list_25979, str_25980)
        # Adding element type (line 253)
        str_25981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 37), 'str', 'setup.cfg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 19), list_25979, str_25981)
        
        # Assigning a type to the variable 'optional' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'optional', list_25979)
        
        # Getting the type of 'optional' (line 254)
        optional_25982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'optional')
        # Testing the type of a for loop iterable (line 254)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 254, 8), optional_25982)
        # Getting the type of the for loop variable (line 254)
        for_loop_var_25983 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 254, 8), optional_25982)
        # Assigning a type to the variable 'pattern' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'pattern', for_loop_var_25983)
        # SSA begins for a for statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 255):
        
        # Assigning a Call to a Name (line 255):
        
        # Call to filter(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'os' (line 255)
        os_25985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 255)
        path_25986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 27), os_25985, 'path')
        # Obtaining the member 'isfile' of a type (line 255)
        isfile_25987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 27), path_25986, 'isfile')
        
        # Call to glob(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'pattern' (line 255)
        pattern_25989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 48), 'pattern', False)
        # Processing the call keyword arguments (line 255)
        kwargs_25990 = {}
        # Getting the type of 'glob' (line 255)
        glob_25988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 43), 'glob', False)
        # Calling glob(args, kwargs) (line 255)
        glob_call_result_25991 = invoke(stypy.reporting.localization.Localization(__file__, 255, 43), glob_25988, *[pattern_25989], **kwargs_25990)
        
        # Processing the call keyword arguments (line 255)
        kwargs_25992 = {}
        # Getting the type of 'filter' (line 255)
        filter_25984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'filter', False)
        # Calling filter(args, kwargs) (line 255)
        filter_call_result_25993 = invoke(stypy.reporting.localization.Localization(__file__, 255, 20), filter_25984, *[isfile_25987, glob_call_result_25991], **kwargs_25992)
        
        # Assigning a type to the variable 'files' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'files', filter_call_result_25993)
        
        # Getting the type of 'files' (line 256)
        files_25994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'files')
        # Testing the type of an if condition (line 256)
        if_condition_25995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 12), files_25994)
        # Assigning a type to the variable 'if_condition_25995' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'if_condition_25995', if_condition_25995)
        # SSA begins for if statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'files' (line 257)
        files_25999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 37), 'files', False)
        # Processing the call keyword arguments (line 257)
        kwargs_26000 = {}
        # Getting the type of 'self' (line 257)
        self_25996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'self', False)
        # Obtaining the member 'filelist' of a type (line 257)
        filelist_25997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 16), self_25996, 'filelist')
        # Obtaining the member 'extend' of a type (line 257)
        extend_25998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 16), filelist_25997, 'extend')
        # Calling extend(args, kwargs) (line 257)
        extend_call_result_26001 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), extend_25998, *[files_25999], **kwargs_26000)
        
        # SSA join for if statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 262):
        
        # Assigning a Call to a Name (line 262):
        
        # Call to get_finalized_command(...): (line 262)
        # Processing the call arguments (line 262)
        str_26004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 46), 'str', 'build_py')
        # Processing the call keyword arguments (line 262)
        kwargs_26005 = {}
        # Getting the type of 'self' (line 262)
        self_26002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 262)
        get_finalized_command_26003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 19), self_26002, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 262)
        get_finalized_command_call_result_26006 = invoke(stypy.reporting.localization.Localization(__file__, 262, 19), get_finalized_command_26003, *[str_26004], **kwargs_26005)
        
        # Assigning a type to the variable 'build_py' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'build_py', get_finalized_command_call_result_26006)
        
        
        # Call to has_pure_modules(...): (line 265)
        # Processing the call keyword arguments (line 265)
        kwargs_26010 = {}
        # Getting the type of 'self' (line 265)
        self_26007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 265)
        distribution_26008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), self_26007, 'distribution')
        # Obtaining the member 'has_pure_modules' of a type (line 265)
        has_pure_modules_26009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), distribution_26008, 'has_pure_modules')
        # Calling has_pure_modules(args, kwargs) (line 265)
        has_pure_modules_call_result_26011 = invoke(stypy.reporting.localization.Localization(__file__, 265, 11), has_pure_modules_26009, *[], **kwargs_26010)
        
        # Testing the type of an if condition (line 265)
        if_condition_26012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), has_pure_modules_call_result_26011)
        # Assigning a type to the variable 'if_condition_26012' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_26012', if_condition_26012)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 266)
        # Processing the call arguments (line 266)
        
        # Call to get_source_files(...): (line 266)
        # Processing the call keyword arguments (line 266)
        kwargs_26018 = {}
        # Getting the type of 'build_py' (line 266)
        build_py_26016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'build_py', False)
        # Obtaining the member 'get_source_files' of a type (line 266)
        get_source_files_26017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 33), build_py_26016, 'get_source_files')
        # Calling get_source_files(args, kwargs) (line 266)
        get_source_files_call_result_26019 = invoke(stypy.reporting.localization.Localization(__file__, 266, 33), get_source_files_26017, *[], **kwargs_26018)
        
        # Processing the call keyword arguments (line 266)
        kwargs_26020 = {}
        # Getting the type of 'self' (line 266)
        self_26013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'self', False)
        # Obtaining the member 'filelist' of a type (line 266)
        filelist_26014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), self_26013, 'filelist')
        # Obtaining the member 'extend' of a type (line 266)
        extend_26015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), filelist_26014, 'extend')
        # Calling extend(args, kwargs) (line 266)
        extend_call_result_26021 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), extend_26015, *[get_source_files_call_result_26019], **kwargs_26020)
        
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'build_py' (line 270)
        build_py_26022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 50), 'build_py')
        # Obtaining the member 'data_files' of a type (line 270)
        data_files_26023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 50), build_py_26022, 'data_files')
        # Testing the type of a for loop iterable (line 270)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 270, 8), data_files_26023)
        # Getting the type of the for loop variable (line 270)
        for_loop_var_26024 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 270, 8), data_files_26023)
        # Assigning a type to the variable 'pkg' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'pkg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 8), for_loop_var_26024))
        # Assigning a type to the variable 'src_dir' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'src_dir', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 8), for_loop_var_26024))
        # Assigning a type to the variable 'build_dir' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'build_dir', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 8), for_loop_var_26024))
        # Assigning a type to the variable 'filenames' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'filenames', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 8), for_loop_var_26024))
        # SSA begins for a for statement (line 270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'filenames' (line 271)
        filenames_26025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 28), 'filenames')
        # Testing the type of a for loop iterable (line 271)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 271, 12), filenames_26025)
        # Getting the type of the for loop variable (line 271)
        for_loop_var_26026 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 271, 12), filenames_26025)
        # Assigning a type to the variable 'filename' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'filename', for_loop_var_26026)
        # SSA begins for a for statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 272)
        # Processing the call arguments (line 272)
        
        # Call to join(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'src_dir' (line 272)
        src_dir_26033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 50), 'src_dir', False)
        # Getting the type of 'filename' (line 272)
        filename_26034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 59), 'filename', False)
        # Processing the call keyword arguments (line 272)
        kwargs_26035 = {}
        # Getting the type of 'os' (line 272)
        os_26030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 272)
        path_26031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 37), os_26030, 'path')
        # Obtaining the member 'join' of a type (line 272)
        join_26032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 37), path_26031, 'join')
        # Calling join(args, kwargs) (line 272)
        join_call_result_26036 = invoke(stypy.reporting.localization.Localization(__file__, 272, 37), join_26032, *[src_dir_26033, filename_26034], **kwargs_26035)
        
        # Processing the call keyword arguments (line 272)
        kwargs_26037 = {}
        # Getting the type of 'self' (line 272)
        self_26027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'self', False)
        # Obtaining the member 'filelist' of a type (line 272)
        filelist_26028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), self_26027, 'filelist')
        # Obtaining the member 'append' of a type (line 272)
        append_26029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), filelist_26028, 'append')
        # Calling append(args, kwargs) (line 272)
        append_call_result_26038 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), append_26029, *[join_call_result_26036], **kwargs_26037)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_data_files(...): (line 275)
        # Processing the call keyword arguments (line 275)
        kwargs_26042 = {}
        # Getting the type of 'self' (line 275)
        self_26039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 275)
        distribution_26040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 11), self_26039, 'distribution')
        # Obtaining the member 'has_data_files' of a type (line 275)
        has_data_files_26041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 11), distribution_26040, 'has_data_files')
        # Calling has_data_files(args, kwargs) (line 275)
        has_data_files_call_result_26043 = invoke(stypy.reporting.localization.Localization(__file__, 275, 11), has_data_files_26041, *[], **kwargs_26042)
        
        # Testing the type of an if condition (line 275)
        if_condition_26044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 8), has_data_files_call_result_26043)
        # Assigning a type to the variable 'if_condition_26044' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'if_condition_26044', if_condition_26044)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 276)
        self_26045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 24), 'self')
        # Obtaining the member 'distribution' of a type (line 276)
        distribution_26046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 24), self_26045, 'distribution')
        # Obtaining the member 'data_files' of a type (line 276)
        data_files_26047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 24), distribution_26046, 'data_files')
        # Testing the type of a for loop iterable (line 276)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 276, 12), data_files_26047)
        # Getting the type of the for loop variable (line 276)
        for_loop_var_26048 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 276, 12), data_files_26047)
        # Assigning a type to the variable 'item' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'item', for_loop_var_26048)
        # SSA begins for a for statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 277)
        # Getting the type of 'str' (line 277)
        str_26049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 36), 'str')
        # Getting the type of 'item' (line 277)
        item_26050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 30), 'item')
        
        (may_be_26051, more_types_in_union_26052) = may_be_subtype(str_26049, item_26050)

        if may_be_26051:

            if more_types_in_union_26052:
                # Runtime conditional SSA (line 277)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'item' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'item', remove_not_subtype_from_union(item_26050, str))
            
            # Assigning a Call to a Name (line 278):
            
            # Assigning a Call to a Name (line 278):
            
            # Call to convert_path(...): (line 278)
            # Processing the call arguments (line 278)
            # Getting the type of 'item' (line 278)
            item_26054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 40), 'item', False)
            # Processing the call keyword arguments (line 278)
            kwargs_26055 = {}
            # Getting the type of 'convert_path' (line 278)
            convert_path_26053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 27), 'convert_path', False)
            # Calling convert_path(args, kwargs) (line 278)
            convert_path_call_result_26056 = invoke(stypy.reporting.localization.Localization(__file__, 278, 27), convert_path_26053, *[item_26054], **kwargs_26055)
            
            # Assigning a type to the variable 'item' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 20), 'item', convert_path_call_result_26056)
            
            
            # Call to isfile(...): (line 279)
            # Processing the call arguments (line 279)
            # Getting the type of 'item' (line 279)
            item_26060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 38), 'item', False)
            # Processing the call keyword arguments (line 279)
            kwargs_26061 = {}
            # Getting the type of 'os' (line 279)
            os_26057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 279)
            path_26058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 23), os_26057, 'path')
            # Obtaining the member 'isfile' of a type (line 279)
            isfile_26059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 23), path_26058, 'isfile')
            # Calling isfile(args, kwargs) (line 279)
            isfile_call_result_26062 = invoke(stypy.reporting.localization.Localization(__file__, 279, 23), isfile_26059, *[item_26060], **kwargs_26061)
            
            # Testing the type of an if condition (line 279)
            if_condition_26063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 20), isfile_call_result_26062)
            # Assigning a type to the variable 'if_condition_26063' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'if_condition_26063', if_condition_26063)
            # SSA begins for if statement (line 279)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 280)
            # Processing the call arguments (line 280)
            # Getting the type of 'item' (line 280)
            item_26067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 45), 'item', False)
            # Processing the call keyword arguments (line 280)
            kwargs_26068 = {}
            # Getting the type of 'self' (line 280)
            self_26064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'self', False)
            # Obtaining the member 'filelist' of a type (line 280)
            filelist_26065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 24), self_26064, 'filelist')
            # Obtaining the member 'append' of a type (line 280)
            append_26066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 24), filelist_26065, 'append')
            # Calling append(args, kwargs) (line 280)
            append_call_result_26069 = invoke(stypy.reporting.localization.Localization(__file__, 280, 24), append_26066, *[item_26067], **kwargs_26068)
            
            # SSA join for if statement (line 279)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_26052:
                # Runtime conditional SSA for else branch (line 277)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_26051) or more_types_in_union_26052):
            # Assigning a type to the variable 'item' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'item', remove_subtype_from_union(item_26050, str))
            
            # Assigning a Name to a Tuple (line 282):
            
            # Assigning a Subscript to a Name (line 282):
            
            # Obtaining the type of the subscript
            int_26070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 20), 'int')
            # Getting the type of 'item' (line 282)
            item_26071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 41), 'item')
            # Obtaining the member '__getitem__' of a type (line 282)
            getitem___26072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 20), item_26071, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 282)
            subscript_call_result_26073 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), getitem___26072, int_26070)
            
            # Assigning a type to the variable 'tuple_var_assignment_25646' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'tuple_var_assignment_25646', subscript_call_result_26073)
            
            # Assigning a Subscript to a Name (line 282):
            
            # Obtaining the type of the subscript
            int_26074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 20), 'int')
            # Getting the type of 'item' (line 282)
            item_26075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 41), 'item')
            # Obtaining the member '__getitem__' of a type (line 282)
            getitem___26076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 20), item_26075, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 282)
            subscript_call_result_26077 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), getitem___26076, int_26074)
            
            # Assigning a type to the variable 'tuple_var_assignment_25647' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'tuple_var_assignment_25647', subscript_call_result_26077)
            
            # Assigning a Name to a Name (line 282):
            # Getting the type of 'tuple_var_assignment_25646' (line 282)
            tuple_var_assignment_25646_26078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'tuple_var_assignment_25646')
            # Assigning a type to the variable 'dirname' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'dirname', tuple_var_assignment_25646_26078)
            
            # Assigning a Name to a Name (line 282):
            # Getting the type of 'tuple_var_assignment_25647' (line 282)
            tuple_var_assignment_25647_26079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'tuple_var_assignment_25647')
            # Assigning a type to the variable 'filenames' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 29), 'filenames', tuple_var_assignment_25647_26079)
            
            # Getting the type of 'filenames' (line 283)
            filenames_26080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 29), 'filenames')
            # Testing the type of a for loop iterable (line 283)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 20), filenames_26080)
            # Getting the type of the for loop variable (line 283)
            for_loop_var_26081 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 20), filenames_26080)
            # Assigning a type to the variable 'f' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 20), 'f', for_loop_var_26081)
            # SSA begins for a for statement (line 283)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 284):
            
            # Assigning a Call to a Name (line 284):
            
            # Call to convert_path(...): (line 284)
            # Processing the call arguments (line 284)
            # Getting the type of 'f' (line 284)
            f_26083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 41), 'f', False)
            # Processing the call keyword arguments (line 284)
            kwargs_26084 = {}
            # Getting the type of 'convert_path' (line 284)
            convert_path_26082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 28), 'convert_path', False)
            # Calling convert_path(args, kwargs) (line 284)
            convert_path_call_result_26085 = invoke(stypy.reporting.localization.Localization(__file__, 284, 28), convert_path_26082, *[f_26083], **kwargs_26084)
            
            # Assigning a type to the variable 'f' (line 284)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'f', convert_path_call_result_26085)
            
            
            # Call to isfile(...): (line 285)
            # Processing the call arguments (line 285)
            # Getting the type of 'f' (line 285)
            f_26089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 42), 'f', False)
            # Processing the call keyword arguments (line 285)
            kwargs_26090 = {}
            # Getting the type of 'os' (line 285)
            os_26086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 27), 'os', False)
            # Obtaining the member 'path' of a type (line 285)
            path_26087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 27), os_26086, 'path')
            # Obtaining the member 'isfile' of a type (line 285)
            isfile_26088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 27), path_26087, 'isfile')
            # Calling isfile(args, kwargs) (line 285)
            isfile_call_result_26091 = invoke(stypy.reporting.localization.Localization(__file__, 285, 27), isfile_26088, *[f_26089], **kwargs_26090)
            
            # Testing the type of an if condition (line 285)
            if_condition_26092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 24), isfile_call_result_26091)
            # Assigning a type to the variable 'if_condition_26092' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'if_condition_26092', if_condition_26092)
            # SSA begins for if statement (line 285)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 286)
            # Processing the call arguments (line 286)
            # Getting the type of 'f' (line 286)
            f_26096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 49), 'f', False)
            # Processing the call keyword arguments (line 286)
            kwargs_26097 = {}
            # Getting the type of 'self' (line 286)
            self_26093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 28), 'self', False)
            # Obtaining the member 'filelist' of a type (line 286)
            filelist_26094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 28), self_26093, 'filelist')
            # Obtaining the member 'append' of a type (line 286)
            append_26095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 28), filelist_26094, 'append')
            # Calling append(args, kwargs) (line 286)
            append_call_result_26098 = invoke(stypy.reporting.localization.Localization(__file__, 286, 28), append_26095, *[f_26096], **kwargs_26097)
            
            # SSA join for if statement (line 285)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_26051 and more_types_in_union_26052):
                # SSA join for if statement (line 277)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_ext_modules(...): (line 288)
        # Processing the call keyword arguments (line 288)
        kwargs_26102 = {}
        # Getting the type of 'self' (line 288)
        self_26099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 288)
        distribution_26100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 11), self_26099, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 288)
        has_ext_modules_26101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 11), distribution_26100, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 288)
        has_ext_modules_call_result_26103 = invoke(stypy.reporting.localization.Localization(__file__, 288, 11), has_ext_modules_26101, *[], **kwargs_26102)
        
        # Testing the type of an if condition (line 288)
        if_condition_26104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 8), has_ext_modules_call_result_26103)
        # Assigning a type to the variable 'if_condition_26104' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'if_condition_26104', if_condition_26104)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to get_finalized_command(...): (line 289)
        # Processing the call arguments (line 289)
        str_26107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 51), 'str', 'build_ext')
        # Processing the call keyword arguments (line 289)
        kwargs_26108 = {}
        # Getting the type of 'self' (line 289)
        self_26105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 24), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 289)
        get_finalized_command_26106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 24), self_26105, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 289)
        get_finalized_command_call_result_26109 = invoke(stypy.reporting.localization.Localization(__file__, 289, 24), get_finalized_command_26106, *[str_26107], **kwargs_26108)
        
        # Assigning a type to the variable 'build_ext' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'build_ext', get_finalized_command_call_result_26109)
        
        # Call to extend(...): (line 290)
        # Processing the call arguments (line 290)
        
        # Call to get_source_files(...): (line 290)
        # Processing the call keyword arguments (line 290)
        kwargs_26115 = {}
        # Getting the type of 'build_ext' (line 290)
        build_ext_26113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 33), 'build_ext', False)
        # Obtaining the member 'get_source_files' of a type (line 290)
        get_source_files_26114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 33), build_ext_26113, 'get_source_files')
        # Calling get_source_files(args, kwargs) (line 290)
        get_source_files_call_result_26116 = invoke(stypy.reporting.localization.Localization(__file__, 290, 33), get_source_files_26114, *[], **kwargs_26115)
        
        # Processing the call keyword arguments (line 290)
        kwargs_26117 = {}
        # Getting the type of 'self' (line 290)
        self_26110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'self', False)
        # Obtaining the member 'filelist' of a type (line 290)
        filelist_26111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), self_26110, 'filelist')
        # Obtaining the member 'extend' of a type (line 290)
        extend_26112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), filelist_26111, 'extend')
        # Calling extend(args, kwargs) (line 290)
        extend_call_result_26118 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), extend_26112, *[get_source_files_call_result_26116], **kwargs_26117)
        
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_c_libraries(...): (line 292)
        # Processing the call keyword arguments (line 292)
        kwargs_26122 = {}
        # Getting the type of 'self' (line 292)
        self_26119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 292)
        distribution_26120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 11), self_26119, 'distribution')
        # Obtaining the member 'has_c_libraries' of a type (line 292)
        has_c_libraries_26121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 11), distribution_26120, 'has_c_libraries')
        # Calling has_c_libraries(args, kwargs) (line 292)
        has_c_libraries_call_result_26123 = invoke(stypy.reporting.localization.Localization(__file__, 292, 11), has_c_libraries_26121, *[], **kwargs_26122)
        
        # Testing the type of an if condition (line 292)
        if_condition_26124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 8), has_c_libraries_call_result_26123)
        # Assigning a type to the variable 'if_condition_26124' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'if_condition_26124', if_condition_26124)
        # SSA begins for if statement (line 292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to get_finalized_command(...): (line 293)
        # Processing the call arguments (line 293)
        str_26127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 52), 'str', 'build_clib')
        # Processing the call keyword arguments (line 293)
        kwargs_26128 = {}
        # Getting the type of 'self' (line 293)
        self_26125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 25), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 293)
        get_finalized_command_26126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 25), self_26125, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 293)
        get_finalized_command_call_result_26129 = invoke(stypy.reporting.localization.Localization(__file__, 293, 25), get_finalized_command_26126, *[str_26127], **kwargs_26128)
        
        # Assigning a type to the variable 'build_clib' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'build_clib', get_finalized_command_call_result_26129)
        
        # Call to extend(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to get_source_files(...): (line 294)
        # Processing the call keyword arguments (line 294)
        kwargs_26135 = {}
        # Getting the type of 'build_clib' (line 294)
        build_clib_26133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 33), 'build_clib', False)
        # Obtaining the member 'get_source_files' of a type (line 294)
        get_source_files_26134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 33), build_clib_26133, 'get_source_files')
        # Calling get_source_files(args, kwargs) (line 294)
        get_source_files_call_result_26136 = invoke(stypy.reporting.localization.Localization(__file__, 294, 33), get_source_files_26134, *[], **kwargs_26135)
        
        # Processing the call keyword arguments (line 294)
        kwargs_26137 = {}
        # Getting the type of 'self' (line 294)
        self_26130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'self', False)
        # Obtaining the member 'filelist' of a type (line 294)
        filelist_26131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), self_26130, 'filelist')
        # Obtaining the member 'extend' of a type (line 294)
        extend_26132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), filelist_26131, 'extend')
        # Calling extend(args, kwargs) (line 294)
        extend_call_result_26138 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), extend_26132, *[get_source_files_call_result_26136], **kwargs_26137)
        
        # SSA join for if statement (line 292)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_scripts(...): (line 296)
        # Processing the call keyword arguments (line 296)
        kwargs_26142 = {}
        # Getting the type of 'self' (line 296)
        self_26139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 296)
        distribution_26140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 11), self_26139, 'distribution')
        # Obtaining the member 'has_scripts' of a type (line 296)
        has_scripts_26141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 11), distribution_26140, 'has_scripts')
        # Calling has_scripts(args, kwargs) (line 296)
        has_scripts_call_result_26143 = invoke(stypy.reporting.localization.Localization(__file__, 296, 11), has_scripts_26141, *[], **kwargs_26142)
        
        # Testing the type of an if condition (line 296)
        if_condition_26144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 8), has_scripts_call_result_26143)
        # Assigning a type to the variable 'if_condition_26144' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'if_condition_26144', if_condition_26144)
        # SSA begins for if statement (line 296)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 297):
        
        # Assigning a Call to a Name (line 297):
        
        # Call to get_finalized_command(...): (line 297)
        # Processing the call arguments (line 297)
        str_26147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 55), 'str', 'build_scripts')
        # Processing the call keyword arguments (line 297)
        kwargs_26148 = {}
        # Getting the type of 'self' (line 297)
        self_26145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 28), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 297)
        get_finalized_command_26146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 28), self_26145, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 297)
        get_finalized_command_call_result_26149 = invoke(stypy.reporting.localization.Localization(__file__, 297, 28), get_finalized_command_26146, *[str_26147], **kwargs_26148)
        
        # Assigning a type to the variable 'build_scripts' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'build_scripts', get_finalized_command_call_result_26149)
        
        # Call to extend(...): (line 298)
        # Processing the call arguments (line 298)
        
        # Call to get_source_files(...): (line 298)
        # Processing the call keyword arguments (line 298)
        kwargs_26155 = {}
        # Getting the type of 'build_scripts' (line 298)
        build_scripts_26153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 33), 'build_scripts', False)
        # Obtaining the member 'get_source_files' of a type (line 298)
        get_source_files_26154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 33), build_scripts_26153, 'get_source_files')
        # Calling get_source_files(args, kwargs) (line 298)
        get_source_files_call_result_26156 = invoke(stypy.reporting.localization.Localization(__file__, 298, 33), get_source_files_26154, *[], **kwargs_26155)
        
        # Processing the call keyword arguments (line 298)
        kwargs_26157 = {}
        # Getting the type of 'self' (line 298)
        self_26150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'self', False)
        # Obtaining the member 'filelist' of a type (line 298)
        filelist_26151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), self_26150, 'filelist')
        # Obtaining the member 'extend' of a type (line 298)
        extend_26152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), filelist_26151, 'extend')
        # Calling extend(args, kwargs) (line 298)
        extend_call_result_26158 = invoke(stypy.reporting.localization.Localization(__file__, 298, 12), extend_26152, *[get_source_files_call_result_26156], **kwargs_26157)
        
        # SSA join for if statement (line 296)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'add_defaults(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_defaults' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_26159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_defaults'
        return stypy_return_type_26159


    @norecursion
    def read_template(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_template'
        module_type_store = module_type_store.open_function_context('read_template', 300, 4, False)
        # Assigning a type to the variable 'self' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.read_template.__dict__.__setitem__('stypy_localization', localization)
        sdist.read_template.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.read_template.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.read_template.__dict__.__setitem__('stypy_function_name', 'sdist.read_template')
        sdist.read_template.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.read_template.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.read_template.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.read_template.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.read_template.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.read_template.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.read_template.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.read_template', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_template', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_template(...)' code ##################

        str_26160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, (-1)), 'str', 'Read and parse manifest template file named by self.template.\n\n        (usually "MANIFEST.in") The parsing and processing is done by\n        \'self.filelist\', which updates itself accordingly.\n        ')
        
        # Call to info(...): (line 306)
        # Processing the call arguments (line 306)
        str_26163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 17), 'str', "reading manifest template '%s'")
        # Getting the type of 'self' (line 306)
        self_26164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 51), 'self', False)
        # Obtaining the member 'template' of a type (line 306)
        template_26165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 51), self_26164, 'template')
        # Processing the call keyword arguments (line 306)
        kwargs_26166 = {}
        # Getting the type of 'log' (line 306)
        log_26161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 306)
        info_26162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), log_26161, 'info')
        # Calling info(args, kwargs) (line 306)
        info_call_result_26167 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), info_26162, *[str_26163, template_26165], **kwargs_26166)
        
        
        # Assigning a Call to a Name (line 307):
        
        # Assigning a Call to a Name (line 307):
        
        # Call to TextFile(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'self' (line 307)
        self_26169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 28), 'self', False)
        # Obtaining the member 'template' of a type (line 307)
        template_26170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 28), self_26169, 'template')
        # Processing the call keyword arguments (line 307)
        int_26171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 43), 'int')
        keyword_26172 = int_26171
        int_26173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 40), 'int')
        keyword_26174 = int_26173
        int_26175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 39), 'int')
        keyword_26176 = int_26175
        int_26177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 38), 'int')
        keyword_26178 = int_26177
        int_26179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 38), 'int')
        keyword_26180 = int_26179
        int_26181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 42), 'int')
        keyword_26182 = int_26181
        kwargs_26183 = {'lstrip_ws': keyword_26178, 'skip_blanks': keyword_26174, 'strip_comments': keyword_26172, 'rstrip_ws': keyword_26180, 'collapse_join': keyword_26182, 'join_lines': keyword_26176}
        # Getting the type of 'TextFile' (line 307)
        TextFile_26168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'TextFile', False)
        # Calling TextFile(args, kwargs) (line 307)
        TextFile_call_result_26184 = invoke(stypy.reporting.localization.Localization(__file__, 307, 19), TextFile_26168, *[template_26170], **kwargs_26183)
        
        # Assigning a type to the variable 'template' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'template', TextFile_call_result_26184)
        
        # Try-finally block (line 315)
        
        int_26185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 18), 'int')
        # Testing the type of an if condition (line 316)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 12), int_26185)
        # SSA begins for while statement (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to readline(...): (line 317)
        # Processing the call keyword arguments (line 317)
        kwargs_26188 = {}
        # Getting the type of 'template' (line 317)
        template_26186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 23), 'template', False)
        # Obtaining the member 'readline' of a type (line 317)
        readline_26187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 23), template_26186, 'readline')
        # Calling readline(args, kwargs) (line 317)
        readline_call_result_26189 = invoke(stypy.reporting.localization.Localization(__file__, 317, 23), readline_26187, *[], **kwargs_26188)
        
        # Assigning a type to the variable 'line' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 'line', readline_call_result_26189)
        
        # Type idiom detected: calculating its left and rigth part (line 318)
        # Getting the type of 'line' (line 318)
        line_26190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'line')
        # Getting the type of 'None' (line 318)
        None_26191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 27), 'None')
        
        (may_be_26192, more_types_in_union_26193) = may_be_none(line_26190, None_26191)

        if may_be_26192:

            if more_types_in_union_26193:
                # Runtime conditional SSA (line 318)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            if more_types_in_union_26193:
                # SSA join for if statement (line 318)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # SSA begins for try-except statement (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to process_template_line(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'line' (line 322)
        line_26197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 56), 'line', False)
        # Processing the call keyword arguments (line 322)
        kwargs_26198 = {}
        # Getting the type of 'self' (line 322)
        self_26194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'self', False)
        # Obtaining the member 'filelist' of a type (line 322)
        filelist_26195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 20), self_26194, 'filelist')
        # Obtaining the member 'process_template_line' of a type (line 322)
        process_template_line_26196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 20), filelist_26195, 'process_template_line')
        # Calling process_template_line(args, kwargs) (line 322)
        process_template_line_call_result_26199 = invoke(stypy.reporting.localization.Localization(__file__, 322, 20), process_template_line_26196, *[line_26197], **kwargs_26198)
        
        # SSA branch for the except part of a try statement (line 321)
        # SSA branch for the except 'Tuple' branch of a try statement (line 321)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        
        # Obtaining an instance of the builtin type 'tuple' (line 326)
        tuple_26200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 326)
        # Adding element type (line 326)
        # Getting the type of 'DistutilsTemplateError' (line 326)
        DistutilsTemplateError_26201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 24), 'DistutilsTemplateError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 24), tuple_26200, DistutilsTemplateError_26201)
        # Adding element type (line 326)
        # Getting the type of 'ValueError' (line 326)
        ValueError_26202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 48), 'ValueError')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 24), tuple_26200, ValueError_26202)
        
        # Assigning a type to the variable 'msg' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'msg', tuple_26200)
        
        # Call to warn(...): (line 327)
        # Processing the call arguments (line 327)
        str_26205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 30), 'str', '%s, line %d: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 327)
        tuple_26206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 327)
        # Adding element type (line 327)
        # Getting the type of 'template' (line 327)
        template_26207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 51), 'template', False)
        # Obtaining the member 'filename' of a type (line 327)
        filename_26208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 51), template_26207, 'filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 51), tuple_26206, filename_26208)
        # Adding element type (line 327)
        # Getting the type of 'template' (line 328)
        template_26209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 51), 'template', False)
        # Obtaining the member 'current_line' of a type (line 328)
        current_line_26210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 51), template_26209, 'current_line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 51), tuple_26206, current_line_26210)
        # Adding element type (line 327)
        # Getting the type of 'msg' (line 329)
        msg_26211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 51), 'msg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 51), tuple_26206, msg_26211)
        
        # Applying the binary operator '%' (line 327)
        result_mod_26212 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 30), '%', str_26205, tuple_26206)
        
        # Processing the call keyword arguments (line 327)
        kwargs_26213 = {}
        # Getting the type of 'self' (line 327)
        self_26203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 20), 'self', False)
        # Obtaining the member 'warn' of a type (line 327)
        warn_26204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 20), self_26203, 'warn')
        # Calling warn(args, kwargs) (line 327)
        warn_call_result_26214 = invoke(stypy.reporting.localization.Localization(__file__, 327, 20), warn_26204, *[result_mod_26212], **kwargs_26213)
        
        # SSA join for try-except statement (line 321)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 316)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 315)
        
        # Call to close(...): (line 331)
        # Processing the call keyword arguments (line 331)
        kwargs_26217 = {}
        # Getting the type of 'template' (line 331)
        template_26215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'template', False)
        # Obtaining the member 'close' of a type (line 331)
        close_26216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), template_26215, 'close')
        # Calling close(args, kwargs) (line 331)
        close_call_result_26218 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), close_26216, *[], **kwargs_26217)
        
        
        
        # ################# End of 'read_template(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_template' in the type store
        # Getting the type of 'stypy_return_type' (line 300)
        stypy_return_type_26219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_template'
        return stypy_return_type_26219


    @norecursion
    def prune_file_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'prune_file_list'
        module_type_store = module_type_store.open_function_context('prune_file_list', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.prune_file_list.__dict__.__setitem__('stypy_localization', localization)
        sdist.prune_file_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.prune_file_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.prune_file_list.__dict__.__setitem__('stypy_function_name', 'sdist.prune_file_list')
        sdist.prune_file_list.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.prune_file_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.prune_file_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.prune_file_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.prune_file_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.prune_file_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.prune_file_list.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.prune_file_list', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'prune_file_list', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'prune_file_list(...)' code ##################

        str_26220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, (-1)), 'str', 'Prune off branches that might slip into the file list as created\n        by \'read_template()\', but really don\'t belong there:\n          * the build tree (typically "build")\n          * the release tree itself (only an issue if we ran "sdist"\n            previously with --keep-temp, or it aborted)\n          * any RCS, CVS, .svn, .hg, .git, .bzr, _darcs directories\n        ')
        
        # Assigning a Call to a Name (line 341):
        
        # Assigning a Call to a Name (line 341):
        
        # Call to get_finalized_command(...): (line 341)
        # Processing the call arguments (line 341)
        str_26223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 43), 'str', 'build')
        # Processing the call keyword arguments (line 341)
        kwargs_26224 = {}
        # Getting the type of 'self' (line 341)
        self_26221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 341)
        get_finalized_command_26222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 16), self_26221, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 341)
        get_finalized_command_call_result_26225 = invoke(stypy.reporting.localization.Localization(__file__, 341, 16), get_finalized_command_26222, *[str_26223], **kwargs_26224)
        
        # Assigning a type to the variable 'build' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'build', get_finalized_command_call_result_26225)
        
        # Assigning a Call to a Name (line 342):
        
        # Assigning a Call to a Name (line 342):
        
        # Call to get_fullname(...): (line 342)
        # Processing the call keyword arguments (line 342)
        kwargs_26229 = {}
        # Getting the type of 'self' (line 342)
        self_26226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 'self', False)
        # Obtaining the member 'distribution' of a type (line 342)
        distribution_26227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 19), self_26226, 'distribution')
        # Obtaining the member 'get_fullname' of a type (line 342)
        get_fullname_26228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 19), distribution_26227, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 342)
        get_fullname_call_result_26230 = invoke(stypy.reporting.localization.Localization(__file__, 342, 19), get_fullname_26228, *[], **kwargs_26229)
        
        # Assigning a type to the variable 'base_dir' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'base_dir', get_fullname_call_result_26230)
        
        # Call to exclude_pattern(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'None' (line 344)
        None_26234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 38), 'None', False)
        # Processing the call keyword arguments (line 344)
        # Getting the type of 'build' (line 344)
        build_26235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 51), 'build', False)
        # Obtaining the member 'build_base' of a type (line 344)
        build_base_26236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 51), build_26235, 'build_base')
        keyword_26237 = build_base_26236
        kwargs_26238 = {'prefix': keyword_26237}
        # Getting the type of 'self' (line 344)
        self_26231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'self', False)
        # Obtaining the member 'filelist' of a type (line 344)
        filelist_26232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), self_26231, 'filelist')
        # Obtaining the member 'exclude_pattern' of a type (line 344)
        exclude_pattern_26233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), filelist_26232, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 344)
        exclude_pattern_call_result_26239 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), exclude_pattern_26233, *[None_26234], **kwargs_26238)
        
        
        # Call to exclude_pattern(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'None' (line 345)
        None_26243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 38), 'None', False)
        # Processing the call keyword arguments (line 345)
        # Getting the type of 'base_dir' (line 345)
        base_dir_26244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 51), 'base_dir', False)
        keyword_26245 = base_dir_26244
        kwargs_26246 = {'prefix': keyword_26245}
        # Getting the type of 'self' (line 345)
        self_26240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'self', False)
        # Obtaining the member 'filelist' of a type (line 345)
        filelist_26241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), self_26240, 'filelist')
        # Obtaining the member 'exclude_pattern' of a type (line 345)
        exclude_pattern_26242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), filelist_26241, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 345)
        exclude_pattern_call_result_26247 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), exclude_pattern_26242, *[None_26243], **kwargs_26246)
        
        
        
        # Getting the type of 'sys' (line 349)
        sys_26248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 349)
        platform_26249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 11), sys_26248, 'platform')
        str_26250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 27), 'str', 'win32')
        # Applying the binary operator '==' (line 349)
        result_eq_26251 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 11), '==', platform_26249, str_26250)
        
        # Testing the type of an if condition (line 349)
        if_condition_26252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 8), result_eq_26251)
        # Assigning a type to the variable 'if_condition_26252' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'if_condition_26252', if_condition_26252)
        # SSA begins for if statement (line 349)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 350):
        
        # Assigning a Str to a Name (line 350):
        str_26253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 19), 'str', '/|\\\\')
        # Assigning a type to the variable 'seps' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'seps', str_26253)
        # SSA branch for the else part of an if statement (line 349)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 352):
        
        # Assigning a Str to a Name (line 352):
        str_26254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 19), 'str', '/')
        # Assigning a type to the variable 'seps' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'seps', str_26254)
        # SSA join for if statement (line 349)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 354):
        
        # Assigning a List to a Name (line 354):
        
        # Obtaining an instance of the builtin type 'list' (line 354)
        list_26255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 354)
        # Adding element type (line 354)
        str_26256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 20), 'str', 'RCS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), list_26255, str_26256)
        # Adding element type (line 354)
        str_26257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 27), 'str', 'CVS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), list_26255, str_26257)
        # Adding element type (line 354)
        str_26258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 34), 'str', '\\.svn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), list_26255, str_26258)
        # Adding element type (line 354)
        str_26259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 44), 'str', '\\.hg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), list_26255, str_26259)
        # Adding element type (line 354)
        str_26260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 53), 'str', '\\.git')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), list_26255, str_26260)
        # Adding element type (line 354)
        str_26261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 63), 'str', '\\.bzr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), list_26255, str_26261)
        # Adding element type (line 354)
        str_26262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 20), 'str', '_darcs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 19), list_26255, str_26262)
        
        # Assigning a type to the variable 'vcs_dirs' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'vcs_dirs', list_26255)
        
        # Assigning a BinOp to a Name (line 356):
        
        # Assigning a BinOp to a Name (line 356):
        str_26263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 19), 'str', '(^|%s)(%s)(%s).*')
        
        # Obtaining an instance of the builtin type 'tuple' (line 356)
        tuple_26264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 356)
        # Adding element type (line 356)
        # Getting the type of 'seps' (line 356)
        seps_26265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 42), 'seps')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 42), tuple_26264, seps_26265)
        # Adding element type (line 356)
        
        # Call to join(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'vcs_dirs' (line 356)
        vcs_dirs_26268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 57), 'vcs_dirs', False)
        # Processing the call keyword arguments (line 356)
        kwargs_26269 = {}
        str_26266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 48), 'str', '|')
        # Obtaining the member 'join' of a type (line 356)
        join_26267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 48), str_26266, 'join')
        # Calling join(args, kwargs) (line 356)
        join_call_result_26270 = invoke(stypy.reporting.localization.Localization(__file__, 356, 48), join_26267, *[vcs_dirs_26268], **kwargs_26269)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 42), tuple_26264, join_call_result_26270)
        # Adding element type (line 356)
        # Getting the type of 'seps' (line 356)
        seps_26271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 68), 'seps')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 42), tuple_26264, seps_26271)
        
        # Applying the binary operator '%' (line 356)
        result_mod_26272 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 19), '%', str_26263, tuple_26264)
        
        # Assigning a type to the variable 'vcs_ptrn' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'vcs_ptrn', result_mod_26272)
        
        # Call to exclude_pattern(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'vcs_ptrn' (line 357)
        vcs_ptrn_26276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 38), 'vcs_ptrn', False)
        # Processing the call keyword arguments (line 357)
        int_26277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 57), 'int')
        keyword_26278 = int_26277
        kwargs_26279 = {'is_regex': keyword_26278}
        # Getting the type of 'self' (line 357)
        self_26273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'self', False)
        # Obtaining the member 'filelist' of a type (line 357)
        filelist_26274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), self_26273, 'filelist')
        # Obtaining the member 'exclude_pattern' of a type (line 357)
        exclude_pattern_26275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), filelist_26274, 'exclude_pattern')
        # Calling exclude_pattern(args, kwargs) (line 357)
        exclude_pattern_call_result_26280 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), exclude_pattern_26275, *[vcs_ptrn_26276], **kwargs_26279)
        
        
        # ################# End of 'prune_file_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'prune_file_list' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_26281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26281)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'prune_file_list'
        return stypy_return_type_26281


    @norecursion
    def write_manifest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_manifest'
        module_type_store = module_type_store.open_function_context('write_manifest', 359, 4, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.write_manifest.__dict__.__setitem__('stypy_localization', localization)
        sdist.write_manifest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.write_manifest.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.write_manifest.__dict__.__setitem__('stypy_function_name', 'sdist.write_manifest')
        sdist.write_manifest.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.write_manifest.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.write_manifest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.write_manifest.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.write_manifest.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.write_manifest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.write_manifest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.write_manifest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_manifest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_manifest(...)' code ##################

        str_26282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, (-1)), 'str', "Write the file list in 'self.filelist' (presumably as filled in\n        by 'add_defaults()' and 'read_template()') to the manifest file\n        named by 'self.manifest'.\n        ")
        
        
        # Call to _manifest_is_not_generated(...): (line 364)
        # Processing the call keyword arguments (line 364)
        kwargs_26285 = {}
        # Getting the type of 'self' (line 364)
        self_26283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 11), 'self', False)
        # Obtaining the member '_manifest_is_not_generated' of a type (line 364)
        _manifest_is_not_generated_26284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 11), self_26283, '_manifest_is_not_generated')
        # Calling _manifest_is_not_generated(args, kwargs) (line 364)
        _manifest_is_not_generated_call_result_26286 = invoke(stypy.reporting.localization.Localization(__file__, 364, 11), _manifest_is_not_generated_26284, *[], **kwargs_26285)
        
        # Testing the type of an if condition (line 364)
        if_condition_26287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 8), _manifest_is_not_generated_call_result_26286)
        # Assigning a type to the variable 'if_condition_26287' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'if_condition_26287', if_condition_26287)
        # SSA begins for if statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 365)
        # Processing the call arguments (line 365)
        str_26290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 21), 'str', "not writing to manually maintained manifest file '%s'")
        # Getting the type of 'self' (line 366)
        self_26291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 44), 'self', False)
        # Obtaining the member 'manifest' of a type (line 366)
        manifest_26292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 44), self_26291, 'manifest')
        # Applying the binary operator '%' (line 365)
        result_mod_26293 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 21), '%', str_26290, manifest_26292)
        
        # Processing the call keyword arguments (line 365)
        kwargs_26294 = {}
        # Getting the type of 'log' (line 365)
        log_26288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 365)
        info_26289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 12), log_26288, 'info')
        # Calling info(args, kwargs) (line 365)
        info_call_result_26295 = invoke(stypy.reporting.localization.Localization(__file__, 365, 12), info_26289, *[result_mod_26293], **kwargs_26294)
        
        # Assigning a type to the variable 'stypy_return_type' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 369):
        
        # Assigning a Subscript to a Name (line 369):
        
        # Obtaining the type of the subscript
        slice_26296 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 369, 18), None, None, None)
        # Getting the type of 'self' (line 369)
        self_26297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 18), 'self')
        # Obtaining the member 'filelist' of a type (line 369)
        filelist_26298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 18), self_26297, 'filelist')
        # Obtaining the member 'files' of a type (line 369)
        files_26299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 18), filelist_26298, 'files')
        # Obtaining the member '__getitem__' of a type (line 369)
        getitem___26300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 18), files_26299, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 369)
        subscript_call_result_26301 = invoke(stypy.reporting.localization.Localization(__file__, 369, 18), getitem___26300, slice_26296)
        
        # Assigning a type to the variable 'content' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'content', subscript_call_result_26301)
        
        # Call to insert(...): (line 370)
        # Processing the call arguments (line 370)
        int_26304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 23), 'int')
        str_26305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 26), 'str', '# file GENERATED by distutils, do NOT edit')
        # Processing the call keyword arguments (line 370)
        kwargs_26306 = {}
        # Getting the type of 'content' (line 370)
        content_26302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'content', False)
        # Obtaining the member 'insert' of a type (line 370)
        insert_26303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), content_26302, 'insert')
        # Calling insert(args, kwargs) (line 370)
        insert_call_result_26307 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), insert_26303, *[int_26304, str_26305], **kwargs_26306)
        
        
        # Call to execute(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'file_util' (line 371)
        file_util_26310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'file_util', False)
        # Obtaining the member 'write_file' of a type (line 371)
        write_file_26311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 21), file_util_26310, 'write_file')
        
        # Obtaining an instance of the builtin type 'tuple' (line 371)
        tuple_26312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 371)
        # Adding element type (line 371)
        # Getting the type of 'self' (line 371)
        self_26313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 44), 'self', False)
        # Obtaining the member 'manifest' of a type (line 371)
        manifest_26314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 44), self_26313, 'manifest')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 44), tuple_26312, manifest_26314)
        # Adding element type (line 371)
        # Getting the type of 'content' (line 371)
        content_26315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 59), 'content', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 44), tuple_26312, content_26315)
        
        str_26316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 21), 'str', "writing manifest file '%s'")
        # Getting the type of 'self' (line 372)
        self_26317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 52), 'self', False)
        # Obtaining the member 'manifest' of a type (line 372)
        manifest_26318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 52), self_26317, 'manifest')
        # Applying the binary operator '%' (line 372)
        result_mod_26319 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 21), '%', str_26316, manifest_26318)
        
        # Processing the call keyword arguments (line 371)
        kwargs_26320 = {}
        # Getting the type of 'self' (line 371)
        self_26308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'self', False)
        # Obtaining the member 'execute' of a type (line 371)
        execute_26309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), self_26308, 'execute')
        # Calling execute(args, kwargs) (line 371)
        execute_call_result_26321 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), execute_26309, *[write_file_26311, tuple_26312, result_mod_26319], **kwargs_26320)
        
        
        # ################# End of 'write_manifest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_manifest' in the type store
        # Getting the type of 'stypy_return_type' (line 359)
        stypy_return_type_26322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_manifest'
        return stypy_return_type_26322


    @norecursion
    def _manifest_is_not_generated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_manifest_is_not_generated'
        module_type_store = module_type_store.open_function_context('_manifest_is_not_generated', 374, 4, False)
        # Assigning a type to the variable 'self' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_localization', localization)
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_function_name', 'sdist._manifest_is_not_generated')
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_param_names_list', [])
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist._manifest_is_not_generated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist._manifest_is_not_generated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_manifest_is_not_generated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_manifest_is_not_generated(...)' code ##################

        
        
        
        # Call to isfile(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'self' (line 376)
        self_26326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 30), 'self', False)
        # Obtaining the member 'manifest' of a type (line 376)
        manifest_26327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 30), self_26326, 'manifest')
        # Processing the call keyword arguments (line 376)
        kwargs_26328 = {}
        # Getting the type of 'os' (line 376)
        os_26323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 376)
        path_26324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 15), os_26323, 'path')
        # Obtaining the member 'isfile' of a type (line 376)
        isfile_26325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 15), path_26324, 'isfile')
        # Calling isfile(args, kwargs) (line 376)
        isfile_call_result_26329 = invoke(stypy.reporting.localization.Localization(__file__, 376, 15), isfile_26325, *[manifest_26327], **kwargs_26328)
        
        # Applying the 'not' unary operator (line 376)
        result_not__26330 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 11), 'not', isfile_call_result_26329)
        
        # Testing the type of an if condition (line 376)
        if_condition_26331 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 8), result_not__26330)
        # Assigning a type to the variable 'if_condition_26331' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'if_condition_26331', if_condition_26331)
        # SSA begins for if statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 377)
        False_26332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'stypy_return_type', False_26332)
        # SSA join for if statement (line 376)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 379):
        
        # Assigning a Call to a Name (line 379):
        
        # Call to open(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'self' (line 379)
        self_26334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 18), 'self', False)
        # Obtaining the member 'manifest' of a type (line 379)
        manifest_26335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 18), self_26334, 'manifest')
        str_26336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 33), 'str', 'rU')
        # Processing the call keyword arguments (line 379)
        kwargs_26337 = {}
        # Getting the type of 'open' (line 379)
        open_26333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 13), 'open', False)
        # Calling open(args, kwargs) (line 379)
        open_call_result_26338 = invoke(stypy.reporting.localization.Localization(__file__, 379, 13), open_26333, *[manifest_26335, str_26336], **kwargs_26337)
        
        # Assigning a type to the variable 'fp' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'fp', open_call_result_26338)
        
        # Try-finally block (line 380)
        
        # Assigning a Call to a Name (line 381):
        
        # Assigning a Call to a Name (line 381):
        
        # Call to readline(...): (line 381)
        # Processing the call keyword arguments (line 381)
        kwargs_26341 = {}
        # Getting the type of 'fp' (line 381)
        fp_26339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 25), 'fp', False)
        # Obtaining the member 'readline' of a type (line 381)
        readline_26340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 25), fp_26339, 'readline')
        # Calling readline(args, kwargs) (line 381)
        readline_call_result_26342 = invoke(stypy.reporting.localization.Localization(__file__, 381, 25), readline_26340, *[], **kwargs_26341)
        
        # Assigning a type to the variable 'first_line' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'first_line', readline_call_result_26342)
        
        # finally branch of the try-finally block (line 380)
        
        # Call to close(...): (line 383)
        # Processing the call keyword arguments (line 383)
        kwargs_26345 = {}
        # Getting the type of 'fp' (line 383)
        fp_26343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'fp', False)
        # Obtaining the member 'close' of a type (line 383)
        close_26344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 12), fp_26343, 'close')
        # Calling close(args, kwargs) (line 383)
        close_call_result_26346 = invoke(stypy.reporting.localization.Localization(__file__, 383, 12), close_26344, *[], **kwargs_26345)
        
        
        
        # Getting the type of 'first_line' (line 384)
        first_line_26347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'first_line')
        str_26348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 29), 'str', '# file GENERATED by distutils, do NOT edit\n')
        # Applying the binary operator '!=' (line 384)
        result_ne_26349 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 15), '!=', first_line_26347, str_26348)
        
        # Assigning a type to the variable 'stypy_return_type' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'stypy_return_type', result_ne_26349)
        
        # ################# End of '_manifest_is_not_generated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_manifest_is_not_generated' in the type store
        # Getting the type of 'stypy_return_type' (line 374)
        stypy_return_type_26350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26350)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_manifest_is_not_generated'
        return stypy_return_type_26350


    @norecursion
    def read_manifest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'read_manifest'
        module_type_store = module_type_store.open_function_context('read_manifest', 386, 4, False)
        # Assigning a type to the variable 'self' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.read_manifest.__dict__.__setitem__('stypy_localization', localization)
        sdist.read_manifest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.read_manifest.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.read_manifest.__dict__.__setitem__('stypy_function_name', 'sdist.read_manifest')
        sdist.read_manifest.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.read_manifest.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.read_manifest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.read_manifest.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.read_manifest.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.read_manifest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.read_manifest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.read_manifest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'read_manifest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'read_manifest(...)' code ##################

        str_26351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, (-1)), 'str', "Read the manifest file (named by 'self.manifest') and use it to\n        fill in 'self.filelist', the list of files to include in the source\n        distribution.\n        ")
        
        # Call to info(...): (line 391)
        # Processing the call arguments (line 391)
        str_26354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 17), 'str', "reading manifest file '%s'")
        # Getting the type of 'self' (line 391)
        self_26355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 47), 'self', False)
        # Obtaining the member 'manifest' of a type (line 391)
        manifest_26356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 47), self_26355, 'manifest')
        # Processing the call keyword arguments (line 391)
        kwargs_26357 = {}
        # Getting the type of 'log' (line 391)
        log_26352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 391)
        info_26353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), log_26352, 'info')
        # Calling info(args, kwargs) (line 391)
        info_call_result_26358 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), info_26353, *[str_26354, manifest_26356], **kwargs_26357)
        
        
        # Assigning a Call to a Name (line 392):
        
        # Assigning a Call to a Name (line 392):
        
        # Call to open(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'self' (line 392)
        self_26360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 24), 'self', False)
        # Obtaining the member 'manifest' of a type (line 392)
        manifest_26361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 24), self_26360, 'manifest')
        # Processing the call keyword arguments (line 392)
        kwargs_26362 = {}
        # Getting the type of 'open' (line 392)
        open_26359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'open', False)
        # Calling open(args, kwargs) (line 392)
        open_call_result_26363 = invoke(stypy.reporting.localization.Localization(__file__, 392, 19), open_26359, *[manifest_26361], **kwargs_26362)
        
        # Assigning a type to the variable 'manifest' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'manifest', open_call_result_26363)
        
        # Getting the type of 'manifest' (line 393)
        manifest_26364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 'manifest')
        # Testing the type of a for loop iterable (line 393)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 393, 8), manifest_26364)
        # Getting the type of the for loop variable (line 393)
        for_loop_var_26365 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 393, 8), manifest_26364)
        # Assigning a type to the variable 'line' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'line', for_loop_var_26365)
        # SSA begins for a for statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to strip(...): (line 395)
        # Processing the call keyword arguments (line 395)
        kwargs_26368 = {}
        # Getting the type of 'line' (line 395)
        line_26366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 19), 'line', False)
        # Obtaining the member 'strip' of a type (line 395)
        strip_26367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 19), line_26366, 'strip')
        # Calling strip(args, kwargs) (line 395)
        strip_call_result_26369 = invoke(stypy.reporting.localization.Localization(__file__, 395, 19), strip_26367, *[], **kwargs_26368)
        
        # Assigning a type to the variable 'line' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'line', strip_call_result_26369)
        
        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 396)
        # Processing the call arguments (line 396)
        str_26372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 31), 'str', '#')
        # Processing the call keyword arguments (line 396)
        kwargs_26373 = {}
        # Getting the type of 'line' (line 396)
        line_26370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 15), 'line', False)
        # Obtaining the member 'startswith' of a type (line 396)
        startswith_26371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 15), line_26370, 'startswith')
        # Calling startswith(args, kwargs) (line 396)
        startswith_call_result_26374 = invoke(stypy.reporting.localization.Localization(__file__, 396, 15), startswith_26371, *[str_26372], **kwargs_26373)
        
        
        # Getting the type of 'line' (line 396)
        line_26375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 43), 'line')
        # Applying the 'not' unary operator (line 396)
        result_not__26376 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 39), 'not', line_26375)
        
        # Applying the binary operator 'or' (line 396)
        result_or_keyword_26377 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 15), 'or', startswith_call_result_26374, result_not__26376)
        
        # Testing the type of an if condition (line 396)
        if_condition_26378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 12), result_or_keyword_26377)
        # Assigning a type to the variable 'if_condition_26378' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'if_condition_26378', if_condition_26378)
        # SSA begins for if statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'line' (line 398)
        line_26382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 33), 'line', False)
        # Processing the call keyword arguments (line 398)
        kwargs_26383 = {}
        # Getting the type of 'self' (line 398)
        self_26379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'self', False)
        # Obtaining the member 'filelist' of a type (line 398)
        filelist_26380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), self_26379, 'filelist')
        # Obtaining the member 'append' of a type (line 398)
        append_26381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), filelist_26380, 'append')
        # Calling append(args, kwargs) (line 398)
        append_call_result_26384 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), append_26381, *[line_26382], **kwargs_26383)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to close(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_26387 = {}
        # Getting the type of 'manifest' (line 399)
        manifest_26385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'manifest', False)
        # Obtaining the member 'close' of a type (line 399)
        close_26386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), manifest_26385, 'close')
        # Calling close(args, kwargs) (line 399)
        close_call_result_26388 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), close_26386, *[], **kwargs_26387)
        
        
        # ################# End of 'read_manifest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'read_manifest' in the type store
        # Getting the type of 'stypy_return_type' (line 386)
        stypy_return_type_26389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'read_manifest'
        return stypy_return_type_26389


    @norecursion
    def make_release_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_release_tree'
        module_type_store = module_type_store.open_function_context('make_release_tree', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.make_release_tree.__dict__.__setitem__('stypy_localization', localization)
        sdist.make_release_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.make_release_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.make_release_tree.__dict__.__setitem__('stypy_function_name', 'sdist.make_release_tree')
        sdist.make_release_tree.__dict__.__setitem__('stypy_param_names_list', ['base_dir', 'files'])
        sdist.make_release_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.make_release_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.make_release_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.make_release_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.make_release_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.make_release_tree.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.make_release_tree', ['base_dir', 'files'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_release_tree', localization, ['base_dir', 'files'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_release_tree(...)' code ##################

        str_26390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, (-1)), 'str', "Create the directory tree that will become the source\n        distribution archive.  All directories implied by the filenames in\n        'files' are created under 'base_dir', and then we hard link or copy\n        (if hard linking is unavailable) those files into place.\n        Essentially, this duplicates the developer's source tree, but in a\n        directory named after the distribution, containing only the files\n        to be distributed.\n        ")
        
        # Call to mkpath(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'base_dir' (line 413)
        base_dir_26393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 20), 'base_dir', False)
        # Processing the call keyword arguments (line 413)
        kwargs_26394 = {}
        # Getting the type of 'self' (line 413)
        self_26391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 413)
        mkpath_26392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 8), self_26391, 'mkpath')
        # Calling mkpath(args, kwargs) (line 413)
        mkpath_call_result_26395 = invoke(stypy.reporting.localization.Localization(__file__, 413, 8), mkpath_26392, *[base_dir_26393], **kwargs_26394)
        
        
        # Call to create_tree(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'base_dir' (line 414)
        base_dir_26398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 29), 'base_dir', False)
        # Getting the type of 'files' (line 414)
        files_26399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 39), 'files', False)
        # Processing the call keyword arguments (line 414)
        # Getting the type of 'self' (line 414)
        self_26400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 54), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 414)
        dry_run_26401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 54), self_26400, 'dry_run')
        keyword_26402 = dry_run_26401
        kwargs_26403 = {'dry_run': keyword_26402}
        # Getting the type of 'dir_util' (line 414)
        dir_util_26396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'dir_util', False)
        # Obtaining the member 'create_tree' of a type (line 414)
        create_tree_26397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 8), dir_util_26396, 'create_tree')
        # Calling create_tree(args, kwargs) (line 414)
        create_tree_call_result_26404 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), create_tree_26397, *[base_dir_26398, files_26399], **kwargs_26403)
        
        
        # Type idiom detected: calculating its left and rigth part (line 423)
        str_26405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 23), 'str', 'link')
        # Getting the type of 'os' (line 423)
        os_26406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 19), 'os')
        
        (may_be_26407, more_types_in_union_26408) = may_provide_member(str_26405, os_26406)

        if may_be_26407:

            if more_types_in_union_26408:
                # Runtime conditional SSA (line 423)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'os' (line 423)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'os', remove_not_member_provider_from_union(os_26406, 'link'))
            
            # Assigning a Str to a Name (line 424):
            
            # Assigning a Str to a Name (line 424):
            str_26409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 19), 'str', 'hard')
            # Assigning a type to the variable 'link' (line 424)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'link', str_26409)
            
            # Assigning a BinOp to a Name (line 425):
            
            # Assigning a BinOp to a Name (line 425):
            str_26410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 18), 'str', 'making hard links in %s...')
            # Getting the type of 'base_dir' (line 425)
            base_dir_26411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 49), 'base_dir')
            # Applying the binary operator '%' (line 425)
            result_mod_26412 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 18), '%', str_26410, base_dir_26411)
            
            # Assigning a type to the variable 'msg' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'msg', result_mod_26412)

            if more_types_in_union_26408:
                # Runtime conditional SSA for else branch (line 423)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_26407) or more_types_in_union_26408):
            # Assigning a type to the variable 'os' (line 423)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'os', remove_member_provider_from_union(os_26406, 'link'))
            
            # Assigning a Name to a Name (line 427):
            
            # Assigning a Name to a Name (line 427):
            # Getting the type of 'None' (line 427)
            None_26413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 19), 'None')
            # Assigning a type to the variable 'link' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'link', None_26413)
            
            # Assigning a BinOp to a Name (line 428):
            
            # Assigning a BinOp to a Name (line 428):
            str_26414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 18), 'str', 'copying files to %s...')
            # Getting the type of 'base_dir' (line 428)
            base_dir_26415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 45), 'base_dir')
            # Applying the binary operator '%' (line 428)
            result_mod_26416 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 18), '%', str_26414, base_dir_26415)
            
            # Assigning a type to the variable 'msg' (line 428)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'msg', result_mod_26416)

            if (may_be_26407 and more_types_in_union_26408):
                # SSA join for if statement (line 423)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'files' (line 430)
        files_26417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 15), 'files')
        # Applying the 'not' unary operator (line 430)
        result_not__26418 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), 'not', files_26417)
        
        # Testing the type of an if condition (line 430)
        if_condition_26419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 8), result_not__26418)
        # Assigning a type to the variable 'if_condition_26419' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'if_condition_26419', if_condition_26419)
        # SSA begins for if statement (line 430)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 431)
        # Processing the call arguments (line 431)
        str_26422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 21), 'str', 'no files to distribute -- empty manifest?')
        # Processing the call keyword arguments (line 431)
        kwargs_26423 = {}
        # Getting the type of 'log' (line 431)
        log_26420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'log', False)
        # Obtaining the member 'warn' of a type (line 431)
        warn_26421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 12), log_26420, 'warn')
        # Calling warn(args, kwargs) (line 431)
        warn_call_result_26424 = invoke(stypy.reporting.localization.Localization(__file__, 431, 12), warn_26421, *[str_26422], **kwargs_26423)
        
        # SSA branch for the else part of an if statement (line 430)
        module_type_store.open_ssa_branch('else')
        
        # Call to info(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'msg' (line 433)
        msg_26427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 21), 'msg', False)
        # Processing the call keyword arguments (line 433)
        kwargs_26428 = {}
        # Getting the type of 'log' (line 433)
        log_26425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 433)
        info_26426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 12), log_26425, 'info')
        # Calling info(args, kwargs) (line 433)
        info_call_result_26429 = invoke(stypy.reporting.localization.Localization(__file__, 433, 12), info_26426, *[msg_26427], **kwargs_26428)
        
        # SSA join for if statement (line 430)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'files' (line 434)
        files_26430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'files')
        # Testing the type of a for loop iterable (line 434)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 434, 8), files_26430)
        # Getting the type of the for loop variable (line 434)
        for_loop_var_26431 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 434, 8), files_26430)
        # Assigning a type to the variable 'file' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'file', for_loop_var_26431)
        # SSA begins for a for statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to isfile(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'file' (line 435)
        file_26435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 34), 'file', False)
        # Processing the call keyword arguments (line 435)
        kwargs_26436 = {}
        # Getting the type of 'os' (line 435)
        os_26432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 435)
        path_26433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 19), os_26432, 'path')
        # Obtaining the member 'isfile' of a type (line 435)
        isfile_26434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 19), path_26433, 'isfile')
        # Calling isfile(args, kwargs) (line 435)
        isfile_call_result_26437 = invoke(stypy.reporting.localization.Localization(__file__, 435, 19), isfile_26434, *[file_26435], **kwargs_26436)
        
        # Applying the 'not' unary operator (line 435)
        result_not__26438 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 15), 'not', isfile_call_result_26437)
        
        # Testing the type of an if condition (line 435)
        if_condition_26439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 12), result_not__26438)
        # Assigning a type to the variable 'if_condition_26439' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'if_condition_26439', if_condition_26439)
        # SSA begins for if statement (line 435)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 436)
        # Processing the call arguments (line 436)
        str_26442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 25), 'str', "'%s' not a regular file -- skipping")
        # Getting the type of 'file' (line 436)
        file_26443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 65), 'file', False)
        # Applying the binary operator '%' (line 436)
        result_mod_26444 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 25), '%', str_26442, file_26443)
        
        # Processing the call keyword arguments (line 436)
        kwargs_26445 = {}
        # Getting the type of 'log' (line 436)
        log_26440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 436)
        warn_26441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), log_26440, 'warn')
        # Calling warn(args, kwargs) (line 436)
        warn_call_result_26446 = invoke(stypy.reporting.localization.Localization(__file__, 436, 16), warn_26441, *[result_mod_26444], **kwargs_26445)
        
        # SSA branch for the else part of an if statement (line 435)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 438):
        
        # Assigning a Call to a Name (line 438):
        
        # Call to join(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'base_dir' (line 438)
        base_dir_26450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 36), 'base_dir', False)
        # Getting the type of 'file' (line 438)
        file_26451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 46), 'file', False)
        # Processing the call keyword arguments (line 438)
        kwargs_26452 = {}
        # Getting the type of 'os' (line 438)
        os_26447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 438)
        path_26448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 23), os_26447, 'path')
        # Obtaining the member 'join' of a type (line 438)
        join_26449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 23), path_26448, 'join')
        # Calling join(args, kwargs) (line 438)
        join_call_result_26453 = invoke(stypy.reporting.localization.Localization(__file__, 438, 23), join_26449, *[base_dir_26450, file_26451], **kwargs_26452)
        
        # Assigning a type to the variable 'dest' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'dest', join_call_result_26453)
        
        # Call to copy_file(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'file' (line 439)
        file_26456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 31), 'file', False)
        # Getting the type of 'dest' (line 439)
        dest_26457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 37), 'dest', False)
        # Processing the call keyword arguments (line 439)
        # Getting the type of 'link' (line 439)
        link_26458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 48), 'link', False)
        keyword_26459 = link_26458
        kwargs_26460 = {'link': keyword_26459}
        # Getting the type of 'self' (line 439)
        self_26454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 439)
        copy_file_26455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 16), self_26454, 'copy_file')
        # Calling copy_file(args, kwargs) (line 439)
        copy_file_call_result_26461 = invoke(stypy.reporting.localization.Localization(__file__, 439, 16), copy_file_26455, *[file_26456, dest_26457], **kwargs_26460)
        
        # SSA join for if statement (line 435)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write_pkg_info(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'base_dir' (line 441)
        base_dir_26466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 50), 'base_dir', False)
        # Processing the call keyword arguments (line 441)
        kwargs_26467 = {}
        # Getting the type of 'self' (line 441)
        self_26462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'self', False)
        # Obtaining the member 'distribution' of a type (line 441)
        distribution_26463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), self_26462, 'distribution')
        # Obtaining the member 'metadata' of a type (line 441)
        metadata_26464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), distribution_26463, 'metadata')
        # Obtaining the member 'write_pkg_info' of a type (line 441)
        write_pkg_info_26465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), metadata_26464, 'write_pkg_info')
        # Calling write_pkg_info(args, kwargs) (line 441)
        write_pkg_info_call_result_26468 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), write_pkg_info_26465, *[base_dir_26466], **kwargs_26467)
        
        
        # ################# End of 'make_release_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_release_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_26469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26469)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_release_tree'
        return stypy_return_type_26469


    @norecursion
    def make_distribution(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_distribution'
        module_type_store = module_type_store.open_function_context('make_distribution', 443, 4, False)
        # Assigning a type to the variable 'self' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.make_distribution.__dict__.__setitem__('stypy_localization', localization)
        sdist.make_distribution.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.make_distribution.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.make_distribution.__dict__.__setitem__('stypy_function_name', 'sdist.make_distribution')
        sdist.make_distribution.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.make_distribution.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.make_distribution.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.make_distribution.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.make_distribution.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.make_distribution.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.make_distribution.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.make_distribution', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_distribution', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_distribution(...)' code ##################

        str_26470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, (-1)), 'str', "Create the source distribution(s).  First, we create the release\n        tree with 'make_release_tree()'; then, we create all required\n        archive files (according to 'self.formats') from the release tree.\n        Finally, we clean up by blowing away the release tree (unless\n        'self.keep_temp' is true).  The list of archive files created is\n        stored so it can be retrieved later by 'get_archive_files()'.\n        ")
        
        # Assigning a Call to a Name (line 453):
        
        # Assigning a Call to a Name (line 453):
        
        # Call to get_fullname(...): (line 453)
        # Processing the call keyword arguments (line 453)
        kwargs_26474 = {}
        # Getting the type of 'self' (line 453)
        self_26471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 19), 'self', False)
        # Obtaining the member 'distribution' of a type (line 453)
        distribution_26472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 19), self_26471, 'distribution')
        # Obtaining the member 'get_fullname' of a type (line 453)
        get_fullname_26473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 19), distribution_26472, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 453)
        get_fullname_call_result_26475 = invoke(stypy.reporting.localization.Localization(__file__, 453, 19), get_fullname_26473, *[], **kwargs_26474)
        
        # Assigning a type to the variable 'base_dir' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'base_dir', get_fullname_call_result_26475)
        
        # Assigning a Call to a Name (line 454):
        
        # Assigning a Call to a Name (line 454):
        
        # Call to join(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'self' (line 454)
        self_26479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 33), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 454)
        dist_dir_26480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 33), self_26479, 'dist_dir')
        # Getting the type of 'base_dir' (line 454)
        base_dir_26481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 48), 'base_dir', False)
        # Processing the call keyword arguments (line 454)
        kwargs_26482 = {}
        # Getting the type of 'os' (line 454)
        os_26476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 454)
        path_26477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 20), os_26476, 'path')
        # Obtaining the member 'join' of a type (line 454)
        join_26478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 20), path_26477, 'join')
        # Calling join(args, kwargs) (line 454)
        join_call_result_26483 = invoke(stypy.reporting.localization.Localization(__file__, 454, 20), join_26478, *[dist_dir_26480, base_dir_26481], **kwargs_26482)
        
        # Assigning a type to the variable 'base_name' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'base_name', join_call_result_26483)
        
        # Call to make_release_tree(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'base_dir' (line 456)
        base_dir_26486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 31), 'base_dir', False)
        # Getting the type of 'self' (line 456)
        self_26487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 41), 'self', False)
        # Obtaining the member 'filelist' of a type (line 456)
        filelist_26488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 41), self_26487, 'filelist')
        # Obtaining the member 'files' of a type (line 456)
        files_26489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 41), filelist_26488, 'files')
        # Processing the call keyword arguments (line 456)
        kwargs_26490 = {}
        # Getting the type of 'self' (line 456)
        self_26484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'self', False)
        # Obtaining the member 'make_release_tree' of a type (line 456)
        make_release_tree_26485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), self_26484, 'make_release_tree')
        # Calling make_release_tree(args, kwargs) (line 456)
        make_release_tree_call_result_26491 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), make_release_tree_26485, *[base_dir_26486, files_26489], **kwargs_26490)
        
        
        # Assigning a List to a Name (line 457):
        
        # Assigning a List to a Name (line 457):
        
        # Obtaining an instance of the builtin type 'list' (line 457)
        list_26492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 457)
        
        # Assigning a type to the variable 'archive_files' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'archive_files', list_26492)
        
        
        str_26493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 11), 'str', 'tar')
        # Getting the type of 'self' (line 459)
        self_26494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 20), 'self')
        # Obtaining the member 'formats' of a type (line 459)
        formats_26495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 20), self_26494, 'formats')
        # Applying the binary operator 'in' (line 459)
        result_contains_26496 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 11), 'in', str_26493, formats_26495)
        
        # Testing the type of an if condition (line 459)
        if_condition_26497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 8), result_contains_26496)
        # Assigning a type to the variable 'if_condition_26497' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'if_condition_26497', if_condition_26497)
        # SSA begins for if statement (line 459)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 460)
        # Processing the call arguments (line 460)
        
        # Call to pop(...): (line 460)
        # Processing the call arguments (line 460)
        
        # Call to index(...): (line 460)
        # Processing the call arguments (line 460)
        str_26507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 68), 'str', 'tar')
        # Processing the call keyword arguments (line 460)
        kwargs_26508 = {}
        # Getting the type of 'self' (line 460)
        self_26504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 49), 'self', False)
        # Obtaining the member 'formats' of a type (line 460)
        formats_26505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 49), self_26504, 'formats')
        # Obtaining the member 'index' of a type (line 460)
        index_26506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 49), formats_26505, 'index')
        # Calling index(args, kwargs) (line 460)
        index_call_result_26509 = invoke(stypy.reporting.localization.Localization(__file__, 460, 49), index_26506, *[str_26507], **kwargs_26508)
        
        # Processing the call keyword arguments (line 460)
        kwargs_26510 = {}
        # Getting the type of 'self' (line 460)
        self_26501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 32), 'self', False)
        # Obtaining the member 'formats' of a type (line 460)
        formats_26502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 32), self_26501, 'formats')
        # Obtaining the member 'pop' of a type (line 460)
        pop_26503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 32), formats_26502, 'pop')
        # Calling pop(args, kwargs) (line 460)
        pop_call_result_26511 = invoke(stypy.reporting.localization.Localization(__file__, 460, 32), pop_26503, *[index_call_result_26509], **kwargs_26510)
        
        # Processing the call keyword arguments (line 460)
        kwargs_26512 = {}
        # Getting the type of 'self' (line 460)
        self_26498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'self', False)
        # Obtaining the member 'formats' of a type (line 460)
        formats_26499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 12), self_26498, 'formats')
        # Obtaining the member 'append' of a type (line 460)
        append_26500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 12), formats_26499, 'append')
        # Calling append(args, kwargs) (line 460)
        append_call_result_26513 = invoke(stypy.reporting.localization.Localization(__file__, 460, 12), append_26500, *[pop_call_result_26511], **kwargs_26512)
        
        # SSA join for if statement (line 459)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 462)
        self_26514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 19), 'self')
        # Obtaining the member 'formats' of a type (line 462)
        formats_26515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 19), self_26514, 'formats')
        # Testing the type of a for loop iterable (line 462)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 462, 8), formats_26515)
        # Getting the type of the for loop variable (line 462)
        for_loop_var_26516 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 462, 8), formats_26515)
        # Assigning a type to the variable 'fmt' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'fmt', for_loop_var_26516)
        # SSA begins for a for statement (line 462)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to make_archive(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'base_name' (line 463)
        base_name_26519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 37), 'base_name', False)
        # Getting the type of 'fmt' (line 463)
        fmt_26520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 48), 'fmt', False)
        # Processing the call keyword arguments (line 463)
        # Getting the type of 'base_dir' (line 463)
        base_dir_26521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 62), 'base_dir', False)
        keyword_26522 = base_dir_26521
        # Getting the type of 'self' (line 464)
        self_26523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 43), 'self', False)
        # Obtaining the member 'owner' of a type (line 464)
        owner_26524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 43), self_26523, 'owner')
        keyword_26525 = owner_26524
        # Getting the type of 'self' (line 464)
        self_26526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 61), 'self', False)
        # Obtaining the member 'group' of a type (line 464)
        group_26527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 61), self_26526, 'group')
        keyword_26528 = group_26527
        kwargs_26529 = {'owner': keyword_26525, 'group': keyword_26528, 'base_dir': keyword_26522}
        # Getting the type of 'self' (line 463)
        self_26517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 19), 'self', False)
        # Obtaining the member 'make_archive' of a type (line 463)
        make_archive_26518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 19), self_26517, 'make_archive')
        # Calling make_archive(args, kwargs) (line 463)
        make_archive_call_result_26530 = invoke(stypy.reporting.localization.Localization(__file__, 463, 19), make_archive_26518, *[base_name_26519, fmt_26520], **kwargs_26529)
        
        # Assigning a type to the variable 'file' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'file', make_archive_call_result_26530)
        
        # Call to append(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'file' (line 465)
        file_26533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 33), 'file', False)
        # Processing the call keyword arguments (line 465)
        kwargs_26534 = {}
        # Getting the type of 'archive_files' (line 465)
        archive_files_26531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'archive_files', False)
        # Obtaining the member 'append' of a type (line 465)
        append_26532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 12), archive_files_26531, 'append')
        # Calling append(args, kwargs) (line 465)
        append_call_result_26535 = invoke(stypy.reporting.localization.Localization(__file__, 465, 12), append_26532, *[file_26533], **kwargs_26534)
        
        
        # Call to append(...): (line 466)
        # Processing the call arguments (line 466)
        
        # Obtaining an instance of the builtin type 'tuple' (line 466)
        tuple_26540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 466)
        # Adding element type (line 466)
        str_26541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 49), 'str', 'sdist')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 49), tuple_26540, str_26541)
        # Adding element type (line 466)
        str_26542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 58), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 49), tuple_26540, str_26542)
        # Adding element type (line 466)
        # Getting the type of 'file' (line 466)
        file_26543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 62), 'file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 49), tuple_26540, file_26543)
        
        # Processing the call keyword arguments (line 466)
        kwargs_26544 = {}
        # Getting the type of 'self' (line 466)
        self_26536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'self', False)
        # Obtaining the member 'distribution' of a type (line 466)
        distribution_26537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), self_26536, 'distribution')
        # Obtaining the member 'dist_files' of a type (line 466)
        dist_files_26538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), distribution_26537, 'dist_files')
        # Obtaining the member 'append' of a type (line 466)
        append_26539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), dist_files_26538, 'append')
        # Calling append(args, kwargs) (line 466)
        append_call_result_26545 = invoke(stypy.reporting.localization.Localization(__file__, 466, 12), append_26539, *[tuple_26540], **kwargs_26544)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 468):
        
        # Assigning a Name to a Attribute (line 468):
        # Getting the type of 'archive_files' (line 468)
        archive_files_26546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 29), 'archive_files')
        # Getting the type of 'self' (line 468)
        self_26547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'self')
        # Setting the type of the member 'archive_files' of a type (line 468)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), self_26547, 'archive_files', archive_files_26546)
        
        
        # Getting the type of 'self' (line 470)
        self_26548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'self')
        # Obtaining the member 'keep_temp' of a type (line 470)
        keep_temp_26549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 15), self_26548, 'keep_temp')
        # Applying the 'not' unary operator (line 470)
        result_not__26550 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 11), 'not', keep_temp_26549)
        
        # Testing the type of an if condition (line 470)
        if_condition_26551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 8), result_not__26550)
        # Assigning a type to the variable 'if_condition_26551' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'if_condition_26551', if_condition_26551)
        # SSA begins for if statement (line 470)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove_tree(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'base_dir' (line 471)
        base_dir_26554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 33), 'base_dir', False)
        # Processing the call keyword arguments (line 471)
        # Getting the type of 'self' (line 471)
        self_26555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 51), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 471)
        dry_run_26556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 51), self_26555, 'dry_run')
        keyword_26557 = dry_run_26556
        kwargs_26558 = {'dry_run': keyword_26557}
        # Getting the type of 'dir_util' (line 471)
        dir_util_26552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'dir_util', False)
        # Obtaining the member 'remove_tree' of a type (line 471)
        remove_tree_26553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 12), dir_util_26552, 'remove_tree')
        # Calling remove_tree(args, kwargs) (line 471)
        remove_tree_call_result_26559 = invoke(stypy.reporting.localization.Localization(__file__, 471, 12), remove_tree_26553, *[base_dir_26554], **kwargs_26558)
        
        # SSA join for if statement (line 470)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'make_distribution(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_distribution' in the type store
        # Getting the type of 'stypy_return_type' (line 443)
        stypy_return_type_26560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26560)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_distribution'
        return stypy_return_type_26560


    @norecursion
    def get_archive_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_archive_files'
        module_type_store = module_type_store.open_function_context('get_archive_files', 473, 4, False)
        # Assigning a type to the variable 'self' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sdist.get_archive_files.__dict__.__setitem__('stypy_localization', localization)
        sdist.get_archive_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sdist.get_archive_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        sdist.get_archive_files.__dict__.__setitem__('stypy_function_name', 'sdist.get_archive_files')
        sdist.get_archive_files.__dict__.__setitem__('stypy_param_names_list', [])
        sdist.get_archive_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        sdist.get_archive_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sdist.get_archive_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        sdist.get_archive_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        sdist.get_archive_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sdist.get_archive_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.get_archive_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_archive_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_archive_files(...)' code ##################

        str_26561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, (-1)), 'str', "Return the list of archive files created when the command\n        was run, or None if the command hasn't run yet.\n        ")
        # Getting the type of 'self' (line 477)
        self_26562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'self')
        # Obtaining the member 'archive_files' of a type (line 477)
        archive_files_26563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 15), self_26562, 'archive_files')
        # Assigning a type to the variable 'stypy_return_type' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'stypy_return_type', archive_files_26563)
        
        # ################# End of 'get_archive_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_archive_files' in the type store
        # Getting the type of 'stypy_return_type' (line 473)
        stypy_return_type_26564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26564)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_archive_files'
        return stypy_return_type_26564


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 36, 0, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sdist.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'sdist' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'sdist', sdist)

# Assigning a Str to a Name (line 38):
str_26565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 18), 'str', 'create a source distribution (tarball, zip file, etc.)')
# Getting the type of 'sdist'
sdist_26566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sdist')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sdist_26566, 'description', str_26565)

# Assigning a List to a Name (line 46):

# Obtaining an instance of the builtin type 'list' (line 46)
list_26567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 46)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 47)
tuple_26568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 47)
# Adding element type (line 47)
str_26569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 9), 'str', 'template=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 9), tuple_26568, str_26569)
# Adding element type (line 47)
str_26570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'str', 't')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 9), tuple_26568, str_26570)
# Adding element type (line 47)
str_26571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 9), 'str', 'name of manifest template file [default: MANIFEST.in]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 9), tuple_26568, str_26571)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26568)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 49)
tuple_26572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 49)
# Adding element type (line 49)
str_26573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 9), 'str', 'manifest=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 9), tuple_26572, str_26573)
# Adding element type (line 49)
str_26574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 22), 'str', 'm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 9), tuple_26572, str_26574)
# Adding element type (line 49)
str_26575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'str', 'name of manifest file [default: MANIFEST]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 9), tuple_26572, str_26575)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26572)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 51)
tuple_26576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 51)
# Adding element type (line 51)
str_26577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'str', 'use-defaults')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_26576, str_26577)
# Adding element type (line 51)
# Getting the type of 'None' (line 51)
None_26578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_26576, None_26578)
# Adding element type (line 51)
str_26579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 9), 'str', 'include the default file set in the manifest [default; disable with --no-defaults]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_26576, str_26579)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26576)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 54)
tuple_26580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 54)
# Adding element type (line 54)
str_26581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'str', 'no-defaults')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_26580, str_26581)
# Adding element type (line 54)
# Getting the type of 'None' (line 54)
None_26582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_26580, None_26582)
# Adding element type (line 54)
str_26583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'str', "don't include the default file set")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_26580, str_26583)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26580)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 56)
tuple_26584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 56)
# Adding element type (line 56)
str_26585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'str', 'prune')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_26584, str_26585)
# Adding element type (line 56)
# Getting the type of 'None' (line 56)
None_26586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_26584, None_26586)
# Adding element type (line 56)
str_26587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'str', 'specifically exclude files/directories that should not be distributed (build tree, RCS/CVS dirs, etc.) [default; disable with --no-prune]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 9), tuple_26584, str_26587)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26584)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 60)
tuple_26588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 60)
# Adding element type (line 60)
str_26589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 9), 'str', 'no-prune')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 9), tuple_26588, str_26589)
# Adding element type (line 60)
# Getting the type of 'None' (line 60)
None_26590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 9), tuple_26588, None_26590)
# Adding element type (line 60)
str_26591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 9), 'str', "don't automatically exclude anything")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 9), tuple_26588, str_26591)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26588)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 62)
tuple_26592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 62)
# Adding element type (line 62)
str_26593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 9), 'str', 'manifest-only')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 9), tuple_26592, str_26593)
# Adding element type (line 62)
str_26594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 26), 'str', 'o')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 9), tuple_26592, str_26594)
# Adding element type (line 62)
str_26595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 9), 'str', 'just regenerate the manifest and then stop (implies --force-manifest)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 9), tuple_26592, str_26595)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26592)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 65)
tuple_26596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 65)
# Adding element type (line 65)
str_26597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'str', 'force-manifest')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_26596, str_26597)
# Adding element type (line 65)
str_26598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 27), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_26596, str_26598)
# Adding element type (line 65)
str_26599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 9), 'str', 'forcibly regenerate the manifest and carry on as usual. Deprecated: now the manifest is always regenerated.')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_26596, str_26599)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26596)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_26600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)
# Adding element type (line 68)
str_26601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 9), 'str', 'formats=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_26600, str_26601)
# Adding element type (line 68)
# Getting the type of 'None' (line 68)
None_26602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_26600, None_26602)
# Adding element type (line 68)
str_26603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 9), 'str', 'formats for source distribution (comma-separated list)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_26600, str_26603)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26600)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 70)
tuple_26604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 70)
# Adding element type (line 70)
str_26605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 9), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), tuple_26604, str_26605)
# Adding element type (line 70)
str_26606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'str', 'k')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), tuple_26604, str_26606)
# Adding element type (line 70)
str_26607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 9), 'str', 'keep the distribution tree around after creating ')
str_26608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 9), 'str', 'archive file(s)')
# Applying the binary operator '+' (line 71)
result_add_26609 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 9), '+', str_26607, str_26608)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 9), tuple_26604, result_add_26609)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26604)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_26610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
str_26611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 9), 'str', 'dist-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 9), tuple_26610, str_26611)
# Adding element type (line 73)
str_26612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 22), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 9), tuple_26610, str_26612)
# Adding element type (line 73)
str_26613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 9), 'str', 'directory to put the source distribution archive(s) in [default: dist]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 9), tuple_26610, str_26613)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26610)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_26614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
str_26615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'str', 'metadata-check')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_26614, str_26615)
# Adding element type (line 76)
# Getting the type of 'None' (line 76)
None_26616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_26614, None_26616)
# Adding element type (line 76)
str_26617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 9), 'str', 'Ensure that all required elements of meta-data are supplied. Warn if any missing. [default]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 9), tuple_26614, str_26617)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26614)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 79)
tuple_26618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 79)
# Adding element type (line 79)
str_26619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 9), 'str', 'owner=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 9), tuple_26618, str_26619)
# Adding element type (line 79)
str_26620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'str', 'u')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 9), tuple_26618, str_26620)
# Adding element type (line 79)
str_26621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 9), 'str', 'Owner name used when creating a tar file [default: current user]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 9), tuple_26618, str_26621)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26618)
# Adding element type (line 46)

# Obtaining an instance of the builtin type 'tuple' (line 81)
tuple_26622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 81)
# Adding element type (line 81)
str_26623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 9), 'str', 'group=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 9), tuple_26622, str_26623)
# Adding element type (line 81)
str_26624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'str', 'g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 9), tuple_26622, str_26624)
# Adding element type (line 81)
str_26625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 9), 'str', 'Group name used when creating a tar file [default: current group]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 9), tuple_26622, str_26625)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 19), list_26567, tuple_26622)

# Getting the type of 'sdist'
sdist_26626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sdist')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sdist_26626, 'user_options', list_26567)

# Assigning a List to a Name (line 85):

# Obtaining an instance of the builtin type 'list' (line 85)
list_26627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 85)
# Adding element type (line 85)
str_26628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'str', 'use-defaults')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_26627, str_26628)
# Adding element type (line 85)
str_26629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 39), 'str', 'prune')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_26627, str_26629)
# Adding element type (line 85)
str_26630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'str', 'manifest-only')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_26627, str_26630)
# Adding element type (line 85)
str_26631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 40), 'str', 'force-manifest')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_26627, str_26631)
# Adding element type (line 85)
str_26632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_26627, str_26632)
# Adding element type (line 85)
str_26633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 36), 'str', 'metadata-check')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_26627, str_26633)

# Getting the type of 'sdist'
sdist_26634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sdist')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sdist_26634, 'boolean_options', list_26627)

# Assigning a List to a Name (line 89):

# Obtaining an instance of the builtin type 'list' (line 89)
list_26635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 89)
# Adding element type (line 89)

# Obtaining an instance of the builtin type 'tuple' (line 90)
tuple_26636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 90)
# Adding element type (line 90)
str_26637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 9), 'str', 'help-formats')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_26636, str_26637)
# Adding element type (line 90)
# Getting the type of 'None' (line 90)
None_26638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_26636, None_26638)
# Adding element type (line 90)
str_26639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 9), 'str', 'list available distribution formats')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_26636, str_26639)
# Adding element type (line 90)
# Getting the type of 'show_formats' (line 91)
show_formats_26640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'show_formats')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 9), tuple_26636, show_formats_26640)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 19), list_26635, tuple_26636)

# Getting the type of 'sdist'
sdist_26641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sdist')
# Setting the type of the member 'help_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sdist_26641, 'help_options', list_26635)

# Assigning a Dict to a Name (line 94):

# Obtaining an instance of the builtin type 'dict' (line 94)
dict_26642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 94)
# Adding element type (key, value) (line 94)
str_26643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'str', 'no-defaults')
str_26644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 35), 'str', 'use-defaults')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 19), dict_26642, (str_26643, str_26644))
# Adding element type (key, value) (line 94)
str_26645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 20), 'str', 'no-prune')
str_26646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 32), 'str', 'prune')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 19), dict_26642, (str_26645, str_26646))

# Getting the type of 'sdist'
sdist_26647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sdist')
# Setting the type of the member 'negative_opt' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sdist_26647, 'negative_opt', dict_26642)

# Assigning a Dict to a Name (line 97):

# Obtaining an instance of the builtin type 'dict' (line 97)
dict_26648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 97)
# Adding element type (key, value) (line 97)
str_26649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 22), 'str', 'posix')
str_26650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 31), 'str', 'gztar')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 21), dict_26648, (str_26649, str_26650))
# Adding element type (key, value) (line 97)
str_26651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 22), 'str', 'nt')
str_26652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'str', 'zip')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 21), dict_26648, (str_26651, str_26652))

# Getting the type of 'sdist'
sdist_26653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sdist')
# Setting the type of the member 'default_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sdist_26653, 'default_format', dict_26648)

# Assigning a List to a Name (line 100):

# Obtaining an instance of the builtin type 'list' (line 100)
list_26654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 100)
# Adding element type (line 100)

# Obtaining an instance of the builtin type 'tuple' (line 100)
tuple_26655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 100)
# Adding element type (line 100)
str_26656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 21), 'str', 'check')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 21), tuple_26655, str_26656)
# Adding element type (line 100)
# Getting the type of 'sdist'
sdist_26657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sdist')
# Obtaining the member 'checking_metadata' of a type
checking_metadata_26658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sdist_26657, 'checking_metadata')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 21), tuple_26655, checking_metadata_26658)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 19), list_26654, tuple_26655)

# Getting the type of 'sdist'
sdist_26659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'sdist')
# Setting the type of the member 'sub_commands' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sdist_26659, 'sub_commands', list_26654)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
