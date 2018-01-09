
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.build_ext
2: 
3: Implements the Distutils 'build_ext' command, for building extension
4: modules (currently limited to C extensions, should accommodate C++
5: extensions ASAP).'''
6: 
7: # This module should be kept compatible with Python 2.1.
8: 
9: __revision__ = "$Id$"
10: 
11: import sys, os, string, re
12: from types import *
13: from site import USER_BASE, USER_SITE
14: from distutils.core import Command
15: from distutils.errors import *
16: from distutils.sysconfig import customize_compiler, get_python_version
17: from distutils.dep_util import newer_group
18: from distutils.extension import Extension
19: from distutils.util import get_platform
20: from distutils import log
21: 
22: if os.name == 'nt':
23:     from distutils.msvccompiler import get_build_version
24:     MSVC_VERSION = int(get_build_version())
25: 
26: # An extension name is just a dot-separated list of Python NAMEs (ie.
27: # the same as a fully-qualified module name).
28: extension_name_re = re.compile \
29:     (r'^[a-zA-Z_][a-zA-Z_0-9]*(\.[a-zA-Z_][a-zA-Z_0-9]*)*$')
30: 
31: 
32: def show_compilers ():
33:     from distutils.ccompiler import show_compilers
34:     show_compilers()
35: 
36: 
37: class build_ext (Command):
38: 
39:     description = "build C/C++ extensions (compile/link to build directory)"
40: 
41:     # XXX thoughts on how to deal with complex command-line options like
42:     # these, i.e. how to make it so fancy_getopt can suck them off the
43:     # command line and make it look like setup.py defined the appropriate
44:     # lists of tuples of what-have-you.
45:     #   - each command needs a callback to process its command-line options
46:     #   - Command.__init__() needs access to its share of the whole
47:     #     command line (must ultimately come from
48:     #     Distribution.parse_command_line())
49:     #   - it then calls the current command class' option-parsing
50:     #     callback to deal with weird options like -D, which have to
51:     #     parse the option text and churn out some custom data
52:     #     structure
53:     #   - that data structure (in this case, a list of 2-tuples)
54:     #     will then be present in the command object by the time
55:     #     we get to finalize_options() (i.e. the constructor
56:     #     takes care of both command-line and client options
57:     #     in between initialize_options() and finalize_options())
58: 
59:     sep_by = " (separated by '%s')" % os.pathsep
60:     user_options = [
61:         ('build-lib=', 'b',
62:          "directory for compiled extension modules"),
63:         ('build-temp=', 't',
64:          "directory for temporary files (build by-products)"),
65:         ('plat-name=', 'p',
66:          "platform name to cross-compile for, if supported "
67:          "(default: %s)" % get_platform()),
68:         ('inplace', 'i',
69:          "ignore build-lib and put compiled extensions into the source " +
70:          "directory alongside your pure Python modules"),
71:         ('include-dirs=', 'I',
72:          "list of directories to search for header files" + sep_by),
73:         ('define=', 'D',
74:          "C preprocessor macros to define"),
75:         ('undef=', 'U',
76:          "C preprocessor macros to undefine"),
77:         ('libraries=', 'l',
78:          "external C libraries to link with"),
79:         ('library-dirs=', 'L',
80:          "directories to search for external C libraries" + sep_by),
81:         ('rpath=', 'R',
82:          "directories to search for shared C libraries at runtime"),
83:         ('link-objects=', 'O',
84:          "extra explicit link objects to include in the link"),
85:         ('debug', 'g',
86:          "compile/link with debugging information"),
87:         ('force', 'f',
88:          "forcibly build everything (ignore file timestamps)"),
89:         ('compiler=', 'c',
90:          "specify the compiler type"),
91:         ('swig-cpp', None,
92:          "make SWIG create C++ files (default is C)"),
93:         ('swig-opts=', None,
94:          "list of SWIG command line options"),
95:         ('swig=', None,
96:          "path to the SWIG executable"),
97:         ('user', None,
98:          "add user include, library and rpath"),
99:         ]
100: 
101:     boolean_options = ['inplace', 'debug', 'force', 'swig-cpp', 'user']
102: 
103:     help_options = [
104:         ('help-compiler', None,
105:          "list available compilers", show_compilers),
106:         ]
107: 
108:     def initialize_options (self):
109:         self.extensions = None
110:         self.build_lib = None
111:         self.plat_name = None
112:         self.build_temp = None
113:         self.inplace = 0
114:         self.package = None
115: 
116:         self.include_dirs = None
117:         self.define = None
118:         self.undef = None
119:         self.libraries = None
120:         self.library_dirs = None
121:         self.rpath = None
122:         self.link_objects = None
123:         self.debug = None
124:         self.force = None
125:         self.compiler = None
126:         self.swig = None
127:         self.swig_cpp = None
128:         self.swig_opts = None
129:         self.user = None
130: 
131:     def finalize_options(self):
132:         from distutils import sysconfig
133: 
134:         self.set_undefined_options('build',
135:                                    ('build_lib', 'build_lib'),
136:                                    ('build_temp', 'build_temp'),
137:                                    ('compiler', 'compiler'),
138:                                    ('debug', 'debug'),
139:                                    ('force', 'force'),
140:                                    ('plat_name', 'plat_name'),
141:                                    )
142: 
143:         if self.package is None:
144:             self.package = self.distribution.ext_package
145: 
146:         self.extensions = self.distribution.ext_modules
147: 
148:         # Make sure Python's include directories (for Python.h, pyconfig.h,
149:         # etc.) are in the include search path.
150:         py_include = sysconfig.get_python_inc()
151:         plat_py_include = sysconfig.get_python_inc(plat_specific=1)
152:         if self.include_dirs is None:
153:             self.include_dirs = self.distribution.include_dirs or []
154:         if isinstance(self.include_dirs, str):
155:             self.include_dirs = self.include_dirs.split(os.pathsep)
156: 
157:         # Put the Python "system" include dir at the end, so that
158:         # any local include dirs take precedence.
159:         self.include_dirs.append(py_include)
160:         if plat_py_include != py_include:
161:             self.include_dirs.append(plat_py_include)
162: 
163:         self.ensure_string_list('libraries')
164:         self.ensure_string_list('link_objects')
165: 
166:         # Life is easier if we're not forever checking for None, so
167:         # simplify these options to empty lists if unset
168:         if self.libraries is None:
169:             self.libraries = []
170:         if self.library_dirs is None:
171:             self.library_dirs = []
172:         elif type(self.library_dirs) is StringType:
173:             self.library_dirs = string.split(self.library_dirs, os.pathsep)
174: 
175:         if self.rpath is None:
176:             self.rpath = []
177:         elif type(self.rpath) is StringType:
178:             self.rpath = string.split(self.rpath, os.pathsep)
179: 
180:         # for extensions under windows use different directories
181:         # for Release and Debug builds.
182:         # also Python's library directory must be appended to library_dirs
183:         if os.name == 'nt':
184:             # the 'libs' directory is for binary installs - we assume that
185:             # must be the *native* platform.  But we don't really support
186:             # cross-compiling via a binary install anyway, so we let it go.
187:             self.library_dirs.append(os.path.join(sys.exec_prefix, 'libs'))
188:             if self.debug:
189:                 self.build_temp = os.path.join(self.build_temp, "Debug")
190:             else:
191:                 self.build_temp = os.path.join(self.build_temp, "Release")
192: 
193:             # Append the source distribution include and library directories,
194:             # this allows distutils on windows to work in the source tree
195:             self.include_dirs.append(os.path.join(sys.exec_prefix, 'PC'))
196:             if MSVC_VERSION == 9:
197:                 # Use the .lib files for the correct architecture
198:                 if self.plat_name == 'win32':
199:                     suffix = ''
200:                 else:
201:                     # win-amd64 or win-ia64
202:                     suffix = self.plat_name[4:]
203:                 # We could have been built in one of two places; add both
204:                 for d in ('PCbuild',), ('PC', 'VS9.0'):
205:                     new_lib = os.path.join(sys.exec_prefix, *d)
206:                     if suffix:
207:                         new_lib = os.path.join(new_lib, suffix)
208:                     self.library_dirs.append(new_lib)
209: 
210:             elif MSVC_VERSION == 8:
211:                 self.library_dirs.append(os.path.join(sys.exec_prefix,
212:                                          'PC', 'VS8.0'))
213:             elif MSVC_VERSION == 7:
214:                 self.library_dirs.append(os.path.join(sys.exec_prefix,
215:                                          'PC', 'VS7.1'))
216:             else:
217:                 self.library_dirs.append(os.path.join(sys.exec_prefix,
218:                                          'PC', 'VC6'))
219: 
220:         # OS/2 (EMX) doesn't support Debug vs Release builds, but has the
221:         # import libraries in its "Config" subdirectory
222:         if os.name == 'os2':
223:             self.library_dirs.append(os.path.join(sys.exec_prefix, 'Config'))
224: 
225:         # for extensions under Cygwin and AtheOS Python's library directory must be
226:         # appended to library_dirs
227:         if sys.platform[:6] == 'cygwin' or sys.platform[:6] == 'atheos':
228:             if sys.executable.startswith(os.path.join(sys.exec_prefix, "bin")):
229:                 # building third party extensions
230:                 self.library_dirs.append(os.path.join(sys.prefix, "lib",
231:                                                       "python" + get_python_version(),
232:                                                       "config"))
233:             else:
234:                 # building python standard extensions
235:                 self.library_dirs.append('.')
236: 
237:         # For building extensions with a shared Python library,
238:         # Python's library directory must be appended to library_dirs
239:         # See Issues: #1600860, #4366
240:         if (sysconfig.get_config_var('Py_ENABLE_SHARED')):
241:             if not sysconfig.python_build:
242:                 # building third party extensions
243:                 self.library_dirs.append(sysconfig.get_config_var('LIBDIR'))
244:             else:
245:                 # building python standard extensions
246:                 self.library_dirs.append('.')
247: 
248:         # The argument parsing will result in self.define being a string, but
249:         # it has to be a list of 2-tuples.  All the preprocessor symbols
250:         # specified by the 'define' option will be set to '1'.  Multiple
251:         # symbols can be separated with commas.
252: 
253:         if self.define:
254:             defines = self.define.split(',')
255:             self.define = map(lambda symbol: (symbol, '1'), defines)
256: 
257:         # The option for macros to undefine is also a string from the
258:         # option parsing, but has to be a list.  Multiple symbols can also
259:         # be separated with commas here.
260:         if self.undef:
261:             self.undef = self.undef.split(',')
262: 
263:         if self.swig_opts is None:
264:             self.swig_opts = []
265:         else:
266:             self.swig_opts = self.swig_opts.split(' ')
267: 
268:         # Finally add the user include and library directories if requested
269:         if self.user:
270:             user_include = os.path.join(USER_BASE, "include")
271:             user_lib = os.path.join(USER_BASE, "lib")
272:             if os.path.isdir(user_include):
273:                 self.include_dirs.append(user_include)
274:             if os.path.isdir(user_lib):
275:                 self.library_dirs.append(user_lib)
276:                 self.rpath.append(user_lib)
277: 
278:     def run(self):
279:         from distutils.ccompiler import new_compiler
280: 
281:         # 'self.extensions', as supplied by setup.py, is a list of
282:         # Extension instances.  See the documentation for Extension (in
283:         # distutils.extension) for details.
284:         #
285:         # For backwards compatibility with Distutils 0.8.2 and earlier, we
286:         # also allow the 'extensions' list to be a list of tuples:
287:         #    (ext_name, build_info)
288:         # where build_info is a dictionary containing everything that
289:         # Extension instances do except the name, with a few things being
290:         # differently named.  We convert these 2-tuples to Extension
291:         # instances as needed.
292: 
293:         if not self.extensions:
294:             return
295: 
296:         # If we were asked to build any C/C++ libraries, make sure that the
297:         # directory where we put them is in the library search path for
298:         # linking extensions.
299:         if self.distribution.has_c_libraries():
300:             build_clib = self.get_finalized_command('build_clib')
301:             self.libraries.extend(build_clib.get_library_names() or [])
302:             self.library_dirs.append(build_clib.build_clib)
303: 
304:         # Setup the CCompiler object that we'll use to do all the
305:         # compiling and linking
306:         self.compiler = new_compiler(compiler=self.compiler,
307:                                      verbose=self.verbose,
308:                                      dry_run=self.dry_run,
309:                                      force=self.force)
310:         customize_compiler(self.compiler)
311:         # If we are cross-compiling, init the compiler now (if we are not
312:         # cross-compiling, init would not hurt, but people may rely on
313:         # late initialization of compiler even if they shouldn't...)
314:         if os.name == 'nt' and self.plat_name != get_platform():
315:             self.compiler.initialize(self.plat_name)
316: 
317:         # And make sure that any compile/link-related options (which might
318:         # come from the command-line or from the setup script) are set in
319:         # that CCompiler object -- that way, they automatically apply to
320:         # all compiling and linking done here.
321:         if self.include_dirs is not None:
322:             self.compiler.set_include_dirs(self.include_dirs)
323:         if self.define is not None:
324:             # 'define' option is a list of (name,value) tuples
325:             for (name, value) in self.define:
326:                 self.compiler.define_macro(name, value)
327:         if self.undef is not None:
328:             for macro in self.undef:
329:                 self.compiler.undefine_macro(macro)
330:         if self.libraries is not None:
331:             self.compiler.set_libraries(self.libraries)
332:         if self.library_dirs is not None:
333:             self.compiler.set_library_dirs(self.library_dirs)
334:         if self.rpath is not None:
335:             self.compiler.set_runtime_library_dirs(self.rpath)
336:         if self.link_objects is not None:
337:             self.compiler.set_link_objects(self.link_objects)
338: 
339:         # Now actually compile and link everything.
340:         self.build_extensions()
341: 
342:     def check_extensions_list(self, extensions):
343:         '''Ensure that the list of extensions (presumably provided as a
344:         command option 'extensions') is valid, i.e. it is a list of
345:         Extension objects.  We also support the old-style list of 2-tuples,
346:         where the tuples are (ext_name, build_info), which are converted to
347:         Extension instances here.
348: 
349:         Raise DistutilsSetupError if the structure is invalid anywhere;
350:         just returns otherwise.
351:         '''
352:         if not isinstance(extensions, list):
353:             raise DistutilsSetupError, \
354:                   "'ext_modules' option must be a list of Extension instances"
355: 
356:         for i, ext in enumerate(extensions):
357:             if isinstance(ext, Extension):
358:                 continue                # OK! (assume type-checking done
359:                                         # by Extension constructor)
360: 
361:             if not isinstance(ext, tuple) or len(ext) != 2:
362:                 raise DistutilsSetupError, \
363:                       ("each element of 'ext_modules' option must be an "
364:                        "Extension instance or 2-tuple")
365: 
366:             ext_name, build_info = ext
367: 
368:             log.warn(("old-style (ext_name, build_info) tuple found in "
369:                       "ext_modules for extension '%s'"
370:                       "-- please convert to Extension instance" % ext_name))
371: 
372:             if not (isinstance(ext_name, str) and
373:                     extension_name_re.match(ext_name)):
374:                 raise DistutilsSetupError, \
375:                       ("first element of each tuple in 'ext_modules' "
376:                        "must be the extension name (a string)")
377: 
378:             if not isinstance(build_info, dict):
379:                 raise DistutilsSetupError, \
380:                       ("second element of each tuple in 'ext_modules' "
381:                        "must be a dictionary (build info)")
382: 
383:             # OK, the (ext_name, build_info) dict is type-safe: convert it
384:             # to an Extension instance.
385:             ext = Extension(ext_name, build_info['sources'])
386: 
387:             # Easy stuff: one-to-one mapping from dict elements to
388:             # instance attributes.
389:             for key in ('include_dirs', 'library_dirs', 'libraries',
390:                         'extra_objects', 'extra_compile_args',
391:                         'extra_link_args'):
392:                 val = build_info.get(key)
393:                 if val is not None:
394:                     setattr(ext, key, val)
395: 
396:             # Medium-easy stuff: same syntax/semantics, different names.
397:             ext.runtime_library_dirs = build_info.get('rpath')
398:             if 'def_file' in build_info:
399:                 log.warn("'def_file' element of build info dict "
400:                          "no longer supported")
401: 
402:             # Non-trivial stuff: 'macros' split into 'define_macros'
403:             # and 'undef_macros'.
404:             macros = build_info.get('macros')
405:             if macros:
406:                 ext.define_macros = []
407:                 ext.undef_macros = []
408:                 for macro in macros:
409:                     if not (isinstance(macro, tuple) and len(macro) in (1, 2)):
410:                         raise DistutilsSetupError, \
411:                               ("'macros' element of build info dict "
412:                                "must be 1- or 2-tuple")
413:                     if len(macro) == 1:
414:                         ext.undef_macros.append(macro[0])
415:                     elif len(macro) == 2:
416:                         ext.define_macros.append(macro)
417: 
418:             extensions[i] = ext
419: 
420:     def get_source_files(self):
421:         self.check_extensions_list(self.extensions)
422:         filenames = []
423: 
424:         # Wouldn't it be neat if we knew the names of header files too...
425:         for ext in self.extensions:
426:             filenames.extend(ext.sources)
427: 
428:         return filenames
429: 
430:     def get_outputs(self):
431:         # Sanity check the 'extensions' list -- can't assume this is being
432:         # done in the same run as a 'build_extensions()' call (in fact, we
433:         # can probably assume that it *isn't*!).
434:         self.check_extensions_list(self.extensions)
435: 
436:         # And build the list of output (built) filenames.  Note that this
437:         # ignores the 'inplace' flag, and assumes everything goes in the
438:         # "build" tree.
439:         outputs = []
440:         for ext in self.extensions:
441:             outputs.append(self.get_ext_fullpath(ext.name))
442:         return outputs
443: 
444:     def build_extensions(self):
445:         # First, sanity-check the 'extensions' list
446:         self.check_extensions_list(self.extensions)
447: 
448:         for ext in self.extensions:
449:             self.build_extension(ext)
450: 
451:     def build_extension(self, ext):
452:         sources = ext.sources
453:         if sources is None or type(sources) not in (ListType, TupleType):
454:             raise DistutilsSetupError, \
455:                   ("in 'ext_modules' option (extension '%s'), " +
456:                    "'sources' must be present and must be " +
457:                    "a list of source filenames") % ext.name
458:         sources = list(sources)
459: 
460:         ext_path = self.get_ext_fullpath(ext.name)
461:         depends = sources + ext.depends
462:         if not (self.force or newer_group(depends, ext_path, 'newer')):
463:             log.debug("skipping '%s' extension (up-to-date)", ext.name)
464:             return
465:         else:
466:             log.info("building '%s' extension", ext.name)
467: 
468:         # First, scan the sources for SWIG definition files (.i), run
469:         # SWIG on 'em to create .c files, and modify the sources list
470:         # accordingly.
471:         sources = self.swig_sources(sources, ext)
472: 
473:         # Next, compile the source code to object files.
474: 
475:         # XXX not honouring 'define_macros' or 'undef_macros' -- the
476:         # CCompiler API needs to change to accommodate this, and I
477:         # want to do one thing at a time!
478: 
479:         # Two possible sources for extra compiler arguments:
480:         #   - 'extra_compile_args' in Extension object
481:         #   - CFLAGS environment variable (not particularly
482:         #     elegant, but people seem to expect it and I
483:         #     guess it's useful)
484:         # The environment variable should take precedence, and
485:         # any sensible compiler will give precedence to later
486:         # command line args.  Hence we combine them in order:
487:         extra_args = ext.extra_compile_args or []
488: 
489:         macros = ext.define_macros[:]
490:         for undef in ext.undef_macros:
491:             macros.append((undef,))
492: 
493:         objects = self.compiler.compile(sources,
494:                                          output_dir=self.build_temp,
495:                                          macros=macros,
496:                                          include_dirs=ext.include_dirs,
497:                                          debug=self.debug,
498:                                          extra_postargs=extra_args,
499:                                          depends=ext.depends)
500: 
501:         # XXX -- this is a Vile HACK!
502:         #
503:         # The setup.py script for Python on Unix needs to be able to
504:         # get this list so it can perform all the clean up needed to
505:         # avoid keeping object files around when cleaning out a failed
506:         # build of an extension module.  Since Distutils does not
507:         # track dependencies, we have to get rid of intermediates to
508:         # ensure all the intermediates will be properly re-built.
509:         #
510:         self._built_objects = objects[:]
511: 
512:         # Now link the object files together into a "shared object" --
513:         # of course, first we have to figure out all the other things
514:         # that go into the mix.
515:         if ext.extra_objects:
516:             objects.extend(ext.extra_objects)
517:         extra_args = ext.extra_link_args or []
518: 
519:         # Detect target language, if not provided
520:         language = ext.language or self.compiler.detect_language(sources)
521: 
522:         self.compiler.link_shared_object(
523:             objects, ext_path,
524:             libraries=self.get_libraries(ext),
525:             library_dirs=ext.library_dirs,
526:             runtime_library_dirs=ext.runtime_library_dirs,
527:             extra_postargs=extra_args,
528:             export_symbols=self.get_export_symbols(ext),
529:             debug=self.debug,
530:             build_temp=self.build_temp,
531:             target_lang=language)
532: 
533: 
534:     def swig_sources (self, sources, extension):
535: 
536:         '''Walk the list of source files in 'sources', looking for SWIG
537:         interface (.i) files.  Run SWIG on all that are found, and
538:         return a modified 'sources' list with SWIG source files replaced
539:         by the generated C (or C++) files.
540:         '''
541: 
542:         new_sources = []
543:         swig_sources = []
544:         swig_targets = {}
545: 
546:         # XXX this drops generated C/C++ files into the source tree, which
547:         # is fine for developers who want to distribute the generated
548:         # source -- but there should be an option to put SWIG output in
549:         # the temp dir.
550: 
551:         if self.swig_cpp:
552:             log.warn("--swig-cpp is deprecated - use --swig-opts=-c++")
553: 
554:         if self.swig_cpp or ('-c++' in self.swig_opts) or \
555:            ('-c++' in extension.swig_opts):
556:             target_ext = '.cpp'
557:         else:
558:             target_ext = '.c'
559: 
560:         for source in sources:
561:             (base, ext) = os.path.splitext(source)
562:             if ext == ".i":             # SWIG interface file
563:                 new_sources.append(base + '_wrap' + target_ext)
564:                 swig_sources.append(source)
565:                 swig_targets[source] = new_sources[-1]
566:             else:
567:                 new_sources.append(source)
568: 
569:         if not swig_sources:
570:             return new_sources
571: 
572:         swig = self.swig or self.find_swig()
573:         swig_cmd = [swig, "-python"]
574:         swig_cmd.extend(self.swig_opts)
575:         if self.swig_cpp:
576:             swig_cmd.append("-c++")
577: 
578:         # Do not override commandline arguments
579:         if not self.swig_opts:
580:             for o in extension.swig_opts:
581:                 swig_cmd.append(o)
582: 
583:         for source in swig_sources:
584:             target = swig_targets[source]
585:             log.info("swigging %s to %s", source, target)
586:             self.spawn(swig_cmd + ["-o", target, source])
587: 
588:         return new_sources
589: 
590:     # swig_sources ()
591: 
592:     def find_swig (self):
593:         '''Return the name of the SWIG executable.  On Unix, this is
594:         just "swig" -- it should be in the PATH.  Tries a bit harder on
595:         Windows.
596:         '''
597: 
598:         if os.name == "posix":
599:             return "swig"
600:         elif os.name == "nt":
601: 
602:             # Look for SWIG in its standard installation directory on
603:             # Windows (or so I presume!).  If we find it there, great;
604:             # if not, act like Unix and assume it's in the PATH.
605:             for vers in ("1.3", "1.2", "1.1"):
606:                 fn = os.path.join("c:\\swig%s" % vers, "swig.exe")
607:                 if os.path.isfile(fn):
608:                     return fn
609:             else:
610:                 return "swig.exe"
611: 
612:         elif os.name == "os2":
613:             # assume swig available in the PATH.
614:             return "swig.exe"
615: 
616:         else:
617:             raise DistutilsPlatformError, \
618:                   ("I don't know how to find (much less run) SWIG "
619:                    "on platform '%s'") % os.name
620: 
621:     # find_swig ()
622: 
623:     # -- Name generators -----------------------------------------------
624:     # (extension names, filenames, whatever)
625:     def get_ext_fullpath(self, ext_name):
626:         '''Returns the path of the filename for a given extension.
627: 
628:         The file is located in `build_lib` or directly in the package
629:         (inplace option).
630:         '''
631:         # makes sure the extension name is only using dots
632:         all_dots = string.maketrans('/'+os.sep, '..')
633:         ext_name = ext_name.translate(all_dots)
634: 
635:         fullname = self.get_ext_fullname(ext_name)
636:         modpath = fullname.split('.')
637:         filename = self.get_ext_filename(ext_name)
638:         filename = os.path.split(filename)[-1]
639: 
640:         if not self.inplace:
641:             # no further work needed
642:             # returning :
643:             #   build_dir/package/path/filename
644:             filename = os.path.join(*modpath[:-1]+[filename])
645:             return os.path.join(self.build_lib, filename)
646: 
647:         # the inplace option requires to find the package directory
648:         # using the build_py command for that
649:         package = '.'.join(modpath[0:-1])
650:         build_py = self.get_finalized_command('build_py')
651:         package_dir = os.path.abspath(build_py.get_package_dir(package))
652: 
653:         # returning
654:         #   package_dir/filename
655:         return os.path.join(package_dir, filename)
656: 
657:     def get_ext_fullname(self, ext_name):
658:         '''Returns the fullname of a given extension name.
659: 
660:         Adds the `package.` prefix'''
661:         if self.package is None:
662:             return ext_name
663:         else:
664:             return self.package + '.' + ext_name
665: 
666:     def get_ext_filename(self, ext_name):
667:         r'''Convert the name of an extension (eg. "foo.bar") into the name
668:         of the file from which it will be loaded (eg. "foo/bar.so", or
669:         "foo\bar.pyd").
670:         '''
671:         from distutils.sysconfig import get_config_var
672:         ext_path = string.split(ext_name, '.')
673:         # OS/2 has an 8 character module (extension) limit :-(
674:         if os.name == "os2":
675:             ext_path[len(ext_path) - 1] = ext_path[len(ext_path) - 1][:8]
676:         # extensions in debug_mode are named 'module_d.pyd' under windows
677:         so_ext = get_config_var('SO')
678:         if os.name == 'nt' and self.debug:
679:             return os.path.join(*ext_path) + '_d' + so_ext
680:         return os.path.join(*ext_path) + so_ext
681: 
682:     def get_export_symbols (self, ext):
683:         '''Return the list of symbols that a shared extension has to
684:         export.  This either uses 'ext.export_symbols' or, if it's not
685:         provided, "init" + module_name.  Only relevant on Windows, where
686:         the .pyd file (DLL) must export the module "init" function.
687:         '''
688:         initfunc_name = "init" + ext.name.split('.')[-1]
689:         if initfunc_name not in ext.export_symbols:
690:             ext.export_symbols.append(initfunc_name)
691:         return ext.export_symbols
692: 
693:     def get_libraries (self, ext):
694:         '''Return the list of libraries to link against when building a
695:         shared extension.  On most platforms, this is just 'ext.libraries';
696:         on Windows and OS/2, we add the Python library (eg. python20.dll).
697:         '''
698:         # The python library is always needed on Windows.  For MSVC, this
699:         # is redundant, since the library is mentioned in a pragma in
700:         # pyconfig.h that MSVC groks.  The other Windows compilers all seem
701:         # to need it mentioned explicitly, though, so that's what we do.
702:         # Append '_d' to the python import library on debug builds.
703:         if sys.platform == "win32":
704:             from distutils.msvccompiler import MSVCCompiler
705:             if not isinstance(self.compiler, MSVCCompiler):
706:                 template = "python%d%d"
707:                 if self.debug:
708:                     template = template + '_d'
709:                 pythonlib = (template %
710:                        (sys.hexversion >> 24, (sys.hexversion >> 16) & 0xff))
711:                 # don't extend ext.libraries, it may be shared with other
712:                 # extensions, it is a reference to the original list
713:                 return ext.libraries + [pythonlib]
714:             else:
715:                 return ext.libraries
716:         elif sys.platform == "os2emx":
717:             # EMX/GCC requires the python library explicitly, and I
718:             # believe VACPP does as well (though not confirmed) - AIM Apr01
719:             template = "python%d%d"
720:             # debug versions of the main DLL aren't supported, at least
721:             # not at this time - AIM Apr01
722:             #if self.debug:
723:             #    template = template + '_d'
724:             pythonlib = (template %
725:                    (sys.hexversion >> 24, (sys.hexversion >> 16) & 0xff))
726:             # don't extend ext.libraries, it may be shared with other
727:             # extensions, it is a reference to the original list
728:             return ext.libraries + [pythonlib]
729:         elif sys.platform[:6] == "cygwin":
730:             template = "python%d.%d"
731:             pythonlib = (template %
732:                    (sys.hexversion >> 24, (sys.hexversion >> 16) & 0xff))
733:             # don't extend ext.libraries, it may be shared with other
734:             # extensions, it is a reference to the original list
735:             return ext.libraries + [pythonlib]
736:         elif sys.platform[:6] == "atheos":
737:             from distutils import sysconfig
738: 
739:             template = "python%d.%d"
740:             pythonlib = (template %
741:                    (sys.hexversion >> 24, (sys.hexversion >> 16) & 0xff))
742:             # Get SHLIBS from Makefile
743:             extra = []
744:             for lib in sysconfig.get_config_var('SHLIBS').split():
745:                 if lib.startswith('-l'):
746:                     extra.append(lib[2:])
747:                 else:
748:                     extra.append(lib)
749:             # don't extend ext.libraries, it may be shared with other
750:             # extensions, it is a reference to the original list
751:             return ext.libraries + [pythonlib, "m"] + extra
752: 
753:         elif sys.platform == 'darwin':
754:             # Don't use the default code below
755:             return ext.libraries
756:         elif sys.platform[:3] == 'aix':
757:             # Don't use the default code below
758:             return ext.libraries
759:         else:
760:             from distutils import sysconfig
761:             if sysconfig.get_config_var('Py_ENABLE_SHARED'):
762:                 template = "python%d.%d"
763:                 pythonlib = (template %
764:                              (sys.hexversion >> 24, (sys.hexversion >> 16) & 0xff))
765:                 return ext.libraries + [pythonlib]
766:             else:
767:                 return ext.libraries
768: 
769: # class build_ext
770: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', "distutils.command.build_ext\n\nImplements the Distutils 'build_ext' command, for building extension\nmodules (currently limited to C extensions, should accommodate C++\nextensions ASAP).")

# Assigning a Str to a Name (line 9):

# Assigning a Str to a Name (line 9):
str_466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__revision__', str_466)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# Multiple import statement. import sys (1/4) (line 11)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/4) (line 11)
import os

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'os', os, module_type_store)
# Multiple import statement. import string (3/4) (line 11)
import string

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'string', string, module_type_store)
# Multiple import statement. import re (4/4) (line 11)
import re

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from types import ' statement (line 12)
try:
    from types import *

except:
    pass
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'types', None, module_type_store, ['*'], None)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from site import USER_BASE, USER_SITE' statement (line 13)
try:
    from site import USER_BASE, USER_SITE

except:
    USER_BASE = UndefinedType
    USER_SITE = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'site', None, module_type_store, ['USER_BASE', 'USER_SITE'], [USER_BASE, USER_SITE])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.core import Command' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_467 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core')

if (type(import_467) is not StypyTypeError):

    if (import_467 != 'pyd_module'):
        __import__(import_467)
        sys_modules_468 = sys.modules[import_467]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', sys_modules_468.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_468, sys_modules_468.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', import_467)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.errors import ' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_469 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors')

if (type(import_469) is not StypyTypeError):

    if (import_469 != 'pyd_module'):
        __import__(import_469)
        sys_modules_470 = sys.modules[import_469]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors', sys_modules_470.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_470, sys_modules_470.module_type_store, module_type_store)
    else:
        from distutils.errors import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'distutils.errors' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors', import_469)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.sysconfig import customize_compiler, get_python_version' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_471 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.sysconfig')

if (type(import_471) is not StypyTypeError):

    if (import_471 != 'pyd_module'):
        __import__(import_471)
        sys_modules_472 = sys.modules[import_471]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.sysconfig', sys_modules_472.module_type_store, module_type_store, ['customize_compiler', 'get_python_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_472, sys_modules_472.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import customize_compiler, get_python_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.sysconfig', None, module_type_store, ['customize_compiler', 'get_python_version'], [customize_compiler, get_python_version])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.sysconfig', import_471)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.dep_util import newer_group' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_473 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.dep_util')

if (type(import_473) is not StypyTypeError):

    if (import_473 != 'pyd_module'):
        __import__(import_473)
        sys_modules_474 = sys.modules[import_473]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.dep_util', sys_modules_474.module_type_store, module_type_store, ['newer_group'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_474, sys_modules_474.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer_group

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.dep_util', None, module_type_store, ['newer_group'], [newer_group])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.dep_util', import_473)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.extension import Extension' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_475 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.extension')

if (type(import_475) is not StypyTypeError):

    if (import_475 != 'pyd_module'):
        __import__(import_475)
        sys_modules_476 = sys.modules[import_475]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.extension', sys_modules_476.module_type_store, module_type_store, ['Extension'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_476, sys_modules_476.module_type_store, module_type_store)
    else:
        from distutils.extension import Extension

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.extension', None, module_type_store, ['Extension'], [Extension])

else:
    # Assigning a type to the variable 'distutils.extension' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.extension', import_475)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils.util import get_platform' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_477 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.util')

if (type(import_477) is not StypyTypeError):

    if (import_477 != 'pyd_module'):
        __import__(import_477)
        sys_modules_478 = sys.modules[import_477]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.util', sys_modules_478.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_478, sys_modules_478.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.util', import_477)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils import log' statement (line 20)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils', None, module_type_store, ['log'], [log])



# Getting the type of 'os' (line 22)
os_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 3), 'os')
# Obtaining the member 'name' of a type (line 22)
name_480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 3), os_479, 'name')
str_481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'str', 'nt')
# Applying the binary operator '==' (line 22)
result_eq_482 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 3), '==', name_480, str_481)

# Testing the type of an if condition (line 22)
if_condition_483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 0), result_eq_482)
# Assigning a type to the variable 'if_condition_483' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'if_condition_483', if_condition_483)
# SSA begins for if statement (line 22)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 4))

# 'from distutils.msvccompiler import get_build_version' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_484 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'distutils.msvccompiler')

if (type(import_484) is not StypyTypeError):

    if (import_484 != 'pyd_module'):
        __import__(import_484)
        sys_modules_485 = sys.modules[import_484]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'distutils.msvccompiler', sys_modules_485.module_type_store, module_type_store, ['get_build_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 4), __file__, sys_modules_485, sys_modules_485.module_type_store, module_type_store)
    else:
        from distutils.msvccompiler import get_build_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'distutils.msvccompiler', None, module_type_store, ['get_build_version'], [get_build_version])

else:
    # Assigning a type to the variable 'distutils.msvccompiler' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'distutils.msvccompiler', import_484)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')


# Assigning a Call to a Name (line 24):

# Assigning a Call to a Name (line 24):

# Call to int(...): (line 24)
# Processing the call arguments (line 24)

# Call to get_build_version(...): (line 24)
# Processing the call keyword arguments (line 24)
kwargs_488 = {}
# Getting the type of 'get_build_version' (line 24)
get_build_version_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'get_build_version', False)
# Calling get_build_version(args, kwargs) (line 24)
get_build_version_call_result_489 = invoke(stypy.reporting.localization.Localization(__file__, 24, 23), get_build_version_487, *[], **kwargs_488)

# Processing the call keyword arguments (line 24)
kwargs_490 = {}
# Getting the type of 'int' (line 24)
int_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'int', False)
# Calling int(args, kwargs) (line 24)
int_call_result_491 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), int_486, *[get_build_version_call_result_489], **kwargs_490)

# Assigning a type to the variable 'MSVC_VERSION' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'MSVC_VERSION', int_call_result_491)
# SSA join for if statement (line 22)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 28):

# Assigning a Call to a Name (line 28):

# Call to compile(...): (line 28)
# Processing the call arguments (line 28)
str_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 5), 'str', '^[a-zA-Z_][a-zA-Z_0-9]*(\\.[a-zA-Z_][a-zA-Z_0-9]*)*$')
# Processing the call keyword arguments (line 28)
kwargs_495 = {}
# Getting the type of 're' (line 28)
re_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 're', False)
# Obtaining the member 'compile' of a type (line 28)
compile_493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 20), re_492, 'compile')
# Calling compile(args, kwargs) (line 28)
compile_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 28, 20), compile_493, *[str_494], **kwargs_495)

# Assigning a type to the variable 'extension_name_re' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'extension_name_re', compile_call_result_496)

@norecursion
def show_compilers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'show_compilers'
    module_type_store = module_type_store.open_function_context('show_compilers', 32, 0, False)
    
    # Passed parameters checking function
    show_compilers.stypy_localization = localization
    show_compilers.stypy_type_of_self = None
    show_compilers.stypy_type_store = module_type_store
    show_compilers.stypy_function_name = 'show_compilers'
    show_compilers.stypy_param_names_list = []
    show_compilers.stypy_varargs_param_name = None
    show_compilers.stypy_kwargs_param_name = None
    show_compilers.stypy_call_defaults = defaults
    show_compilers.stypy_call_varargs = varargs
    show_compilers.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'show_compilers', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'show_compilers', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'show_compilers(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 4))
    
    # 'from distutils.ccompiler import show_compilers' statement (line 33)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
    import_497 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 4), 'distutils.ccompiler')

    if (type(import_497) is not StypyTypeError):

        if (import_497 != 'pyd_module'):
            __import__(import_497)
            sys_modules_498 = sys.modules[import_497]
            import_from_module(stypy.reporting.localization.Localization(__file__, 33, 4), 'distutils.ccompiler', sys_modules_498.module_type_store, module_type_store, ['show_compilers'])
            nest_module(stypy.reporting.localization.Localization(__file__, 33, 4), __file__, sys_modules_498, sys_modules_498.module_type_store, module_type_store)
        else:
            from distutils.ccompiler import show_compilers

            import_from_module(stypy.reporting.localization.Localization(__file__, 33, 4), 'distutils.ccompiler', None, module_type_store, ['show_compilers'], [show_compilers])

    else:
        # Assigning a type to the variable 'distutils.ccompiler' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'distutils.ccompiler', import_497)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
    
    
    # Call to show_compilers(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_500 = {}
    # Getting the type of 'show_compilers' (line 34)
    show_compilers_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'show_compilers', False)
    # Calling show_compilers(args, kwargs) (line 34)
    show_compilers_call_result_501 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), show_compilers_499, *[], **kwargs_500)
    
    
    # ################# End of 'show_compilers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'show_compilers' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_502)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'show_compilers'
    return stypy_return_type_502

# Assigning a type to the variable 'show_compilers' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'show_compilers', show_compilers)
# Declaration of the 'build_ext' class
# Getting the type of 'Command' (line 37)
Command_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'Command')

class build_ext(Command_503, ):
    
    # Assigning a Str to a Name (line 39):
    
    # Assigning a BinOp to a Name (line 59):
    
    # Assigning a List to a Name (line 60):
    
    # Assigning a List to a Name (line 101):
    
    # Assigning a List to a Name (line 103):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        build_ext.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.initialize_options.__dict__.__setitem__('stypy_function_name', 'build_ext.initialize_options')
        build_ext.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 109):
        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'None' (line 109)
        None_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'None')
        # Getting the type of 'self' (line 109)
        self_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'extensions' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_505, 'extensions', None_504)
        
        # Assigning a Name to a Attribute (line 110):
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'None' (line 110)
        None_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'None')
        # Getting the type of 'self' (line 110)
        self_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member 'build_lib' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_507, 'build_lib', None_506)
        
        # Assigning a Name to a Attribute (line 111):
        
        # Assigning a Name to a Attribute (line 111):
        # Getting the type of 'None' (line 111)
        None_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'None')
        # Getting the type of 'self' (line 111)
        self_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member 'plat_name' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_509, 'plat_name', None_508)
        
        # Assigning a Name to a Attribute (line 112):
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'None' (line 112)
        None_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'None')
        # Getting the type of 'self' (line 112)
        self_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self')
        # Setting the type of the member 'build_temp' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_511, 'build_temp', None_510)
        
        # Assigning a Num to a Attribute (line 113):
        
        # Assigning a Num to a Attribute (line 113):
        int_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 23), 'int')
        # Getting the type of 'self' (line 113)
        self_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'inplace' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_513, 'inplace', int_512)
        
        # Assigning a Name to a Attribute (line 114):
        
        # Assigning a Name to a Attribute (line 114):
        # Getting the type of 'None' (line 114)
        None_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'None')
        # Getting the type of 'self' (line 114)
        self_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member 'package' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_515, 'package', None_514)
        
        # Assigning a Name to a Attribute (line 116):
        
        # Assigning a Name to a Attribute (line 116):
        # Getting the type of 'None' (line 116)
        None_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'None')
        # Getting the type of 'self' (line 116)
        self_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self')
        # Setting the type of the member 'include_dirs' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_517, 'include_dirs', None_516)
        
        # Assigning a Name to a Attribute (line 117):
        
        # Assigning a Name to a Attribute (line 117):
        # Getting the type of 'None' (line 117)
        None_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'None')
        # Getting the type of 'self' (line 117)
        self_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self')
        # Setting the type of the member 'define' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_519, 'define', None_518)
        
        # Assigning a Name to a Attribute (line 118):
        
        # Assigning a Name to a Attribute (line 118):
        # Getting the type of 'None' (line 118)
        None_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'None')
        # Getting the type of 'self' (line 118)
        self_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self')
        # Setting the type of the member 'undef' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_521, 'undef', None_520)
        
        # Assigning a Name to a Attribute (line 119):
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'None' (line 119)
        None_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'None')
        # Getting the type of 'self' (line 119)
        self_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_523, 'libraries', None_522)
        
        # Assigning a Name to a Attribute (line 120):
        
        # Assigning a Name to a Attribute (line 120):
        # Getting the type of 'None' (line 120)
        None_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'None')
        # Getting the type of 'self' (line 120)
        self_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member 'library_dirs' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_525, 'library_dirs', None_524)
        
        # Assigning a Name to a Attribute (line 121):
        
        # Assigning a Name to a Attribute (line 121):
        # Getting the type of 'None' (line 121)
        None_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'None')
        # Getting the type of 'self' (line 121)
        self_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member 'rpath' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_527, 'rpath', None_526)
        
        # Assigning a Name to a Attribute (line 122):
        
        # Assigning a Name to a Attribute (line 122):
        # Getting the type of 'None' (line 122)
        None_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'None')
        # Getting the type of 'self' (line 122)
        self_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member 'link_objects' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_529, 'link_objects', None_528)
        
        # Assigning a Name to a Attribute (line 123):
        
        # Assigning a Name to a Attribute (line 123):
        # Getting the type of 'None' (line 123)
        None_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 21), 'None')
        # Getting the type of 'self' (line 123)
        self_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self')
        # Setting the type of the member 'debug' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_531, 'debug', None_530)
        
        # Assigning a Name to a Attribute (line 124):
        
        # Assigning a Name to a Attribute (line 124):
        # Getting the type of 'None' (line 124)
        None_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'None')
        # Getting the type of 'self' (line 124)
        self_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self')
        # Setting the type of the member 'force' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_533, 'force', None_532)
        
        # Assigning a Name to a Attribute (line 125):
        
        # Assigning a Name to a Attribute (line 125):
        # Getting the type of 'None' (line 125)
        None_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'None')
        # Getting the type of 'self' (line 125)
        self_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 125)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_535, 'compiler', None_534)
        
        # Assigning a Name to a Attribute (line 126):
        
        # Assigning a Name to a Attribute (line 126):
        # Getting the type of 'None' (line 126)
        None_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'None')
        # Getting the type of 'self' (line 126)
        self_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'self')
        # Setting the type of the member 'swig' of a type (line 126)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), self_537, 'swig', None_536)
        
        # Assigning a Name to a Attribute (line 127):
        
        # Assigning a Name to a Attribute (line 127):
        # Getting the type of 'None' (line 127)
        None_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'None')
        # Getting the type of 'self' (line 127)
        self_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self')
        # Setting the type of the member 'swig_cpp' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_539, 'swig_cpp', None_538)
        
        # Assigning a Name to a Attribute (line 128):
        
        # Assigning a Name to a Attribute (line 128):
        # Getting the type of 'None' (line 128)
        None_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'None')
        # Getting the type of 'self' (line 128)
        self_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member 'swig_opts' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_541, 'swig_opts', None_540)
        
        # Assigning a Name to a Attribute (line 129):
        
        # Assigning a Name to a Attribute (line 129):
        # Getting the type of 'None' (line 129)
        None_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'None')
        # Getting the type of 'self' (line 129)
        self_543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'user' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_543, 'user', None_542)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_544


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        build_ext.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.finalize_options.__dict__.__setitem__('stypy_function_name', 'build_ext.finalize_options')
        build_ext.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 132, 8))
        
        # 'from distutils import sysconfig' statement (line 132)
        try:
            from distutils import sysconfig

        except:
            sysconfig = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 132, 8), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])
        
        
        # Call to set_undefined_options(...): (line 134)
        # Processing the call arguments (line 134)
        str_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        str_549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 36), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 36), tuple_548, str_549)
        # Adding element type (line 135)
        str_550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 49), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 36), tuple_548, str_550)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 136)
        tuple_551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 136)
        # Adding element type (line 136)
        str_552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 36), 'str', 'build_temp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 36), tuple_551, str_552)
        # Adding element type (line 136)
        str_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 50), 'str', 'build_temp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 36), tuple_551, str_553)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 137)
        tuple_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 137)
        # Adding element type (line 137)
        str_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 36), 'str', 'compiler')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 36), tuple_554, str_555)
        # Adding element type (line 137)
        str_556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 48), 'str', 'compiler')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 36), tuple_554, str_556)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 138)
        tuple_557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 138)
        # Adding element type (line 138)
        str_558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 36), 'str', 'debug')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 36), tuple_557, str_558)
        # Adding element type (line 138)
        str_559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 45), 'str', 'debug')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 36), tuple_557, str_559)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 139)
        tuple_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 139)
        # Adding element type (line 139)
        str_561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 36), tuple_560, str_561)
        # Adding element type (line 139)
        str_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 36), tuple_560, str_562)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 140)
        tuple_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 140)
        # Adding element type (line 140)
        str_564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 36), 'str', 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 36), tuple_563, str_564)
        # Adding element type (line 140)
        str_565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 49), 'str', 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 36), tuple_563, str_565)
        
        # Processing the call keyword arguments (line 134)
        kwargs_566 = {}
        # Getting the type of 'self' (line 134)
        self_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 134)
        set_undefined_options_546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_545, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 134)
        set_undefined_options_call_result_567 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), set_undefined_options_546, *[str_547, tuple_548, tuple_551, tuple_554, tuple_557, tuple_560, tuple_563], **kwargs_566)
        
        
        # Type idiom detected: calculating its left and rigth part (line 143)
        # Getting the type of 'self' (line 143)
        self_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'self')
        # Obtaining the member 'package' of a type (line 143)
        package_569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 11), self_568, 'package')
        # Getting the type of 'None' (line 143)
        None_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'None')
        
        (may_be_571, more_types_in_union_572) = may_be_none(package_569, None_570)

        if may_be_571:

            if more_types_in_union_572:
                # Runtime conditional SSA (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 144):
            
            # Assigning a Attribute to a Attribute (line 144):
            # Getting the type of 'self' (line 144)
            self_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'self')
            # Obtaining the member 'distribution' of a type (line 144)
            distribution_574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 27), self_573, 'distribution')
            # Obtaining the member 'ext_package' of a type (line 144)
            ext_package_575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 27), distribution_574, 'ext_package')
            # Getting the type of 'self' (line 144)
            self_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'self')
            # Setting the type of the member 'package' of a type (line 144)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), self_576, 'package', ext_package_575)

            if more_types_in_union_572:
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Attribute (line 146):
        
        # Assigning a Attribute to a Attribute (line 146):
        # Getting the type of 'self' (line 146)
        self_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'self')
        # Obtaining the member 'distribution' of a type (line 146)
        distribution_578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 26), self_577, 'distribution')
        # Obtaining the member 'ext_modules' of a type (line 146)
        ext_modules_579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 26), distribution_578, 'ext_modules')
        # Getting the type of 'self' (line 146)
        self_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'extensions' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_580, 'extensions', ext_modules_579)
        
        # Assigning a Call to a Name (line 150):
        
        # Assigning a Call to a Name (line 150):
        
        # Call to get_python_inc(...): (line 150)
        # Processing the call keyword arguments (line 150)
        kwargs_583 = {}
        # Getting the type of 'sysconfig' (line 150)
        sysconfig_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'sysconfig', False)
        # Obtaining the member 'get_python_inc' of a type (line 150)
        get_python_inc_582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 21), sysconfig_581, 'get_python_inc')
        # Calling get_python_inc(args, kwargs) (line 150)
        get_python_inc_call_result_584 = invoke(stypy.reporting.localization.Localization(__file__, 150, 21), get_python_inc_582, *[], **kwargs_583)
        
        # Assigning a type to the variable 'py_include' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'py_include', get_python_inc_call_result_584)
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to get_python_inc(...): (line 151)
        # Processing the call keyword arguments (line 151)
        int_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 65), 'int')
        keyword_588 = int_587
        kwargs_589 = {'plat_specific': keyword_588}
        # Getting the type of 'sysconfig' (line 151)
        sysconfig_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'sysconfig', False)
        # Obtaining the member 'get_python_inc' of a type (line 151)
        get_python_inc_586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 26), sysconfig_585, 'get_python_inc')
        # Calling get_python_inc(args, kwargs) (line 151)
        get_python_inc_call_result_590 = invoke(stypy.reporting.localization.Localization(__file__, 151, 26), get_python_inc_586, *[], **kwargs_589)
        
        # Assigning a type to the variable 'plat_py_include' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'plat_py_include', get_python_inc_call_result_590)
        
        # Type idiom detected: calculating its left and rigth part (line 152)
        # Getting the type of 'self' (line 152)
        self_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'self')
        # Obtaining the member 'include_dirs' of a type (line 152)
        include_dirs_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), self_591, 'include_dirs')
        # Getting the type of 'None' (line 152)
        None_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'None')
        
        (may_be_594, more_types_in_union_595) = may_be_none(include_dirs_592, None_593)

        if may_be_594:

            if more_types_in_union_595:
                # Runtime conditional SSA (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BoolOp to a Attribute (line 153):
            
            # Assigning a BoolOp to a Attribute (line 153):
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 153)
            self_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 32), 'self')
            # Obtaining the member 'distribution' of a type (line 153)
            distribution_597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 32), self_596, 'distribution')
            # Obtaining the member 'include_dirs' of a type (line 153)
            include_dirs_598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 32), distribution_597, 'include_dirs')
            
            # Obtaining an instance of the builtin type 'list' (line 153)
            list_599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 66), 'list')
            # Adding type elements to the builtin type 'list' instance (line 153)
            
            # Applying the binary operator 'or' (line 153)
            result_or_keyword_600 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 32), 'or', include_dirs_598, list_599)
            
            # Getting the type of 'self' (line 153)
            self_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'self')
            # Setting the type of the member 'include_dirs' of a type (line 153)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), self_601, 'include_dirs', result_or_keyword_600)

            if more_types_in_union_595:
                # SSA join for if statement (line 152)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 154)
        # Getting the type of 'str' (line 154)
        str_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 41), 'str')
        # Getting the type of 'self' (line 154)
        self_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'self')
        # Obtaining the member 'include_dirs' of a type (line 154)
        include_dirs_604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 22), self_603, 'include_dirs')
        
        (may_be_605, more_types_in_union_606) = may_be_subtype(str_602, include_dirs_604)

        if may_be_605:

            if more_types_in_union_606:
                # Runtime conditional SSA (line 154)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 154)
            self_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
            # Obtaining the member 'include_dirs' of a type (line 154)
            include_dirs_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_607, 'include_dirs')
            # Setting the type of the member 'include_dirs' of a type (line 154)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_607, 'include_dirs', remove_not_subtype_from_union(include_dirs_604, str))
            
            # Assigning a Call to a Attribute (line 155):
            
            # Assigning a Call to a Attribute (line 155):
            
            # Call to split(...): (line 155)
            # Processing the call arguments (line 155)
            # Getting the type of 'os' (line 155)
            os_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 56), 'os', False)
            # Obtaining the member 'pathsep' of a type (line 155)
            pathsep_613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 56), os_612, 'pathsep')
            # Processing the call keyword arguments (line 155)
            kwargs_614 = {}
            # Getting the type of 'self' (line 155)
            self_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'self', False)
            # Obtaining the member 'include_dirs' of a type (line 155)
            include_dirs_610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 32), self_609, 'include_dirs')
            # Obtaining the member 'split' of a type (line 155)
            split_611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 32), include_dirs_610, 'split')
            # Calling split(args, kwargs) (line 155)
            split_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 155, 32), split_611, *[pathsep_613], **kwargs_614)
            
            # Getting the type of 'self' (line 155)
            self_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self')
            # Setting the type of the member 'include_dirs' of a type (line 155)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_616, 'include_dirs', split_call_result_615)

            if more_types_in_union_606:
                # SSA join for if statement (line 154)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to append(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'py_include' (line 159)
        py_include_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'py_include', False)
        # Processing the call keyword arguments (line 159)
        kwargs_621 = {}
        # Getting the type of 'self' (line 159)
        self_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 159)
        include_dirs_618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_617, 'include_dirs')
        # Obtaining the member 'append' of a type (line 159)
        append_619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), include_dirs_618, 'append')
        # Calling append(args, kwargs) (line 159)
        append_call_result_622 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), append_619, *[py_include_620], **kwargs_621)
        
        
        
        # Getting the type of 'plat_py_include' (line 160)
        plat_py_include_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'plat_py_include')
        # Getting the type of 'py_include' (line 160)
        py_include_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 30), 'py_include')
        # Applying the binary operator '!=' (line 160)
        result_ne_625 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), '!=', plat_py_include_623, py_include_624)
        
        # Testing the type of an if condition (line 160)
        if_condition_626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_ne_625)
        # Assigning a type to the variable 'if_condition_626' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_626', if_condition_626)
        # SSA begins for if statement (line 160)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'plat_py_include' (line 161)
        plat_py_include_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'plat_py_include', False)
        # Processing the call keyword arguments (line 161)
        kwargs_631 = {}
        # Getting the type of 'self' (line 161)
        self_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 161)
        include_dirs_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), self_627, 'include_dirs')
        # Obtaining the member 'append' of a type (line 161)
        append_629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), include_dirs_628, 'append')
        # Calling append(args, kwargs) (line 161)
        append_call_result_632 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), append_629, *[plat_py_include_630], **kwargs_631)
        
        # SSA join for if statement (line 160)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to ensure_string_list(...): (line 163)
        # Processing the call arguments (line 163)
        str_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 32), 'str', 'libraries')
        # Processing the call keyword arguments (line 163)
        kwargs_636 = {}
        # Getting the type of 'self' (line 163)
        self_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 163)
        ensure_string_list_634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_633, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 163)
        ensure_string_list_call_result_637 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), ensure_string_list_634, *[str_635], **kwargs_636)
        
        
        # Call to ensure_string_list(...): (line 164)
        # Processing the call arguments (line 164)
        str_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 32), 'str', 'link_objects')
        # Processing the call keyword arguments (line 164)
        kwargs_641 = {}
        # Getting the type of 'self' (line 164)
        self_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 164)
        ensure_string_list_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_638, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 164)
        ensure_string_list_call_result_642 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), ensure_string_list_639, *[str_640], **kwargs_641)
        
        
        # Type idiom detected: calculating its left and rigth part (line 168)
        # Getting the type of 'self' (line 168)
        self_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'self')
        # Obtaining the member 'libraries' of a type (line 168)
        libraries_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 11), self_643, 'libraries')
        # Getting the type of 'None' (line 168)
        None_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 29), 'None')
        
        (may_be_646, more_types_in_union_647) = may_be_none(libraries_644, None_645)

        if may_be_646:

            if more_types_in_union_647:
                # Runtime conditional SSA (line 168)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 169):
            
            # Assigning a List to a Attribute (line 169):
            
            # Obtaining an instance of the builtin type 'list' (line 169)
            list_648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 169)
            
            # Getting the type of 'self' (line 169)
            self_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'self')
            # Setting the type of the member 'libraries' of a type (line 169)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), self_649, 'libraries', list_648)

            if more_types_in_union_647:
                # SSA join for if statement (line 168)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 170)
        # Getting the type of 'self' (line 170)
        self_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'self')
        # Obtaining the member 'library_dirs' of a type (line 170)
        library_dirs_651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), self_650, 'library_dirs')
        # Getting the type of 'None' (line 170)
        None_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 32), 'None')
        
        (may_be_653, more_types_in_union_654) = may_be_none(library_dirs_651, None_652)

        if may_be_653:

            if more_types_in_union_654:
                # Runtime conditional SSA (line 170)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 171):
            
            # Assigning a List to a Attribute (line 171):
            
            # Obtaining an instance of the builtin type 'list' (line 171)
            list_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 32), 'list')
            # Adding type elements to the builtin type 'list' instance (line 171)
            
            # Getting the type of 'self' (line 171)
            self_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'self')
            # Setting the type of the member 'library_dirs' of a type (line 171)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), self_656, 'library_dirs', list_655)

            if more_types_in_union_654:
                # Runtime conditional SSA for else branch (line 170)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_653) or more_types_in_union_654):
            
            
            
            # Call to type(...): (line 172)
            # Processing the call arguments (line 172)
            # Getting the type of 'self' (line 172)
            self_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'self', False)
            # Obtaining the member 'library_dirs' of a type (line 172)
            library_dirs_659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 18), self_658, 'library_dirs')
            # Processing the call keyword arguments (line 172)
            kwargs_660 = {}
            # Getting the type of 'type' (line 172)
            type_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 13), 'type', False)
            # Calling type(args, kwargs) (line 172)
            type_call_result_661 = invoke(stypy.reporting.localization.Localization(__file__, 172, 13), type_657, *[library_dirs_659], **kwargs_660)
            
            # Getting the type of 'StringType' (line 172)
            StringType_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 40), 'StringType')
            # Applying the binary operator 'is' (line 172)
            result_is__663 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 13), 'is', type_call_result_661, StringType_662)
            
            # Testing the type of an if condition (line 172)
            if_condition_664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 13), result_is__663)
            # Assigning a type to the variable 'if_condition_664' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 13), 'if_condition_664', if_condition_664)
            # SSA begins for if statement (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 173):
            
            # Assigning a Call to a Attribute (line 173):
            
            # Call to split(...): (line 173)
            # Processing the call arguments (line 173)
            # Getting the type of 'self' (line 173)
            self_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 45), 'self', False)
            # Obtaining the member 'library_dirs' of a type (line 173)
            library_dirs_668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 45), self_667, 'library_dirs')
            # Getting the type of 'os' (line 173)
            os_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 64), 'os', False)
            # Obtaining the member 'pathsep' of a type (line 173)
            pathsep_670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 64), os_669, 'pathsep')
            # Processing the call keyword arguments (line 173)
            kwargs_671 = {}
            # Getting the type of 'string' (line 173)
            string_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'string', False)
            # Obtaining the member 'split' of a type (line 173)
            split_666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 32), string_665, 'split')
            # Calling split(args, kwargs) (line 173)
            split_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 173, 32), split_666, *[library_dirs_668, pathsep_670], **kwargs_671)
            
            # Getting the type of 'self' (line 173)
            self_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'self')
            # Setting the type of the member 'library_dirs' of a type (line 173)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 12), self_673, 'library_dirs', split_call_result_672)
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_653 and more_types_in_union_654):
                # SSA join for if statement (line 170)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 175)
        # Getting the type of 'self' (line 175)
        self_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'self')
        # Obtaining the member 'rpath' of a type (line 175)
        rpath_675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 11), self_674, 'rpath')
        # Getting the type of 'None' (line 175)
        None_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'None')
        
        (may_be_677, more_types_in_union_678) = may_be_none(rpath_675, None_676)

        if may_be_677:

            if more_types_in_union_678:
                # Runtime conditional SSA (line 175)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 176):
            
            # Assigning a List to a Attribute (line 176):
            
            # Obtaining an instance of the builtin type 'list' (line 176)
            list_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 176)
            
            # Getting the type of 'self' (line 176)
            self_680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'self')
            # Setting the type of the member 'rpath' of a type (line 176)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), self_680, 'rpath', list_679)

            if more_types_in_union_678:
                # Runtime conditional SSA for else branch (line 175)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_677) or more_types_in_union_678):
            
            
            
            # Call to type(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'self' (line 177)
            self_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 18), 'self', False)
            # Obtaining the member 'rpath' of a type (line 177)
            rpath_683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 18), self_682, 'rpath')
            # Processing the call keyword arguments (line 177)
            kwargs_684 = {}
            # Getting the type of 'type' (line 177)
            type_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'type', False)
            # Calling type(args, kwargs) (line 177)
            type_call_result_685 = invoke(stypy.reporting.localization.Localization(__file__, 177, 13), type_681, *[rpath_683], **kwargs_684)
            
            # Getting the type of 'StringType' (line 177)
            StringType_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 33), 'StringType')
            # Applying the binary operator 'is' (line 177)
            result_is__687 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 13), 'is', type_call_result_685, StringType_686)
            
            # Testing the type of an if condition (line 177)
            if_condition_688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 13), result_is__687)
            # Assigning a type to the variable 'if_condition_688' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'if_condition_688', if_condition_688)
            # SSA begins for if statement (line 177)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 178):
            
            # Assigning a Call to a Attribute (line 178):
            
            # Call to split(...): (line 178)
            # Processing the call arguments (line 178)
            # Getting the type of 'self' (line 178)
            self_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 38), 'self', False)
            # Obtaining the member 'rpath' of a type (line 178)
            rpath_692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 38), self_691, 'rpath')
            # Getting the type of 'os' (line 178)
            os_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 50), 'os', False)
            # Obtaining the member 'pathsep' of a type (line 178)
            pathsep_694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 50), os_693, 'pathsep')
            # Processing the call keyword arguments (line 178)
            kwargs_695 = {}
            # Getting the type of 'string' (line 178)
            string_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'string', False)
            # Obtaining the member 'split' of a type (line 178)
            split_690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 25), string_689, 'split')
            # Calling split(args, kwargs) (line 178)
            split_call_result_696 = invoke(stypy.reporting.localization.Localization(__file__, 178, 25), split_690, *[rpath_692, pathsep_694], **kwargs_695)
            
            # Getting the type of 'self' (line 178)
            self_697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'self')
            # Setting the type of the member 'rpath' of a type (line 178)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), self_697, 'rpath', split_call_result_696)
            # SSA join for if statement (line 177)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_677 and more_types_in_union_678):
                # SSA join for if statement (line 175)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'os' (line 183)
        os_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'os')
        # Obtaining the member 'name' of a type (line 183)
        name_699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 11), os_698, 'name')
        str_700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 22), 'str', 'nt')
        # Applying the binary operator '==' (line 183)
        result_eq_701 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 11), '==', name_699, str_700)
        
        # Testing the type of an if condition (line 183)
        if_condition_702 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 8), result_eq_701)
        # Assigning a type to the variable 'if_condition_702' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'if_condition_702', if_condition_702)
        # SSA begins for if statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Call to join(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'sys' (line 187)
        sys_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 50), 'sys', False)
        # Obtaining the member 'exec_prefix' of a type (line 187)
        exec_prefix_710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 50), sys_709, 'exec_prefix')
        str_711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 67), 'str', 'libs')
        # Processing the call keyword arguments (line 187)
        kwargs_712 = {}
        # Getting the type of 'os' (line 187)
        os_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 187)
        path_707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 37), os_706, 'path')
        # Obtaining the member 'join' of a type (line 187)
        join_708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 37), path_707, 'join')
        # Calling join(args, kwargs) (line 187)
        join_call_result_713 = invoke(stypy.reporting.localization.Localization(__file__, 187, 37), join_708, *[exec_prefix_710, str_711], **kwargs_712)
        
        # Processing the call keyword arguments (line 187)
        kwargs_714 = {}
        # Getting the type of 'self' (line 187)
        self_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 187)
        library_dirs_704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 12), self_703, 'library_dirs')
        # Obtaining the member 'append' of a type (line 187)
        append_705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 12), library_dirs_704, 'append')
        # Calling append(args, kwargs) (line 187)
        append_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 187, 12), append_705, *[join_call_result_713], **kwargs_714)
        
        
        # Getting the type of 'self' (line 188)
        self_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'self')
        # Obtaining the member 'debug' of a type (line 188)
        debug_717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 15), self_716, 'debug')
        # Testing the type of an if condition (line 188)
        if_condition_718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 12), debug_717)
        # Assigning a type to the variable 'if_condition_718' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'if_condition_718', if_condition_718)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 189):
        
        # Assigning a Call to a Attribute (line 189):
        
        # Call to join(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'self' (line 189)
        self_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 47), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 189)
        build_temp_723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 47), self_722, 'build_temp')
        str_724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 64), 'str', 'Debug')
        # Processing the call keyword arguments (line 189)
        kwargs_725 = {}
        # Getting the type of 'os' (line 189)
        os_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 189)
        path_720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 34), os_719, 'path')
        # Obtaining the member 'join' of a type (line 189)
        join_721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 34), path_720, 'join')
        # Calling join(args, kwargs) (line 189)
        join_call_result_726 = invoke(stypy.reporting.localization.Localization(__file__, 189, 34), join_721, *[build_temp_723, str_724], **kwargs_725)
        
        # Getting the type of 'self' (line 189)
        self_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'self')
        # Setting the type of the member 'build_temp' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 16), self_727, 'build_temp', join_call_result_726)
        # SSA branch for the else part of an if statement (line 188)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 191):
        
        # Assigning a Call to a Attribute (line 191):
        
        # Call to join(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'self' (line 191)
        self_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 47), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 191)
        build_temp_732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 47), self_731, 'build_temp')
        str_733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 64), 'str', 'Release')
        # Processing the call keyword arguments (line 191)
        kwargs_734 = {}
        # Getting the type of 'os' (line 191)
        os_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 191)
        path_729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 34), os_728, 'path')
        # Obtaining the member 'join' of a type (line 191)
        join_730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 34), path_729, 'join')
        # Calling join(args, kwargs) (line 191)
        join_call_result_735 = invoke(stypy.reporting.localization.Localization(__file__, 191, 34), join_730, *[build_temp_732, str_733], **kwargs_734)
        
        # Getting the type of 'self' (line 191)
        self_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'self')
        # Setting the type of the member 'build_temp' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 16), self_736, 'build_temp', join_call_result_735)
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 195)
        # Processing the call arguments (line 195)
        
        # Call to join(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'sys' (line 195)
        sys_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 50), 'sys', False)
        # Obtaining the member 'exec_prefix' of a type (line 195)
        exec_prefix_744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 50), sys_743, 'exec_prefix')
        str_745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 67), 'str', 'PC')
        # Processing the call keyword arguments (line 195)
        kwargs_746 = {}
        # Getting the type of 'os' (line 195)
        os_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 195)
        path_741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 37), os_740, 'path')
        # Obtaining the member 'join' of a type (line 195)
        join_742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 37), path_741, 'join')
        # Calling join(args, kwargs) (line 195)
        join_call_result_747 = invoke(stypy.reporting.localization.Localization(__file__, 195, 37), join_742, *[exec_prefix_744, str_745], **kwargs_746)
        
        # Processing the call keyword arguments (line 195)
        kwargs_748 = {}
        # Getting the type of 'self' (line 195)
        self_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 195)
        include_dirs_738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_737, 'include_dirs')
        # Obtaining the member 'append' of a type (line 195)
        append_739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), include_dirs_738, 'append')
        # Calling append(args, kwargs) (line 195)
        append_call_result_749 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), append_739, *[join_call_result_747], **kwargs_748)
        
        
        
        # Getting the type of 'MSVC_VERSION' (line 196)
        MSVC_VERSION_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'MSVC_VERSION')
        int_751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 31), 'int')
        # Applying the binary operator '==' (line 196)
        result_eq_752 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 15), '==', MSVC_VERSION_750, int_751)
        
        # Testing the type of an if condition (line 196)
        if_condition_753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 12), result_eq_752)
        # Assigning a type to the variable 'if_condition_753' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'if_condition_753', if_condition_753)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 198)
        self_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'self')
        # Obtaining the member 'plat_name' of a type (line 198)
        plat_name_755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), self_754, 'plat_name')
        str_756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 37), 'str', 'win32')
        # Applying the binary operator '==' (line 198)
        result_eq_757 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 19), '==', plat_name_755, str_756)
        
        # Testing the type of an if condition (line 198)
        if_condition_758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 16), result_eq_757)
        # Assigning a type to the variable 'if_condition_758' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'if_condition_758', if_condition_758)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 199):
        
        # Assigning a Str to a Name (line 199):
        str_759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'str', '')
        # Assigning a type to the variable 'suffix' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'suffix', str_759)
        # SSA branch for the else part of an if statement (line 198)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 202):
        
        # Assigning a Subscript to a Name (line 202):
        
        # Obtaining the type of the subscript
        int_760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 44), 'int')
        slice_761 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 202, 29), int_760, None, None)
        # Getting the type of 'self' (line 202)
        self_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'self')
        # Obtaining the member 'plat_name' of a type (line 202)
        plat_name_763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 29), self_762, 'plat_name')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 29), plat_name_763, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_765 = invoke(stypy.reporting.localization.Localization(__file__, 202, 29), getitem___764, slice_761)
        
        # Assigning a type to the variable 'suffix' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'suffix', subscript_call_result_765)
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 204)
        tuple_766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 204)
        # Adding element type (line 204)
        
        # Obtaining an instance of the builtin type 'tuple' (line 204)
        tuple_767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 204)
        # Adding element type (line 204)
        str_768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 26), 'str', 'PCbuild')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 26), tuple_767, str_768)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), tuple_766, tuple_767)
        # Adding element type (line 204)
        
        # Obtaining an instance of the builtin type 'tuple' (line 204)
        tuple_769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 204)
        # Adding element type (line 204)
        str_770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 40), 'str', 'PC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 40), tuple_769, str_770)
        # Adding element type (line 204)
        str_771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 46), 'str', 'VS9.0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 40), tuple_769, str_771)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), tuple_766, tuple_769)
        
        # Testing the type of a for loop iterable (line 204)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 204, 16), tuple_766)
        # Getting the type of the for loop variable (line 204)
        for_loop_var_772 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 204, 16), tuple_766)
        # Assigning a type to the variable 'd' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'd', for_loop_var_772)
        # SSA begins for a for statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to join(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'sys' (line 205)
        sys_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 43), 'sys', False)
        # Obtaining the member 'exec_prefix' of a type (line 205)
        exec_prefix_777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 43), sys_776, 'exec_prefix')
        # Getting the type of 'd' (line 205)
        d_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 61), 'd', False)
        # Processing the call keyword arguments (line 205)
        kwargs_779 = {}
        # Getting the type of 'os' (line 205)
        os_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 205)
        path_774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 30), os_773, 'path')
        # Obtaining the member 'join' of a type (line 205)
        join_775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 30), path_774, 'join')
        # Calling join(args, kwargs) (line 205)
        join_call_result_780 = invoke(stypy.reporting.localization.Localization(__file__, 205, 30), join_775, *[exec_prefix_777, d_778], **kwargs_779)
        
        # Assigning a type to the variable 'new_lib' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'new_lib', join_call_result_780)
        
        # Getting the type of 'suffix' (line 206)
        suffix_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 'suffix')
        # Testing the type of an if condition (line 206)
        if_condition_782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 20), suffix_781)
        # Assigning a type to the variable 'if_condition_782' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'if_condition_782', if_condition_782)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to join(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'new_lib' (line 207)
        new_lib_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 47), 'new_lib', False)
        # Getting the type of 'suffix' (line 207)
        suffix_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 56), 'suffix', False)
        # Processing the call keyword arguments (line 207)
        kwargs_788 = {}
        # Getting the type of 'os' (line 207)
        os_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 207)
        path_784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 34), os_783, 'path')
        # Obtaining the member 'join' of a type (line 207)
        join_785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 34), path_784, 'join')
        # Calling join(args, kwargs) (line 207)
        join_call_result_789 = invoke(stypy.reporting.localization.Localization(__file__, 207, 34), join_785, *[new_lib_786, suffix_787], **kwargs_788)
        
        # Assigning a type to the variable 'new_lib' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 24), 'new_lib', join_call_result_789)
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'new_lib' (line 208)
        new_lib_793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 45), 'new_lib', False)
        # Processing the call keyword arguments (line 208)
        kwargs_794 = {}
        # Getting the type of 'self' (line 208)
        self_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 208)
        library_dirs_791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), self_790, 'library_dirs')
        # Obtaining the member 'append' of a type (line 208)
        append_792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), library_dirs_791, 'append')
        # Calling append(args, kwargs) (line 208)
        append_call_result_795 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), append_792, *[new_lib_793], **kwargs_794)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 196)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'MSVC_VERSION' (line 210)
        MSVC_VERSION_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'MSVC_VERSION')
        int_797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 33), 'int')
        # Applying the binary operator '==' (line 210)
        result_eq_798 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 17), '==', MSVC_VERSION_796, int_797)
        
        # Testing the type of an if condition (line 210)
        if_condition_799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 17), result_eq_798)
        # Assigning a type to the variable 'if_condition_799' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 17), 'if_condition_799', if_condition_799)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Call to join(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'sys' (line 211)
        sys_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 54), 'sys', False)
        # Obtaining the member 'exec_prefix' of a type (line 211)
        exec_prefix_807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 54), sys_806, 'exec_prefix')
        str_808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 41), 'str', 'PC')
        str_809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 47), 'str', 'VS8.0')
        # Processing the call keyword arguments (line 211)
        kwargs_810 = {}
        # Getting the type of 'os' (line 211)
        os_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 41), 'os', False)
        # Obtaining the member 'path' of a type (line 211)
        path_804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 41), os_803, 'path')
        # Obtaining the member 'join' of a type (line 211)
        join_805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 41), path_804, 'join')
        # Calling join(args, kwargs) (line 211)
        join_call_result_811 = invoke(stypy.reporting.localization.Localization(__file__, 211, 41), join_805, *[exec_prefix_807, str_808, str_809], **kwargs_810)
        
        # Processing the call keyword arguments (line 211)
        kwargs_812 = {}
        # Getting the type of 'self' (line 211)
        self_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 211)
        library_dirs_801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 16), self_800, 'library_dirs')
        # Obtaining the member 'append' of a type (line 211)
        append_802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 16), library_dirs_801, 'append')
        # Calling append(args, kwargs) (line 211)
        append_call_result_813 = invoke(stypy.reporting.localization.Localization(__file__, 211, 16), append_802, *[join_call_result_811], **kwargs_812)
        
        # SSA branch for the else part of an if statement (line 210)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'MSVC_VERSION' (line 213)
        MSVC_VERSION_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'MSVC_VERSION')
        int_815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 33), 'int')
        # Applying the binary operator '==' (line 213)
        result_eq_816 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 17), '==', MSVC_VERSION_814, int_815)
        
        # Testing the type of an if condition (line 213)
        if_condition_817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 17), result_eq_816)
        # Assigning a type to the variable 'if_condition_817' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'if_condition_817', if_condition_817)
        # SSA begins for if statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Call to join(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'sys' (line 214)
        sys_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 54), 'sys', False)
        # Obtaining the member 'exec_prefix' of a type (line 214)
        exec_prefix_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 54), sys_824, 'exec_prefix')
        str_826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 41), 'str', 'PC')
        str_827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 47), 'str', 'VS7.1')
        # Processing the call keyword arguments (line 214)
        kwargs_828 = {}
        # Getting the type of 'os' (line 214)
        os_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 41), 'os', False)
        # Obtaining the member 'path' of a type (line 214)
        path_822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 41), os_821, 'path')
        # Obtaining the member 'join' of a type (line 214)
        join_823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 41), path_822, 'join')
        # Calling join(args, kwargs) (line 214)
        join_call_result_829 = invoke(stypy.reporting.localization.Localization(__file__, 214, 41), join_823, *[exec_prefix_825, str_826, str_827], **kwargs_828)
        
        # Processing the call keyword arguments (line 214)
        kwargs_830 = {}
        # Getting the type of 'self' (line 214)
        self_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 214)
        library_dirs_819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), self_818, 'library_dirs')
        # Obtaining the member 'append' of a type (line 214)
        append_820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), library_dirs_819, 'append')
        # Calling append(args, kwargs) (line 214)
        append_call_result_831 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), append_820, *[join_call_result_829], **kwargs_830)
        
        # SSA branch for the else part of an if statement (line 213)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Call to join(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'sys' (line 217)
        sys_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 54), 'sys', False)
        # Obtaining the member 'exec_prefix' of a type (line 217)
        exec_prefix_839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 54), sys_838, 'exec_prefix')
        str_840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 41), 'str', 'PC')
        str_841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 47), 'str', 'VC6')
        # Processing the call keyword arguments (line 217)
        kwargs_842 = {}
        # Getting the type of 'os' (line 217)
        os_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 41), 'os', False)
        # Obtaining the member 'path' of a type (line 217)
        path_836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 41), os_835, 'path')
        # Obtaining the member 'join' of a type (line 217)
        join_837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 41), path_836, 'join')
        # Calling join(args, kwargs) (line 217)
        join_call_result_843 = invoke(stypy.reporting.localization.Localization(__file__, 217, 41), join_837, *[exec_prefix_839, str_840, str_841], **kwargs_842)
        
        # Processing the call keyword arguments (line 217)
        kwargs_844 = {}
        # Getting the type of 'self' (line 217)
        self_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 217)
        library_dirs_833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 16), self_832, 'library_dirs')
        # Obtaining the member 'append' of a type (line 217)
        append_834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 16), library_dirs_833, 'append')
        # Calling append(args, kwargs) (line 217)
        append_call_result_845 = invoke(stypy.reporting.localization.Localization(__file__, 217, 16), append_834, *[join_call_result_843], **kwargs_844)
        
        # SSA join for if statement (line 213)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'os' (line 222)
        os_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'os')
        # Obtaining the member 'name' of a type (line 222)
        name_847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 11), os_846, 'name')
        str_848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 22), 'str', 'os2')
        # Applying the binary operator '==' (line 222)
        result_eq_849 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 11), '==', name_847, str_848)
        
        # Testing the type of an if condition (line 222)
        if_condition_850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 8), result_eq_849)
        # Assigning a type to the variable 'if_condition_850' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'if_condition_850', if_condition_850)
        # SSA begins for if statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Call to join(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'sys' (line 223)
        sys_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 50), 'sys', False)
        # Obtaining the member 'exec_prefix' of a type (line 223)
        exec_prefix_858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 50), sys_857, 'exec_prefix')
        str_859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 67), 'str', 'Config')
        # Processing the call keyword arguments (line 223)
        kwargs_860 = {}
        # Getting the type of 'os' (line 223)
        os_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 223)
        path_855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 37), os_854, 'path')
        # Obtaining the member 'join' of a type (line 223)
        join_856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 37), path_855, 'join')
        # Calling join(args, kwargs) (line 223)
        join_call_result_861 = invoke(stypy.reporting.localization.Localization(__file__, 223, 37), join_856, *[exec_prefix_858, str_859], **kwargs_860)
        
        # Processing the call keyword arguments (line 223)
        kwargs_862 = {}
        # Getting the type of 'self' (line 223)
        self_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 223)
        library_dirs_852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), self_851, 'library_dirs')
        # Obtaining the member 'append' of a type (line 223)
        append_853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), library_dirs_852, 'append')
        # Calling append(args, kwargs) (line 223)
        append_call_result_863 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), append_853, *[join_call_result_861], **kwargs_862)
        
        # SSA join for if statement (line 222)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 25), 'int')
        slice_865 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 227, 11), None, int_864, None)
        # Getting the type of 'sys' (line 227)
        sys_866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 227)
        platform_867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), sys_866, 'platform')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), platform_867, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_869 = invoke(stypy.reporting.localization.Localization(__file__, 227, 11), getitem___868, slice_865)
        
        str_870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 31), 'str', 'cygwin')
        # Applying the binary operator '==' (line 227)
        result_eq_871 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), '==', subscript_call_result_869, str_870)
        
        
        
        # Obtaining the type of the subscript
        int_872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 57), 'int')
        slice_873 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 227, 43), None, int_872, None)
        # Getting the type of 'sys' (line 227)
        sys_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 43), 'sys')
        # Obtaining the member 'platform' of a type (line 227)
        platform_875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 43), sys_874, 'platform')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 43), platform_875, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_877 = invoke(stypy.reporting.localization.Localization(__file__, 227, 43), getitem___876, slice_873)
        
        str_878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 63), 'str', 'atheos')
        # Applying the binary operator '==' (line 227)
        result_eq_879 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 43), '==', subscript_call_result_877, str_878)
        
        # Applying the binary operator 'or' (line 227)
        result_or_keyword_880 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), 'or', result_eq_871, result_eq_879)
        
        # Testing the type of an if condition (line 227)
        if_condition_881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), result_or_keyword_880)
        # Assigning a type to the variable 'if_condition_881' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_881', if_condition_881)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to startswith(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Call to join(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'sys' (line 228)
        sys_888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 54), 'sys', False)
        # Obtaining the member 'exec_prefix' of a type (line 228)
        exec_prefix_889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 54), sys_888, 'exec_prefix')
        str_890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 71), 'str', 'bin')
        # Processing the call keyword arguments (line 228)
        kwargs_891 = {}
        # Getting the type of 'os' (line 228)
        os_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 41), 'os', False)
        # Obtaining the member 'path' of a type (line 228)
        path_886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 41), os_885, 'path')
        # Obtaining the member 'join' of a type (line 228)
        join_887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 41), path_886, 'join')
        # Calling join(args, kwargs) (line 228)
        join_call_result_892 = invoke(stypy.reporting.localization.Localization(__file__, 228, 41), join_887, *[exec_prefix_889, str_890], **kwargs_891)
        
        # Processing the call keyword arguments (line 228)
        kwargs_893 = {}
        # Getting the type of 'sys' (line 228)
        sys_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'sys', False)
        # Obtaining the member 'executable' of a type (line 228)
        executable_883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 15), sys_882, 'executable')
        # Obtaining the member 'startswith' of a type (line 228)
        startswith_884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 15), executable_883, 'startswith')
        # Calling startswith(args, kwargs) (line 228)
        startswith_call_result_894 = invoke(stypy.reporting.localization.Localization(__file__, 228, 15), startswith_884, *[join_call_result_892], **kwargs_893)
        
        # Testing the type of an if condition (line 228)
        if_condition_895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 12), startswith_call_result_894)
        # Assigning a type to the variable 'if_condition_895' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'if_condition_895', if_condition_895)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 230)
        # Processing the call arguments (line 230)
        
        # Call to join(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'sys' (line 230)
        sys_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 54), 'sys', False)
        # Obtaining the member 'prefix' of a type (line 230)
        prefix_903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 54), sys_902, 'prefix')
        str_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 66), 'str', 'lib')
        str_905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 54), 'str', 'python')
        
        # Call to get_python_version(...): (line 231)
        # Processing the call keyword arguments (line 231)
        kwargs_907 = {}
        # Getting the type of 'get_python_version' (line 231)
        get_python_version_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 65), 'get_python_version', False)
        # Calling get_python_version(args, kwargs) (line 231)
        get_python_version_call_result_908 = invoke(stypy.reporting.localization.Localization(__file__, 231, 65), get_python_version_906, *[], **kwargs_907)
        
        # Applying the binary operator '+' (line 231)
        result_add_909 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 54), '+', str_905, get_python_version_call_result_908)
        
        str_910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 54), 'str', 'config')
        # Processing the call keyword arguments (line 230)
        kwargs_911 = {}
        # Getting the type of 'os' (line 230)
        os_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 41), 'os', False)
        # Obtaining the member 'path' of a type (line 230)
        path_900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 41), os_899, 'path')
        # Obtaining the member 'join' of a type (line 230)
        join_901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 41), path_900, 'join')
        # Calling join(args, kwargs) (line 230)
        join_call_result_912 = invoke(stypy.reporting.localization.Localization(__file__, 230, 41), join_901, *[prefix_903, str_904, result_add_909, str_910], **kwargs_911)
        
        # Processing the call keyword arguments (line 230)
        kwargs_913 = {}
        # Getting the type of 'self' (line 230)
        self_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 230)
        library_dirs_897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 16), self_896, 'library_dirs')
        # Obtaining the member 'append' of a type (line 230)
        append_898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 16), library_dirs_897, 'append')
        # Calling append(args, kwargs) (line 230)
        append_call_result_914 = invoke(stypy.reporting.localization.Localization(__file__, 230, 16), append_898, *[join_call_result_912], **kwargs_913)
        
        # SSA branch for the else part of an if statement (line 228)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 235)
        # Processing the call arguments (line 235)
        str_918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 41), 'str', '.')
        # Processing the call keyword arguments (line 235)
        kwargs_919 = {}
        # Getting the type of 'self' (line 235)
        self_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 235)
        library_dirs_916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), self_915, 'library_dirs')
        # Obtaining the member 'append' of a type (line 235)
        append_917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), library_dirs_916, 'append')
        # Calling append(args, kwargs) (line 235)
        append_call_result_920 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), append_917, *[str_918], **kwargs_919)
        
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get_config_var(...): (line 240)
        # Processing the call arguments (line 240)
        str_923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 37), 'str', 'Py_ENABLE_SHARED')
        # Processing the call keyword arguments (line 240)
        kwargs_924 = {}
        # Getting the type of 'sysconfig' (line 240)
        sysconfig_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 240)
        get_config_var_922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), sysconfig_921, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 240)
        get_config_var_call_result_925 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), get_config_var_922, *[str_923], **kwargs_924)
        
        # Testing the type of an if condition (line 240)
        if_condition_926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), get_config_var_call_result_925)
        # Assigning a type to the variable 'if_condition_926' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_926', if_condition_926)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'sysconfig' (line 241)
        sysconfig_927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 19), 'sysconfig')
        # Obtaining the member 'python_build' of a type (line 241)
        python_build_928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 19), sysconfig_927, 'python_build')
        # Applying the 'not' unary operator (line 241)
        result_not__929 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 15), 'not', python_build_928)
        
        # Testing the type of an if condition (line 241)
        if_condition_930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 12), result_not__929)
        # Assigning a type to the variable 'if_condition_930' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'if_condition_930', if_condition_930)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 243)
        # Processing the call arguments (line 243)
        
        # Call to get_config_var(...): (line 243)
        # Processing the call arguments (line 243)
        str_936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 66), 'str', 'LIBDIR')
        # Processing the call keyword arguments (line 243)
        kwargs_937 = {}
        # Getting the type of 'sysconfig' (line 243)
        sysconfig_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 41), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 243)
        get_config_var_935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 41), sysconfig_934, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 243)
        get_config_var_call_result_938 = invoke(stypy.reporting.localization.Localization(__file__, 243, 41), get_config_var_935, *[str_936], **kwargs_937)
        
        # Processing the call keyword arguments (line 243)
        kwargs_939 = {}
        # Getting the type of 'self' (line 243)
        self_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 243)
        library_dirs_932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 16), self_931, 'library_dirs')
        # Obtaining the member 'append' of a type (line 243)
        append_933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 16), library_dirs_932, 'append')
        # Calling append(args, kwargs) (line 243)
        append_call_result_940 = invoke(stypy.reporting.localization.Localization(__file__, 243, 16), append_933, *[get_config_var_call_result_938], **kwargs_939)
        
        # SSA branch for the else part of an if statement (line 241)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 246)
        # Processing the call arguments (line 246)
        str_944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 41), 'str', '.')
        # Processing the call keyword arguments (line 246)
        kwargs_945 = {}
        # Getting the type of 'self' (line 246)
        self_941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 246)
        library_dirs_942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), self_941, 'library_dirs')
        # Obtaining the member 'append' of a type (line 246)
        append_943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), library_dirs_942, 'append')
        # Calling append(args, kwargs) (line 246)
        append_call_result_946 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), append_943, *[str_944], **kwargs_945)
        
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 253)
        self_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'self')
        # Obtaining the member 'define' of a type (line 253)
        define_948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 11), self_947, 'define')
        # Testing the type of an if condition (line 253)
        if_condition_949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 8), define_948)
        # Assigning a type to the variable 'if_condition_949' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'if_condition_949', if_condition_949)
        # SSA begins for if statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 254):
        
        # Assigning a Call to a Name (line 254):
        
        # Call to split(...): (line 254)
        # Processing the call arguments (line 254)
        str_953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 40), 'str', ',')
        # Processing the call keyword arguments (line 254)
        kwargs_954 = {}
        # Getting the type of 'self' (line 254)
        self_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 22), 'self', False)
        # Obtaining the member 'define' of a type (line 254)
        define_951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 22), self_950, 'define')
        # Obtaining the member 'split' of a type (line 254)
        split_952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 22), define_951, 'split')
        # Calling split(args, kwargs) (line 254)
        split_call_result_955 = invoke(stypy.reporting.localization.Localization(__file__, 254, 22), split_952, *[str_953], **kwargs_954)
        
        # Assigning a type to the variable 'defines' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'defines', split_call_result_955)
        
        # Assigning a Call to a Attribute (line 255):
        
        # Assigning a Call to a Attribute (line 255):
        
        # Call to map(...): (line 255)
        # Processing the call arguments (line 255)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 255, 30, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['symbol']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['symbol'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['symbol'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining an instance of the builtin type 'tuple' (line 255)
            tuple_957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 46), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 255)
            # Adding element type (line 255)
            # Getting the type of 'symbol' (line 255)
            symbol_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 46), 'symbol', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 46), tuple_957, symbol_958)
            # Adding element type (line 255)
            str_959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 54), 'str', '1')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 46), tuple_957, str_959)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 30), 'stypy_return_type', tuple_957)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 255)
            stypy_return_type_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 30), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_960)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_960

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 30), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 255)
        _stypy_temp_lambda_1_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 30), '_stypy_temp_lambda_1')
        # Getting the type of 'defines' (line 255)
        defines_962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 60), 'defines', False)
        # Processing the call keyword arguments (line 255)
        kwargs_963 = {}
        # Getting the type of 'map' (line 255)
        map_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 26), 'map', False)
        # Calling map(args, kwargs) (line 255)
        map_call_result_964 = invoke(stypy.reporting.localization.Localization(__file__, 255, 26), map_956, *[_stypy_temp_lambda_1_961, defines_962], **kwargs_963)
        
        # Getting the type of 'self' (line 255)
        self_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self')
        # Setting the type of the member 'define' of a type (line 255)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_965, 'define', map_call_result_964)
        # SSA join for if statement (line 253)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 260)
        self_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'self')
        # Obtaining the member 'undef' of a type (line 260)
        undef_967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 11), self_966, 'undef')
        # Testing the type of an if condition (line 260)
        if_condition_968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 8), undef_967)
        # Assigning a type to the variable 'if_condition_968' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'if_condition_968', if_condition_968)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 261):
        
        # Assigning a Call to a Attribute (line 261):
        
        # Call to split(...): (line 261)
        # Processing the call arguments (line 261)
        str_972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 42), 'str', ',')
        # Processing the call keyword arguments (line 261)
        kwargs_973 = {}
        # Getting the type of 'self' (line 261)
        self_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 25), 'self', False)
        # Obtaining the member 'undef' of a type (line 261)
        undef_970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 25), self_969, 'undef')
        # Obtaining the member 'split' of a type (line 261)
        split_971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 25), undef_970, 'split')
        # Calling split(args, kwargs) (line 261)
        split_call_result_974 = invoke(stypy.reporting.localization.Localization(__file__, 261, 25), split_971, *[str_972], **kwargs_973)
        
        # Getting the type of 'self' (line 261)
        self_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'self')
        # Setting the type of the member 'undef' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 12), self_975, 'undef', split_call_result_974)
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 263)
        # Getting the type of 'self' (line 263)
        self_976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'self')
        # Obtaining the member 'swig_opts' of a type (line 263)
        swig_opts_977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 11), self_976, 'swig_opts')
        # Getting the type of 'None' (line 263)
        None_978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 29), 'None')
        
        (may_be_979, more_types_in_union_980) = may_be_none(swig_opts_977, None_978)

        if may_be_979:

            if more_types_in_union_980:
                # Runtime conditional SSA (line 263)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Attribute (line 264):
            
            # Assigning a List to a Attribute (line 264):
            
            # Obtaining an instance of the builtin type 'list' (line 264)
            list_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 264)
            
            # Getting the type of 'self' (line 264)
            self_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'self')
            # Setting the type of the member 'swig_opts' of a type (line 264)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), self_982, 'swig_opts', list_981)

            if more_types_in_union_980:
                # Runtime conditional SSA for else branch (line 263)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_979) or more_types_in_union_980):
            
            # Assigning a Call to a Attribute (line 266):
            
            # Assigning a Call to a Attribute (line 266):
            
            # Call to split(...): (line 266)
            # Processing the call arguments (line 266)
            str_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 50), 'str', ' ')
            # Processing the call keyword arguments (line 266)
            kwargs_987 = {}
            # Getting the type of 'self' (line 266)
            self_983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 29), 'self', False)
            # Obtaining the member 'swig_opts' of a type (line 266)
            swig_opts_984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 29), self_983, 'swig_opts')
            # Obtaining the member 'split' of a type (line 266)
            split_985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 29), swig_opts_984, 'split')
            # Calling split(args, kwargs) (line 266)
            split_call_result_988 = invoke(stypy.reporting.localization.Localization(__file__, 266, 29), split_985, *[str_986], **kwargs_987)
            
            # Getting the type of 'self' (line 266)
            self_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'self')
            # Setting the type of the member 'swig_opts' of a type (line 266)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), self_989, 'swig_opts', split_call_result_988)

            if (may_be_979 and more_types_in_union_980):
                # SSA join for if statement (line 263)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 269)
        self_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'self')
        # Obtaining the member 'user' of a type (line 269)
        user_991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 11), self_990, 'user')
        # Testing the type of an if condition (line 269)
        if_condition_992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 8), user_991)
        # Assigning a type to the variable 'if_condition_992' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'if_condition_992', if_condition_992)
        # SSA begins for if statement (line 269)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to join(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'USER_BASE' (line 270)
        USER_BASE_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 40), 'USER_BASE', False)
        str_997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 51), 'str', 'include')
        # Processing the call keyword arguments (line 270)
        kwargs_998 = {}
        # Getting the type of 'os' (line 270)
        os_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 270)
        path_994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 27), os_993, 'path')
        # Obtaining the member 'join' of a type (line 270)
        join_995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 27), path_994, 'join')
        # Calling join(args, kwargs) (line 270)
        join_call_result_999 = invoke(stypy.reporting.localization.Localization(__file__, 270, 27), join_995, *[USER_BASE_996, str_997], **kwargs_998)
        
        # Assigning a type to the variable 'user_include' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'user_include', join_call_result_999)
        
        # Assigning a Call to a Name (line 271):
        
        # Assigning a Call to a Name (line 271):
        
        # Call to join(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'USER_BASE' (line 271)
        USER_BASE_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 36), 'USER_BASE', False)
        str_1004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 47), 'str', 'lib')
        # Processing the call keyword arguments (line 271)
        kwargs_1005 = {}
        # Getting the type of 'os' (line 271)
        os_1000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 271)
        path_1001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 23), os_1000, 'path')
        # Obtaining the member 'join' of a type (line 271)
        join_1002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 23), path_1001, 'join')
        # Calling join(args, kwargs) (line 271)
        join_call_result_1006 = invoke(stypy.reporting.localization.Localization(__file__, 271, 23), join_1002, *[USER_BASE_1003, str_1004], **kwargs_1005)
        
        # Assigning a type to the variable 'user_lib' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'user_lib', join_call_result_1006)
        
        
        # Call to isdir(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'user_include' (line 272)
        user_include_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 'user_include', False)
        # Processing the call keyword arguments (line 272)
        kwargs_1011 = {}
        # Getting the type of 'os' (line 272)
        os_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 272)
        path_1008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), os_1007, 'path')
        # Obtaining the member 'isdir' of a type (line 272)
        isdir_1009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), path_1008, 'isdir')
        # Calling isdir(args, kwargs) (line 272)
        isdir_call_result_1012 = invoke(stypy.reporting.localization.Localization(__file__, 272, 15), isdir_1009, *[user_include_1010], **kwargs_1011)
        
        # Testing the type of an if condition (line 272)
        if_condition_1013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), isdir_call_result_1012)
        # Assigning a type to the variable 'if_condition_1013' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_1013', if_condition_1013)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'user_include' (line 273)
        user_include_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 41), 'user_include', False)
        # Processing the call keyword arguments (line 273)
        kwargs_1018 = {}
        # Getting the type of 'self' (line 273)
        self_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 273)
        include_dirs_1015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), self_1014, 'include_dirs')
        # Obtaining the member 'append' of a type (line 273)
        append_1016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), include_dirs_1015, 'append')
        # Calling append(args, kwargs) (line 273)
        append_call_result_1019 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), append_1016, *[user_include_1017], **kwargs_1018)
        
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isdir(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'user_lib' (line 274)
        user_lib_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'user_lib', False)
        # Processing the call keyword arguments (line 274)
        kwargs_1024 = {}
        # Getting the type of 'os' (line 274)
        os_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 274)
        path_1021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 15), os_1020, 'path')
        # Obtaining the member 'isdir' of a type (line 274)
        isdir_1022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 15), path_1021, 'isdir')
        # Calling isdir(args, kwargs) (line 274)
        isdir_call_result_1025 = invoke(stypy.reporting.localization.Localization(__file__, 274, 15), isdir_1022, *[user_lib_1023], **kwargs_1024)
        
        # Testing the type of an if condition (line 274)
        if_condition_1026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 12), isdir_call_result_1025)
        # Assigning a type to the variable 'if_condition_1026' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'if_condition_1026', if_condition_1026)
        # SSA begins for if statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'user_lib' (line 275)
        user_lib_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 41), 'user_lib', False)
        # Processing the call keyword arguments (line 275)
        kwargs_1031 = {}
        # Getting the type of 'self' (line 275)
        self_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 275)
        library_dirs_1028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), self_1027, 'library_dirs')
        # Obtaining the member 'append' of a type (line 275)
        append_1029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), library_dirs_1028, 'append')
        # Calling append(args, kwargs) (line 275)
        append_call_result_1032 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), append_1029, *[user_lib_1030], **kwargs_1031)
        
        
        # Call to append(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'user_lib' (line 276)
        user_lib_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'user_lib', False)
        # Processing the call keyword arguments (line 276)
        kwargs_1037 = {}
        # Getting the type of 'self' (line 276)
        self_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'self', False)
        # Obtaining the member 'rpath' of a type (line 276)
        rpath_1034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), self_1033, 'rpath')
        # Obtaining the member 'append' of a type (line 276)
        append_1035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), rpath_1034, 'append')
        # Calling append(args, kwargs) (line 276)
        append_call_result_1038 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), append_1035, *[user_lib_1036], **kwargs_1037)
        
        # SSA join for if statement (line 274)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 269)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_1039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1039)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_1039


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.run.__dict__.__setitem__('stypy_localization', localization)
        build_ext.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.run.__dict__.__setitem__('stypy_function_name', 'build_ext.run')
        build_ext.run.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.run', [], None, None, defaults, varargs, kwargs)

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

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 279, 8))
        
        # 'from distutils.ccompiler import new_compiler' statement (line 279)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_1040 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 279, 8), 'distutils.ccompiler')

        if (type(import_1040) is not StypyTypeError):

            if (import_1040 != 'pyd_module'):
                __import__(import_1040)
                sys_modules_1041 = sys.modules[import_1040]
                import_from_module(stypy.reporting.localization.Localization(__file__, 279, 8), 'distutils.ccompiler', sys_modules_1041.module_type_store, module_type_store, ['new_compiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 279, 8), __file__, sys_modules_1041, sys_modules_1041.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import new_compiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 279, 8), 'distutils.ccompiler', None, module_type_store, ['new_compiler'], [new_compiler])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'distutils.ccompiler', import_1040)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        
        # Getting the type of 'self' (line 293)
        self_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'self')
        # Obtaining the member 'extensions' of a type (line 293)
        extensions_1043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 15), self_1042, 'extensions')
        # Applying the 'not' unary operator (line 293)
        result_not__1044 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 11), 'not', extensions_1043)
        
        # Testing the type of an if condition (line 293)
        if_condition_1045 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 8), result_not__1044)
        # Assigning a type to the variable 'if_condition_1045' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'if_condition_1045', if_condition_1045)
        # SSA begins for if statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 293)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_c_libraries(...): (line 299)
        # Processing the call keyword arguments (line 299)
        kwargs_1049 = {}
        # Getting the type of 'self' (line 299)
        self_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 299)
        distribution_1047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), self_1046, 'distribution')
        # Obtaining the member 'has_c_libraries' of a type (line 299)
        has_c_libraries_1048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), distribution_1047, 'has_c_libraries')
        # Calling has_c_libraries(args, kwargs) (line 299)
        has_c_libraries_call_result_1050 = invoke(stypy.reporting.localization.Localization(__file__, 299, 11), has_c_libraries_1048, *[], **kwargs_1049)
        
        # Testing the type of an if condition (line 299)
        if_condition_1051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 8), has_c_libraries_call_result_1050)
        # Assigning a type to the variable 'if_condition_1051' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'if_condition_1051', if_condition_1051)
        # SSA begins for if statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to get_finalized_command(...): (line 300)
        # Processing the call arguments (line 300)
        str_1054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 52), 'str', 'build_clib')
        # Processing the call keyword arguments (line 300)
        kwargs_1055 = {}
        # Getting the type of 'self' (line 300)
        self_1052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 25), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 300)
        get_finalized_command_1053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 25), self_1052, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 300)
        get_finalized_command_call_result_1056 = invoke(stypy.reporting.localization.Localization(__file__, 300, 25), get_finalized_command_1053, *[str_1054], **kwargs_1055)
        
        # Assigning a type to the variable 'build_clib' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'build_clib', get_finalized_command_call_result_1056)
        
        # Call to extend(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Evaluating a boolean operation
        
        # Call to get_library_names(...): (line 301)
        # Processing the call keyword arguments (line 301)
        kwargs_1062 = {}
        # Getting the type of 'build_clib' (line 301)
        build_clib_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 34), 'build_clib', False)
        # Obtaining the member 'get_library_names' of a type (line 301)
        get_library_names_1061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 34), build_clib_1060, 'get_library_names')
        # Calling get_library_names(args, kwargs) (line 301)
        get_library_names_call_result_1063 = invoke(stypy.reporting.localization.Localization(__file__, 301, 34), get_library_names_1061, *[], **kwargs_1062)
        
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_1064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        
        # Applying the binary operator 'or' (line 301)
        result_or_keyword_1065 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 34), 'or', get_library_names_call_result_1063, list_1064)
        
        # Processing the call keyword arguments (line 301)
        kwargs_1066 = {}
        # Getting the type of 'self' (line 301)
        self_1057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'self', False)
        # Obtaining the member 'libraries' of a type (line 301)
        libraries_1058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), self_1057, 'libraries')
        # Obtaining the member 'extend' of a type (line 301)
        extend_1059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), libraries_1058, 'extend')
        # Calling extend(args, kwargs) (line 301)
        extend_call_result_1067 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), extend_1059, *[result_or_keyword_1065], **kwargs_1066)
        
        
        # Call to append(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'build_clib' (line 302)
        build_clib_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 37), 'build_clib', False)
        # Obtaining the member 'build_clib' of a type (line 302)
        build_clib_1072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 37), build_clib_1071, 'build_clib')
        # Processing the call keyword arguments (line 302)
        kwargs_1073 = {}
        # Getting the type of 'self' (line 302)
        self_1068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 302)
        library_dirs_1069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 12), self_1068, 'library_dirs')
        # Obtaining the member 'append' of a type (line 302)
        append_1070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 12), library_dirs_1069, 'append')
        # Calling append(args, kwargs) (line 302)
        append_call_result_1074 = invoke(stypy.reporting.localization.Localization(__file__, 302, 12), append_1070, *[build_clib_1072], **kwargs_1073)
        
        # SSA join for if statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 306):
        
        # Assigning a Call to a Attribute (line 306):
        
        # Call to new_compiler(...): (line 306)
        # Processing the call keyword arguments (line 306)
        # Getting the type of 'self' (line 306)
        self_1076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 46), 'self', False)
        # Obtaining the member 'compiler' of a type (line 306)
        compiler_1077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 46), self_1076, 'compiler')
        keyword_1078 = compiler_1077
        # Getting the type of 'self' (line 307)
        self_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 45), 'self', False)
        # Obtaining the member 'verbose' of a type (line 307)
        verbose_1080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 45), self_1079, 'verbose')
        keyword_1081 = verbose_1080
        # Getting the type of 'self' (line 308)
        self_1082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 45), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 308)
        dry_run_1083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 45), self_1082, 'dry_run')
        keyword_1084 = dry_run_1083
        # Getting the type of 'self' (line 309)
        self_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 43), 'self', False)
        # Obtaining the member 'force' of a type (line 309)
        force_1086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 43), self_1085, 'force')
        keyword_1087 = force_1086
        kwargs_1088 = {'force': keyword_1087, 'verbose': keyword_1081, 'dry_run': keyword_1084, 'compiler': keyword_1078}
        # Getting the type of 'new_compiler' (line 306)
        new_compiler_1075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 24), 'new_compiler', False)
        # Calling new_compiler(args, kwargs) (line 306)
        new_compiler_call_result_1089 = invoke(stypy.reporting.localization.Localization(__file__, 306, 24), new_compiler_1075, *[], **kwargs_1088)
        
        # Getting the type of 'self' (line 306)
        self_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 306)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), self_1090, 'compiler', new_compiler_call_result_1089)
        
        # Call to customize_compiler(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'self' (line 310)
        self_1092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'self', False)
        # Obtaining the member 'compiler' of a type (line 310)
        compiler_1093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 27), self_1092, 'compiler')
        # Processing the call keyword arguments (line 310)
        kwargs_1094 = {}
        # Getting the type of 'customize_compiler' (line 310)
        customize_compiler_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'customize_compiler', False)
        # Calling customize_compiler(args, kwargs) (line 310)
        customize_compiler_call_result_1095 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), customize_compiler_1091, *[compiler_1093], **kwargs_1094)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'os' (line 314)
        os_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'os')
        # Obtaining the member 'name' of a type (line 314)
        name_1097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 11), os_1096, 'name')
        str_1098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 22), 'str', 'nt')
        # Applying the binary operator '==' (line 314)
        result_eq_1099 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 11), '==', name_1097, str_1098)
        
        
        # Getting the type of 'self' (line 314)
        self_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 31), 'self')
        # Obtaining the member 'plat_name' of a type (line 314)
        plat_name_1101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 31), self_1100, 'plat_name')
        
        # Call to get_platform(...): (line 314)
        # Processing the call keyword arguments (line 314)
        kwargs_1103 = {}
        # Getting the type of 'get_platform' (line 314)
        get_platform_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 49), 'get_platform', False)
        # Calling get_platform(args, kwargs) (line 314)
        get_platform_call_result_1104 = invoke(stypy.reporting.localization.Localization(__file__, 314, 49), get_platform_1102, *[], **kwargs_1103)
        
        # Applying the binary operator '!=' (line 314)
        result_ne_1105 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 31), '!=', plat_name_1101, get_platform_call_result_1104)
        
        # Applying the binary operator 'and' (line 314)
        result_and_keyword_1106 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 11), 'and', result_eq_1099, result_ne_1105)
        
        # Testing the type of an if condition (line 314)
        if_condition_1107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 8), result_and_keyword_1106)
        # Assigning a type to the variable 'if_condition_1107' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'if_condition_1107', if_condition_1107)
        # SSA begins for if statement (line 314)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to initialize(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'self' (line 315)
        self_1111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 37), 'self', False)
        # Obtaining the member 'plat_name' of a type (line 315)
        plat_name_1112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 37), self_1111, 'plat_name')
        # Processing the call keyword arguments (line 315)
        kwargs_1113 = {}
        # Getting the type of 'self' (line 315)
        self_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'self', False)
        # Obtaining the member 'compiler' of a type (line 315)
        compiler_1109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), self_1108, 'compiler')
        # Obtaining the member 'initialize' of a type (line 315)
        initialize_1110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), compiler_1109, 'initialize')
        # Calling initialize(args, kwargs) (line 315)
        initialize_call_result_1114 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), initialize_1110, *[plat_name_1112], **kwargs_1113)
        
        # SSA join for if statement (line 314)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 321)
        self_1115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 'self')
        # Obtaining the member 'include_dirs' of a type (line 321)
        include_dirs_1116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 11), self_1115, 'include_dirs')
        # Getting the type of 'None' (line 321)
        None_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 36), 'None')
        # Applying the binary operator 'isnot' (line 321)
        result_is_not_1118 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 11), 'isnot', include_dirs_1116, None_1117)
        
        # Testing the type of an if condition (line 321)
        if_condition_1119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 8), result_is_not_1118)
        # Assigning a type to the variable 'if_condition_1119' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'if_condition_1119', if_condition_1119)
        # SSA begins for if statement (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_include_dirs(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'self' (line 322)
        self_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 43), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 322)
        include_dirs_1124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 43), self_1123, 'include_dirs')
        # Processing the call keyword arguments (line 322)
        kwargs_1125 = {}
        # Getting the type of 'self' (line 322)
        self_1120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'self', False)
        # Obtaining the member 'compiler' of a type (line 322)
        compiler_1121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), self_1120, 'compiler')
        # Obtaining the member 'set_include_dirs' of a type (line 322)
        set_include_dirs_1122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), compiler_1121, 'set_include_dirs')
        # Calling set_include_dirs(args, kwargs) (line 322)
        set_include_dirs_call_result_1126 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), set_include_dirs_1122, *[include_dirs_1124], **kwargs_1125)
        
        # SSA join for if statement (line 321)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 323)
        self_1127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 11), 'self')
        # Obtaining the member 'define' of a type (line 323)
        define_1128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 11), self_1127, 'define')
        # Getting the type of 'None' (line 323)
        None_1129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 30), 'None')
        # Applying the binary operator 'isnot' (line 323)
        result_is_not_1130 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 11), 'isnot', define_1128, None_1129)
        
        # Testing the type of an if condition (line 323)
        if_condition_1131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 8), result_is_not_1130)
        # Assigning a type to the variable 'if_condition_1131' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'if_condition_1131', if_condition_1131)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 325)
        self_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 33), 'self')
        # Obtaining the member 'define' of a type (line 325)
        define_1133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 33), self_1132, 'define')
        # Testing the type of a for loop iterable (line 325)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 325, 12), define_1133)
        # Getting the type of the for loop variable (line 325)
        for_loop_var_1134 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 325, 12), define_1133)
        # Assigning a type to the variable 'name' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 12), for_loop_var_1134))
        # Assigning a type to the variable 'value' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 12), for_loop_var_1134))
        # SSA begins for a for statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to define_macro(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'name' (line 326)
        name_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 43), 'name', False)
        # Getting the type of 'value' (line 326)
        value_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 49), 'value', False)
        # Processing the call keyword arguments (line 326)
        kwargs_1140 = {}
        # Getting the type of 'self' (line 326)
        self_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'self', False)
        # Obtaining the member 'compiler' of a type (line 326)
        compiler_1136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 16), self_1135, 'compiler')
        # Obtaining the member 'define_macro' of a type (line 326)
        define_macro_1137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 16), compiler_1136, 'define_macro')
        # Calling define_macro(args, kwargs) (line 326)
        define_macro_call_result_1141 = invoke(stypy.reporting.localization.Localization(__file__, 326, 16), define_macro_1137, *[name_1138, value_1139], **kwargs_1140)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 327)
        self_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'self')
        # Obtaining the member 'undef' of a type (line 327)
        undef_1143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 11), self_1142, 'undef')
        # Getting the type of 'None' (line 327)
        None_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 29), 'None')
        # Applying the binary operator 'isnot' (line 327)
        result_is_not_1145 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 11), 'isnot', undef_1143, None_1144)
        
        # Testing the type of an if condition (line 327)
        if_condition_1146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 8), result_is_not_1145)
        # Assigning a type to the variable 'if_condition_1146' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'if_condition_1146', if_condition_1146)
        # SSA begins for if statement (line 327)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 328)
        self_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 25), 'self')
        # Obtaining the member 'undef' of a type (line 328)
        undef_1148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 25), self_1147, 'undef')
        # Testing the type of a for loop iterable (line 328)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 328, 12), undef_1148)
        # Getting the type of the for loop variable (line 328)
        for_loop_var_1149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 328, 12), undef_1148)
        # Assigning a type to the variable 'macro' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'macro', for_loop_var_1149)
        # SSA begins for a for statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to undefine_macro(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'macro' (line 329)
        macro_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 45), 'macro', False)
        # Processing the call keyword arguments (line 329)
        kwargs_1154 = {}
        # Getting the type of 'self' (line 329)
        self_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'self', False)
        # Obtaining the member 'compiler' of a type (line 329)
        compiler_1151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 16), self_1150, 'compiler')
        # Obtaining the member 'undefine_macro' of a type (line 329)
        undefine_macro_1152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 16), compiler_1151, 'undefine_macro')
        # Calling undefine_macro(args, kwargs) (line 329)
        undefine_macro_call_result_1155 = invoke(stypy.reporting.localization.Localization(__file__, 329, 16), undefine_macro_1152, *[macro_1153], **kwargs_1154)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 327)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 330)
        self_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'self')
        # Obtaining the member 'libraries' of a type (line 330)
        libraries_1157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 11), self_1156, 'libraries')
        # Getting the type of 'None' (line 330)
        None_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 33), 'None')
        # Applying the binary operator 'isnot' (line 330)
        result_is_not_1159 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 11), 'isnot', libraries_1157, None_1158)
        
        # Testing the type of an if condition (line 330)
        if_condition_1160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 8), result_is_not_1159)
        # Assigning a type to the variable 'if_condition_1160' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'if_condition_1160', if_condition_1160)
        # SSA begins for if statement (line 330)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_libraries(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'self' (line 331)
        self_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 40), 'self', False)
        # Obtaining the member 'libraries' of a type (line 331)
        libraries_1165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 40), self_1164, 'libraries')
        # Processing the call keyword arguments (line 331)
        kwargs_1166 = {}
        # Getting the type of 'self' (line 331)
        self_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'self', False)
        # Obtaining the member 'compiler' of a type (line 331)
        compiler_1162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), self_1161, 'compiler')
        # Obtaining the member 'set_libraries' of a type (line 331)
        set_libraries_1163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), compiler_1162, 'set_libraries')
        # Calling set_libraries(args, kwargs) (line 331)
        set_libraries_call_result_1167 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), set_libraries_1163, *[libraries_1165], **kwargs_1166)
        
        # SSA join for if statement (line 330)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 332)
        self_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 11), 'self')
        # Obtaining the member 'library_dirs' of a type (line 332)
        library_dirs_1169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 11), self_1168, 'library_dirs')
        # Getting the type of 'None' (line 332)
        None_1170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 36), 'None')
        # Applying the binary operator 'isnot' (line 332)
        result_is_not_1171 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 11), 'isnot', library_dirs_1169, None_1170)
        
        # Testing the type of an if condition (line 332)
        if_condition_1172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 8), result_is_not_1171)
        # Assigning a type to the variable 'if_condition_1172' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'if_condition_1172', if_condition_1172)
        # SSA begins for if statement (line 332)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_library_dirs(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'self' (line 333)
        self_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 43), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 333)
        library_dirs_1177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 43), self_1176, 'library_dirs')
        # Processing the call keyword arguments (line 333)
        kwargs_1178 = {}
        # Getting the type of 'self' (line 333)
        self_1173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'self', False)
        # Obtaining the member 'compiler' of a type (line 333)
        compiler_1174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), self_1173, 'compiler')
        # Obtaining the member 'set_library_dirs' of a type (line 333)
        set_library_dirs_1175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), compiler_1174, 'set_library_dirs')
        # Calling set_library_dirs(args, kwargs) (line 333)
        set_library_dirs_call_result_1179 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), set_library_dirs_1175, *[library_dirs_1177], **kwargs_1178)
        
        # SSA join for if statement (line 332)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 334)
        self_1180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'self')
        # Obtaining the member 'rpath' of a type (line 334)
        rpath_1181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 11), self_1180, 'rpath')
        # Getting the type of 'None' (line 334)
        None_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 29), 'None')
        # Applying the binary operator 'isnot' (line 334)
        result_is_not_1183 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), 'isnot', rpath_1181, None_1182)
        
        # Testing the type of an if condition (line 334)
        if_condition_1184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), result_is_not_1183)
        # Assigning a type to the variable 'if_condition_1184' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_1184', if_condition_1184)
        # SSA begins for if statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_runtime_library_dirs(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'self' (line 335)
        self_1188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 51), 'self', False)
        # Obtaining the member 'rpath' of a type (line 335)
        rpath_1189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 51), self_1188, 'rpath')
        # Processing the call keyword arguments (line 335)
        kwargs_1190 = {}
        # Getting the type of 'self' (line 335)
        self_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'self', False)
        # Obtaining the member 'compiler' of a type (line 335)
        compiler_1186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), self_1185, 'compiler')
        # Obtaining the member 'set_runtime_library_dirs' of a type (line 335)
        set_runtime_library_dirs_1187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), compiler_1186, 'set_runtime_library_dirs')
        # Calling set_runtime_library_dirs(args, kwargs) (line 335)
        set_runtime_library_dirs_call_result_1191 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), set_runtime_library_dirs_1187, *[rpath_1189], **kwargs_1190)
        
        # SSA join for if statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 336)
        self_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'self')
        # Obtaining the member 'link_objects' of a type (line 336)
        link_objects_1193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 11), self_1192, 'link_objects')
        # Getting the type of 'None' (line 336)
        None_1194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 36), 'None')
        # Applying the binary operator 'isnot' (line 336)
        result_is_not_1195 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 11), 'isnot', link_objects_1193, None_1194)
        
        # Testing the type of an if condition (line 336)
        if_condition_1196 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), result_is_not_1195)
        # Assigning a type to the variable 'if_condition_1196' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_1196', if_condition_1196)
        # SSA begins for if statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_link_objects(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'self' (line 337)
        self_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 43), 'self', False)
        # Obtaining the member 'link_objects' of a type (line 337)
        link_objects_1201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 43), self_1200, 'link_objects')
        # Processing the call keyword arguments (line 337)
        kwargs_1202 = {}
        # Getting the type of 'self' (line 337)
        self_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'self', False)
        # Obtaining the member 'compiler' of a type (line 337)
        compiler_1198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), self_1197, 'compiler')
        # Obtaining the member 'set_link_objects' of a type (line 337)
        set_link_objects_1199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), compiler_1198, 'set_link_objects')
        # Calling set_link_objects(args, kwargs) (line 337)
        set_link_objects_call_result_1203 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), set_link_objects_1199, *[link_objects_1201], **kwargs_1202)
        
        # SSA join for if statement (line 336)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to build_extensions(...): (line 340)
        # Processing the call keyword arguments (line 340)
        kwargs_1206 = {}
        # Getting the type of 'self' (line 340)
        self_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self', False)
        # Obtaining the member 'build_extensions' of a type (line 340)
        build_extensions_1205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_1204, 'build_extensions')
        # Calling build_extensions(args, kwargs) (line 340)
        build_extensions_call_result_1207 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), build_extensions_1205, *[], **kwargs_1206)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_1208


    @norecursion
    def check_extensions_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_extensions_list'
        module_type_store = module_type_store.open_function_context('check_extensions_list', 342, 4, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_localization', localization)
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_function_name', 'build_ext.check_extensions_list')
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_param_names_list', ['extensions'])
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.check_extensions_list.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.check_extensions_list', ['extensions'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_extensions_list', localization, ['extensions'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_extensions_list(...)' code ##################

        str_1209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, (-1)), 'str', "Ensure that the list of extensions (presumably provided as a\n        command option 'extensions') is valid, i.e. it is a list of\n        Extension objects.  We also support the old-style list of 2-tuples,\n        where the tuples are (ext_name, build_info), which are converted to\n        Extension instances here.\n\n        Raise DistutilsSetupError if the structure is invalid anywhere;\n        just returns otherwise.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 352)
        # Getting the type of 'list' (line 352)
        list_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 38), 'list')
        # Getting the type of 'extensions' (line 352)
        extensions_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 26), 'extensions')
        
        (may_be_1212, more_types_in_union_1213) = may_not_be_subtype(list_1210, extensions_1211)

        if may_be_1212:

            if more_types_in_union_1213:
                # Runtime conditional SSA (line 352)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'extensions' (line 352)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'extensions', remove_subtype_from_union(extensions_1211, list))
            # Getting the type of 'DistutilsSetupError' (line 353)
            DistutilsSetupError_1214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 18), 'DistutilsSetupError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 353, 12), DistutilsSetupError_1214, 'raise parameter', BaseException)

            if more_types_in_union_1213:
                # SSA join for if statement (line 352)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to enumerate(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'extensions' (line 356)
        extensions_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 32), 'extensions', False)
        # Processing the call keyword arguments (line 356)
        kwargs_1217 = {}
        # Getting the type of 'enumerate' (line 356)
        enumerate_1215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 356)
        enumerate_call_result_1218 = invoke(stypy.reporting.localization.Localization(__file__, 356, 22), enumerate_1215, *[extensions_1216], **kwargs_1217)
        
        # Testing the type of a for loop iterable (line 356)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 356, 8), enumerate_call_result_1218)
        # Getting the type of the for loop variable (line 356)
        for_loop_var_1219 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 356, 8), enumerate_call_result_1218)
        # Assigning a type to the variable 'i' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 8), for_loop_var_1219))
        # Assigning a type to the variable 'ext' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'ext', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 8), for_loop_var_1219))
        # SSA begins for a for statement (line 356)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to isinstance(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'ext' (line 357)
        ext_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 26), 'ext', False)
        # Getting the type of 'Extension' (line 357)
        Extension_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'Extension', False)
        # Processing the call keyword arguments (line 357)
        kwargs_1223 = {}
        # Getting the type of 'isinstance' (line 357)
        isinstance_1220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 357)
        isinstance_call_result_1224 = invoke(stypy.reporting.localization.Localization(__file__, 357, 15), isinstance_1220, *[ext_1221, Extension_1222], **kwargs_1223)
        
        # Testing the type of an if condition (line 357)
        if_condition_1225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 12), isinstance_call_result_1224)
        # Assigning a type to the variable 'if_condition_1225' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'if_condition_1225', if_condition_1225)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'ext' (line 361)
        ext_1227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 30), 'ext', False)
        # Getting the type of 'tuple' (line 361)
        tuple_1228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 35), 'tuple', False)
        # Processing the call keyword arguments (line 361)
        kwargs_1229 = {}
        # Getting the type of 'isinstance' (line 361)
        isinstance_1226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 361)
        isinstance_call_result_1230 = invoke(stypy.reporting.localization.Localization(__file__, 361, 19), isinstance_1226, *[ext_1227, tuple_1228], **kwargs_1229)
        
        # Applying the 'not' unary operator (line 361)
        result_not__1231 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 15), 'not', isinstance_call_result_1230)
        
        
        
        # Call to len(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'ext' (line 361)
        ext_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 49), 'ext', False)
        # Processing the call keyword arguments (line 361)
        kwargs_1234 = {}
        # Getting the type of 'len' (line 361)
        len_1232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 45), 'len', False)
        # Calling len(args, kwargs) (line 361)
        len_call_result_1235 = invoke(stypy.reporting.localization.Localization(__file__, 361, 45), len_1232, *[ext_1233], **kwargs_1234)
        
        int_1236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 57), 'int')
        # Applying the binary operator '!=' (line 361)
        result_ne_1237 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 45), '!=', len_call_result_1235, int_1236)
        
        # Applying the binary operator 'or' (line 361)
        result_or_keyword_1238 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 15), 'or', result_not__1231, result_ne_1237)
        
        # Testing the type of an if condition (line 361)
        if_condition_1239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 12), result_or_keyword_1238)
        # Assigning a type to the variable 'if_condition_1239' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'if_condition_1239', if_condition_1239)
        # SSA begins for if statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsSetupError' (line 362)
        DistutilsSetupError_1240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 22), 'DistutilsSetupError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 362, 16), DistutilsSetupError_1240, 'raise parameter', BaseException)
        # SSA join for if statement (line 361)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Tuple (line 366):
        
        # Assigning a Subscript to a Name (line 366):
        
        # Obtaining the type of the subscript
        int_1241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 12), 'int')
        # Getting the type of 'ext' (line 366)
        ext_1242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 35), 'ext')
        # Obtaining the member '__getitem__' of a type (line 366)
        getitem___1243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), ext_1242, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 366)
        subscript_call_result_1244 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), getitem___1243, int_1241)
        
        # Assigning a type to the variable 'tuple_var_assignment_461' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'tuple_var_assignment_461', subscript_call_result_1244)
        
        # Assigning a Subscript to a Name (line 366):
        
        # Obtaining the type of the subscript
        int_1245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 12), 'int')
        # Getting the type of 'ext' (line 366)
        ext_1246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 35), 'ext')
        # Obtaining the member '__getitem__' of a type (line 366)
        getitem___1247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), ext_1246, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 366)
        subscript_call_result_1248 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), getitem___1247, int_1245)
        
        # Assigning a type to the variable 'tuple_var_assignment_462' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'tuple_var_assignment_462', subscript_call_result_1248)
        
        # Assigning a Name to a Name (line 366):
        # Getting the type of 'tuple_var_assignment_461' (line 366)
        tuple_var_assignment_461_1249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'tuple_var_assignment_461')
        # Assigning a type to the variable 'ext_name' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'ext_name', tuple_var_assignment_461_1249)
        
        # Assigning a Name to a Name (line 366):
        # Getting the type of 'tuple_var_assignment_462' (line 366)
        tuple_var_assignment_462_1250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'tuple_var_assignment_462')
        # Assigning a type to the variable 'build_info' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'build_info', tuple_var_assignment_462_1250)
        
        # Call to warn(...): (line 368)
        # Processing the call arguments (line 368)
        str_1253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 22), 'str', "old-style (ext_name, build_info) tuple found in ext_modules for extension '%s'-- please convert to Extension instance")
        # Getting the type of 'ext_name' (line 370)
        ext_name_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 66), 'ext_name', False)
        # Applying the binary operator '%' (line 368)
        result_mod_1255 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 22), '%', str_1253, ext_name_1254)
        
        # Processing the call keyword arguments (line 368)
        kwargs_1256 = {}
        # Getting the type of 'log' (line 368)
        log_1251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'log', False)
        # Obtaining the member 'warn' of a type (line 368)
        warn_1252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 12), log_1251, 'warn')
        # Calling warn(args, kwargs) (line 368)
        warn_call_result_1257 = invoke(stypy.reporting.localization.Localization(__file__, 368, 12), warn_1252, *[result_mod_1255], **kwargs_1256)
        
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'ext_name' (line 372)
        ext_name_1259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 31), 'ext_name', False)
        # Getting the type of 'str' (line 372)
        str_1260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 41), 'str', False)
        # Processing the call keyword arguments (line 372)
        kwargs_1261 = {}
        # Getting the type of 'isinstance' (line 372)
        isinstance_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 372)
        isinstance_call_result_1262 = invoke(stypy.reporting.localization.Localization(__file__, 372, 20), isinstance_1258, *[ext_name_1259, str_1260], **kwargs_1261)
        
        
        # Call to match(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'ext_name' (line 373)
        ext_name_1265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 44), 'ext_name', False)
        # Processing the call keyword arguments (line 373)
        kwargs_1266 = {}
        # Getting the type of 'extension_name_re' (line 373)
        extension_name_re_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 20), 'extension_name_re', False)
        # Obtaining the member 'match' of a type (line 373)
        match_1264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 20), extension_name_re_1263, 'match')
        # Calling match(args, kwargs) (line 373)
        match_call_result_1267 = invoke(stypy.reporting.localization.Localization(__file__, 373, 20), match_1264, *[ext_name_1265], **kwargs_1266)
        
        # Applying the binary operator 'and' (line 372)
        result_and_keyword_1268 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 20), 'and', isinstance_call_result_1262, match_call_result_1267)
        
        # Applying the 'not' unary operator (line 372)
        result_not__1269 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 15), 'not', result_and_keyword_1268)
        
        # Testing the type of an if condition (line 372)
        if_condition_1270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 12), result_not__1269)
        # Assigning a type to the variable 'if_condition_1270' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'if_condition_1270', if_condition_1270)
        # SSA begins for if statement (line 372)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsSetupError' (line 374)
        DistutilsSetupError_1271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 22), 'DistutilsSetupError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 374, 16), DistutilsSetupError_1271, 'raise parameter', BaseException)
        # SSA join for if statement (line 372)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 378)
        # Getting the type of 'dict' (line 378)
        dict_1272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 42), 'dict')
        # Getting the type of 'build_info' (line 378)
        build_info_1273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 30), 'build_info')
        
        (may_be_1274, more_types_in_union_1275) = may_not_be_subtype(dict_1272, build_info_1273)

        if may_be_1274:

            if more_types_in_union_1275:
                # Runtime conditional SSA (line 378)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'build_info' (line 378)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'build_info', remove_subtype_from_union(build_info_1273, dict))
            # Getting the type of 'DistutilsSetupError' (line 379)
            DistutilsSetupError_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 22), 'DistutilsSetupError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 379, 16), DistutilsSetupError_1276, 'raise parameter', BaseException)

            if more_types_in_union_1275:
                # SSA join for if statement (line 378)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Call to Extension(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'ext_name' (line 385)
        ext_name_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 28), 'ext_name', False)
        
        # Obtaining the type of the subscript
        str_1279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 49), 'str', 'sources')
        # Getting the type of 'build_info' (line 385)
        build_info_1280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 38), 'build_info', False)
        # Obtaining the member '__getitem__' of a type (line 385)
        getitem___1281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 38), build_info_1280, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 385)
        subscript_call_result_1282 = invoke(stypy.reporting.localization.Localization(__file__, 385, 38), getitem___1281, str_1279)
        
        # Processing the call keyword arguments (line 385)
        kwargs_1283 = {}
        # Getting the type of 'Extension' (line 385)
        Extension_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'Extension', False)
        # Calling Extension(args, kwargs) (line 385)
        Extension_call_result_1284 = invoke(stypy.reporting.localization.Localization(__file__, 385, 18), Extension_1277, *[ext_name_1278, subscript_call_result_1282], **kwargs_1283)
        
        # Assigning a type to the variable 'ext' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'ext', Extension_call_result_1284)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 389)
        tuple_1285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 389)
        # Adding element type (line 389)
        str_1286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 24), 'str', 'include_dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 24), tuple_1285, str_1286)
        # Adding element type (line 389)
        str_1287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 40), 'str', 'library_dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 24), tuple_1285, str_1287)
        # Adding element type (line 389)
        str_1288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 56), 'str', 'libraries')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 24), tuple_1285, str_1288)
        # Adding element type (line 389)
        str_1289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 24), 'str', 'extra_objects')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 24), tuple_1285, str_1289)
        # Adding element type (line 389)
        str_1290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 41), 'str', 'extra_compile_args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 24), tuple_1285, str_1290)
        # Adding element type (line 389)
        str_1291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 24), 'str', 'extra_link_args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 24), tuple_1285, str_1291)
        
        # Testing the type of a for loop iterable (line 389)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 389, 12), tuple_1285)
        # Getting the type of the for loop variable (line 389)
        for_loop_var_1292 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 389, 12), tuple_1285)
        # Assigning a type to the variable 'key' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'key', for_loop_var_1292)
        # SSA begins for a for statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 392):
        
        # Assigning a Call to a Name (line 392):
        
        # Call to get(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'key' (line 392)
        key_1295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 37), 'key', False)
        # Processing the call keyword arguments (line 392)
        kwargs_1296 = {}
        # Getting the type of 'build_info' (line 392)
        build_info_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 22), 'build_info', False)
        # Obtaining the member 'get' of a type (line 392)
        get_1294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 22), build_info_1293, 'get')
        # Calling get(args, kwargs) (line 392)
        get_call_result_1297 = invoke(stypy.reporting.localization.Localization(__file__, 392, 22), get_1294, *[key_1295], **kwargs_1296)
        
        # Assigning a type to the variable 'val' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'val', get_call_result_1297)
        
        # Type idiom detected: calculating its left and rigth part (line 393)
        # Getting the type of 'val' (line 393)
        val_1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'val')
        # Getting the type of 'None' (line 393)
        None_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 30), 'None')
        
        (may_be_1300, more_types_in_union_1301) = may_not_be_none(val_1298, None_1299)

        if may_be_1300:

            if more_types_in_union_1301:
                # Runtime conditional SSA (line 393)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setattr(...): (line 394)
            # Processing the call arguments (line 394)
            # Getting the type of 'ext' (line 394)
            ext_1303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 28), 'ext', False)
            # Getting the type of 'key' (line 394)
            key_1304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 33), 'key', False)
            # Getting the type of 'val' (line 394)
            val_1305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 38), 'val', False)
            # Processing the call keyword arguments (line 394)
            kwargs_1306 = {}
            # Getting the type of 'setattr' (line 394)
            setattr_1302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 20), 'setattr', False)
            # Calling setattr(args, kwargs) (line 394)
            setattr_call_result_1307 = invoke(stypy.reporting.localization.Localization(__file__, 394, 20), setattr_1302, *[ext_1303, key_1304, val_1305], **kwargs_1306)
            

            if more_types_in_union_1301:
                # SSA join for if statement (line 393)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 397):
        
        # Assigning a Call to a Attribute (line 397):
        
        # Call to get(...): (line 397)
        # Processing the call arguments (line 397)
        str_1310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 54), 'str', 'rpath')
        # Processing the call keyword arguments (line 397)
        kwargs_1311 = {}
        # Getting the type of 'build_info' (line 397)
        build_info_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 39), 'build_info', False)
        # Obtaining the member 'get' of a type (line 397)
        get_1309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 39), build_info_1308, 'get')
        # Calling get(args, kwargs) (line 397)
        get_call_result_1312 = invoke(stypy.reporting.localization.Localization(__file__, 397, 39), get_1309, *[str_1310], **kwargs_1311)
        
        # Getting the type of 'ext' (line 397)
        ext_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'ext')
        # Setting the type of the member 'runtime_library_dirs' of a type (line 397)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), ext_1313, 'runtime_library_dirs', get_call_result_1312)
        
        
        str_1314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 15), 'str', 'def_file')
        # Getting the type of 'build_info' (line 398)
        build_info_1315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 29), 'build_info')
        # Applying the binary operator 'in' (line 398)
        result_contains_1316 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 15), 'in', str_1314, build_info_1315)
        
        # Testing the type of an if condition (line 398)
        if_condition_1317 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 12), result_contains_1316)
        # Assigning a type to the variable 'if_condition_1317' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'if_condition_1317', if_condition_1317)
        # SSA begins for if statement (line 398)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 399)
        # Processing the call arguments (line 399)
        str_1320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 25), 'str', "'def_file' element of build info dict no longer supported")
        # Processing the call keyword arguments (line 399)
        kwargs_1321 = {}
        # Getting the type of 'log' (line 399)
        log_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 399)
        warn_1319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 16), log_1318, 'warn')
        # Calling warn(args, kwargs) (line 399)
        warn_call_result_1322 = invoke(stypy.reporting.localization.Localization(__file__, 399, 16), warn_1319, *[str_1320], **kwargs_1321)
        
        # SSA join for if statement (line 398)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 404):
        
        # Assigning a Call to a Name (line 404):
        
        # Call to get(...): (line 404)
        # Processing the call arguments (line 404)
        str_1325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 36), 'str', 'macros')
        # Processing the call keyword arguments (line 404)
        kwargs_1326 = {}
        # Getting the type of 'build_info' (line 404)
        build_info_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 21), 'build_info', False)
        # Obtaining the member 'get' of a type (line 404)
        get_1324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 21), build_info_1323, 'get')
        # Calling get(args, kwargs) (line 404)
        get_call_result_1327 = invoke(stypy.reporting.localization.Localization(__file__, 404, 21), get_1324, *[str_1325], **kwargs_1326)
        
        # Assigning a type to the variable 'macros' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'macros', get_call_result_1327)
        
        # Getting the type of 'macros' (line 405)
        macros_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'macros')
        # Testing the type of an if condition (line 405)
        if_condition_1329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 405, 12), macros_1328)
        # Assigning a type to the variable 'if_condition_1329' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'if_condition_1329', if_condition_1329)
        # SSA begins for if statement (line 405)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 406):
        
        # Assigning a List to a Attribute (line 406):
        
        # Obtaining an instance of the builtin type 'list' (line 406)
        list_1330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 406)
        
        # Getting the type of 'ext' (line 406)
        ext_1331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'ext')
        # Setting the type of the member 'define_macros' of a type (line 406)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 16), ext_1331, 'define_macros', list_1330)
        
        # Assigning a List to a Attribute (line 407):
        
        # Assigning a List to a Attribute (line 407):
        
        # Obtaining an instance of the builtin type 'list' (line 407)
        list_1332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 407)
        
        # Getting the type of 'ext' (line 407)
        ext_1333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'ext')
        # Setting the type of the member 'undef_macros' of a type (line 407)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), ext_1333, 'undef_macros', list_1332)
        
        # Getting the type of 'macros' (line 408)
        macros_1334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 29), 'macros')
        # Testing the type of a for loop iterable (line 408)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 408, 16), macros_1334)
        # Getting the type of the for loop variable (line 408)
        for_loop_var_1335 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 408, 16), macros_1334)
        # Assigning a type to the variable 'macro' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 16), 'macro', for_loop_var_1335)
        # SSA begins for a for statement (line 408)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'macro' (line 409)
        macro_1337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 39), 'macro', False)
        # Getting the type of 'tuple' (line 409)
        tuple_1338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 46), 'tuple', False)
        # Processing the call keyword arguments (line 409)
        kwargs_1339 = {}
        # Getting the type of 'isinstance' (line 409)
        isinstance_1336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 28), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 409)
        isinstance_call_result_1340 = invoke(stypy.reporting.localization.Localization(__file__, 409, 28), isinstance_1336, *[macro_1337, tuple_1338], **kwargs_1339)
        
        
        
        # Call to len(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'macro' (line 409)
        macro_1342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 61), 'macro', False)
        # Processing the call keyword arguments (line 409)
        kwargs_1343 = {}
        # Getting the type of 'len' (line 409)
        len_1341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 57), 'len', False)
        # Calling len(args, kwargs) (line 409)
        len_call_result_1344 = invoke(stypy.reporting.localization.Localization(__file__, 409, 57), len_1341, *[macro_1342], **kwargs_1343)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 409)
        tuple_1345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 72), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 409)
        # Adding element type (line 409)
        int_1346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 72), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 72), tuple_1345, int_1346)
        # Adding element type (line 409)
        int_1347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 75), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 72), tuple_1345, int_1347)
        
        # Applying the binary operator 'in' (line 409)
        result_contains_1348 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 57), 'in', len_call_result_1344, tuple_1345)
        
        # Applying the binary operator 'and' (line 409)
        result_and_keyword_1349 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 28), 'and', isinstance_call_result_1340, result_contains_1348)
        
        # Applying the 'not' unary operator (line 409)
        result_not__1350 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 23), 'not', result_and_keyword_1349)
        
        # Testing the type of an if condition (line 409)
        if_condition_1351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 20), result_not__1350)
        # Assigning a type to the variable 'if_condition_1351' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 20), 'if_condition_1351', if_condition_1351)
        # SSA begins for if statement (line 409)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsSetupError' (line 410)
        DistutilsSetupError_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 30), 'DistutilsSetupError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 410, 24), DistutilsSetupError_1352, 'raise parameter', BaseException)
        # SSA join for if statement (line 409)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'macro' (line 413)
        macro_1354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 27), 'macro', False)
        # Processing the call keyword arguments (line 413)
        kwargs_1355 = {}
        # Getting the type of 'len' (line 413)
        len_1353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 23), 'len', False)
        # Calling len(args, kwargs) (line 413)
        len_call_result_1356 = invoke(stypy.reporting.localization.Localization(__file__, 413, 23), len_1353, *[macro_1354], **kwargs_1355)
        
        int_1357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 37), 'int')
        # Applying the binary operator '==' (line 413)
        result_eq_1358 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 23), '==', len_call_result_1356, int_1357)
        
        # Testing the type of an if condition (line 413)
        if_condition_1359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 20), result_eq_1358)
        # Assigning a type to the variable 'if_condition_1359' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 20), 'if_condition_1359', if_condition_1359)
        # SSA begins for if statement (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 414)
        # Processing the call arguments (line 414)
        
        # Obtaining the type of the subscript
        int_1363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 54), 'int')
        # Getting the type of 'macro' (line 414)
        macro_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 48), 'macro', False)
        # Obtaining the member '__getitem__' of a type (line 414)
        getitem___1365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 48), macro_1364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 414)
        subscript_call_result_1366 = invoke(stypy.reporting.localization.Localization(__file__, 414, 48), getitem___1365, int_1363)
        
        # Processing the call keyword arguments (line 414)
        kwargs_1367 = {}
        # Getting the type of 'ext' (line 414)
        ext_1360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 24), 'ext', False)
        # Obtaining the member 'undef_macros' of a type (line 414)
        undef_macros_1361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 24), ext_1360, 'undef_macros')
        # Obtaining the member 'append' of a type (line 414)
        append_1362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 24), undef_macros_1361, 'append')
        # Calling append(args, kwargs) (line 414)
        append_call_result_1368 = invoke(stypy.reporting.localization.Localization(__file__, 414, 24), append_1362, *[subscript_call_result_1366], **kwargs_1367)
        
        # SSA branch for the else part of an if statement (line 413)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'macro' (line 415)
        macro_1370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 29), 'macro', False)
        # Processing the call keyword arguments (line 415)
        kwargs_1371 = {}
        # Getting the type of 'len' (line 415)
        len_1369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 25), 'len', False)
        # Calling len(args, kwargs) (line 415)
        len_call_result_1372 = invoke(stypy.reporting.localization.Localization(__file__, 415, 25), len_1369, *[macro_1370], **kwargs_1371)
        
        int_1373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 39), 'int')
        # Applying the binary operator '==' (line 415)
        result_eq_1374 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 25), '==', len_call_result_1372, int_1373)
        
        # Testing the type of an if condition (line 415)
        if_condition_1375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 25), result_eq_1374)
        # Assigning a type to the variable 'if_condition_1375' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 25), 'if_condition_1375', if_condition_1375)
        # SSA begins for if statement (line 415)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'macro' (line 416)
        macro_1379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 49), 'macro', False)
        # Processing the call keyword arguments (line 416)
        kwargs_1380 = {}
        # Getting the type of 'ext' (line 416)
        ext_1376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 24), 'ext', False)
        # Obtaining the member 'define_macros' of a type (line 416)
        define_macros_1377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 24), ext_1376, 'define_macros')
        # Obtaining the member 'append' of a type (line 416)
        append_1378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 24), define_macros_1377, 'append')
        # Calling append(args, kwargs) (line 416)
        append_call_result_1381 = invoke(stypy.reporting.localization.Localization(__file__, 416, 24), append_1378, *[macro_1379], **kwargs_1380)
        
        # SSA join for if statement (line 415)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 413)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 405)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 418):
        
        # Assigning a Name to a Subscript (line 418):
        # Getting the type of 'ext' (line 418)
        ext_1382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 28), 'ext')
        # Getting the type of 'extensions' (line 418)
        extensions_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'extensions')
        # Getting the type of 'i' (line 418)
        i_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'i')
        # Storing an element on a container (line 418)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 12), extensions_1383, (i_1384, ext_1382))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_extensions_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_extensions_list' in the type store
        # Getting the type of 'stypy_return_type' (line 342)
        stypy_return_type_1385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_extensions_list'
        return stypy_return_type_1385


    @norecursion
    def get_source_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_source_files'
        module_type_store = module_type_store.open_function_context('get_source_files', 420, 4, False)
        # Assigning a type to the variable 'self' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_source_files.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_source_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_source_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_source_files.__dict__.__setitem__('stypy_function_name', 'build_ext.get_source_files')
        build_ext.get_source_files.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.get_source_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_source_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_source_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_source_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_source_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_source_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_source_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_source_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_source_files(...)' code ##################

        
        # Call to check_extensions_list(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'self' (line 421)
        self_1388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 35), 'self', False)
        # Obtaining the member 'extensions' of a type (line 421)
        extensions_1389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 35), self_1388, 'extensions')
        # Processing the call keyword arguments (line 421)
        kwargs_1390 = {}
        # Getting the type of 'self' (line 421)
        self_1386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'self', False)
        # Obtaining the member 'check_extensions_list' of a type (line 421)
        check_extensions_list_1387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), self_1386, 'check_extensions_list')
        # Calling check_extensions_list(args, kwargs) (line 421)
        check_extensions_list_call_result_1391 = invoke(stypy.reporting.localization.Localization(__file__, 421, 8), check_extensions_list_1387, *[extensions_1389], **kwargs_1390)
        
        
        # Assigning a List to a Name (line 422):
        
        # Assigning a List to a Name (line 422):
        
        # Obtaining an instance of the builtin type 'list' (line 422)
        list_1392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 422)
        
        # Assigning a type to the variable 'filenames' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'filenames', list_1392)
        
        # Getting the type of 'self' (line 425)
        self_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 425)
        extensions_1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 19), self_1393, 'extensions')
        # Testing the type of a for loop iterable (line 425)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 425, 8), extensions_1394)
        # Getting the type of the for loop variable (line 425)
        for_loop_var_1395 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 425, 8), extensions_1394)
        # Assigning a type to the variable 'ext' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'ext', for_loop_var_1395)
        # SSA begins for a for statement (line 425)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extend(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'ext' (line 426)
        ext_1398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 29), 'ext', False)
        # Obtaining the member 'sources' of a type (line 426)
        sources_1399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 29), ext_1398, 'sources')
        # Processing the call keyword arguments (line 426)
        kwargs_1400 = {}
        # Getting the type of 'filenames' (line 426)
        filenames_1396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'filenames', False)
        # Obtaining the member 'extend' of a type (line 426)
        extend_1397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 12), filenames_1396, 'extend')
        # Calling extend(args, kwargs) (line 426)
        extend_call_result_1401 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), extend_1397, *[sources_1399], **kwargs_1400)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'filenames' (line 428)
        filenames_1402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'filenames')
        # Assigning a type to the variable 'stypy_return_type' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'stypy_return_type', filenames_1402)
        
        # ################# End of 'get_source_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_source_files' in the type store
        # Getting the type of 'stypy_return_type' (line 420)
        stypy_return_type_1403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1403)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_source_files'
        return stypy_return_type_1403


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 430, 4, False)
        # Assigning a type to the variable 'self' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_outputs.__dict__.__setitem__('stypy_function_name', 'build_ext.get_outputs')
        build_ext.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_outputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_outputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_outputs(...)' code ##################

        
        # Call to check_extensions_list(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'self' (line 434)
        self_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 35), 'self', False)
        # Obtaining the member 'extensions' of a type (line 434)
        extensions_1407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 35), self_1406, 'extensions')
        # Processing the call keyword arguments (line 434)
        kwargs_1408 = {}
        # Getting the type of 'self' (line 434)
        self_1404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'self', False)
        # Obtaining the member 'check_extensions_list' of a type (line 434)
        check_extensions_list_1405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), self_1404, 'check_extensions_list')
        # Calling check_extensions_list(args, kwargs) (line 434)
        check_extensions_list_call_result_1409 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), check_extensions_list_1405, *[extensions_1407], **kwargs_1408)
        
        
        # Assigning a List to a Name (line 439):
        
        # Assigning a List to a Name (line 439):
        
        # Obtaining an instance of the builtin type 'list' (line 439)
        list_1410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 439)
        
        # Assigning a type to the variable 'outputs' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'outputs', list_1410)
        
        # Getting the type of 'self' (line 440)
        self_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 440)
        extensions_1412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 19), self_1411, 'extensions')
        # Testing the type of a for loop iterable (line 440)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 440, 8), extensions_1412)
        # Getting the type of the for loop variable (line 440)
        for_loop_var_1413 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 440, 8), extensions_1412)
        # Assigning a type to the variable 'ext' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'ext', for_loop_var_1413)
        # SSA begins for a for statement (line 440)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 441)
        # Processing the call arguments (line 441)
        
        # Call to get_ext_fullpath(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'ext' (line 441)
        ext_1418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 49), 'ext', False)
        # Obtaining the member 'name' of a type (line 441)
        name_1419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 49), ext_1418, 'name')
        # Processing the call keyword arguments (line 441)
        kwargs_1420 = {}
        # Getting the type of 'self' (line 441)
        self_1416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 27), 'self', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 441)
        get_ext_fullpath_1417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 27), self_1416, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 441)
        get_ext_fullpath_call_result_1421 = invoke(stypy.reporting.localization.Localization(__file__, 441, 27), get_ext_fullpath_1417, *[name_1419], **kwargs_1420)
        
        # Processing the call keyword arguments (line 441)
        kwargs_1422 = {}
        # Getting the type of 'outputs' (line 441)
        outputs_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'outputs', False)
        # Obtaining the member 'append' of a type (line 441)
        append_1415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 12), outputs_1414, 'append')
        # Calling append(args, kwargs) (line 441)
        append_call_result_1423 = invoke(stypy.reporting.localization.Localization(__file__, 441, 12), append_1415, *[get_ext_fullpath_call_result_1421], **kwargs_1422)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'outputs' (line 442)
        outputs_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), 'outputs')
        # Assigning a type to the variable 'stypy_return_type' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'stypy_return_type', outputs_1424)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 430)
        stypy_return_type_1425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1425)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_1425


    @norecursion
    def build_extensions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_extensions'
        module_type_store = module_type_store.open_function_context('build_extensions', 444, 4, False)
        # Assigning a type to the variable 'self' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.build_extensions.__dict__.__setitem__('stypy_localization', localization)
        build_ext.build_extensions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.build_extensions.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.build_extensions.__dict__.__setitem__('stypy_function_name', 'build_ext.build_extensions')
        build_ext.build_extensions.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.build_extensions.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.build_extensions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.build_extensions.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.build_extensions.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.build_extensions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.build_extensions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.build_extensions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_extensions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_extensions(...)' code ##################

        
        # Call to check_extensions_list(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'self' (line 446)
        self_1428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 35), 'self', False)
        # Obtaining the member 'extensions' of a type (line 446)
        extensions_1429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 35), self_1428, 'extensions')
        # Processing the call keyword arguments (line 446)
        kwargs_1430 = {}
        # Getting the type of 'self' (line 446)
        self_1426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'self', False)
        # Obtaining the member 'check_extensions_list' of a type (line 446)
        check_extensions_list_1427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), self_1426, 'check_extensions_list')
        # Calling check_extensions_list(args, kwargs) (line 446)
        check_extensions_list_call_result_1431 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), check_extensions_list_1427, *[extensions_1429], **kwargs_1430)
        
        
        # Getting the type of 'self' (line 448)
        self_1432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 448)
        extensions_1433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 19), self_1432, 'extensions')
        # Testing the type of a for loop iterable (line 448)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 448, 8), extensions_1433)
        # Getting the type of the for loop variable (line 448)
        for_loop_var_1434 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 448, 8), extensions_1433)
        # Assigning a type to the variable 'ext' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'ext', for_loop_var_1434)
        # SSA begins for a for statement (line 448)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to build_extension(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'ext' (line 449)
        ext_1437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 33), 'ext', False)
        # Processing the call keyword arguments (line 449)
        kwargs_1438 = {}
        # Getting the type of 'self' (line 449)
        self_1435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'self', False)
        # Obtaining the member 'build_extension' of a type (line 449)
        build_extension_1436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 12), self_1435, 'build_extension')
        # Calling build_extension(args, kwargs) (line 449)
        build_extension_call_result_1439 = invoke(stypy.reporting.localization.Localization(__file__, 449, 12), build_extension_1436, *[ext_1437], **kwargs_1438)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'build_extensions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_extensions' in the type store
        # Getting the type of 'stypy_return_type' (line 444)
        stypy_return_type_1440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_extensions'
        return stypy_return_type_1440


    @norecursion
    def build_extension(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_extension'
        module_type_store = module_type_store.open_function_context('build_extension', 451, 4, False)
        # Assigning a type to the variable 'self' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.build_extension.__dict__.__setitem__('stypy_localization', localization)
        build_ext.build_extension.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.build_extension.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.build_extension.__dict__.__setitem__('stypy_function_name', 'build_ext.build_extension')
        build_ext.build_extension.__dict__.__setitem__('stypy_param_names_list', ['ext'])
        build_ext.build_extension.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.build_extension.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.build_extension.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.build_extension.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.build_extension.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.build_extension.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.build_extension', ['ext'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_extension', localization, ['ext'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_extension(...)' code ##################

        
        # Assigning a Attribute to a Name (line 452):
        
        # Assigning a Attribute to a Name (line 452):
        # Getting the type of 'ext' (line 452)
        ext_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 18), 'ext')
        # Obtaining the member 'sources' of a type (line 452)
        sources_1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 18), ext_1441, 'sources')
        # Assigning a type to the variable 'sources' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'sources', sources_1442)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sources' (line 453)
        sources_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'sources')
        # Getting the type of 'None' (line 453)
        None_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 22), 'None')
        # Applying the binary operator 'is' (line 453)
        result_is__1445 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 11), 'is', sources_1443, None_1444)
        
        
        
        # Call to type(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'sources' (line 453)
        sources_1447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 35), 'sources', False)
        # Processing the call keyword arguments (line 453)
        kwargs_1448 = {}
        # Getting the type of 'type' (line 453)
        type_1446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 30), 'type', False)
        # Calling type(args, kwargs) (line 453)
        type_call_result_1449 = invoke(stypy.reporting.localization.Localization(__file__, 453, 30), type_1446, *[sources_1447], **kwargs_1448)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 453)
        tuple_1450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 453)
        # Adding element type (line 453)
        # Getting the type of 'ListType' (line 453)
        ListType_1451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 52), 'ListType')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 52), tuple_1450, ListType_1451)
        # Adding element type (line 453)
        # Getting the type of 'TupleType' (line 453)
        TupleType_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 62), 'TupleType')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 52), tuple_1450, TupleType_1452)
        
        # Applying the binary operator 'notin' (line 453)
        result_contains_1453 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 30), 'notin', type_call_result_1449, tuple_1450)
        
        # Applying the binary operator 'or' (line 453)
        result_or_keyword_1454 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 11), 'or', result_is__1445, result_contains_1453)
        
        # Testing the type of an if condition (line 453)
        if_condition_1455 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 8), result_or_keyword_1454)
        # Assigning a type to the variable 'if_condition_1455' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'if_condition_1455', if_condition_1455)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsSetupError' (line 454)
        DistutilsSetupError_1456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 18), 'DistutilsSetupError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 454, 12), DistutilsSetupError_1456, 'raise parameter', BaseException)
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 458):
        
        # Assigning a Call to a Name (line 458):
        
        # Call to list(...): (line 458)
        # Processing the call arguments (line 458)
        # Getting the type of 'sources' (line 458)
        sources_1458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 23), 'sources', False)
        # Processing the call keyword arguments (line 458)
        kwargs_1459 = {}
        # Getting the type of 'list' (line 458)
        list_1457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 18), 'list', False)
        # Calling list(args, kwargs) (line 458)
        list_call_result_1460 = invoke(stypy.reporting.localization.Localization(__file__, 458, 18), list_1457, *[sources_1458], **kwargs_1459)
        
        # Assigning a type to the variable 'sources' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'sources', list_call_result_1460)
        
        # Assigning a Call to a Name (line 460):
        
        # Assigning a Call to a Name (line 460):
        
        # Call to get_ext_fullpath(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'ext' (line 460)
        ext_1463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 41), 'ext', False)
        # Obtaining the member 'name' of a type (line 460)
        name_1464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 41), ext_1463, 'name')
        # Processing the call keyword arguments (line 460)
        kwargs_1465 = {}
        # Getting the type of 'self' (line 460)
        self_1461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 19), 'self', False)
        # Obtaining the member 'get_ext_fullpath' of a type (line 460)
        get_ext_fullpath_1462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 19), self_1461, 'get_ext_fullpath')
        # Calling get_ext_fullpath(args, kwargs) (line 460)
        get_ext_fullpath_call_result_1466 = invoke(stypy.reporting.localization.Localization(__file__, 460, 19), get_ext_fullpath_1462, *[name_1464], **kwargs_1465)
        
        # Assigning a type to the variable 'ext_path' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'ext_path', get_ext_fullpath_call_result_1466)
        
        # Assigning a BinOp to a Name (line 461):
        
        # Assigning a BinOp to a Name (line 461):
        # Getting the type of 'sources' (line 461)
        sources_1467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 18), 'sources')
        # Getting the type of 'ext' (line 461)
        ext_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 28), 'ext')
        # Obtaining the member 'depends' of a type (line 461)
        depends_1469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 28), ext_1468, 'depends')
        # Applying the binary operator '+' (line 461)
        result_add_1470 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 18), '+', sources_1467, depends_1469)
        
        # Assigning a type to the variable 'depends' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'depends', result_add_1470)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 462)
        self_1471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 16), 'self')
        # Obtaining the member 'force' of a type (line 462)
        force_1472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 16), self_1471, 'force')
        
        # Call to newer_group(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'depends' (line 462)
        depends_1474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 42), 'depends', False)
        # Getting the type of 'ext_path' (line 462)
        ext_path_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 51), 'ext_path', False)
        str_1476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 61), 'str', 'newer')
        # Processing the call keyword arguments (line 462)
        kwargs_1477 = {}
        # Getting the type of 'newer_group' (line 462)
        newer_group_1473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 30), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 462)
        newer_group_call_result_1478 = invoke(stypy.reporting.localization.Localization(__file__, 462, 30), newer_group_1473, *[depends_1474, ext_path_1475, str_1476], **kwargs_1477)
        
        # Applying the binary operator 'or' (line 462)
        result_or_keyword_1479 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 16), 'or', force_1472, newer_group_call_result_1478)
        
        # Applying the 'not' unary operator (line 462)
        result_not__1480 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 11), 'not', result_or_keyword_1479)
        
        # Testing the type of an if condition (line 462)
        if_condition_1481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 8), result_not__1480)
        # Assigning a type to the variable 'if_condition_1481' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'if_condition_1481', if_condition_1481)
        # SSA begins for if statement (line 462)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug(...): (line 463)
        # Processing the call arguments (line 463)
        str_1484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 22), 'str', "skipping '%s' extension (up-to-date)")
        # Getting the type of 'ext' (line 463)
        ext_1485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 62), 'ext', False)
        # Obtaining the member 'name' of a type (line 463)
        name_1486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 62), ext_1485, 'name')
        # Processing the call keyword arguments (line 463)
        kwargs_1487 = {}
        # Getting the type of 'log' (line 463)
        log_1482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 463)
        debug_1483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 12), log_1482, 'debug')
        # Calling debug(args, kwargs) (line 463)
        debug_call_result_1488 = invoke(stypy.reporting.localization.Localization(__file__, 463, 12), debug_1483, *[str_1484, name_1486], **kwargs_1487)
        
        # Assigning a type to the variable 'stypy_return_type' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'stypy_return_type', types.NoneType)
        # SSA branch for the else part of an if statement (line 462)
        module_type_store.open_ssa_branch('else')
        
        # Call to info(...): (line 466)
        # Processing the call arguments (line 466)
        str_1491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 21), 'str', "building '%s' extension")
        # Getting the type of 'ext' (line 466)
        ext_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 48), 'ext', False)
        # Obtaining the member 'name' of a type (line 466)
        name_1493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 48), ext_1492, 'name')
        # Processing the call keyword arguments (line 466)
        kwargs_1494 = {}
        # Getting the type of 'log' (line 466)
        log_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 466)
        info_1490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), log_1489, 'info')
        # Calling info(args, kwargs) (line 466)
        info_call_result_1495 = invoke(stypy.reporting.localization.Localization(__file__, 466, 12), info_1490, *[str_1491, name_1493], **kwargs_1494)
        
        # SSA join for if statement (line 462)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Call to swig_sources(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'sources' (line 471)
        sources_1498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 36), 'sources', False)
        # Getting the type of 'ext' (line 471)
        ext_1499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 45), 'ext', False)
        # Processing the call keyword arguments (line 471)
        kwargs_1500 = {}
        # Getting the type of 'self' (line 471)
        self_1496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 18), 'self', False)
        # Obtaining the member 'swig_sources' of a type (line 471)
        swig_sources_1497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 18), self_1496, 'swig_sources')
        # Calling swig_sources(args, kwargs) (line 471)
        swig_sources_call_result_1501 = invoke(stypy.reporting.localization.Localization(__file__, 471, 18), swig_sources_1497, *[sources_1498, ext_1499], **kwargs_1500)
        
        # Assigning a type to the variable 'sources' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'sources', swig_sources_call_result_1501)
        
        # Assigning a BoolOp to a Name (line 487):
        
        # Assigning a BoolOp to a Name (line 487):
        
        # Evaluating a boolean operation
        # Getting the type of 'ext' (line 487)
        ext_1502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 21), 'ext')
        # Obtaining the member 'extra_compile_args' of a type (line 487)
        extra_compile_args_1503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 21), ext_1502, 'extra_compile_args')
        
        # Obtaining an instance of the builtin type 'list' (line 487)
        list_1504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 487)
        
        # Applying the binary operator 'or' (line 487)
        result_or_keyword_1505 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 21), 'or', extra_compile_args_1503, list_1504)
        
        # Assigning a type to the variable 'extra_args' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'extra_args', result_or_keyword_1505)
        
        # Assigning a Subscript to a Name (line 489):
        
        # Assigning a Subscript to a Name (line 489):
        
        # Obtaining the type of the subscript
        slice_1506 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 489, 17), None, None, None)
        # Getting the type of 'ext' (line 489)
        ext_1507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'ext')
        # Obtaining the member 'define_macros' of a type (line 489)
        define_macros_1508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 17), ext_1507, 'define_macros')
        # Obtaining the member '__getitem__' of a type (line 489)
        getitem___1509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 17), define_macros_1508, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 489)
        subscript_call_result_1510 = invoke(stypy.reporting.localization.Localization(__file__, 489, 17), getitem___1509, slice_1506)
        
        # Assigning a type to the variable 'macros' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'macros', subscript_call_result_1510)
        
        # Getting the type of 'ext' (line 490)
        ext_1511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 21), 'ext')
        # Obtaining the member 'undef_macros' of a type (line 490)
        undef_macros_1512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 21), ext_1511, 'undef_macros')
        # Testing the type of a for loop iterable (line 490)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 490, 8), undef_macros_1512)
        # Getting the type of the for loop variable (line 490)
        for_loop_var_1513 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 490, 8), undef_macros_1512)
        # Assigning a type to the variable 'undef' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'undef', for_loop_var_1513)
        # SSA begins for a for statement (line 490)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 491)
        # Processing the call arguments (line 491)
        
        # Obtaining an instance of the builtin type 'tuple' (line 491)
        tuple_1516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 491)
        # Adding element type (line 491)
        # Getting the type of 'undef' (line 491)
        undef_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 27), 'undef', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 27), tuple_1516, undef_1517)
        
        # Processing the call keyword arguments (line 491)
        kwargs_1518 = {}
        # Getting the type of 'macros' (line 491)
        macros_1514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'macros', False)
        # Obtaining the member 'append' of a type (line 491)
        append_1515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 12), macros_1514, 'append')
        # Calling append(args, kwargs) (line 491)
        append_call_result_1519 = invoke(stypy.reporting.localization.Localization(__file__, 491, 12), append_1515, *[tuple_1516], **kwargs_1518)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 493):
        
        # Assigning a Call to a Name (line 493):
        
        # Call to compile(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'sources' (line 493)
        sources_1523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 40), 'sources', False)
        # Processing the call keyword arguments (line 493)
        # Getting the type of 'self' (line 494)
        self_1524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 52), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 494)
        build_temp_1525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 52), self_1524, 'build_temp')
        keyword_1526 = build_temp_1525
        # Getting the type of 'macros' (line 495)
        macros_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 48), 'macros', False)
        keyword_1528 = macros_1527
        # Getting the type of 'ext' (line 496)
        ext_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 54), 'ext', False)
        # Obtaining the member 'include_dirs' of a type (line 496)
        include_dirs_1530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 54), ext_1529, 'include_dirs')
        keyword_1531 = include_dirs_1530
        # Getting the type of 'self' (line 497)
        self_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 47), 'self', False)
        # Obtaining the member 'debug' of a type (line 497)
        debug_1533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 47), self_1532, 'debug')
        keyword_1534 = debug_1533
        # Getting the type of 'extra_args' (line 498)
        extra_args_1535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 56), 'extra_args', False)
        keyword_1536 = extra_args_1535
        # Getting the type of 'ext' (line 499)
        ext_1537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 49), 'ext', False)
        # Obtaining the member 'depends' of a type (line 499)
        depends_1538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 49), ext_1537, 'depends')
        keyword_1539 = depends_1538
        kwargs_1540 = {'depends': keyword_1539, 'macros': keyword_1528, 'extra_postargs': keyword_1536, 'output_dir': keyword_1526, 'debug': keyword_1534, 'include_dirs': keyword_1531}
        # Getting the type of 'self' (line 493)
        self_1520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 18), 'self', False)
        # Obtaining the member 'compiler' of a type (line 493)
        compiler_1521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 18), self_1520, 'compiler')
        # Obtaining the member 'compile' of a type (line 493)
        compile_1522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 18), compiler_1521, 'compile')
        # Calling compile(args, kwargs) (line 493)
        compile_call_result_1541 = invoke(stypy.reporting.localization.Localization(__file__, 493, 18), compile_1522, *[sources_1523], **kwargs_1540)
        
        # Assigning a type to the variable 'objects' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'objects', compile_call_result_1541)
        
        # Assigning a Subscript to a Attribute (line 510):
        
        # Assigning a Subscript to a Attribute (line 510):
        
        # Obtaining the type of the subscript
        slice_1542 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 510, 30), None, None, None)
        # Getting the type of 'objects' (line 510)
        objects_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 30), 'objects')
        # Obtaining the member '__getitem__' of a type (line 510)
        getitem___1544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 30), objects_1543, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 510)
        subscript_call_result_1545 = invoke(stypy.reporting.localization.Localization(__file__, 510, 30), getitem___1544, slice_1542)
        
        # Getting the type of 'self' (line 510)
        self_1546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'self')
        # Setting the type of the member '_built_objects' of a type (line 510)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), self_1546, '_built_objects', subscript_call_result_1545)
        
        # Getting the type of 'ext' (line 515)
        ext_1547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 11), 'ext')
        # Obtaining the member 'extra_objects' of a type (line 515)
        extra_objects_1548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 11), ext_1547, 'extra_objects')
        # Testing the type of an if condition (line 515)
        if_condition_1549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 8), extra_objects_1548)
        # Assigning a type to the variable 'if_condition_1549' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'if_condition_1549', if_condition_1549)
        # SSA begins for if statement (line 515)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'ext' (line 516)
        ext_1552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 27), 'ext', False)
        # Obtaining the member 'extra_objects' of a type (line 516)
        extra_objects_1553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 27), ext_1552, 'extra_objects')
        # Processing the call keyword arguments (line 516)
        kwargs_1554 = {}
        # Getting the type of 'objects' (line 516)
        objects_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'objects', False)
        # Obtaining the member 'extend' of a type (line 516)
        extend_1551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), objects_1550, 'extend')
        # Calling extend(args, kwargs) (line 516)
        extend_call_result_1555 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), extend_1551, *[extra_objects_1553], **kwargs_1554)
        
        # SSA join for if statement (line 515)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 517):
        
        # Assigning a BoolOp to a Name (line 517):
        
        # Evaluating a boolean operation
        # Getting the type of 'ext' (line 517)
        ext_1556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 21), 'ext')
        # Obtaining the member 'extra_link_args' of a type (line 517)
        extra_link_args_1557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 21), ext_1556, 'extra_link_args')
        
        # Obtaining an instance of the builtin type 'list' (line 517)
        list_1558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 517)
        
        # Applying the binary operator 'or' (line 517)
        result_or_keyword_1559 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 21), 'or', extra_link_args_1557, list_1558)
        
        # Assigning a type to the variable 'extra_args' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'extra_args', result_or_keyword_1559)
        
        # Assigning a BoolOp to a Name (line 520):
        
        # Assigning a BoolOp to a Name (line 520):
        
        # Evaluating a boolean operation
        # Getting the type of 'ext' (line 520)
        ext_1560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 19), 'ext')
        # Obtaining the member 'language' of a type (line 520)
        language_1561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 19), ext_1560, 'language')
        
        # Call to detect_language(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'sources' (line 520)
        sources_1565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 65), 'sources', False)
        # Processing the call keyword arguments (line 520)
        kwargs_1566 = {}
        # Getting the type of 'self' (line 520)
        self_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 35), 'self', False)
        # Obtaining the member 'compiler' of a type (line 520)
        compiler_1563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 35), self_1562, 'compiler')
        # Obtaining the member 'detect_language' of a type (line 520)
        detect_language_1564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 35), compiler_1563, 'detect_language')
        # Calling detect_language(args, kwargs) (line 520)
        detect_language_call_result_1567 = invoke(stypy.reporting.localization.Localization(__file__, 520, 35), detect_language_1564, *[sources_1565], **kwargs_1566)
        
        # Applying the binary operator 'or' (line 520)
        result_or_keyword_1568 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 19), 'or', language_1561, detect_language_call_result_1567)
        
        # Assigning a type to the variable 'language' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'language', result_or_keyword_1568)
        
        # Call to link_shared_object(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'objects' (line 523)
        objects_1572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'objects', False)
        # Getting the type of 'ext_path' (line 523)
        ext_path_1573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 21), 'ext_path', False)
        # Processing the call keyword arguments (line 522)
        
        # Call to get_libraries(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'ext' (line 524)
        ext_1576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 41), 'ext', False)
        # Processing the call keyword arguments (line 524)
        kwargs_1577 = {}
        # Getting the type of 'self' (line 524)
        self_1574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 22), 'self', False)
        # Obtaining the member 'get_libraries' of a type (line 524)
        get_libraries_1575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 22), self_1574, 'get_libraries')
        # Calling get_libraries(args, kwargs) (line 524)
        get_libraries_call_result_1578 = invoke(stypy.reporting.localization.Localization(__file__, 524, 22), get_libraries_1575, *[ext_1576], **kwargs_1577)
        
        keyword_1579 = get_libraries_call_result_1578
        # Getting the type of 'ext' (line 525)
        ext_1580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'ext', False)
        # Obtaining the member 'library_dirs' of a type (line 525)
        library_dirs_1581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 25), ext_1580, 'library_dirs')
        keyword_1582 = library_dirs_1581
        # Getting the type of 'ext' (line 526)
        ext_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 33), 'ext', False)
        # Obtaining the member 'runtime_library_dirs' of a type (line 526)
        runtime_library_dirs_1584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 33), ext_1583, 'runtime_library_dirs')
        keyword_1585 = runtime_library_dirs_1584
        # Getting the type of 'extra_args' (line 527)
        extra_args_1586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 27), 'extra_args', False)
        keyword_1587 = extra_args_1586
        
        # Call to get_export_symbols(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'ext' (line 528)
        ext_1590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 51), 'ext', False)
        # Processing the call keyword arguments (line 528)
        kwargs_1591 = {}
        # Getting the type of 'self' (line 528)
        self_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 27), 'self', False)
        # Obtaining the member 'get_export_symbols' of a type (line 528)
        get_export_symbols_1589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 27), self_1588, 'get_export_symbols')
        # Calling get_export_symbols(args, kwargs) (line 528)
        get_export_symbols_call_result_1592 = invoke(stypy.reporting.localization.Localization(__file__, 528, 27), get_export_symbols_1589, *[ext_1590], **kwargs_1591)
        
        keyword_1593 = get_export_symbols_call_result_1592
        # Getting the type of 'self' (line 529)
        self_1594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 18), 'self', False)
        # Obtaining the member 'debug' of a type (line 529)
        debug_1595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 18), self_1594, 'debug')
        keyword_1596 = debug_1595
        # Getting the type of 'self' (line 530)
        self_1597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 23), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 530)
        build_temp_1598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 23), self_1597, 'build_temp')
        keyword_1599 = build_temp_1598
        # Getting the type of 'language' (line 531)
        language_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 24), 'language', False)
        keyword_1601 = language_1600
        kwargs_1602 = {'target_lang': keyword_1601, 'export_symbols': keyword_1593, 'runtime_library_dirs': keyword_1585, 'libraries': keyword_1579, 'extra_postargs': keyword_1587, 'debug': keyword_1596, 'build_temp': keyword_1599, 'library_dirs': keyword_1582}
        # Getting the type of 'self' (line 522)
        self_1569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 522)
        compiler_1570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 8), self_1569, 'compiler')
        # Obtaining the member 'link_shared_object' of a type (line 522)
        link_shared_object_1571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 8), compiler_1570, 'link_shared_object')
        # Calling link_shared_object(args, kwargs) (line 522)
        link_shared_object_call_result_1603 = invoke(stypy.reporting.localization.Localization(__file__, 522, 8), link_shared_object_1571, *[objects_1572, ext_path_1573], **kwargs_1602)
        
        
        # ################# End of 'build_extension(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_extension' in the type store
        # Getting the type of 'stypy_return_type' (line 451)
        stypy_return_type_1604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1604)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_extension'
        return stypy_return_type_1604


    @norecursion
    def swig_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'swig_sources'
        module_type_store = module_type_store.open_function_context('swig_sources', 534, 4, False)
        # Assigning a type to the variable 'self' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.swig_sources.__dict__.__setitem__('stypy_localization', localization)
        build_ext.swig_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.swig_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.swig_sources.__dict__.__setitem__('stypy_function_name', 'build_ext.swig_sources')
        build_ext.swig_sources.__dict__.__setitem__('stypy_param_names_list', ['sources', 'extension'])
        build_ext.swig_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.swig_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.swig_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.swig_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.swig_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.swig_sources.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.swig_sources', ['sources', 'extension'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'swig_sources', localization, ['sources', 'extension'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'swig_sources(...)' code ##################

        str_1605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, (-1)), 'str', "Walk the list of source files in 'sources', looking for SWIG\n        interface (.i) files.  Run SWIG on all that are found, and\n        return a modified 'sources' list with SWIG source files replaced\n        by the generated C (or C++) files.\n        ")
        
        # Assigning a List to a Name (line 542):
        
        # Assigning a List to a Name (line 542):
        
        # Obtaining an instance of the builtin type 'list' (line 542)
        list_1606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 542)
        
        # Assigning a type to the variable 'new_sources' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'new_sources', list_1606)
        
        # Assigning a List to a Name (line 543):
        
        # Assigning a List to a Name (line 543):
        
        # Obtaining an instance of the builtin type 'list' (line 543)
        list_1607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 543)
        
        # Assigning a type to the variable 'swig_sources' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'swig_sources', list_1607)
        
        # Assigning a Dict to a Name (line 544):
        
        # Assigning a Dict to a Name (line 544):
        
        # Obtaining an instance of the builtin type 'dict' (line 544)
        dict_1608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 544)
        
        # Assigning a type to the variable 'swig_targets' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'swig_targets', dict_1608)
        
        # Getting the type of 'self' (line 551)
        self_1609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 11), 'self')
        # Obtaining the member 'swig_cpp' of a type (line 551)
        swig_cpp_1610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 11), self_1609, 'swig_cpp')
        # Testing the type of an if condition (line 551)
        if_condition_1611 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 551, 8), swig_cpp_1610)
        # Assigning a type to the variable 'if_condition_1611' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'if_condition_1611', if_condition_1611)
        # SSA begins for if statement (line 551)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 552)
        # Processing the call arguments (line 552)
        str_1614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 21), 'str', '--swig-cpp is deprecated - use --swig-opts=-c++')
        # Processing the call keyword arguments (line 552)
        kwargs_1615 = {}
        # Getting the type of 'log' (line 552)
        log_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 12), 'log', False)
        # Obtaining the member 'warn' of a type (line 552)
        warn_1613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 12), log_1612, 'warn')
        # Calling warn(args, kwargs) (line 552)
        warn_call_result_1616 = invoke(stypy.reporting.localization.Localization(__file__, 552, 12), warn_1613, *[str_1614], **kwargs_1615)
        
        # SSA join for if statement (line 551)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 554)
        self_1617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 'self')
        # Obtaining the member 'swig_cpp' of a type (line 554)
        swig_cpp_1618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 11), self_1617, 'swig_cpp')
        
        str_1619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 29), 'str', '-c++')
        # Getting the type of 'self' (line 554)
        self_1620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 39), 'self')
        # Obtaining the member 'swig_opts' of a type (line 554)
        swig_opts_1621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 39), self_1620, 'swig_opts')
        # Applying the binary operator 'in' (line 554)
        result_contains_1622 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 29), 'in', str_1619, swig_opts_1621)
        
        # Applying the binary operator 'or' (line 554)
        result_or_keyword_1623 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 11), 'or', swig_cpp_1618, result_contains_1622)
        
        str_1624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 12), 'str', '-c++')
        # Getting the type of 'extension' (line 555)
        extension_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 22), 'extension')
        # Obtaining the member 'swig_opts' of a type (line 555)
        swig_opts_1626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 22), extension_1625, 'swig_opts')
        # Applying the binary operator 'in' (line 555)
        result_contains_1627 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 12), 'in', str_1624, swig_opts_1626)
        
        # Applying the binary operator 'or' (line 554)
        result_or_keyword_1628 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 11), 'or', result_or_keyword_1623, result_contains_1627)
        
        # Testing the type of an if condition (line 554)
        if_condition_1629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 8), result_or_keyword_1628)
        # Assigning a type to the variable 'if_condition_1629' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'if_condition_1629', if_condition_1629)
        # SSA begins for if statement (line 554)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 556):
        
        # Assigning a Str to a Name (line 556):
        str_1630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 25), 'str', '.cpp')
        # Assigning a type to the variable 'target_ext' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'target_ext', str_1630)
        # SSA branch for the else part of an if statement (line 554)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 558):
        
        # Assigning a Str to a Name (line 558):
        str_1631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 25), 'str', '.c')
        # Assigning a type to the variable 'target_ext' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'target_ext', str_1631)
        # SSA join for if statement (line 554)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'sources' (line 560)
        sources_1632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 22), 'sources')
        # Testing the type of a for loop iterable (line 560)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 560, 8), sources_1632)
        # Getting the type of the for loop variable (line 560)
        for_loop_var_1633 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 560, 8), sources_1632)
        # Assigning a type to the variable 'source' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'source', for_loop_var_1633)
        # SSA begins for a for statement (line 560)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 561):
        
        # Assigning a Subscript to a Name (line 561):
        
        # Obtaining the type of the subscript
        int_1634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 12), 'int')
        
        # Call to splitext(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'source' (line 561)
        source_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 43), 'source', False)
        # Processing the call keyword arguments (line 561)
        kwargs_1639 = {}
        # Getting the type of 'os' (line 561)
        os_1635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 561)
        path_1636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 26), os_1635, 'path')
        # Obtaining the member 'splitext' of a type (line 561)
        splitext_1637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 26), path_1636, 'splitext')
        # Calling splitext(args, kwargs) (line 561)
        splitext_call_result_1640 = invoke(stypy.reporting.localization.Localization(__file__, 561, 26), splitext_1637, *[source_1638], **kwargs_1639)
        
        # Obtaining the member '__getitem__' of a type (line 561)
        getitem___1641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 12), splitext_call_result_1640, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 561)
        subscript_call_result_1642 = invoke(stypy.reporting.localization.Localization(__file__, 561, 12), getitem___1641, int_1634)
        
        # Assigning a type to the variable 'tuple_var_assignment_463' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'tuple_var_assignment_463', subscript_call_result_1642)
        
        # Assigning a Subscript to a Name (line 561):
        
        # Obtaining the type of the subscript
        int_1643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 12), 'int')
        
        # Call to splitext(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'source' (line 561)
        source_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 43), 'source', False)
        # Processing the call keyword arguments (line 561)
        kwargs_1648 = {}
        # Getting the type of 'os' (line 561)
        os_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 561)
        path_1645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 26), os_1644, 'path')
        # Obtaining the member 'splitext' of a type (line 561)
        splitext_1646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 26), path_1645, 'splitext')
        # Calling splitext(args, kwargs) (line 561)
        splitext_call_result_1649 = invoke(stypy.reporting.localization.Localization(__file__, 561, 26), splitext_1646, *[source_1647], **kwargs_1648)
        
        # Obtaining the member '__getitem__' of a type (line 561)
        getitem___1650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 12), splitext_call_result_1649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 561)
        subscript_call_result_1651 = invoke(stypy.reporting.localization.Localization(__file__, 561, 12), getitem___1650, int_1643)
        
        # Assigning a type to the variable 'tuple_var_assignment_464' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'tuple_var_assignment_464', subscript_call_result_1651)
        
        # Assigning a Name to a Name (line 561):
        # Getting the type of 'tuple_var_assignment_463' (line 561)
        tuple_var_assignment_463_1652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'tuple_var_assignment_463')
        # Assigning a type to the variable 'base' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 13), 'base', tuple_var_assignment_463_1652)
        
        # Assigning a Name to a Name (line 561):
        # Getting the type of 'tuple_var_assignment_464' (line 561)
        tuple_var_assignment_464_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'tuple_var_assignment_464')
        # Assigning a type to the variable 'ext' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 19), 'ext', tuple_var_assignment_464_1653)
        
        
        # Getting the type of 'ext' (line 562)
        ext_1654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 15), 'ext')
        str_1655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 22), 'str', '.i')
        # Applying the binary operator '==' (line 562)
        result_eq_1656 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 15), '==', ext_1654, str_1655)
        
        # Testing the type of an if condition (line 562)
        if_condition_1657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 12), result_eq_1656)
        # Assigning a type to the variable 'if_condition_1657' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'if_condition_1657', if_condition_1657)
        # SSA begins for if statement (line 562)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'base' (line 563)
        base_1660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 35), 'base', False)
        str_1661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 42), 'str', '_wrap')
        # Applying the binary operator '+' (line 563)
        result_add_1662 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 35), '+', base_1660, str_1661)
        
        # Getting the type of 'target_ext' (line 563)
        target_ext_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 52), 'target_ext', False)
        # Applying the binary operator '+' (line 563)
        result_add_1664 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 50), '+', result_add_1662, target_ext_1663)
        
        # Processing the call keyword arguments (line 563)
        kwargs_1665 = {}
        # Getting the type of 'new_sources' (line 563)
        new_sources_1658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 563)
        append_1659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 16), new_sources_1658, 'append')
        # Calling append(args, kwargs) (line 563)
        append_call_result_1666 = invoke(stypy.reporting.localization.Localization(__file__, 563, 16), append_1659, *[result_add_1664], **kwargs_1665)
        
        
        # Call to append(...): (line 564)
        # Processing the call arguments (line 564)
        # Getting the type of 'source' (line 564)
        source_1669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 36), 'source', False)
        # Processing the call keyword arguments (line 564)
        kwargs_1670 = {}
        # Getting the type of 'swig_sources' (line 564)
        swig_sources_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 16), 'swig_sources', False)
        # Obtaining the member 'append' of a type (line 564)
        append_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 16), swig_sources_1667, 'append')
        # Calling append(args, kwargs) (line 564)
        append_call_result_1671 = invoke(stypy.reporting.localization.Localization(__file__, 564, 16), append_1668, *[source_1669], **kwargs_1670)
        
        
        # Assigning a Subscript to a Subscript (line 565):
        
        # Assigning a Subscript to a Subscript (line 565):
        
        # Obtaining the type of the subscript
        int_1672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 51), 'int')
        # Getting the type of 'new_sources' (line 565)
        new_sources_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 39), 'new_sources')
        # Obtaining the member '__getitem__' of a type (line 565)
        getitem___1674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 39), new_sources_1673, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 565)
        subscript_call_result_1675 = invoke(stypy.reporting.localization.Localization(__file__, 565, 39), getitem___1674, int_1672)
        
        # Getting the type of 'swig_targets' (line 565)
        swig_targets_1676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'swig_targets')
        # Getting the type of 'source' (line 565)
        source_1677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 29), 'source')
        # Storing an element on a container (line 565)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 16), swig_targets_1676, (source_1677, subscript_call_result_1675))
        # SSA branch for the else part of an if statement (line 562)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'source' (line 567)
        source_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 35), 'source', False)
        # Processing the call keyword arguments (line 567)
        kwargs_1681 = {}
        # Getting the type of 'new_sources' (line 567)
        new_sources_1678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 16), 'new_sources', False)
        # Obtaining the member 'append' of a type (line 567)
        append_1679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 16), new_sources_1678, 'append')
        # Calling append(args, kwargs) (line 567)
        append_call_result_1682 = invoke(stypy.reporting.localization.Localization(__file__, 567, 16), append_1679, *[source_1680], **kwargs_1681)
        
        # SSA join for if statement (line 562)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'swig_sources' (line 569)
        swig_sources_1683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 15), 'swig_sources')
        # Applying the 'not' unary operator (line 569)
        result_not__1684 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 11), 'not', swig_sources_1683)
        
        # Testing the type of an if condition (line 569)
        if_condition_1685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 569, 8), result_not__1684)
        # Assigning a type to the variable 'if_condition_1685' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'if_condition_1685', if_condition_1685)
        # SSA begins for if statement (line 569)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'new_sources' (line 570)
        new_sources_1686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 19), 'new_sources')
        # Assigning a type to the variable 'stypy_return_type' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'stypy_return_type', new_sources_1686)
        # SSA join for if statement (line 569)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 572):
        
        # Assigning a BoolOp to a Name (line 572):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 572)
        self_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'self')
        # Obtaining the member 'swig' of a type (line 572)
        swig_1688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 15), self_1687, 'swig')
        
        # Call to find_swig(...): (line 572)
        # Processing the call keyword arguments (line 572)
        kwargs_1691 = {}
        # Getting the type of 'self' (line 572)
        self_1689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 28), 'self', False)
        # Obtaining the member 'find_swig' of a type (line 572)
        find_swig_1690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 28), self_1689, 'find_swig')
        # Calling find_swig(args, kwargs) (line 572)
        find_swig_call_result_1692 = invoke(stypy.reporting.localization.Localization(__file__, 572, 28), find_swig_1690, *[], **kwargs_1691)
        
        # Applying the binary operator 'or' (line 572)
        result_or_keyword_1693 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 15), 'or', swig_1688, find_swig_call_result_1692)
        
        # Assigning a type to the variable 'swig' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'swig', result_or_keyword_1693)
        
        # Assigning a List to a Name (line 573):
        
        # Assigning a List to a Name (line 573):
        
        # Obtaining an instance of the builtin type 'list' (line 573)
        list_1694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 573)
        # Adding element type (line 573)
        # Getting the type of 'swig' (line 573)
        swig_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 20), 'swig')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 19), list_1694, swig_1695)
        # Adding element type (line 573)
        str_1696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 26), 'str', '-python')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 573, 19), list_1694, str_1696)
        
        # Assigning a type to the variable 'swig_cmd' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'swig_cmd', list_1694)
        
        # Call to extend(...): (line 574)
        # Processing the call arguments (line 574)
        # Getting the type of 'self' (line 574)
        self_1699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 24), 'self', False)
        # Obtaining the member 'swig_opts' of a type (line 574)
        swig_opts_1700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 24), self_1699, 'swig_opts')
        # Processing the call keyword arguments (line 574)
        kwargs_1701 = {}
        # Getting the type of 'swig_cmd' (line 574)
        swig_cmd_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'swig_cmd', False)
        # Obtaining the member 'extend' of a type (line 574)
        extend_1698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 8), swig_cmd_1697, 'extend')
        # Calling extend(args, kwargs) (line 574)
        extend_call_result_1702 = invoke(stypy.reporting.localization.Localization(__file__, 574, 8), extend_1698, *[swig_opts_1700], **kwargs_1701)
        
        
        # Getting the type of 'self' (line 575)
        self_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), 'self')
        # Obtaining the member 'swig_cpp' of a type (line 575)
        swig_cpp_1704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 11), self_1703, 'swig_cpp')
        # Testing the type of an if condition (line 575)
        if_condition_1705 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 8), swig_cpp_1704)
        # Assigning a type to the variable 'if_condition_1705' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'if_condition_1705', if_condition_1705)
        # SSA begins for if statement (line 575)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 576)
        # Processing the call arguments (line 576)
        str_1708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 28), 'str', '-c++')
        # Processing the call keyword arguments (line 576)
        kwargs_1709 = {}
        # Getting the type of 'swig_cmd' (line 576)
        swig_cmd_1706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'swig_cmd', False)
        # Obtaining the member 'append' of a type (line 576)
        append_1707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 12), swig_cmd_1706, 'append')
        # Calling append(args, kwargs) (line 576)
        append_call_result_1710 = invoke(stypy.reporting.localization.Localization(__file__, 576, 12), append_1707, *[str_1708], **kwargs_1709)
        
        # SSA join for if statement (line 575)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 579)
        self_1711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 15), 'self')
        # Obtaining the member 'swig_opts' of a type (line 579)
        swig_opts_1712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 15), self_1711, 'swig_opts')
        # Applying the 'not' unary operator (line 579)
        result_not__1713 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 11), 'not', swig_opts_1712)
        
        # Testing the type of an if condition (line 579)
        if_condition_1714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 8), result_not__1713)
        # Assigning a type to the variable 'if_condition_1714' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'if_condition_1714', if_condition_1714)
        # SSA begins for if statement (line 579)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'extension' (line 580)
        extension_1715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 21), 'extension')
        # Obtaining the member 'swig_opts' of a type (line 580)
        swig_opts_1716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 21), extension_1715, 'swig_opts')
        # Testing the type of a for loop iterable (line 580)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 580, 12), swig_opts_1716)
        # Getting the type of the for loop variable (line 580)
        for_loop_var_1717 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 580, 12), swig_opts_1716)
        # Assigning a type to the variable 'o' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'o', for_loop_var_1717)
        # SSA begins for a for statement (line 580)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'o' (line 581)
        o_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 32), 'o', False)
        # Processing the call keyword arguments (line 581)
        kwargs_1721 = {}
        # Getting the type of 'swig_cmd' (line 581)
        swig_cmd_1718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'swig_cmd', False)
        # Obtaining the member 'append' of a type (line 581)
        append_1719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 16), swig_cmd_1718, 'append')
        # Calling append(args, kwargs) (line 581)
        append_call_result_1722 = invoke(stypy.reporting.localization.Localization(__file__, 581, 16), append_1719, *[o_1720], **kwargs_1721)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 579)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'swig_sources' (line 583)
        swig_sources_1723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 22), 'swig_sources')
        # Testing the type of a for loop iterable (line 583)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 583, 8), swig_sources_1723)
        # Getting the type of the for loop variable (line 583)
        for_loop_var_1724 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 583, 8), swig_sources_1723)
        # Assigning a type to the variable 'source' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'source', for_loop_var_1724)
        # SSA begins for a for statement (line 583)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 584):
        
        # Assigning a Subscript to a Name (line 584):
        
        # Obtaining the type of the subscript
        # Getting the type of 'source' (line 584)
        source_1725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 34), 'source')
        # Getting the type of 'swig_targets' (line 584)
        swig_targets_1726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 21), 'swig_targets')
        # Obtaining the member '__getitem__' of a type (line 584)
        getitem___1727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 21), swig_targets_1726, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 584)
        subscript_call_result_1728 = invoke(stypy.reporting.localization.Localization(__file__, 584, 21), getitem___1727, source_1725)
        
        # Assigning a type to the variable 'target' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'target', subscript_call_result_1728)
        
        # Call to info(...): (line 585)
        # Processing the call arguments (line 585)
        str_1731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 21), 'str', 'swigging %s to %s')
        # Getting the type of 'source' (line 585)
        source_1732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 42), 'source', False)
        # Getting the type of 'target' (line 585)
        target_1733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 50), 'target', False)
        # Processing the call keyword arguments (line 585)
        kwargs_1734 = {}
        # Getting the type of 'log' (line 585)
        log_1729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 585)
        info_1730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 12), log_1729, 'info')
        # Calling info(args, kwargs) (line 585)
        info_call_result_1735 = invoke(stypy.reporting.localization.Localization(__file__, 585, 12), info_1730, *[str_1731, source_1732, target_1733], **kwargs_1734)
        
        
        # Call to spawn(...): (line 586)
        # Processing the call arguments (line 586)
        # Getting the type of 'swig_cmd' (line 586)
        swig_cmd_1738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 23), 'swig_cmd', False)
        
        # Obtaining an instance of the builtin type 'list' (line 586)
        list_1739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 586)
        # Adding element type (line 586)
        str_1740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 35), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 34), list_1739, str_1740)
        # Adding element type (line 586)
        # Getting the type of 'target' (line 586)
        target_1741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 41), 'target', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 34), list_1739, target_1741)
        # Adding element type (line 586)
        # Getting the type of 'source' (line 586)
        source_1742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 49), 'source', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 34), list_1739, source_1742)
        
        # Applying the binary operator '+' (line 586)
        result_add_1743 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 23), '+', swig_cmd_1738, list_1739)
        
        # Processing the call keyword arguments (line 586)
        kwargs_1744 = {}
        # Getting the type of 'self' (line 586)
        self_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'self', False)
        # Obtaining the member 'spawn' of a type (line 586)
        spawn_1737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 12), self_1736, 'spawn')
        # Calling spawn(args, kwargs) (line 586)
        spawn_call_result_1745 = invoke(stypy.reporting.localization.Localization(__file__, 586, 12), spawn_1737, *[result_add_1743], **kwargs_1744)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_sources' (line 588)
        new_sources_1746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 15), 'new_sources')
        # Assigning a type to the variable 'stypy_return_type' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'stypy_return_type', new_sources_1746)
        
        # ################# End of 'swig_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'swig_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 534)
        stypy_return_type_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'swig_sources'
        return stypy_return_type_1747


    @norecursion
    def find_swig(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_swig'
        module_type_store = module_type_store.open_function_context('find_swig', 592, 4, False)
        # Assigning a type to the variable 'self' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.find_swig.__dict__.__setitem__('stypy_localization', localization)
        build_ext.find_swig.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.find_swig.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.find_swig.__dict__.__setitem__('stypy_function_name', 'build_ext.find_swig')
        build_ext.find_swig.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.find_swig.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.find_swig.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.find_swig.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.find_swig.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.find_swig.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.find_swig.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.find_swig', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_swig', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_swig(...)' code ##################

        str_1748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, (-1)), 'str', 'Return the name of the SWIG executable.  On Unix, this is\n        just "swig" -- it should be in the PATH.  Tries a bit harder on\n        Windows.\n        ')
        
        
        # Getting the type of 'os' (line 598)
        os_1749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 11), 'os')
        # Obtaining the member 'name' of a type (line 598)
        name_1750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 11), os_1749, 'name')
        str_1751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 22), 'str', 'posix')
        # Applying the binary operator '==' (line 598)
        result_eq_1752 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 11), '==', name_1750, str_1751)
        
        # Testing the type of an if condition (line 598)
        if_condition_1753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 598, 8), result_eq_1752)
        # Assigning a type to the variable 'if_condition_1753' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'if_condition_1753', if_condition_1753)
        # SSA begins for if statement (line 598)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_1754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 19), 'str', 'swig')
        # Assigning a type to the variable 'stypy_return_type' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'stypy_return_type', str_1754)
        # SSA branch for the else part of an if statement (line 598)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'os' (line 600)
        os_1755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 13), 'os')
        # Obtaining the member 'name' of a type (line 600)
        name_1756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 13), os_1755, 'name')
        str_1757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 24), 'str', 'nt')
        # Applying the binary operator '==' (line 600)
        result_eq_1758 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 13), '==', name_1756, str_1757)
        
        # Testing the type of an if condition (line 600)
        if_condition_1759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 600, 13), result_eq_1758)
        # Assigning a type to the variable 'if_condition_1759' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 13), 'if_condition_1759', if_condition_1759)
        # SSA begins for if statement (line 600)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 605)
        tuple_1760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 605)
        # Adding element type (line 605)
        str_1761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 25), 'str', '1.3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 25), tuple_1760, str_1761)
        # Adding element type (line 605)
        str_1762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 32), 'str', '1.2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 25), tuple_1760, str_1762)
        # Adding element type (line 605)
        str_1763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 39), 'str', '1.1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 25), tuple_1760, str_1763)
        
        # Testing the type of a for loop iterable (line 605)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 605, 12), tuple_1760)
        # Getting the type of the for loop variable (line 605)
        for_loop_var_1764 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 605, 12), tuple_1760)
        # Assigning a type to the variable 'vers' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'vers', for_loop_var_1764)
        # SSA begins for a for statement (line 605)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 606):
        
        # Assigning a Call to a Name (line 606):
        
        # Call to join(...): (line 606)
        # Processing the call arguments (line 606)
        str_1768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 34), 'str', 'c:\\swig%s')
        # Getting the type of 'vers' (line 606)
        vers_1769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 49), 'vers', False)
        # Applying the binary operator '%' (line 606)
        result_mod_1770 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 34), '%', str_1768, vers_1769)
        
        str_1771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 55), 'str', 'swig.exe')
        # Processing the call keyword arguments (line 606)
        kwargs_1772 = {}
        # Getting the type of 'os' (line 606)
        os_1765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 606)
        path_1766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 21), os_1765, 'path')
        # Obtaining the member 'join' of a type (line 606)
        join_1767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 21), path_1766, 'join')
        # Calling join(args, kwargs) (line 606)
        join_call_result_1773 = invoke(stypy.reporting.localization.Localization(__file__, 606, 21), join_1767, *[result_mod_1770, str_1771], **kwargs_1772)
        
        # Assigning a type to the variable 'fn' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 16), 'fn', join_call_result_1773)
        
        
        # Call to isfile(...): (line 607)
        # Processing the call arguments (line 607)
        # Getting the type of 'fn' (line 607)
        fn_1777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 34), 'fn', False)
        # Processing the call keyword arguments (line 607)
        kwargs_1778 = {}
        # Getting the type of 'os' (line 607)
        os_1774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 607)
        path_1775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 19), os_1774, 'path')
        # Obtaining the member 'isfile' of a type (line 607)
        isfile_1776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 19), path_1775, 'isfile')
        # Calling isfile(args, kwargs) (line 607)
        isfile_call_result_1779 = invoke(stypy.reporting.localization.Localization(__file__, 607, 19), isfile_1776, *[fn_1777], **kwargs_1778)
        
        # Testing the type of an if condition (line 607)
        if_condition_1780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 607, 16), isfile_call_result_1779)
        # Assigning a type to the variable 'if_condition_1780' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'if_condition_1780', if_condition_1780)
        # SSA begins for if statement (line 607)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'fn' (line 608)
        fn_1781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 27), 'fn')
        # Assigning a type to the variable 'stypy_return_type' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 20), 'stypy_return_type', fn_1781)
        # SSA join for if statement (line 607)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 605)
        module_type_store.open_ssa_branch('for loop else')
        str_1782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 23), 'str', 'swig.exe')
        # Assigning a type to the variable 'stypy_return_type' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 16), 'stypy_return_type', str_1782)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 600)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'os' (line 612)
        os_1783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 13), 'os')
        # Obtaining the member 'name' of a type (line 612)
        name_1784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 13), os_1783, 'name')
        str_1785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 24), 'str', 'os2')
        # Applying the binary operator '==' (line 612)
        result_eq_1786 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 13), '==', name_1784, str_1785)
        
        # Testing the type of an if condition (line 612)
        if_condition_1787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 13), result_eq_1786)
        # Assigning a type to the variable 'if_condition_1787' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 13), 'if_condition_1787', if_condition_1787)
        # SSA begins for if statement (line 612)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_1788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 19), 'str', 'swig.exe')
        # Assigning a type to the variable 'stypy_return_type' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'stypy_return_type', str_1788)
        # SSA branch for the else part of an if statement (line 612)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'DistutilsPlatformError' (line 617)
        DistutilsPlatformError_1789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 18), 'DistutilsPlatformError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 617, 12), DistutilsPlatformError_1789, 'raise parameter', BaseException)
        # SSA join for if statement (line 612)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 600)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 598)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'find_swig(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_swig' in the type store
        # Getting the type of 'stypy_return_type' (line 592)
        stypy_return_type_1790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1790)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_swig'
        return stypy_return_type_1790


    @norecursion
    def get_ext_fullpath(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_ext_fullpath'
        module_type_store = module_type_store.open_function_context('get_ext_fullpath', 625, 4, False)
        # Assigning a type to the variable 'self' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_function_name', 'build_ext.get_ext_fullpath')
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_param_names_list', ['ext_name'])
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_ext_fullpath.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_ext_fullpath', ['ext_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_ext_fullpath', localization, ['ext_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_ext_fullpath(...)' code ##################

        str_1791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, (-1)), 'str', 'Returns the path of the filename for a given extension.\n\n        The file is located in `build_lib` or directly in the package\n        (inplace option).\n        ')
        
        # Assigning a Call to a Name (line 632):
        
        # Assigning a Call to a Name (line 632):
        
        # Call to maketrans(...): (line 632)
        # Processing the call arguments (line 632)
        str_1794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 36), 'str', '/')
        # Getting the type of 'os' (line 632)
        os_1795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 40), 'os', False)
        # Obtaining the member 'sep' of a type (line 632)
        sep_1796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 40), os_1795, 'sep')
        # Applying the binary operator '+' (line 632)
        result_add_1797 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 36), '+', str_1794, sep_1796)
        
        str_1798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 48), 'str', '..')
        # Processing the call keyword arguments (line 632)
        kwargs_1799 = {}
        # Getting the type of 'string' (line 632)
        string_1792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 19), 'string', False)
        # Obtaining the member 'maketrans' of a type (line 632)
        maketrans_1793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 19), string_1792, 'maketrans')
        # Calling maketrans(args, kwargs) (line 632)
        maketrans_call_result_1800 = invoke(stypy.reporting.localization.Localization(__file__, 632, 19), maketrans_1793, *[result_add_1797, str_1798], **kwargs_1799)
        
        # Assigning a type to the variable 'all_dots' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'all_dots', maketrans_call_result_1800)
        
        # Assigning a Call to a Name (line 633):
        
        # Assigning a Call to a Name (line 633):
        
        # Call to translate(...): (line 633)
        # Processing the call arguments (line 633)
        # Getting the type of 'all_dots' (line 633)
        all_dots_1803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 38), 'all_dots', False)
        # Processing the call keyword arguments (line 633)
        kwargs_1804 = {}
        # Getting the type of 'ext_name' (line 633)
        ext_name_1801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 19), 'ext_name', False)
        # Obtaining the member 'translate' of a type (line 633)
        translate_1802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 19), ext_name_1801, 'translate')
        # Calling translate(args, kwargs) (line 633)
        translate_call_result_1805 = invoke(stypy.reporting.localization.Localization(__file__, 633, 19), translate_1802, *[all_dots_1803], **kwargs_1804)
        
        # Assigning a type to the variable 'ext_name' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'ext_name', translate_call_result_1805)
        
        # Assigning a Call to a Name (line 635):
        
        # Assigning a Call to a Name (line 635):
        
        # Call to get_ext_fullname(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'ext_name' (line 635)
        ext_name_1808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 41), 'ext_name', False)
        # Processing the call keyword arguments (line 635)
        kwargs_1809 = {}
        # Getting the type of 'self' (line 635)
        self_1806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 19), 'self', False)
        # Obtaining the member 'get_ext_fullname' of a type (line 635)
        get_ext_fullname_1807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 19), self_1806, 'get_ext_fullname')
        # Calling get_ext_fullname(args, kwargs) (line 635)
        get_ext_fullname_call_result_1810 = invoke(stypy.reporting.localization.Localization(__file__, 635, 19), get_ext_fullname_1807, *[ext_name_1808], **kwargs_1809)
        
        # Assigning a type to the variable 'fullname' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'fullname', get_ext_fullname_call_result_1810)
        
        # Assigning a Call to a Name (line 636):
        
        # Assigning a Call to a Name (line 636):
        
        # Call to split(...): (line 636)
        # Processing the call arguments (line 636)
        str_1813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 33), 'str', '.')
        # Processing the call keyword arguments (line 636)
        kwargs_1814 = {}
        # Getting the type of 'fullname' (line 636)
        fullname_1811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 18), 'fullname', False)
        # Obtaining the member 'split' of a type (line 636)
        split_1812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 18), fullname_1811, 'split')
        # Calling split(args, kwargs) (line 636)
        split_call_result_1815 = invoke(stypy.reporting.localization.Localization(__file__, 636, 18), split_1812, *[str_1813], **kwargs_1814)
        
        # Assigning a type to the variable 'modpath' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'modpath', split_call_result_1815)
        
        # Assigning a Call to a Name (line 637):
        
        # Assigning a Call to a Name (line 637):
        
        # Call to get_ext_filename(...): (line 637)
        # Processing the call arguments (line 637)
        # Getting the type of 'ext_name' (line 637)
        ext_name_1818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 41), 'ext_name', False)
        # Processing the call keyword arguments (line 637)
        kwargs_1819 = {}
        # Getting the type of 'self' (line 637)
        self_1816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 19), 'self', False)
        # Obtaining the member 'get_ext_filename' of a type (line 637)
        get_ext_filename_1817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 19), self_1816, 'get_ext_filename')
        # Calling get_ext_filename(args, kwargs) (line 637)
        get_ext_filename_call_result_1820 = invoke(stypy.reporting.localization.Localization(__file__, 637, 19), get_ext_filename_1817, *[ext_name_1818], **kwargs_1819)
        
        # Assigning a type to the variable 'filename' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'filename', get_ext_filename_call_result_1820)
        
        # Assigning a Subscript to a Name (line 638):
        
        # Assigning a Subscript to a Name (line 638):
        
        # Obtaining the type of the subscript
        int_1821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 43), 'int')
        
        # Call to split(...): (line 638)
        # Processing the call arguments (line 638)
        # Getting the type of 'filename' (line 638)
        filename_1825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 33), 'filename', False)
        # Processing the call keyword arguments (line 638)
        kwargs_1826 = {}
        # Getting the type of 'os' (line 638)
        os_1822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 638)
        path_1823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 19), os_1822, 'path')
        # Obtaining the member 'split' of a type (line 638)
        split_1824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 19), path_1823, 'split')
        # Calling split(args, kwargs) (line 638)
        split_call_result_1827 = invoke(stypy.reporting.localization.Localization(__file__, 638, 19), split_1824, *[filename_1825], **kwargs_1826)
        
        # Obtaining the member '__getitem__' of a type (line 638)
        getitem___1828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 19), split_call_result_1827, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 638)
        subscript_call_result_1829 = invoke(stypy.reporting.localization.Localization(__file__, 638, 19), getitem___1828, int_1821)
        
        # Assigning a type to the variable 'filename' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'filename', subscript_call_result_1829)
        
        
        # Getting the type of 'self' (line 640)
        self_1830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 15), 'self')
        # Obtaining the member 'inplace' of a type (line 640)
        inplace_1831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 15), self_1830, 'inplace')
        # Applying the 'not' unary operator (line 640)
        result_not__1832 = python_operator(stypy.reporting.localization.Localization(__file__, 640, 11), 'not', inplace_1831)
        
        # Testing the type of an if condition (line 640)
        if_condition_1833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 640, 8), result_not__1832)
        # Assigning a type to the variable 'if_condition_1833' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'if_condition_1833', if_condition_1833)
        # SSA begins for if statement (line 640)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 644):
        
        # Assigning a Call to a Name (line 644):
        
        # Call to join(...): (line 644)
        
        # Obtaining the type of the subscript
        int_1837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 46), 'int')
        slice_1838 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 644, 37), None, int_1837, None)
        # Getting the type of 'modpath' (line 644)
        modpath_1839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 37), 'modpath', False)
        # Obtaining the member '__getitem__' of a type (line 644)
        getitem___1840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 37), modpath_1839, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 644)
        subscript_call_result_1841 = invoke(stypy.reporting.localization.Localization(__file__, 644, 37), getitem___1840, slice_1838)
        
        
        # Obtaining an instance of the builtin type 'list' (line 644)
        list_1842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 644)
        # Adding element type (line 644)
        # Getting the type of 'filename' (line 644)
        filename_1843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 51), 'filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 50), list_1842, filename_1843)
        
        # Applying the binary operator '+' (line 644)
        result_add_1844 = python_operator(stypy.reporting.localization.Localization(__file__, 644, 37), '+', subscript_call_result_1841, list_1842)
        
        # Processing the call keyword arguments (line 644)
        kwargs_1845 = {}
        # Getting the type of 'os' (line 644)
        os_1834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 644)
        path_1835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 23), os_1834, 'path')
        # Obtaining the member 'join' of a type (line 644)
        join_1836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 23), path_1835, 'join')
        # Calling join(args, kwargs) (line 644)
        join_call_result_1846 = invoke(stypy.reporting.localization.Localization(__file__, 644, 23), join_1836, *[result_add_1844], **kwargs_1845)
        
        # Assigning a type to the variable 'filename' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'filename', join_call_result_1846)
        
        # Call to join(...): (line 645)
        # Processing the call arguments (line 645)
        # Getting the type of 'self' (line 645)
        self_1850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 32), 'self', False)
        # Obtaining the member 'build_lib' of a type (line 645)
        build_lib_1851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 32), self_1850, 'build_lib')
        # Getting the type of 'filename' (line 645)
        filename_1852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 48), 'filename', False)
        # Processing the call keyword arguments (line 645)
        kwargs_1853 = {}
        # Getting the type of 'os' (line 645)
        os_1847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 645)
        path_1848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 19), os_1847, 'path')
        # Obtaining the member 'join' of a type (line 645)
        join_1849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 19), path_1848, 'join')
        # Calling join(args, kwargs) (line 645)
        join_call_result_1854 = invoke(stypy.reporting.localization.Localization(__file__, 645, 19), join_1849, *[build_lib_1851, filename_1852], **kwargs_1853)
        
        # Assigning a type to the variable 'stypy_return_type' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 12), 'stypy_return_type', join_call_result_1854)
        # SSA join for if statement (line 640)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 649):
        
        # Assigning a Call to a Name (line 649):
        
        # Call to join(...): (line 649)
        # Processing the call arguments (line 649)
        
        # Obtaining the type of the subscript
        int_1857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 35), 'int')
        int_1858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 37), 'int')
        slice_1859 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 649, 27), int_1857, int_1858, None)
        # Getting the type of 'modpath' (line 649)
        modpath_1860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 27), 'modpath', False)
        # Obtaining the member '__getitem__' of a type (line 649)
        getitem___1861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 27), modpath_1860, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 649)
        subscript_call_result_1862 = invoke(stypy.reporting.localization.Localization(__file__, 649, 27), getitem___1861, slice_1859)
        
        # Processing the call keyword arguments (line 649)
        kwargs_1863 = {}
        str_1855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 18), 'str', '.')
        # Obtaining the member 'join' of a type (line 649)
        join_1856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 18), str_1855, 'join')
        # Calling join(args, kwargs) (line 649)
        join_call_result_1864 = invoke(stypy.reporting.localization.Localization(__file__, 649, 18), join_1856, *[subscript_call_result_1862], **kwargs_1863)
        
        # Assigning a type to the variable 'package' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'package', join_call_result_1864)
        
        # Assigning a Call to a Name (line 650):
        
        # Assigning a Call to a Name (line 650):
        
        # Call to get_finalized_command(...): (line 650)
        # Processing the call arguments (line 650)
        str_1867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 46), 'str', 'build_py')
        # Processing the call keyword arguments (line 650)
        kwargs_1868 = {}
        # Getting the type of 'self' (line 650)
        self_1865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 19), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 650)
        get_finalized_command_1866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 19), self_1865, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 650)
        get_finalized_command_call_result_1869 = invoke(stypy.reporting.localization.Localization(__file__, 650, 19), get_finalized_command_1866, *[str_1867], **kwargs_1868)
        
        # Assigning a type to the variable 'build_py' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'build_py', get_finalized_command_call_result_1869)
        
        # Assigning a Call to a Name (line 651):
        
        # Assigning a Call to a Name (line 651):
        
        # Call to abspath(...): (line 651)
        # Processing the call arguments (line 651)
        
        # Call to get_package_dir(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'package' (line 651)
        package_1875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 63), 'package', False)
        # Processing the call keyword arguments (line 651)
        kwargs_1876 = {}
        # Getting the type of 'build_py' (line 651)
        build_py_1873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 38), 'build_py', False)
        # Obtaining the member 'get_package_dir' of a type (line 651)
        get_package_dir_1874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 38), build_py_1873, 'get_package_dir')
        # Calling get_package_dir(args, kwargs) (line 651)
        get_package_dir_call_result_1877 = invoke(stypy.reporting.localization.Localization(__file__, 651, 38), get_package_dir_1874, *[package_1875], **kwargs_1876)
        
        # Processing the call keyword arguments (line 651)
        kwargs_1878 = {}
        # Getting the type of 'os' (line 651)
        os_1870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 651)
        path_1871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 22), os_1870, 'path')
        # Obtaining the member 'abspath' of a type (line 651)
        abspath_1872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 22), path_1871, 'abspath')
        # Calling abspath(args, kwargs) (line 651)
        abspath_call_result_1879 = invoke(stypy.reporting.localization.Localization(__file__, 651, 22), abspath_1872, *[get_package_dir_call_result_1877], **kwargs_1878)
        
        # Assigning a type to the variable 'package_dir' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'package_dir', abspath_call_result_1879)
        
        # Call to join(...): (line 655)
        # Processing the call arguments (line 655)
        # Getting the type of 'package_dir' (line 655)
        package_dir_1883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 28), 'package_dir', False)
        # Getting the type of 'filename' (line 655)
        filename_1884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 41), 'filename', False)
        # Processing the call keyword arguments (line 655)
        kwargs_1885 = {}
        # Getting the type of 'os' (line 655)
        os_1880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 655)
        path_1881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 15), os_1880, 'path')
        # Obtaining the member 'join' of a type (line 655)
        join_1882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 15), path_1881, 'join')
        # Calling join(args, kwargs) (line 655)
        join_call_result_1886 = invoke(stypy.reporting.localization.Localization(__file__, 655, 15), join_1882, *[package_dir_1883, filename_1884], **kwargs_1885)
        
        # Assigning a type to the variable 'stypy_return_type' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'stypy_return_type', join_call_result_1886)
        
        # ################# End of 'get_ext_fullpath(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_ext_fullpath' in the type store
        # Getting the type of 'stypy_return_type' (line 625)
        stypy_return_type_1887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_ext_fullpath'
        return stypy_return_type_1887


    @norecursion
    def get_ext_fullname(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_ext_fullname'
        module_type_store = module_type_store.open_function_context('get_ext_fullname', 657, 4, False)
        # Assigning a type to the variable 'self' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_function_name', 'build_ext.get_ext_fullname')
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_param_names_list', ['ext_name'])
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_ext_fullname.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_ext_fullname', ['ext_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_ext_fullname', localization, ['ext_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_ext_fullname(...)' code ##################

        str_1888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, (-1)), 'str', 'Returns the fullname of a given extension name.\n\n        Adds the `package.` prefix')
        
        # Type idiom detected: calculating its left and rigth part (line 661)
        # Getting the type of 'self' (line 661)
        self_1889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 11), 'self')
        # Obtaining the member 'package' of a type (line 661)
        package_1890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 11), self_1889, 'package')
        # Getting the type of 'None' (line 661)
        None_1891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 27), 'None')
        
        (may_be_1892, more_types_in_union_1893) = may_be_none(package_1890, None_1891)

        if may_be_1892:

            if more_types_in_union_1893:
                # Runtime conditional SSA (line 661)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'ext_name' (line 662)
            ext_name_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 19), 'ext_name')
            # Assigning a type to the variable 'stypy_return_type' (line 662)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'stypy_return_type', ext_name_1894)

            if more_types_in_union_1893:
                # Runtime conditional SSA for else branch (line 661)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_1892) or more_types_in_union_1893):
            # Getting the type of 'self' (line 664)
            self_1895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 19), 'self')
            # Obtaining the member 'package' of a type (line 664)
            package_1896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 19), self_1895, 'package')
            str_1897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 34), 'str', '.')
            # Applying the binary operator '+' (line 664)
            result_add_1898 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 19), '+', package_1896, str_1897)
            
            # Getting the type of 'ext_name' (line 664)
            ext_name_1899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 40), 'ext_name')
            # Applying the binary operator '+' (line 664)
            result_add_1900 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 38), '+', result_add_1898, ext_name_1899)
            
            # Assigning a type to the variable 'stypy_return_type' (line 664)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'stypy_return_type', result_add_1900)

            if (may_be_1892 and more_types_in_union_1893):
                # SSA join for if statement (line 661)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_ext_fullname(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_ext_fullname' in the type store
        # Getting the type of 'stypy_return_type' (line 657)
        stypy_return_type_1901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1901)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_ext_fullname'
        return stypy_return_type_1901


    @norecursion
    def get_ext_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_ext_filename'
        module_type_store = module_type_store.open_function_context('get_ext_filename', 666, 4, False)
        # Assigning a type to the variable 'self' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_function_name', 'build_ext.get_ext_filename')
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_param_names_list', ['ext_name'])
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_ext_filename', ['ext_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_ext_filename', localization, ['ext_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_ext_filename(...)' code ##################

        str_1902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, (-1)), 'str', 'Convert the name of an extension (eg. "foo.bar") into the name\n        of the file from which it will be loaded (eg. "foo/bar.so", or\n        "foo\\bar.pyd").\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 671, 8))
        
        # 'from distutils.sysconfig import get_config_var' statement (line 671)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_1903 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 671, 8), 'distutils.sysconfig')

        if (type(import_1903) is not StypyTypeError):

            if (import_1903 != 'pyd_module'):
                __import__(import_1903)
                sys_modules_1904 = sys.modules[import_1903]
                import_from_module(stypy.reporting.localization.Localization(__file__, 671, 8), 'distutils.sysconfig', sys_modules_1904.module_type_store, module_type_store, ['get_config_var'])
                nest_module(stypy.reporting.localization.Localization(__file__, 671, 8), __file__, sys_modules_1904, sys_modules_1904.module_type_store, module_type_store)
            else:
                from distutils.sysconfig import get_config_var

                import_from_module(stypy.reporting.localization.Localization(__file__, 671, 8), 'distutils.sysconfig', None, module_type_store, ['get_config_var'], [get_config_var])

        else:
            # Assigning a type to the variable 'distutils.sysconfig' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'distutils.sysconfig', import_1903)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        # Assigning a Call to a Name (line 672):
        
        # Assigning a Call to a Name (line 672):
        
        # Call to split(...): (line 672)
        # Processing the call arguments (line 672)
        # Getting the type of 'ext_name' (line 672)
        ext_name_1907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 32), 'ext_name', False)
        str_1908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 42), 'str', '.')
        # Processing the call keyword arguments (line 672)
        kwargs_1909 = {}
        # Getting the type of 'string' (line 672)
        string_1905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 19), 'string', False)
        # Obtaining the member 'split' of a type (line 672)
        split_1906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 19), string_1905, 'split')
        # Calling split(args, kwargs) (line 672)
        split_call_result_1910 = invoke(stypy.reporting.localization.Localization(__file__, 672, 19), split_1906, *[ext_name_1907, str_1908], **kwargs_1909)
        
        # Assigning a type to the variable 'ext_path' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'ext_path', split_call_result_1910)
        
        
        # Getting the type of 'os' (line 674)
        os_1911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 11), 'os')
        # Obtaining the member 'name' of a type (line 674)
        name_1912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 11), os_1911, 'name')
        str_1913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 22), 'str', 'os2')
        # Applying the binary operator '==' (line 674)
        result_eq_1914 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 11), '==', name_1912, str_1913)
        
        # Testing the type of an if condition (line 674)
        if_condition_1915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 674, 8), result_eq_1914)
        # Assigning a type to the variable 'if_condition_1915' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'if_condition_1915', if_condition_1915)
        # SSA begins for if statement (line 674)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 675):
        
        # Assigning a Subscript to a Subscript (line 675):
        
        # Obtaining the type of the subscript
        int_1916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 71), 'int')
        slice_1917 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 675, 42), None, int_1916, None)
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'ext_path' (line 675)
        ext_path_1919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 55), 'ext_path', False)
        # Processing the call keyword arguments (line 675)
        kwargs_1920 = {}
        # Getting the type of 'len' (line 675)
        len_1918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 51), 'len', False)
        # Calling len(args, kwargs) (line 675)
        len_call_result_1921 = invoke(stypy.reporting.localization.Localization(__file__, 675, 51), len_1918, *[ext_path_1919], **kwargs_1920)
        
        int_1922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 67), 'int')
        # Applying the binary operator '-' (line 675)
        result_sub_1923 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 51), '-', len_call_result_1921, int_1922)
        
        # Getting the type of 'ext_path' (line 675)
        ext_path_1924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 42), 'ext_path')
        # Obtaining the member '__getitem__' of a type (line 675)
        getitem___1925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 42), ext_path_1924, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 675)
        subscript_call_result_1926 = invoke(stypy.reporting.localization.Localization(__file__, 675, 42), getitem___1925, result_sub_1923)
        
        # Obtaining the member '__getitem__' of a type (line 675)
        getitem___1927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 42), subscript_call_result_1926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 675)
        subscript_call_result_1928 = invoke(stypy.reporting.localization.Localization(__file__, 675, 42), getitem___1927, slice_1917)
        
        # Getting the type of 'ext_path' (line 675)
        ext_path_1929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 12), 'ext_path')
        
        # Call to len(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'ext_path' (line 675)
        ext_path_1931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 25), 'ext_path', False)
        # Processing the call keyword arguments (line 675)
        kwargs_1932 = {}
        # Getting the type of 'len' (line 675)
        len_1930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 21), 'len', False)
        # Calling len(args, kwargs) (line 675)
        len_call_result_1933 = invoke(stypy.reporting.localization.Localization(__file__, 675, 21), len_1930, *[ext_path_1931], **kwargs_1932)
        
        int_1934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 37), 'int')
        # Applying the binary operator '-' (line 675)
        result_sub_1935 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 21), '-', len_call_result_1933, int_1934)
        
        # Storing an element on a container (line 675)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 12), ext_path_1929, (result_sub_1935, subscript_call_result_1928))
        # SSA join for if statement (line 674)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 677):
        
        # Assigning a Call to a Name (line 677):
        
        # Call to get_config_var(...): (line 677)
        # Processing the call arguments (line 677)
        str_1937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 32), 'str', 'SO')
        # Processing the call keyword arguments (line 677)
        kwargs_1938 = {}
        # Getting the type of 'get_config_var' (line 677)
        get_config_var_1936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 17), 'get_config_var', False)
        # Calling get_config_var(args, kwargs) (line 677)
        get_config_var_call_result_1939 = invoke(stypy.reporting.localization.Localization(__file__, 677, 17), get_config_var_1936, *[str_1937], **kwargs_1938)
        
        # Assigning a type to the variable 'so_ext' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'so_ext', get_config_var_call_result_1939)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'os' (line 678)
        os_1940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 11), 'os')
        # Obtaining the member 'name' of a type (line 678)
        name_1941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 11), os_1940, 'name')
        str_1942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 22), 'str', 'nt')
        # Applying the binary operator '==' (line 678)
        result_eq_1943 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 11), '==', name_1941, str_1942)
        
        # Getting the type of 'self' (line 678)
        self_1944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 31), 'self')
        # Obtaining the member 'debug' of a type (line 678)
        debug_1945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 31), self_1944, 'debug')
        # Applying the binary operator 'and' (line 678)
        result_and_keyword_1946 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 11), 'and', result_eq_1943, debug_1945)
        
        # Testing the type of an if condition (line 678)
        if_condition_1947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 678, 8), result_and_keyword_1946)
        # Assigning a type to the variable 'if_condition_1947' (line 678)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 8), 'if_condition_1947', if_condition_1947)
        # SSA begins for if statement (line 678)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to join(...): (line 679)
        # Getting the type of 'ext_path' (line 679)
        ext_path_1951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 33), 'ext_path', False)
        # Processing the call keyword arguments (line 679)
        kwargs_1952 = {}
        # Getting the type of 'os' (line 679)
        os_1948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 679)
        path_1949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 19), os_1948, 'path')
        # Obtaining the member 'join' of a type (line 679)
        join_1950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 19), path_1949, 'join')
        # Calling join(args, kwargs) (line 679)
        join_call_result_1953 = invoke(stypy.reporting.localization.Localization(__file__, 679, 19), join_1950, *[ext_path_1951], **kwargs_1952)
        
        str_1954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 45), 'str', '_d')
        # Applying the binary operator '+' (line 679)
        result_add_1955 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 19), '+', join_call_result_1953, str_1954)
        
        # Getting the type of 'so_ext' (line 679)
        so_ext_1956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 52), 'so_ext')
        # Applying the binary operator '+' (line 679)
        result_add_1957 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 50), '+', result_add_1955, so_ext_1956)
        
        # Assigning a type to the variable 'stypy_return_type' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 'stypy_return_type', result_add_1957)
        # SSA join for if statement (line 678)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 680)
        # Getting the type of 'ext_path' (line 680)
        ext_path_1961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 29), 'ext_path', False)
        # Processing the call keyword arguments (line 680)
        kwargs_1962 = {}
        # Getting the type of 'os' (line 680)
        os_1958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 680)
        path_1959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 15), os_1958, 'path')
        # Obtaining the member 'join' of a type (line 680)
        join_1960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 15), path_1959, 'join')
        # Calling join(args, kwargs) (line 680)
        join_call_result_1963 = invoke(stypy.reporting.localization.Localization(__file__, 680, 15), join_1960, *[ext_path_1961], **kwargs_1962)
        
        # Getting the type of 'so_ext' (line 680)
        so_ext_1964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 41), 'so_ext')
        # Applying the binary operator '+' (line 680)
        result_add_1965 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 15), '+', join_call_result_1963, so_ext_1964)
        
        # Assigning a type to the variable 'stypy_return_type' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'stypy_return_type', result_add_1965)
        
        # ################# End of 'get_ext_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_ext_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 666)
        stypy_return_type_1966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1966)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_ext_filename'
        return stypy_return_type_1966


    @norecursion
    def get_export_symbols(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_export_symbols'
        module_type_store = module_type_store.open_function_context('get_export_symbols', 682, 4, False)
        # Assigning a type to the variable 'self' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_function_name', 'build_ext.get_export_symbols')
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_param_names_list', ['ext'])
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_export_symbols.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_export_symbols', ['ext'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_export_symbols', localization, ['ext'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_export_symbols(...)' code ##################

        str_1967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, (-1)), 'str', 'Return the list of symbols that a shared extension has to\n        export.  This either uses \'ext.export_symbols\' or, if it\'s not\n        provided, "init" + module_name.  Only relevant on Windows, where\n        the .pyd file (DLL) must export the module "init" function.\n        ')
        
        # Assigning a BinOp to a Name (line 688):
        
        # Assigning a BinOp to a Name (line 688):
        str_1968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 24), 'str', 'init')
        
        # Obtaining the type of the subscript
        int_1969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 53), 'int')
        
        # Call to split(...): (line 688)
        # Processing the call arguments (line 688)
        str_1973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 48), 'str', '.')
        # Processing the call keyword arguments (line 688)
        kwargs_1974 = {}
        # Getting the type of 'ext' (line 688)
        ext_1970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 33), 'ext', False)
        # Obtaining the member 'name' of a type (line 688)
        name_1971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 33), ext_1970, 'name')
        # Obtaining the member 'split' of a type (line 688)
        split_1972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 33), name_1971, 'split')
        # Calling split(args, kwargs) (line 688)
        split_call_result_1975 = invoke(stypy.reporting.localization.Localization(__file__, 688, 33), split_1972, *[str_1973], **kwargs_1974)
        
        # Obtaining the member '__getitem__' of a type (line 688)
        getitem___1976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 33), split_call_result_1975, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 688)
        subscript_call_result_1977 = invoke(stypy.reporting.localization.Localization(__file__, 688, 33), getitem___1976, int_1969)
        
        # Applying the binary operator '+' (line 688)
        result_add_1978 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 24), '+', str_1968, subscript_call_result_1977)
        
        # Assigning a type to the variable 'initfunc_name' (line 688)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'initfunc_name', result_add_1978)
        
        
        # Getting the type of 'initfunc_name' (line 689)
        initfunc_name_1979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 11), 'initfunc_name')
        # Getting the type of 'ext' (line 689)
        ext_1980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 32), 'ext')
        # Obtaining the member 'export_symbols' of a type (line 689)
        export_symbols_1981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 32), ext_1980, 'export_symbols')
        # Applying the binary operator 'notin' (line 689)
        result_contains_1982 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 11), 'notin', initfunc_name_1979, export_symbols_1981)
        
        # Testing the type of an if condition (line 689)
        if_condition_1983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 689, 8), result_contains_1982)
        # Assigning a type to the variable 'if_condition_1983' (line 689)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'if_condition_1983', if_condition_1983)
        # SSA begins for if statement (line 689)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 690)
        # Processing the call arguments (line 690)
        # Getting the type of 'initfunc_name' (line 690)
        initfunc_name_1987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 38), 'initfunc_name', False)
        # Processing the call keyword arguments (line 690)
        kwargs_1988 = {}
        # Getting the type of 'ext' (line 690)
        ext_1984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 12), 'ext', False)
        # Obtaining the member 'export_symbols' of a type (line 690)
        export_symbols_1985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 12), ext_1984, 'export_symbols')
        # Obtaining the member 'append' of a type (line 690)
        append_1986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 12), export_symbols_1985, 'append')
        # Calling append(args, kwargs) (line 690)
        append_call_result_1989 = invoke(stypy.reporting.localization.Localization(__file__, 690, 12), append_1986, *[initfunc_name_1987], **kwargs_1988)
        
        # SSA join for if statement (line 689)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'ext' (line 691)
        ext_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 15), 'ext')
        # Obtaining the member 'export_symbols' of a type (line 691)
        export_symbols_1991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 15), ext_1990, 'export_symbols')
        # Assigning a type to the variable 'stypy_return_type' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'stypy_return_type', export_symbols_1991)
        
        # ################# End of 'get_export_symbols(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_export_symbols' in the type store
        # Getting the type of 'stypy_return_type' (line 682)
        stypy_return_type_1992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_export_symbols'
        return stypy_return_type_1992


    @norecursion
    def get_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_libraries'
        module_type_store = module_type_store.open_function_context('get_libraries', 693, 4, False)
        # Assigning a type to the variable 'self' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_libraries.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_libraries.__dict__.__setitem__('stypy_function_name', 'build_ext.get_libraries')
        build_ext.get_libraries.__dict__.__setitem__('stypy_param_names_list', ['ext'])
        build_ext.get_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_libraries.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_libraries', ['ext'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_libraries', localization, ['ext'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_libraries(...)' code ##################

        str_1993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, (-1)), 'str', "Return the list of libraries to link against when building a\n        shared extension.  On most platforms, this is just 'ext.libraries';\n        on Windows and OS/2, we add the Python library (eg. python20.dll).\n        ")
        
        
        # Getting the type of 'sys' (line 703)
        sys_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 703)
        platform_1995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 11), sys_1994, 'platform')
        str_1996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 27), 'str', 'win32')
        # Applying the binary operator '==' (line 703)
        result_eq_1997 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 11), '==', platform_1995, str_1996)
        
        # Testing the type of an if condition (line 703)
        if_condition_1998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 703, 8), result_eq_1997)
        # Assigning a type to the variable 'if_condition_1998' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'if_condition_1998', if_condition_1998)
        # SSA begins for if statement (line 703)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 704, 12))
        
        # 'from distutils.msvccompiler import MSVCCompiler' statement (line 704)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_1999 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 704, 12), 'distutils.msvccompiler')

        if (type(import_1999) is not StypyTypeError):

            if (import_1999 != 'pyd_module'):
                __import__(import_1999)
                sys_modules_2000 = sys.modules[import_1999]
                import_from_module(stypy.reporting.localization.Localization(__file__, 704, 12), 'distutils.msvccompiler', sys_modules_2000.module_type_store, module_type_store, ['MSVCCompiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 704, 12), __file__, sys_modules_2000, sys_modules_2000.module_type_store, module_type_store)
            else:
                from distutils.msvccompiler import MSVCCompiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 704, 12), 'distutils.msvccompiler', None, module_type_store, ['MSVCCompiler'], [MSVCCompiler])

        else:
            # Assigning a type to the variable 'distutils.msvccompiler' (line 704)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 12), 'distutils.msvccompiler', import_1999)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        
        
        # Call to isinstance(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'self' (line 705)
        self_2002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 30), 'self', False)
        # Obtaining the member 'compiler' of a type (line 705)
        compiler_2003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 30), self_2002, 'compiler')
        # Getting the type of 'MSVCCompiler' (line 705)
        MSVCCompiler_2004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 45), 'MSVCCompiler', False)
        # Processing the call keyword arguments (line 705)
        kwargs_2005 = {}
        # Getting the type of 'isinstance' (line 705)
        isinstance_2001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 705)
        isinstance_call_result_2006 = invoke(stypy.reporting.localization.Localization(__file__, 705, 19), isinstance_2001, *[compiler_2003, MSVCCompiler_2004], **kwargs_2005)
        
        # Applying the 'not' unary operator (line 705)
        result_not__2007 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 15), 'not', isinstance_call_result_2006)
        
        # Testing the type of an if condition (line 705)
        if_condition_2008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 705, 12), result_not__2007)
        # Assigning a type to the variable 'if_condition_2008' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 12), 'if_condition_2008', if_condition_2008)
        # SSA begins for if statement (line 705)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 706):
        
        # Assigning a Str to a Name (line 706):
        str_2009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 27), 'str', 'python%d%d')
        # Assigning a type to the variable 'template' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 16), 'template', str_2009)
        
        # Getting the type of 'self' (line 707)
        self_2010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 19), 'self')
        # Obtaining the member 'debug' of a type (line 707)
        debug_2011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 19), self_2010, 'debug')
        # Testing the type of an if condition (line 707)
        if_condition_2012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 707, 16), debug_2011)
        # Assigning a type to the variable 'if_condition_2012' (line 707)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 16), 'if_condition_2012', if_condition_2012)
        # SSA begins for if statement (line 707)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 708):
        
        # Assigning a BinOp to a Name (line 708):
        # Getting the type of 'template' (line 708)
        template_2013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 31), 'template')
        str_2014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 42), 'str', '_d')
        # Applying the binary operator '+' (line 708)
        result_add_2015 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 31), '+', template_2013, str_2014)
        
        # Assigning a type to the variable 'template' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 20), 'template', result_add_2015)
        # SSA join for if statement (line 707)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 709):
        
        # Assigning a BinOp to a Name (line 709):
        # Getting the type of 'template' (line 709)
        template_2016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 29), 'template')
        
        # Obtaining an instance of the builtin type 'tuple' (line 710)
        tuple_2017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 710)
        # Adding element type (line 710)
        # Getting the type of 'sys' (line 710)
        sys_2018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 24), 'sys')
        # Obtaining the member 'hexversion' of a type (line 710)
        hexversion_2019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 24), sys_2018, 'hexversion')
        int_2020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 42), 'int')
        # Applying the binary operator '>>' (line 710)
        result_rshift_2021 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 24), '>>', hexversion_2019, int_2020)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 24), tuple_2017, result_rshift_2021)
        # Adding element type (line 710)
        # Getting the type of 'sys' (line 710)
        sys_2022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 47), 'sys')
        # Obtaining the member 'hexversion' of a type (line 710)
        hexversion_2023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 47), sys_2022, 'hexversion')
        int_2024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 65), 'int')
        # Applying the binary operator '>>' (line 710)
        result_rshift_2025 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 47), '>>', hexversion_2023, int_2024)
        
        int_2026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 71), 'int')
        # Applying the binary operator '&' (line 710)
        result_and__2027 = python_operator(stypy.reporting.localization.Localization(__file__, 710, 46), '&', result_rshift_2025, int_2026)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 24), tuple_2017, result_and__2027)
        
        # Applying the binary operator '%' (line 709)
        result_mod_2028 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 29), '%', template_2016, tuple_2017)
        
        # Assigning a type to the variable 'pythonlib' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 16), 'pythonlib', result_mod_2028)
        # Getting the type of 'ext' (line 713)
        ext_2029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 23), 'ext')
        # Obtaining the member 'libraries' of a type (line 713)
        libraries_2030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 23), ext_2029, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 713)
        list_2031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 713)
        # Adding element type (line 713)
        # Getting the type of 'pythonlib' (line 713)
        pythonlib_2032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 40), 'pythonlib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 39), list_2031, pythonlib_2032)
        
        # Applying the binary operator '+' (line 713)
        result_add_2033 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 23), '+', libraries_2030, list_2031)
        
        # Assigning a type to the variable 'stypy_return_type' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 16), 'stypy_return_type', result_add_2033)
        # SSA branch for the else part of an if statement (line 705)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'ext' (line 715)
        ext_2034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 23), 'ext')
        # Obtaining the member 'libraries' of a type (line 715)
        libraries_2035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 23), ext_2034, 'libraries')
        # Assigning a type to the variable 'stypy_return_type' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 16), 'stypy_return_type', libraries_2035)
        # SSA join for if statement (line 705)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 703)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sys' (line 716)
        sys_2036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 13), 'sys')
        # Obtaining the member 'platform' of a type (line 716)
        platform_2037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 13), sys_2036, 'platform')
        str_2038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 29), 'str', 'os2emx')
        # Applying the binary operator '==' (line 716)
        result_eq_2039 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 13), '==', platform_2037, str_2038)
        
        # Testing the type of an if condition (line 716)
        if_condition_2040 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 716, 13), result_eq_2039)
        # Assigning a type to the variable 'if_condition_2040' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 13), 'if_condition_2040', if_condition_2040)
        # SSA begins for if statement (line 716)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 719):
        
        # Assigning a Str to a Name (line 719):
        str_2041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 23), 'str', 'python%d%d')
        # Assigning a type to the variable 'template' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'template', str_2041)
        
        # Assigning a BinOp to a Name (line 724):
        
        # Assigning a BinOp to a Name (line 724):
        # Getting the type of 'template' (line 724)
        template_2042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 25), 'template')
        
        # Obtaining an instance of the builtin type 'tuple' (line 725)
        tuple_2043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 725)
        # Adding element type (line 725)
        # Getting the type of 'sys' (line 725)
        sys_2044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 20), 'sys')
        # Obtaining the member 'hexversion' of a type (line 725)
        hexversion_2045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 20), sys_2044, 'hexversion')
        int_2046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 38), 'int')
        # Applying the binary operator '>>' (line 725)
        result_rshift_2047 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 20), '>>', hexversion_2045, int_2046)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 20), tuple_2043, result_rshift_2047)
        # Adding element type (line 725)
        # Getting the type of 'sys' (line 725)
        sys_2048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 43), 'sys')
        # Obtaining the member 'hexversion' of a type (line 725)
        hexversion_2049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 43), sys_2048, 'hexversion')
        int_2050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 61), 'int')
        # Applying the binary operator '>>' (line 725)
        result_rshift_2051 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 43), '>>', hexversion_2049, int_2050)
        
        int_2052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 67), 'int')
        # Applying the binary operator '&' (line 725)
        result_and__2053 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 42), '&', result_rshift_2051, int_2052)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 725, 20), tuple_2043, result_and__2053)
        
        # Applying the binary operator '%' (line 724)
        result_mod_2054 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 25), '%', template_2042, tuple_2043)
        
        # Assigning a type to the variable 'pythonlib' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'pythonlib', result_mod_2054)
        # Getting the type of 'ext' (line 728)
        ext_2055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 19), 'ext')
        # Obtaining the member 'libraries' of a type (line 728)
        libraries_2056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 19), ext_2055, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 728)
        list_2057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 728)
        # Adding element type (line 728)
        # Getting the type of 'pythonlib' (line 728)
        pythonlib_2058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 36), 'pythonlib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 728, 35), list_2057, pythonlib_2058)
        
        # Applying the binary operator '+' (line 728)
        result_add_2059 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 19), '+', libraries_2056, list_2057)
        
        # Assigning a type to the variable 'stypy_return_type' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 12), 'stypy_return_type', result_add_2059)
        # SSA branch for the else part of an if statement (line 716)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_2060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 27), 'int')
        slice_2061 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 729, 13), None, int_2060, None)
        # Getting the type of 'sys' (line 729)
        sys_2062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 13), 'sys')
        # Obtaining the member 'platform' of a type (line 729)
        platform_2063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 13), sys_2062, 'platform')
        # Obtaining the member '__getitem__' of a type (line 729)
        getitem___2064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 13), platform_2063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 729)
        subscript_call_result_2065 = invoke(stypy.reporting.localization.Localization(__file__, 729, 13), getitem___2064, slice_2061)
        
        str_2066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 33), 'str', 'cygwin')
        # Applying the binary operator '==' (line 729)
        result_eq_2067 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 13), '==', subscript_call_result_2065, str_2066)
        
        # Testing the type of an if condition (line 729)
        if_condition_2068 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 13), result_eq_2067)
        # Assigning a type to the variable 'if_condition_2068' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 13), 'if_condition_2068', if_condition_2068)
        # SSA begins for if statement (line 729)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 730):
        
        # Assigning a Str to a Name (line 730):
        str_2069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 23), 'str', 'python%d.%d')
        # Assigning a type to the variable 'template' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'template', str_2069)
        
        # Assigning a BinOp to a Name (line 731):
        
        # Assigning a BinOp to a Name (line 731):
        # Getting the type of 'template' (line 731)
        template_2070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 25), 'template')
        
        # Obtaining an instance of the builtin type 'tuple' (line 732)
        tuple_2071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 732)
        # Adding element type (line 732)
        # Getting the type of 'sys' (line 732)
        sys_2072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 20), 'sys')
        # Obtaining the member 'hexversion' of a type (line 732)
        hexversion_2073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 20), sys_2072, 'hexversion')
        int_2074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 38), 'int')
        # Applying the binary operator '>>' (line 732)
        result_rshift_2075 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 20), '>>', hexversion_2073, int_2074)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 20), tuple_2071, result_rshift_2075)
        # Adding element type (line 732)
        # Getting the type of 'sys' (line 732)
        sys_2076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 43), 'sys')
        # Obtaining the member 'hexversion' of a type (line 732)
        hexversion_2077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 43), sys_2076, 'hexversion')
        int_2078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 61), 'int')
        # Applying the binary operator '>>' (line 732)
        result_rshift_2079 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 43), '>>', hexversion_2077, int_2078)
        
        int_2080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 67), 'int')
        # Applying the binary operator '&' (line 732)
        result_and__2081 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 42), '&', result_rshift_2079, int_2080)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 20), tuple_2071, result_and__2081)
        
        # Applying the binary operator '%' (line 731)
        result_mod_2082 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 25), '%', template_2070, tuple_2071)
        
        # Assigning a type to the variable 'pythonlib' (line 731)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 12), 'pythonlib', result_mod_2082)
        # Getting the type of 'ext' (line 735)
        ext_2083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 19), 'ext')
        # Obtaining the member 'libraries' of a type (line 735)
        libraries_2084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 19), ext_2083, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 735)
        list_2085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 735)
        # Adding element type (line 735)
        # Getting the type of 'pythonlib' (line 735)
        pythonlib_2086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 36), 'pythonlib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 735, 35), list_2085, pythonlib_2086)
        
        # Applying the binary operator '+' (line 735)
        result_add_2087 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 19), '+', libraries_2084, list_2085)
        
        # Assigning a type to the variable 'stypy_return_type' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 12), 'stypy_return_type', result_add_2087)
        # SSA branch for the else part of an if statement (line 729)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_2088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 27), 'int')
        slice_2089 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 736, 13), None, int_2088, None)
        # Getting the type of 'sys' (line 736)
        sys_2090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 13), 'sys')
        # Obtaining the member 'platform' of a type (line 736)
        platform_2091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 13), sys_2090, 'platform')
        # Obtaining the member '__getitem__' of a type (line 736)
        getitem___2092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 13), platform_2091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 736)
        subscript_call_result_2093 = invoke(stypy.reporting.localization.Localization(__file__, 736, 13), getitem___2092, slice_2089)
        
        str_2094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 33), 'str', 'atheos')
        # Applying the binary operator '==' (line 736)
        result_eq_2095 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 13), '==', subscript_call_result_2093, str_2094)
        
        # Testing the type of an if condition (line 736)
        if_condition_2096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 13), result_eq_2095)
        # Assigning a type to the variable 'if_condition_2096' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 13), 'if_condition_2096', if_condition_2096)
        # SSA begins for if statement (line 736)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 737, 12))
        
        # 'from distutils import sysconfig' statement (line 737)
        try:
            from distutils import sysconfig

        except:
            sysconfig = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 737, 12), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])
        
        
        # Assigning a Str to a Name (line 739):
        
        # Assigning a Str to a Name (line 739):
        str_2097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 23), 'str', 'python%d.%d')
        # Assigning a type to the variable 'template' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'template', str_2097)
        
        # Assigning a BinOp to a Name (line 740):
        
        # Assigning a BinOp to a Name (line 740):
        # Getting the type of 'template' (line 740)
        template_2098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 25), 'template')
        
        # Obtaining an instance of the builtin type 'tuple' (line 741)
        tuple_2099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 741)
        # Adding element type (line 741)
        # Getting the type of 'sys' (line 741)
        sys_2100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 20), 'sys')
        # Obtaining the member 'hexversion' of a type (line 741)
        hexversion_2101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 20), sys_2100, 'hexversion')
        int_2102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 38), 'int')
        # Applying the binary operator '>>' (line 741)
        result_rshift_2103 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 20), '>>', hexversion_2101, int_2102)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 741, 20), tuple_2099, result_rshift_2103)
        # Adding element type (line 741)
        # Getting the type of 'sys' (line 741)
        sys_2104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 43), 'sys')
        # Obtaining the member 'hexversion' of a type (line 741)
        hexversion_2105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 43), sys_2104, 'hexversion')
        int_2106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 61), 'int')
        # Applying the binary operator '>>' (line 741)
        result_rshift_2107 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 43), '>>', hexversion_2105, int_2106)
        
        int_2108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 67), 'int')
        # Applying the binary operator '&' (line 741)
        result_and__2109 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 42), '&', result_rshift_2107, int_2108)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 741, 20), tuple_2099, result_and__2109)
        
        # Applying the binary operator '%' (line 740)
        result_mod_2110 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 25), '%', template_2098, tuple_2099)
        
        # Assigning a type to the variable 'pythonlib' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'pythonlib', result_mod_2110)
        
        # Assigning a List to a Name (line 743):
        
        # Assigning a List to a Name (line 743):
        
        # Obtaining an instance of the builtin type 'list' (line 743)
        list_2111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 743)
        
        # Assigning a type to the variable 'extra' (line 743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 12), 'extra', list_2111)
        
        
        # Call to split(...): (line 744)
        # Processing the call keyword arguments (line 744)
        kwargs_2118 = {}
        
        # Call to get_config_var(...): (line 744)
        # Processing the call arguments (line 744)
        str_2114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 48), 'str', 'SHLIBS')
        # Processing the call keyword arguments (line 744)
        kwargs_2115 = {}
        # Getting the type of 'sysconfig' (line 744)
        sysconfig_2112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 23), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 744)
        get_config_var_2113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 23), sysconfig_2112, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 744)
        get_config_var_call_result_2116 = invoke(stypy.reporting.localization.Localization(__file__, 744, 23), get_config_var_2113, *[str_2114], **kwargs_2115)
        
        # Obtaining the member 'split' of a type (line 744)
        split_2117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 23), get_config_var_call_result_2116, 'split')
        # Calling split(args, kwargs) (line 744)
        split_call_result_2119 = invoke(stypy.reporting.localization.Localization(__file__, 744, 23), split_2117, *[], **kwargs_2118)
        
        # Testing the type of a for loop iterable (line 744)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 744, 12), split_call_result_2119)
        # Getting the type of the for loop variable (line 744)
        for_loop_var_2120 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 744, 12), split_call_result_2119)
        # Assigning a type to the variable 'lib' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 12), 'lib', for_loop_var_2120)
        # SSA begins for a for statement (line 744)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to startswith(...): (line 745)
        # Processing the call arguments (line 745)
        str_2123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 34), 'str', '-l')
        # Processing the call keyword arguments (line 745)
        kwargs_2124 = {}
        # Getting the type of 'lib' (line 745)
        lib_2121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 19), 'lib', False)
        # Obtaining the member 'startswith' of a type (line 745)
        startswith_2122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 19), lib_2121, 'startswith')
        # Calling startswith(args, kwargs) (line 745)
        startswith_call_result_2125 = invoke(stypy.reporting.localization.Localization(__file__, 745, 19), startswith_2122, *[str_2123], **kwargs_2124)
        
        # Testing the type of an if condition (line 745)
        if_condition_2126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 745, 16), startswith_call_result_2125)
        # Assigning a type to the variable 'if_condition_2126' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 16), 'if_condition_2126', if_condition_2126)
        # SSA begins for if statement (line 745)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 746)
        # Processing the call arguments (line 746)
        
        # Obtaining the type of the subscript
        int_2129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 37), 'int')
        slice_2130 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 746, 33), int_2129, None, None)
        # Getting the type of 'lib' (line 746)
        lib_2131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 33), 'lib', False)
        # Obtaining the member '__getitem__' of a type (line 746)
        getitem___2132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 33), lib_2131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 746)
        subscript_call_result_2133 = invoke(stypy.reporting.localization.Localization(__file__, 746, 33), getitem___2132, slice_2130)
        
        # Processing the call keyword arguments (line 746)
        kwargs_2134 = {}
        # Getting the type of 'extra' (line 746)
        extra_2127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 20), 'extra', False)
        # Obtaining the member 'append' of a type (line 746)
        append_2128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 20), extra_2127, 'append')
        # Calling append(args, kwargs) (line 746)
        append_call_result_2135 = invoke(stypy.reporting.localization.Localization(__file__, 746, 20), append_2128, *[subscript_call_result_2133], **kwargs_2134)
        
        # SSA branch for the else part of an if statement (line 745)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 748)
        # Processing the call arguments (line 748)
        # Getting the type of 'lib' (line 748)
        lib_2138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 33), 'lib', False)
        # Processing the call keyword arguments (line 748)
        kwargs_2139 = {}
        # Getting the type of 'extra' (line 748)
        extra_2136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 20), 'extra', False)
        # Obtaining the member 'append' of a type (line 748)
        append_2137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 20), extra_2136, 'append')
        # Calling append(args, kwargs) (line 748)
        append_call_result_2140 = invoke(stypy.reporting.localization.Localization(__file__, 748, 20), append_2137, *[lib_2138], **kwargs_2139)
        
        # SSA join for if statement (line 745)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'ext' (line 751)
        ext_2141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 19), 'ext')
        # Obtaining the member 'libraries' of a type (line 751)
        libraries_2142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 19), ext_2141, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 751)
        list_2143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 751)
        # Adding element type (line 751)
        # Getting the type of 'pythonlib' (line 751)
        pythonlib_2144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 36), 'pythonlib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 35), list_2143, pythonlib_2144)
        # Adding element type (line 751)
        str_2145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 47), 'str', 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 35), list_2143, str_2145)
        
        # Applying the binary operator '+' (line 751)
        result_add_2146 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 19), '+', libraries_2142, list_2143)
        
        # Getting the type of 'extra' (line 751)
        extra_2147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 54), 'extra')
        # Applying the binary operator '+' (line 751)
        result_add_2148 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 52), '+', result_add_2146, extra_2147)
        
        # Assigning a type to the variable 'stypy_return_type' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 12), 'stypy_return_type', result_add_2148)
        # SSA branch for the else part of an if statement (line 736)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'sys' (line 753)
        sys_2149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 13), 'sys')
        # Obtaining the member 'platform' of a type (line 753)
        platform_2150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 13), sys_2149, 'platform')
        str_2151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 29), 'str', 'darwin')
        # Applying the binary operator '==' (line 753)
        result_eq_2152 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 13), '==', platform_2150, str_2151)
        
        # Testing the type of an if condition (line 753)
        if_condition_2153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 753, 13), result_eq_2152)
        # Assigning a type to the variable 'if_condition_2153' (line 753)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 13), 'if_condition_2153', if_condition_2153)
        # SSA begins for if statement (line 753)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ext' (line 755)
        ext_2154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 19), 'ext')
        # Obtaining the member 'libraries' of a type (line 755)
        libraries_2155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 19), ext_2154, 'libraries')
        # Assigning a type to the variable 'stypy_return_type' (line 755)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 12), 'stypy_return_type', libraries_2155)
        # SSA branch for the else part of an if statement (line 753)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_2156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 27), 'int')
        slice_2157 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 756, 13), None, int_2156, None)
        # Getting the type of 'sys' (line 756)
        sys_2158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 13), 'sys')
        # Obtaining the member 'platform' of a type (line 756)
        platform_2159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 13), sys_2158, 'platform')
        # Obtaining the member '__getitem__' of a type (line 756)
        getitem___2160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 13), platform_2159, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 756)
        subscript_call_result_2161 = invoke(stypy.reporting.localization.Localization(__file__, 756, 13), getitem___2160, slice_2157)
        
        str_2162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 33), 'str', 'aix')
        # Applying the binary operator '==' (line 756)
        result_eq_2163 = python_operator(stypy.reporting.localization.Localization(__file__, 756, 13), '==', subscript_call_result_2161, str_2162)
        
        # Testing the type of an if condition (line 756)
        if_condition_2164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 756, 13), result_eq_2163)
        # Assigning a type to the variable 'if_condition_2164' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 13), 'if_condition_2164', if_condition_2164)
        # SSA begins for if statement (line 756)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ext' (line 758)
        ext_2165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 19), 'ext')
        # Obtaining the member 'libraries' of a type (line 758)
        libraries_2166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 19), ext_2165, 'libraries')
        # Assigning a type to the variable 'stypy_return_type' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 12), 'stypy_return_type', libraries_2166)
        # SSA branch for the else part of an if statement (line 756)
        module_type_store.open_ssa_branch('else')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 760, 12))
        
        # 'from distutils import sysconfig' statement (line 760)
        try:
            from distutils import sysconfig

        except:
            sysconfig = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 760, 12), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])
        
        
        
        # Call to get_config_var(...): (line 761)
        # Processing the call arguments (line 761)
        str_2169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 40), 'str', 'Py_ENABLE_SHARED')
        # Processing the call keyword arguments (line 761)
        kwargs_2170 = {}
        # Getting the type of 'sysconfig' (line 761)
        sysconfig_2167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 15), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 761)
        get_config_var_2168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 15), sysconfig_2167, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 761)
        get_config_var_call_result_2171 = invoke(stypy.reporting.localization.Localization(__file__, 761, 15), get_config_var_2168, *[str_2169], **kwargs_2170)
        
        # Testing the type of an if condition (line 761)
        if_condition_2172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 761, 12), get_config_var_call_result_2171)
        # Assigning a type to the variable 'if_condition_2172' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'if_condition_2172', if_condition_2172)
        # SSA begins for if statement (line 761)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 762):
        
        # Assigning a Str to a Name (line 762):
        str_2173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 27), 'str', 'python%d.%d')
        # Assigning a type to the variable 'template' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 16), 'template', str_2173)
        
        # Assigning a BinOp to a Name (line 763):
        
        # Assigning a BinOp to a Name (line 763):
        # Getting the type of 'template' (line 763)
        template_2174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 29), 'template')
        
        # Obtaining an instance of the builtin type 'tuple' (line 764)
        tuple_2175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 764)
        # Adding element type (line 764)
        # Getting the type of 'sys' (line 764)
        sys_2176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 30), 'sys')
        # Obtaining the member 'hexversion' of a type (line 764)
        hexversion_2177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 30), sys_2176, 'hexversion')
        int_2178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 48), 'int')
        # Applying the binary operator '>>' (line 764)
        result_rshift_2179 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 30), '>>', hexversion_2177, int_2178)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 30), tuple_2175, result_rshift_2179)
        # Adding element type (line 764)
        # Getting the type of 'sys' (line 764)
        sys_2180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 53), 'sys')
        # Obtaining the member 'hexversion' of a type (line 764)
        hexversion_2181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 53), sys_2180, 'hexversion')
        int_2182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 71), 'int')
        # Applying the binary operator '>>' (line 764)
        result_rshift_2183 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 53), '>>', hexversion_2181, int_2182)
        
        int_2184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 77), 'int')
        # Applying the binary operator '&' (line 764)
        result_and__2185 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 52), '&', result_rshift_2183, int_2184)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 30), tuple_2175, result_and__2185)
        
        # Applying the binary operator '%' (line 763)
        result_mod_2186 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 29), '%', template_2174, tuple_2175)
        
        # Assigning a type to the variable 'pythonlib' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), 'pythonlib', result_mod_2186)
        # Getting the type of 'ext' (line 765)
        ext_2187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 23), 'ext')
        # Obtaining the member 'libraries' of a type (line 765)
        libraries_2188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 23), ext_2187, 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 765)
        list_2189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 765)
        # Adding element type (line 765)
        # Getting the type of 'pythonlib' (line 765)
        pythonlib_2190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 40), 'pythonlib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 765, 39), list_2189, pythonlib_2190)
        
        # Applying the binary operator '+' (line 765)
        result_add_2191 = python_operator(stypy.reporting.localization.Localization(__file__, 765, 23), '+', libraries_2188, list_2189)
        
        # Assigning a type to the variable 'stypy_return_type' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'stypy_return_type', result_add_2191)
        # SSA branch for the else part of an if statement (line 761)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'ext' (line 767)
        ext_2192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 23), 'ext')
        # Obtaining the member 'libraries' of a type (line 767)
        libraries_2193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 23), ext_2192, 'libraries')
        # Assigning a type to the variable 'stypy_return_type' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'stypy_return_type', libraries_2193)
        # SSA join for if statement (line 761)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 756)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 753)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 736)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 729)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 716)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 703)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 693)
        stypy_return_type_2194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2194)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_libraries'
        return stypy_return_type_2194


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 37, 0, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build_ext' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'build_ext', build_ext)

# Assigning a Str to a Name (line 39):
str_2195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'str', 'build C/C++ extensions (compile/link to build directory)')
# Getting the type of 'build_ext'
build_ext_2196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_2196, 'description', str_2195)

# Assigning a BinOp to a Name (line 59):
str_2197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 13), 'str', " (separated by '%s')")
# Getting the type of 'os' (line 59)
os_2198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 38), 'os')
# Obtaining the member 'pathsep' of a type (line 59)
pathsep_2199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 38), os_2198, 'pathsep')
# Applying the binary operator '%' (line 59)
result_mod_2200 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 13), '%', str_2197, pathsep_2199)

# Getting the type of 'build_ext'
build_ext_2201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Setting the type of the member 'sep_by' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_2201, 'sep_by', result_mod_2200)

# Assigning a List to a Name (line 60):

# Obtaining an instance of the builtin type 'list' (line 60)
list_2202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 61)
tuple_2203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 61)
# Adding element type (line 61)
str_2204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 9), 'str', 'build-lib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 9), tuple_2203, str_2204)
# Adding element type (line 61)
str_2205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 9), tuple_2203, str_2205)
# Adding element type (line 61)
str_2206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 9), 'str', 'directory for compiled extension modules')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 9), tuple_2203, str_2206)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2203)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 63)
tuple_2207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 63)
# Adding element type (line 63)
str_2208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 9), 'str', 'build-temp=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 9), tuple_2207, str_2208)
# Adding element type (line 63)
str_2209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'str', 't')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 9), tuple_2207, str_2209)
# Adding element type (line 63)
str_2210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 9), 'str', 'directory for temporary files (build by-products)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 9), tuple_2207, str_2210)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2207)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 65)
tuple_2211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 65)
# Adding element type (line 65)
str_2212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 9), 'str', 'plat-name=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_2211, str_2212)
# Adding element type (line 65)
str_2213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'str', 'p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_2211, str_2213)
# Adding element type (line 65)
str_2214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 9), 'str', 'platform name to cross-compile for, if supported (default: %s)')

# Call to get_platform(...): (line 67)
# Processing the call keyword arguments (line 67)
kwargs_2216 = {}
# Getting the type of 'get_platform' (line 67)
get_platform_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 27), 'get_platform', False)
# Calling get_platform(args, kwargs) (line 67)
get_platform_call_result_2217 = invoke(stypy.reporting.localization.Localization(__file__, 67, 27), get_platform_2215, *[], **kwargs_2216)

# Applying the binary operator '%' (line 66)
result_mod_2218 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 9), '%', str_2214, get_platform_call_result_2217)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 9), tuple_2211, result_mod_2218)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2211)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_2219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)
# Adding element type (line 68)
str_2220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 9), 'str', 'inplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_2219, str_2220)
# Adding element type (line 68)
str_2221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'str', 'i')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_2219, str_2221)
# Adding element type (line 68)
str_2222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 9), 'str', 'ignore build-lib and put compiled extensions into the source ')
str_2223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 9), 'str', 'directory alongside your pure Python modules')
# Applying the binary operator '+' (line 69)
result_add_2224 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 9), '+', str_2222, str_2223)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 9), tuple_2219, result_add_2224)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2219)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 71)
tuple_2225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 71)
# Adding element type (line 71)
str_2226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 9), 'str', 'include-dirs=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 9), tuple_2225, str_2226)
# Adding element type (line 71)
str_2227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'str', 'I')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 9), tuple_2225, str_2227)
# Adding element type (line 71)
str_2228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 9), 'str', 'list of directories to search for header files')
# Getting the type of 'build_ext'
build_ext_2229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Obtaining the member 'sep_by' of a type
sep_by_2230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_2229, 'sep_by')
# Applying the binary operator '+' (line 72)
result_add_2231 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 9), '+', str_2228, sep_by_2230)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 9), tuple_2225, result_add_2231)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2225)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_2232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
str_2233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 9), 'str', 'define=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 9), tuple_2232, str_2233)
# Adding element type (line 73)
str_2234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'str', 'D')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 9), tuple_2232, str_2234)
# Adding element type (line 73)
str_2235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 9), 'str', 'C preprocessor macros to define')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 9), tuple_2232, str_2235)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2232)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 75)
tuple_2236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 75)
# Adding element type (line 75)
str_2237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 9), 'str', 'undef=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 9), tuple_2236, str_2237)
# Adding element type (line 75)
str_2238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'str', 'U')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 9), tuple_2236, str_2238)
# Adding element type (line 75)
str_2239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 9), 'str', 'C preprocessor macros to undefine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 9), tuple_2236, str_2239)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2236)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_2240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)
str_2241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 9), 'str', 'libraries=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 9), tuple_2240, str_2241)
# Adding element type (line 77)
str_2242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 23), 'str', 'l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 9), tuple_2240, str_2242)
# Adding element type (line 77)
str_2243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 9), 'str', 'external C libraries to link with')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 9), tuple_2240, str_2243)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2240)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 79)
tuple_2244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 79)
# Adding element type (line 79)
str_2245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 9), 'str', 'library-dirs=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 9), tuple_2244, str_2245)
# Adding element type (line 79)
str_2246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'str', 'L')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 9), tuple_2244, str_2246)
# Adding element type (line 79)
str_2247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 9), 'str', 'directories to search for external C libraries')
# Getting the type of 'build_ext'
build_ext_2248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Obtaining the member 'sep_by' of a type
sep_by_2249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_2248, 'sep_by')
# Applying the binary operator '+' (line 80)
result_add_2250 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 9), '+', str_2247, sep_by_2249)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 9), tuple_2244, result_add_2250)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2244)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 81)
tuple_2251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 81)
# Adding element type (line 81)
str_2252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 9), 'str', 'rpath=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 9), tuple_2251, str_2252)
# Adding element type (line 81)
str_2253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'str', 'R')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 9), tuple_2251, str_2253)
# Adding element type (line 81)
str_2254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 9), 'str', 'directories to search for shared C libraries at runtime')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 9), tuple_2251, str_2254)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2251)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 83)
tuple_2255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 83)
# Adding element type (line 83)
str_2256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 9), 'str', 'link-objects=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 9), tuple_2255, str_2256)
# Adding element type (line 83)
str_2257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'str', 'O')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 9), tuple_2255, str_2257)
# Adding element type (line 83)
str_2258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 9), 'str', 'extra explicit link objects to include in the link')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 9), tuple_2255, str_2258)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2255)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 85)
tuple_2259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 85)
# Adding element type (line 85)
str_2260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 9), 'str', 'debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 9), tuple_2259, str_2260)
# Adding element type (line 85)
str_2261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'str', 'g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 9), tuple_2259, str_2261)
# Adding element type (line 85)
str_2262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 9), 'str', 'compile/link with debugging information')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 9), tuple_2259, str_2262)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2259)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 87)
tuple_2263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 87)
# Adding element type (line 87)
str_2264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 9), tuple_2263, str_2264)
# Adding element type (line 87)
str_2265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 9), tuple_2263, str_2265)
# Adding element type (line 87)
str_2266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 9), 'str', 'forcibly build everything (ignore file timestamps)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 9), tuple_2263, str_2266)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2263)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 89)
tuple_2267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 89)
# Adding element type (line 89)
str_2268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 9), 'str', 'compiler=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 9), tuple_2267, str_2268)
# Adding element type (line 89)
str_2269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 9), tuple_2267, str_2269)
# Adding element type (line 89)
str_2270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 9), 'str', 'specify the compiler type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 9), tuple_2267, str_2270)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2267)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 91)
tuple_2271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 91)
# Adding element type (line 91)
str_2272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 9), 'str', 'swig-cpp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 9), tuple_2271, str_2272)
# Adding element type (line 91)
# Getting the type of 'None' (line 91)
None_2273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 9), tuple_2271, None_2273)
# Adding element type (line 91)
str_2274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 9), 'str', 'make SWIG create C++ files (default is C)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 9), tuple_2271, str_2274)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2271)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 93)
tuple_2275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 93)
# Adding element type (line 93)
str_2276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 9), 'str', 'swig-opts=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 9), tuple_2275, str_2276)
# Adding element type (line 93)
# Getting the type of 'None' (line 93)
None_2277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 9), tuple_2275, None_2277)
# Adding element type (line 93)
str_2278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 9), 'str', 'list of SWIG command line options')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 9), tuple_2275, str_2278)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2275)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 95)
tuple_2279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 95)
# Adding element type (line 95)
str_2280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 9), 'str', 'swig=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 9), tuple_2279, str_2280)
# Adding element type (line 95)
# Getting the type of 'None' (line 95)
None_2281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 9), tuple_2279, None_2281)
# Adding element type (line 95)
str_2282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 9), 'str', 'path to the SWIG executable')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 9), tuple_2279, str_2282)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2279)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'tuple' (line 97)
tuple_2283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 97)
# Adding element type (line 97)
str_2284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 9), 'str', 'user')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 9), tuple_2283, str_2284)
# Adding element type (line 97)
# Getting the type of 'None' (line 97)
None_2285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 9), tuple_2283, None_2285)
# Adding element type (line 97)
str_2286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 9), 'str', 'add user include, library and rpath')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 9), tuple_2283, str_2286)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), list_2202, tuple_2283)

# Getting the type of 'build_ext'
build_ext_2287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_2287, 'user_options', list_2202)

# Assigning a List to a Name (line 101):

# Obtaining an instance of the builtin type 'list' (line 101)
list_2288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 101)
# Adding element type (line 101)
str_2289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'str', 'inplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), list_2288, str_2289)
# Adding element type (line 101)
str_2290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 34), 'str', 'debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), list_2288, str_2290)
# Adding element type (line 101)
str_2291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 43), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), list_2288, str_2291)
# Adding element type (line 101)
str_2292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 52), 'str', 'swig-cpp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), list_2288, str_2292)
# Adding element type (line 101)
str_2293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 64), 'str', 'user')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), list_2288, str_2293)

# Getting the type of 'build_ext'
build_ext_2294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_2294, 'boolean_options', list_2288)

# Assigning a List to a Name (line 103):

# Obtaining an instance of the builtin type 'list' (line 103)
list_2295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 103)
# Adding element type (line 103)

# Obtaining an instance of the builtin type 'tuple' (line 104)
tuple_2296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 104)
# Adding element type (line 104)
str_2297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'str', 'help-compiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_2296, str_2297)
# Adding element type (line 104)
# Getting the type of 'None' (line 104)
None_2298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_2296, None_2298)
# Adding element type (line 104)
str_2299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 9), 'str', 'list available compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_2296, str_2299)
# Adding element type (line 104)
# Getting the type of 'show_compilers' (line 105)
show_compilers_2300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 37), 'show_compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), tuple_2296, show_compilers_2300)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), list_2295, tuple_2296)

# Getting the type of 'build_ext'
build_ext_2301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_ext')
# Setting the type of the member 'help_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_ext_2301, 'help_options', list_2295)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
